"""Fixed-vector robustness scorecard; never selects model parameters or weights."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.validation.compare import logit,spearman
from src.validation.private_sim import RankedAUC
from scripts.assemble_sol_foundation_test import verify_primary,checked_probability
from scripts.audit_sol_state import pair_diagnostic


def paired_bootstrap(y,vectors,reference='aux10',repeats=200,seed=20261010):
    scorers={name:RankedAUC(y,p) for name,p in vectors.items()}
    rng=np.random.default_rng(seed);deltas={name:[] for name in vectors if name!=reference}
    for _ in range(repeats):
        weights=np.bincount(rng.integers(0,len(y),size=len(y)),minlength=len(y))
        baseline=scorers[reference].auc(weights)
        for name,rows in deltas.items():rows.append(scorers[name].auc(weights)-baseline)
    return {'seed':seed,'repeats':repeats,'unit':'paired rows resampled with replacement',
        'limitation':'Conditional on saved validation predictions; does not simulate refitting or unknown distribution shift',
        'reference':reference,'comparisons':{n:{'mean':float(np.mean(v)),
            'percentile_95_interval':np.quantile(v,[.025,.975]).tolist(),'positive_fraction':float(np.mean(np.array(v)>0))}
            for n,v in deltas.items()}}


def correlations(a,b,y=None):
    result={'logit':float(np.corrcoef(logit(a),logit(b))[0,1]),'spearman':spearman(a,b)}
    if y is not None:result['probability_residual']=float(np.corrcoef(y-a,y-b)[0,1])
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-report',default='reports/sol_route_aux10_primary.json')
    ap.add_argument('--test-certificate',default='reports/sol_route_aux10_primary_test_certificate.json')
    ap.add_argument('--confirmation',action='append',required=True)
    ap.add_argument('--simulation',default='reports/sol_private_sim_primary.json')
    ap.add_argument('--seed-diagnostic-report',required=True)
    ap.add_argument('--name',default='sol_route_aux10_scorecard')
    args=ap.parse_args()
    output=REPORTS/(args.name+'.json')
    if output.exists():raise FileExistsError('Preserve the existing scorecard')
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();test_ids=te.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    primary,vectors,folds=verify_primary(args.primary_report,ids,y,data_hash)
    cert=json.loads(Path(args.test_certificate).read_text(encoding='utf-8'))
    assert cert['status']=='CERTIFIED_FIXED_OOF_TEST_REQUIRES_ROBUSTNESS_SCORECARD'
    assert cert['primary_report_sha256']==file_sha256(args.primary_report)
    assert cert['data_sha256']==data_hash and cert['ordered_test_ids_sha256']==arr_sha256(test_ids)
    assert cert['oof_sha256']=={n:arr_sha256(p) for n,p in vectors.items()}
    assert cert['weights']=={'route':.5,'equal_logit_auxiliary10':.5}
    root=Path(cert['test_artifact_root'])
    tests={n:checked_probability(root/(n+'.npy'),root/'test_ids.npy',h,test_ids) for n,h in cert['test_sha256'].items()}
    confirms=[]
    for path in args.confirmation:
        r=json.loads(Path(path).read_text(encoding='utf-8'))
        assert r['scheme'] in ('shadow','block10') and r['scope']==f'full {r["scheme"]} OOF'
        assert r['data_sha256']==data_hash and r['ordered_train_ids_sha256']==arr_sha256(ids)
        assert r['predeclared_hypothesis_sha256']==primary['predeclared_hypothesis_sha256']
        assert r['verdict']=='INDEPENDENT_CONFIRMATION_GATE_PASS'
        assert r['admission_gate_against_clean_auxiliary_stack']['admit']
        confirms.append({'report':path,'sha256':file_sha256(path),'scheme':r['scheme'],
            'mean_paired_gain':r['mean_paired_gain_vs_strict_aux10'],'paired_se':r['paired_fold_se'],
            'positive_folds':r['positive_folds'],'pooled_auc':r['pooled_oof_auc']})
    assert any(r['scheme']=='block10' for r in confirms), 'Finalist contract requires block10 confirmation'
    sim=json.loads(Path(args.simulation).read_text(encoding='utf-8'))
    assert sim['portfolio_contract_sha256']==file_sha256(args.primary_report)
    assert sim['reference']=='route_aux10' and sim['population']==len(te) and sim['seed']==20261010
    assert sim['fold_sha256']==arr_sha256(folds) and sim['ids_sha256']==arr_sha256(ids)
    assert all(file_sha256(p)==h for p,h in sim['source_sha256'].items())
    assert sim['oof_prediction_sha256']['route_aux10']==arr_sha256(logit(vectors['candidate']))
    assert sim['oof_prediction_sha256']['strict_aux10']==arr_sha256(logit(vectors['aux10']))
    seed_path=Path(args.seed_diagnostic_report);seed_record=json.loads(seed_path.read_text(encoding='utf-8'))
    assert seed_record['status']=='FIXED_SEED_SENSITIVITY_DIAGNOSTIC_ONLY'
    assert seed_record['seed_diagnostic'] and seed_record['scheme']=='primary' and seed_record['fold']==0 and seed_record['seed']==1202
    assert seed_record['data_sha256']==data_hash and seed_record['fold_sha256']==arr_sha256(folds)
    va=np.flatnonzero(folds==0);fi=np.flatnonzero(folds!=0)
    assert seed_record['fit_ids_sha256']==arr_sha256(ids[fi]) and seed_record['validation_ids_sha256']==arr_sha256(ids[va])
    assert seed_record['train_rows']==len(fi) and seed_record['validation_rows']==len(va) and seed_record['full_intended_population']
    assert all(file_sha256(p)==h for p,h in seed_record['source_sha256'].items())
    assert seed_record['seed_protocol_sha256']==file_sha256('research/sol_seed_diagnostic.md')
    first=next(r for r in primary['folds'] if r['fold']==0)
    source=json.loads((REPORTS/first['foundation']['tag']/'route_f0.json').read_text(encoding='utf-8'))
    assert seed_record['params']==dict(source['params'],random_state=1202)
    seed_root=ARTIFACTS/seed_path.parent.name
    seed_route=checked_probability(seed_root/'route_f0.npy',seed_root/'ids_f0.npy',seed_record['prediction_sha256'],ids[va])
    assert abs(float(roc_auc_score(y[va],seed_route))-seed_record['auc'])<1e-14
    seed_candidate=(1/(1+np.exp(-(logit(seed_route)+logit(vectors['aux10'][va]))/2))).astype('float32')
    seed_control=(1/(1+np.exp(-(logit(vectors['route'][va])+logit(vectors['aux10'][va]))/2))).astype('float32')
    recomposed_gap=float(np.abs(seed_control-vectors['candidate'][va]).max())
    assert recomposed_gap<=2e-6
    seed_evidence={'report_sha256':file_sha256(seed_path),'fold':0,'seeds':[1201,1202],
        'scope':'One fixed seed replay; no seed selection or five-fold seed-robustness claim',
        'route_auc_seed1202_minus1201':float(roc_auc_score(y[va],seed_route)-roc_auc_score(y[va],vectors['route'][va])),
        'candidate_auc_seed1202_minus1201':float(roc_auc_score(y[va],seed_candidate)-roc_auc_score(y[va],seed_control)),
        'route_correlation':correlations(vectors['route'][va],seed_route,y[va]),
        'candidate_correlation':correlations(seed_control,seed_candidate,y[va]),
        'candidate_pair_rescue_damage':pair_diagnostic(y[va],seed_candidate,seed_control),
        'common_f32_auxiliary_recomposition_max_gap':recomposed_gap}
    comparisons={}
    for name in ('aux10','route'):
        comparisons[name]={'pooled_auc':float(roc_auc_score(y,vectors[name])),
            'delta_candidate_minus_component':float(roc_auc_score(y,vectors['candidate'])-roc_auc_score(y,vectors[name])),
            'oof_correlations':correlations(vectors['candidate'],vectors[name],y),
            'test_correlations':correlations(tests['candidate'],tests[name]),
            'pair_rescue_damage':pair_diagnostic(y,vectors['candidate'],vectors[name])}
    component_corr={'oof':correlations(vectors['route'],vectors['aux10'],y),
        'test':correlations(tests['route'],tests['aux10'])}
    scorecard={'utc':datetime.now(timezone.utc).isoformat(),'git':git_commit(),
        'status':'COMPLETE_FIXED_PORTFOLIO_ROBUSTNESS_EVIDENCE_REQUIRES_SCIENTIFIC_DECISION',
        'primary_report_sha256':file_sha256(args.primary_report),'test_certificate_sha256':file_sha256(args.test_certificate),
        'simulation_report_sha256':file_sha256(args.simulation),'confirmations':confirms,
        'candidate_auc':float(roc_auc_score(y,vectors['candidate'])),
        'paired_fold_deltas':[r['delta_vs_strict_aux10'] for r in primary['folds']],
        'mean_paired_gain':primary['mean_paired_gain_vs_strict_aux10'],'paired_se':primary['paired_fold_se'],
        'admission_gate':primary['admission_gate_against_clean_auxiliary_stack'],
        'comparisons':comparisons,'method_component_correlations':component_corr,
        'paired_bootstrap':paired_bootstrap(y,vectors),
        'private_simulation':sim['candidates']['strict_aux10'],
        'model_seed_stability':seed_evidence,
        'data_sha256':data_hash,'fold_sha256':arr_sha256(folds),'oof_sha256':cert['oof_sha256'],'test_sha256':cert['test_sha256'],
        'source_sha256':{p:file_sha256(p) for p in ('scripts/score_sol_foundation.py','scripts/assemble_sol_foundation_test.py','src/validation/private_sim.py')},
        'limitations':['Private simulation conditions on trained OOF vectors; it is not a probability of winning.',
            'The auxiliary reference is a fully corrected fixed ten-role stack; legacy59 is not certified.']}
    save_json(scorecard,output)
    print(json.dumps({k:scorecard[k] for k in ('status','candidate_auc','mean_paired_gain','paired_se')},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
