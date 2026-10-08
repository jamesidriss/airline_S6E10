"""Verify frozen expected-rating TabPFN predictions and staged admission."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.compare import logit
from src.ensemble.lab import admission_gate
from scripts.assemble_sol_foundation_test import verify_primary, checked_probability
from scripts.run_sol_tabpfn import frames
from scripts.sol_phase15_auxpfn import expected_values
from scripts.score_sol_foundation import paired_bootstrap, correlations
from scripts.audit_sol_state import pair_diagnostic


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',required=True);ap.add_argument('--name',required=True)
    args=ap.parse_args();ks=list(map(int,args.folds.split(',')));assert ks==list(range(len(ks))) and len(ks)<=5
    path=REPORTS/(args.name+'.json');root=ARTIFACTS/args.name
    if path.exists() or root.exists():raise FileExistsError('Preserve previous evidence')
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    dh={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    _,base,folds=verify_primary('reports/sol_route_aux10_primary_final.json',ids,y,dh)
    x,_,names,cats,maps=frames(tr,te,True)
    source_path=REPORTS/'sol_tabpfn35_route/route_f0.json';source=json.loads(source_path.read_text())
    records=[];oof=np.full(len(y),np.nan,dtype='float32');foundation=oof.copy();root.mkdir()
    for k in ks:
        report_path=REPORTS/'sol_phase15_auxpfn'/f'auxpfn_f{k}.json';r=json.loads(report_path.read_text());c=r['contract']
        assert 'auc' in r and not c['timing_only'] and c['scheme']=='primary' and c['fold']==k and c['seed']==1201
        assert c['data_sha256']==dh and c['fold_sha256']==arr_sha256(folds)
        fi,va=np.flatnonzero(folds!=k),np.flatnonzero(folds==k)
        assert c['fit_ids_sha256']==arr_sha256(ids[fi]) and c['apply_ids_sha256']==arr_sha256(ids[va])
        assert c['train_rows']==len(fi) and c['apply_rows']==len(va) and not np.intersect1d(ids[fi],ids[va]).size
        assert c['params']==source['params'] and c['checkpoint']==source['checkpoint']
        assert c['arm']=='route_plus13_auxEV' and c['route_reference_sha256']==file_sha256(source_path)
        assert c['scope_sha256']==file_sha256('research/sol_phase15_scope_20261008.json')
        assert c['decision_sha256']==file_sha256('reports/sol_phase15_third_branch_decision.json')
        assert file_sha256(c['params']['model_path'])==c['checkpoint']['checkpoint_sha256']
        assert c['categorical_indices']==cats and c['label_free_category_maps']==maps
        assert c['feature_names']==names+[f'aux_ev_{j}' for j in range(13)]
        assert c['weights']=={'new_tabpfn':.125,'v6':.875}
        assert all(file_sha256(p)==h for p,h in c['source_sha256'].items())
        assert c['library_versions']=={p:version(p) for p in c['library_versions']}
        assert c['protocol_sha256']==file_sha256('research/sol_phase15_auxpfn_protocol.md')
        control_path=REPORTS/'sol_clean_aux10_primary'/f'z3_cat_d8_s2_f{k}.json'
        assert c['control_report_sha256']==file_sha256(control_path)
        control=json.loads(control_path.read_text())
        a,b,proof=expected_values(control['contract']['aux'],x[fi],x[va],names,ids[fi],ids[va],arr_sha256(folds))
        assert proof==c['auxiliary']
        assert arr_sha256(np.column_stack([x[fi],a]))==c['feature_fit_sha256']
        assert arr_sha256(np.column_stack([x[va],b]))==c['feature_apply_sha256']
        p=checked_probability(ARTIFACTS/'sol_phase15_auxpfn'/f'auxpfn_f{k}.npy',
                              ARTIFACTS/'sol_phase15_auxpfn'/f'auxpfn_f{k}_ids.npy',r['prediction_sha256'],ids[va])
        candidate=expit(.125*logit(p)+.875*logit(base['candidate'][va])).astype('float32')
        saved=checked_probability(ARTIFACTS/'sol_phase15_auxpfn'/f'portfolio_f{k}.npy',
                                  ARTIFACTS/'sol_phase15_auxpfn'/f'auxpfn_f{k}_ids.npy',r['portfolio_prediction_sha256'],ids[va])
        assert np.array_equal(candidate,saved)
        assert abs(roc_auc_score(y[va],p)-r['auc'])<1e-14 and abs(roc_auc_score(y[va],candidate)-r['candidate_auc'])<1e-14
        append_control=expit(.125*logit(base['route'][va])+.875*logit(base['candidate'][va])).astype('float32')
        v6_auc=float(roc_auc_score(y[va],base['candidate'][va]));auc=float(roc_auc_score(y[va],candidate))
        assert abs((auc-v6_auc)-r['v6_gain'])<1e-14
        assert abs((roc_auc_score(y[va],p)-roc_auc_score(y[va],base['route'][va]))-r['standalone_gain_vs_route'])<1e-14
        records.append({'fold':k,'v6_auc':v6_auc,'candidate_auc':auc,'delta_vs_v6':auc-v6_auc,
                        'standalone_gain_vs_route':r['standalone_gain_vs_route'],'standalone_auc':r['auc'],
                        'matched_append_route_control_auc':float(roc_auc_score(y[va],append_control)),
                        'gain_vs_matched_append_control':float(auc-roc_auc_score(y[va],append_control)),
                        'correlation_vs_v6':correlations(candidate,base['candidate'][va]),
                        'pair_rescue_damage':pair_diagnostic(y[va],candidate,base['candidate'][va]),
                        'source_report_sha256':file_sha256(report_path),'candidate_sha256':arr_sha256(candidate),'ids_sha256':arr_sha256(ids[va])})
        np.save(root/f'candidate_f{k}.npy',candidate);np.save(root/f'ids_f{k}.npy',ids[va]);oof[va]=candidate;foundation[va]=p
    ds=np.array([r['delta_vs_v6'] for r in records]);gate=None
    if len(ks)==1:
        verdict='DISCOVERY_PASS_REQUIRES_REPLICATION' if records[0]['standalone_gain_vs_route']>=5e-5 or ds[0]>=1e-5 else 'DISCOVERY_FAIL_STOP'
    elif len(ks)==2:
        verdict='POSITIVE_PORTFOLIO_REPLICATION' if (ds>0).all() else 'PORTFOLIO_NOT_REPLICATED_STOP'
    elif len(ks)==3:
        verdict='THREE_FOLD_CONTINUATION_PASS' if (ds>0).all() and ds.mean()>=1.5e-5 else 'THREE_FOLD_CONTINUATION_FAIL_STOP'
    else:
        assert len(ks)==5
        gate=admission_gate('expected-rating TabPFN vs v6',[r['v6_auc'] for r in records],[r['candidate_auc'] for r in records])
        verdict='PRIMARY_GATE_PASS_REQUIRES_INDEPENDENT_AND_TEST' if gate['admit'] else 'PRIMARY_GATE_FAIL_NO_PROMOTION'
    mask=np.isin(folds,ks)
    np.save(root/'candidate_selected.npy',oof[mask]);np.save(root/'ids_selected.npy',ids[mask])
    if len(ks)==5:
        np.save(root/'candidate_oof.npy',oof);np.save(root/'foundation_oof.npy',foundation);np.save(root/'train_ids.npy',ids)
    result={'utc':datetime.now(timezone.utc).isoformat(),'git':git_commit(),'scope':'full primary OOF' if len(ks)==5 else 'selected primary folds; not full OOF',
            'folds':records,'weights':{'new_tabpfn':.125,'v6':.875},'data_sha256':dh,'fold_sha256':arr_sha256(folds),
            'ordered_selected_ids_sha256':arr_sha256(ids[mask]),'candidate_sha256':arr_sha256(oof[mask]),
            'selected_pooled_auc':float(roc_auc_score(y[mask],oof[mask])),
            'selected_pooled_delta_vs_v6':float(roc_auc_score(y[mask],oof[mask])-roc_auc_score(y[mask],base['candidate'][mask])),
            'mean_paired_gain':float(ds.mean()),'paired_se':float(ds.std(ddof=1)/np.sqrt(len(ds))) if len(ds)>1 else None,
            'positive_folds':int((ds>0).sum()),'admission_gate':gate,'verdict':verdict,
            'paired_bootstrap':paired_bootstrap(y[mask],{'v6':base['candidate'][mask],'candidate':oof[mask]},reference='v6'),
            'source_sha256':{'scripts/evaluate_sol_phase15_auxpfn.py':file_sha256(__file__)},
            'matched_append_control_protocol_sha256':file_sha256('research/sol_phase15_auxpfn_append_control.json')}
    save_json(result,path)
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8') as ledger:
        ledger.write(json.dumps({'exp_id':args.name,'ts':result['utc'],'kind':'VERIFIED_EXPECTED_RATING_PFN','report':str(path),
                                'report_sha256':file_sha256(path),'paired_fold_deltas':ds.tolist(),
                                'corr_with_champion':[r['correlation_vs_v6'] for r in records],'verdict':verdict})+'\n')
    print(json.dumps({k:result[k] for k in ('scope','selected_pooled_auc','selected_pooled_delta_vs_v6','mean_paired_gain','verdict')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
