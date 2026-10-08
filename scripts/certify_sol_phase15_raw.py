"""Certify the fixed raw/route candidate's independent check or full test vector."""
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.assemble_sol_foundation_test import checked_probability, verify_primary
from scripts.evaluate_sol_foundation import foundation, classical
from scripts.replay_sol_classical import frozen_roles
from scripts.score_sol_foundation import paired_bootstrap, correlations
from scripts.audit_sol_state import pair_diagnostic
from scripts.audit_sol_phase15 import evidence
from scripts.run_sol_tabpfn import frames


def verify_context(path, ids, y, folds, apply_ids, data_hash, mode, k):
    r=json.loads(Path(path).read_text());c=r['contract'];fi=np.flatnonzero(folds!=k)
    assert c['mode']==mode and c['arm']=='raw' and c['fold']==k and c['seed']==1201
    assert c['train_rows']==len(fi) and c['apply_rows']==len(apply_ids)
    assert c['fit_ids_sha256']==arr_sha256(ids[fi]) and c['apply_ids_sha256']==arr_sha256(apply_ids)
    assert not np.intersect1d(ids[fi],apply_ids).size
    assert c['data_sha256']==data_hash and c['fold_sha256']==arr_sha256(folds)
    assert c['scheme']==('shadow' if mode=='shadow' else 'primary')
    assert c['protocol_sha256']==file_sha256('research/sol_phase15_raw_protocol.md')
    assert c['scope_sha256']==file_sha256('research/sol_phase15_scope_20261008.json')
    assert all(file_sha256(p)==h for p,h in c['source_sha256'].items())
    assert c['library_versions']=={p:version(p) for p in c['library_versions']}
    source_path=REPORTS/'sol_tabpfn35_predict_guard/raw_f0.json';source=json.loads(source_path.read_text())
    assert c['source_reference_sha256']==file_sha256(source_path) and c['params']==source['params']
    assert c['checkpoint']==source['checkpoint'] and c['feature_names']==source['feature_names']
    assert file_sha256(c['params']['model_path'])==c['checkpoint']['checkpoint_sha256']
    tag=Path(path).parent.name
    p=checked_probability(ARTIFACTS/tag/f'{mode}_f{k}.npy',ARTIFACTS/tag/f'ids_{mode}_f{k}.npy',r['prediction_sha256'],apply_ids)
    if mode=='shadow':assert abs(float(roc_auc_score(y[folds==k],p))-r['auc'])<1e-14
    else:
        proof=c['cv_proof'];cv_path=REPORTS/proof['tag']/f'raw_f{k}.json';cv=json.loads(cv_path.read_text())
        assert file_sha256(cv_path)==proof['report_sha256'] and cv['prediction_sha256']==proof['prediction_sha256']
        assert c['feature_fit_sha256']==cv['feature_fit_sha256'] and c['fit_ids_sha256']==cv['fit_ids_sha256']
    return p,{'report':str(path),'report_sha256':file_sha256(path),'prediction_sha256':arr_sha256(p),'contract':c}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['shadow','test'],required=True)
    ap.add_argument('--raw-tag',required=True);ap.add_argument('--route-tag');ap.add_argument('--aux-tag')
    ap.add_argument('--primary',default='reports/sol_phase15_raw_primary.json');ap.add_argument('--name',required=True)
    args=ap.parse_args();path=REPORTS/(args.name+'.json');root=ARTIFACTS/args.name
    if path.exists() or root.exists():raise FileExistsError('Preserve prior certificate')
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();test_ids=te.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    scope=json.loads(Path('research/sol_phase15_scope_20261008.json').read_text());root.mkdir()
    x,xt,names,cats,maps=frames(tr,te,False)
    if args.mode=='shadow':
        assert args.route_tag and args.aux_tag
        folds=get_scheme('shadow',y,ids).folds;k=2;va=np.flatnonzero(folds==k)
        raw,proof=verify_context(REPORTS/args.raw_tag/f'shadow_f{k}.json',ids,y,folds,ids[va],data_hash,'shadow',k)
        c=proof['contract']
        assert c['feature_fit_sha256']==arr_sha256(x[folds!=k]) and c['feature_apply_sha256']==arr_sha256(x[va])
        assert c['feature_names']==names and c['params']['categorical_features_indices']==cats and c['label_free_category_maps']==maps
        route,_,route_proof=foundation(k,[args.route_tag],ids,folds,y,data_hash,'shadow')
        roles=[dict(r,source_scheme=r['scheme'],scheme='shadow') for r in frozen_roles() if r['aux_arm']]
        aux,aux_proofs=classical(k,args.aux_tag,roles,ids,folds,y,data_hash)
        base=expit(.5*logit(route)+.5*aux).astype('float32')
        candidate=expit(.375*logit(route)+.125*logit(raw)+.5*aux).astype('float32')
        metrics=evidence(y[va],np.zeros(len(va),dtype=int),candidate,base,tr.iloc[va],scope)
        metrics['paired_bootstrap']=paired_bootstrap(y[va],{'v6':base,'candidate':candidate},reference='v6')
        np.save(root/'candidate_f2.npy',candidate);np.save(root/'v6_f2.npy',base);np.save(root/'ids_f2.npy',ids[va])
        result={'mode':'shadow','scope':'one predeclared shadow fold2; not full shadow OOF','fold':2,
                'data_sha256':data_hash,'fold_sha256':arr_sha256(folds),'raw_proof':proof,'route_proof':route_proof,
                'auxiliary_proofs':aux_proofs,'candidate_auc':metrics['pooled_auc'],'v6_auc':float(roc_auc_score(y[va],base)),
                'delta_vs_v6':metrics['pooled_delta_vs_v6'],'metrics':metrics,
                'verdict':'INDEPENDENT_FROZEN_CHECK_PASS' if metrics['pooled_delta_vs_v6']>=1e-5 else 'INDEPENDENT_CHECK_FAILED_NO_PERFORMANCE_SUBMISSION'}
    else:
        ppath=Path(args.primary);p=json.loads(ppath.read_text())
        assert p['scope']=='full primary OOF' and p['weights']=={'route':.375,'raw':.125,'aux10':.5}
        ds=np.array([r['delta_vs_v6'] for r in p['folds']]);se=float(ds.std(ddof=1)/np.sqrt(5))
        assert (ds>0).sum()>=4 and ds.mean()>=1.5e-5 and ds.mean()>=2.5*se
        assert p['selected_pooled_delta_vs_v6']>=3e-5, 'Performance submission needs material pooled gain'
        baseline,_,folds=verify_primary('reports/sol_route_aux10_primary_final.json',ids,y,data_hash)
        cert_path=REPORTS/'sol_route_aux10_primary_final_test_certificate.json';cert=json.loads(cert_path.read_text())
        assert cert['data_sha256']==data_hash and cert['ordered_test_ids_sha256']==arr_sha256(test_ids)
        assert cert['primary_report_sha256']==file_sha256('reports/sol_route_aux10_primary_final.json')
        oldroot=Path(cert['test_artifact_root'])
        old={n:checked_probability(oldroot/(n+'.npy'),oldroot/'test_ids.npy',h,test_ids) for n,h in cert['test_sha256'].items()}
        total=np.zeros(len(test_ids));proofs=[]
        for k in range(5):
            raw,proof=verify_context(REPORTS/args.raw_tag/f'test_f{k}.json',ids,y,folds,test_ids,data_hash,'test',k)
            c=proof['contract']
            assert c['feature_fit_sha256']==arr_sha256(x[folds!=k]) and c['feature_apply_sha256']==arr_sha256(xt)
            assert c['feature_names']==names and c['params']['categorical_features_indices']==cats and c['label_free_category_maps']==maps
            total+=raw/5;proofs.append(proof)
        raw=total.astype('float32');candidate=expit(.375*logit(old['route'])+.125*logit(raw)+.5*logit(old['aux10'])).astype('float32')
        assert candidate.shape==(len(test_ids),) and np.isfinite(candidate).all()
        np.save(root/'candidate.npy',candidate);np.save(root/'raw.npy',raw);np.save(root/'test_ids.npy',test_ids)
        result={'mode':'test','status':'CERTIFIED_ALL_RAW_TEST_CONTEXTS_FROZEN_METHOD_WEIGHTS',
            'data_sha256':data_hash,'fold_sha256':arr_sha256(folds),'primary_report_sha256':file_sha256(ppath),
            'v6_test_certificate_sha256':file_sha256(cert_path),'raw_context_proofs':proofs,'test_artifact_root':str(root),
            'test_rows':len(test_ids),'ordered_test_ids_sha256':arr_sha256(test_ids),
            'test_sha256':{'candidate':arr_sha256(candidate),'raw':arr_sha256(raw)},
            'test_correlation_vs_v6':correlations(candidate,old['candidate']),
            'test_policy':'Equal probability average within each TabPFN representation across5 primary FIT contexts; ten exact auxiliary full-label refits; frozen three-method logit blend'}
    result.update(utc=datetime.now(timezone.utc).isoformat(),git=git_commit(),weights={'route':.375,'raw':.125,'aux10':.5},
                  scope_sha256=file_sha256('research/sol_phase15_scope_20261008.json'),
                  source_sha256={p:file_sha256(p) for p in ('scripts/certify_sol_phase15_raw.py','scripts/run_sol_phase15_raw_context.py')})
    save_json(result,path)
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'exp_id':args.name,'kind':'FROZEN_INDEPENDENT_CHECK' if args.mode=='shadow' else 'FULL_TEST_CERTIFICATE',
                           'ts':result['utc'],'report':str(path),'report_sha256':file_sha256(path),
                           'paired_fold_deltas':[result['delta_vs_v6']] if args.mode=='shadow' else None,
                           'corr_with_champion':result.get('test_correlation_vs_v6',result.get('metrics',{}).get('oof_correlation_vs_v6')),
                           'verdict':result.get('verdict',result.get('status'))})+'\n')
    print(json.dumps({k:result[k] for k in ('mode','scope','candidate_auc','delta_vs_v6','verdict','status','test_rows','test_correlation_vs_v6') if k in result}))
    return 0


if __name__=='__main__':raise SystemExit(main())
