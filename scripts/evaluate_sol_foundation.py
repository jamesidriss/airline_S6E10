"""Verify and score the frozen route/auxiliary10 portfolio without weight fitting."""
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
from src.submission import store
from src.validation.folds import get_scheme
from src.validation.compare import logit
from src.ensemble.lab import admission_gate
from scripts.audit_sol_state import reconstruct_v5
from scripts.replay_sol_classical import frozen_roles,resolved_params,PROTOCOL


def foundation(k,tags,ids,folds,y,data_hash,scheme='primary'):
    matches=[]
    for tag in tags:
        path=REPORTS/tag/f'route_f{k}.json'
        if path.exists():
            r=json.loads(path.read_text(encoding='utf-8'))
            if 'prediction_sha256' in r:
                matches.append((tag,path,r))
    assert matches and len({r['prediction_sha256'] for _,_,r in matches})==1, 'Missing or ambiguous route result'
    tag,path,r=matches[0]
    va=np.flatnonzero(folds==k); fi=np.flatnonzero(folds!=k)
    assert r['full_intended_population'] and not r['timing_only'] and r['arm']=='route'
    assert r['train_rows']==len(fi) and r['validation_rows']==len(va)
    assert r['fold_sha256']==arr_sha256(folds) and r['fit_ids_sha256']==arr_sha256(ids[fi])
    assert r['validation_ids_sha256']==arr_sha256(ids[va])
    assert r['data_sha256']==data_hash
    if scheme=='primary':
        for file,key in [('scripts/run_sol_tabpfn.py','source_sha256'),
            ('src/models/windows_attention.py','backend_source_sha256'),
            ('src/models/pointwise_inference.py','decoder_chunk_source_sha256'),
            ('src/models/resource_guard.py','resource_guard_source_sha256')]:
            assert file_sha256(file)==r[key]
    else:
        assert r['scheme']==scheme
        assert all(file_sha256(p)==h for p,h in r['source_sha256'].items())
        assert 'scripts/run_sol_tabpfn_confirm.py' in r['source_sha256']
    p=np.load(ARTIFACTS/tag/f'route_f{k}.npy')
    assert arr_sha256(p)==r['prediction_sha256'] and p.shape==(len(va),)
    assert np.array_equal(np.load(ARTIFACTS/tag/f'ids_f{k}.npy'),ids[va])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert abs(float(roc_auc_score(y[va],p))-r['auc'])<1e-14
    return p,r,{'tag':tag,'report_sha256':file_sha256(path),'prediction_sha256':arr_sha256(p)}


def classical(k,tag,roles,ids,folds,y,data_hash):
    va=np.flatnonzero(folds==k); fi=np.flatnonzero(folds!=k)
    total=np.zeros(len(va)); proofs=[]
    for role in roles:
        stem=f'{role["member"]}_f{k}'
        path=REPORTS/tag/(stem+'.json'); r=json.loads(path.read_text(encoding='utf-8')); c=r['contract']
        assert c['protocol']==PROTOCOL and c['role']==role and c['params']==resolved_params(role)
        assert c['fold']==k and c['fold_sha256']==arr_sha256(folds)
        assert c['data_sha256']==data_hash
        assert c['outer_fit_ids_sha256']==arr_sha256(ids[fi]) and c['validation_ids_sha256']==arr_sha256(ids[va])
        assert all(file_sha256(s)==h for s,h in c['source_sha256'].items())
        p=np.load(ARTIFACTS/tag/(stem+'.npy'))
        assert p.shape==(len(va),) and arr_sha256(p)==r['prediction_sha256']
        assert np.array_equal(np.load(ARTIFACTS/tag/(stem+'_ids.npy')),ids[va])
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert abs(float(roc_auc_score(y[va],p))-r['auc'])<1e-14
        if k==0 and role['scheme']=='primary':
            old_path=REPORTS/'sol_repair'/f'{role["aux_arm"]}_A0_f0.json'
            old=json.loads(old_path.read_text(encoding='utf-8'))
            assert c['feature_fit_sha256']==old['contract']['feature_fit_sha256']
            assert c['feature_val_sha256']==old['contract']['feature_val_sha256']
            control=np.load(ARTIFACTS/'sol_repair'/f'{role["aux_arm"]}_A0_f0.npy')
            gap=float(np.abs(p-control).max())
            assert gap<=2e-6, 'Corrected auxiliary control did not reproduce; stop portfolio scoring'
            assert r['n_trees']==old['n_trees'], 'Selected capacity control changed'
        else:
            gap=None
        total+=logit(p)
        proofs.append({'member':role['member'],'report_sha256':file_sha256(path),
            'prediction_sha256':arr_sha256(p),'fold0_control_max_gap':gap})
    return total/len(roles),proofs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folds',default='0,1')
    ap.add_argument('--scheme',choices=['primary','shadow'],default='primary')
    ap.add_argument('--classical-tag',default='sol_clean_aux10_isolated')
    ap.add_argument('--foundation-tags',default='sol_tabpfn35_route,sol_tabpfn35_route_reserve')
    ap.add_argument('--name',default='sol_route_aux10_twofold')
    args=ap.parse_args()
    output=REPORTS/(args.name+'.json'); root=ARTIFACTS/args.name
    if output.exists() or root.exists():
        raise FileExistsError('Preserve the existing portfolio result; use a new name')
    ks=list(map(int,args.folds.split(','))); assert len(set(ks))==len(ks) and set(ks)<=set(range(5))
    roles=[r for r in frozen_roles() if r['aux_arm']]; assert len(roles)==10
    if args.scheme=='shadow':
        for role in roles:
            role.update(source_scheme=role['scheme'],scheme='shadow')
    tr,_=load_cached_parquet(); y=tr.satisfaction.to_numpy(dtype='int8'); ids=tr.id.to_numpy()
    folds=get_scheme(args.scheme,y,ids).folds
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    legacy=reconstruct_v5(y,folds)[1] if args.scheme=='primary' else None
    records=[]; first=None; verified=[]
    for k in ks:
        p,r,proof=foundation(k,args.foundation_tags.split(','),ids,folds,y,data_hash,args.scheme)
        if first is None:
            first=r
        for field in ('params','seed','feature_names','library_version','batch_size','precision','icl_bf16',
            'inference_chunk_cells','inference_col_chunk_size','decoder_inplace_gelu','reuse_query_output'):
            assert r[field]==first[field], 'Frozen foundation recipe changed between folds'
        aux,certificates=classical(k,args.classical_tag,roles,ids,folds,y,data_hash)
        pred=(1/(1+np.exp(-(logit(p)+aux)/2))).astype('float32')
        base=(1/(1+np.exp(-aux))).astype('float32')
        verified.append((k,pred,base,p,proof,certificates))
    root.mkdir()
    for k,pred,base,route,proof,certificates in verified:
        va=np.flatnonzero(folds==k)
        np.save(root/f'candidate_f{k}.npy',pred); np.save(root/f'ids_f{k}.npy',ids[va])
        np.save(root/f'aux10_f{k}.npy',base)
        ca=float(roc_auc_score(y[va],pred)); ba=float(roc_auc_score(y[va],base))
        la=float(roc_auc_score(y[va],legacy[va])) if legacy is not None else None
        records.append({'fold':k,'candidate_auc':ca,'strict_aux10_auc':ba,'legacy_v5_auc_diagnostic':la,
            'route_auc':float(roc_auc_score(y[va],route)),'strict_aux10_sha256':arr_sha256(base),
            'delta_vs_strict_aux10':ca-ba,'delta_vs_legacy_v5_diagnostic':ca-la if la is not None else None,
            'candidate_sha256':arr_sha256(pred),'ids_sha256':arr_sha256(ids[va]),
            'logit_correlation_vs_legacy_v5':float(np.corrcoef(logit(pred),legacy[va])[0,1]) if legacy is not None else None,
            'logit_correlation_vs_strict_aux10':float(np.corrcoef(logit(pred),logit(base))[0,1]),
            'foundation':proof,'classical_certificates':certificates})
    deltas=np.array([r['delta_vs_strict_aux10'] for r in records])
    full=set(ks)==set(range(5))
    gate=admission_gate(args.name,[r['strict_aux10_auc'] for r in records],[r['candidate_auc'] for r in records]) if full else None
    report={'hypothesis':'One frozen 50/50 mean-logit vote: route and equal-logit auxiliary10',
        'predeclared_hypothesis_sha256':file_sha256('research/sol_foundation_portfolio.md'),
        'scope':f'full {args.scheme} OOF' if full else f'selected {args.scheme} folds; not full OOF',
        'scheme':args.scheme,
        'folds':records,'mean_paired_gain_vs_strict_aux10':float(deltas.mean()),
        'paired_fold_se':float(deltas.std(ddof=1)/np.sqrt(len(deltas))) if len(deltas)>1 else None,
        'positive_folds':int((deltas>0).sum()),'admission_gate_against_clean_auxiliary_stack':gate,
        'verdict':('INDEPENDENT_CONFIRMATION_GATE_PASS' if args.scheme=='shadow' and gate and gate['admit'] else
            'INDEPENDENT_CONFIRMATION_GATE_FAIL' if args.scheme=='shadow' and full else
            'INDEPENDENT_CONFIRMATION_INCOMPLETE' if args.scheme=='shadow' else
            'FULL_PRIMARY_GATE_PASS_REQUIRES_TEST_AND_CONFIRMATION' if gate and gate['admit'] else
            'REPLICATED_PROMOTE_NEXT_FOLD' if not full and len(ks)>=2 and (deltas>0).all() and deltas.mean()>=5e-5 else
            'NO_PROMOTION'),
        'limitation':'Legacy v5 is diagnostic only; final selection also requires test, private simulation and independent confirmation'}
    if full:
        oof=np.full(len(y),np.nan,dtype='float32')
        base=np.full(len(y),np.nan,dtype='float32')
        route_oof=np.full(len(y),np.nan,dtype='float32')
        for k,pred,base_fold,route,_,_ in verified:
            oof[folds==k]=pred
            base[folds==k]=base_fold
            route_oof[folds==k]=route
        assert np.isfinite(oof).all() and np.isfinite(base).all() and np.isfinite(route_oof).all()
        np.save(root/'candidate_oof.npy',oof); np.save(root/'aux10_oof.npy',base); np.save(root/'train_ids.npy',ids)
        np.save(root/'route_oof.npy',route_oof)
        report.update(pooled_oof_auc=float(roc_auc_score(y,oof)),strict_aux10_pooled_auc=float(roc_auc_score(y,base)),
            route_pooled_auc=float(roc_auc_score(y,route_oof)),route_oof_sha256=arr_sha256(route_oof),
            strict_aux10_oof_sha256=arr_sha256(base),oof_sha256=arr_sha256(oof),ordered_train_ids_sha256=arr_sha256(ids))
        store.save(args.name,oof,None,fold_scheme=args.scheme,meta={'family':'blend','auc':report['pooled_oof_auc'],
            'training_protocol':'sol_route_aux10_fixed_50_50_v1','test_policy':'pending; no finalist claim',
            'geometry':report['hypothesis'],'ordered_train_ids_sha256':arr_sha256(ids)})
    save_json(report,output)
    ledger=Path('experiments/ledger.jsonl')
    existing={r.get('id',r.get('exp_id')) for r in map(json.loads,ledger.read_text(encoding='utf-8').splitlines())}
    if args.name not in existing:
        with ledger.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps({'exp_id':args.name,'ts':datetime.now(timezone.utc).isoformat(),'git':git_commit(),
                'kind':'INDEPENDENT_CONFIRMATION' if args.scheme=='shadow' else 'PREDECLARED_METHOD_BLEND','scope':report['scope'],
                'fold_aucs':[r['candidate_auc'] for r in records],
                'paired_fold_deltas':[r['delta_vs_strict_aux10'] for r in records],
                'legacy_paired_fold_deltas_diagnostic':[r['delta_vs_legacy_v5_diagnostic'] for r in records],
                'corr_with_champion':[r['logit_correlation_vs_legacy_v5'] if args.scheme=='primary' else r['logit_correlation_vs_strict_aux10'] for r in records],
                'seed':{'route':first['seed'],'classical':{r['member']:r['seed'] for r in roles}},
                'params':{'route':first['params'],'classical_roles':roles,'weights':{'route':.5,'equal_logit_auxiliary10':.5}},
                'data_hash':data_hash,'fold_hash':arr_sha256(folds),'report':str(output),
                'report_sha256':file_sha256(output),'verdict':report['verdict']})+'\n')
    print(json.dumps({k:report[k] for k in ('scope','mean_paired_gain_vs_strict_aux10','positive_folds','verdict')},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
