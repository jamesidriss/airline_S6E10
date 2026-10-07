"""Replay the frozen 47 classical v5 roles with strict TE and consistent refits.

This repairs contracts; it neither selects members by their new AUC nor changes
the historical role weights. Original immutable validation schemes are retained.
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.features.view import ViewBuilder
from src.submission import store
from src.validation.folds import get_scheme
from scripts.run_views import _inner_es_split, _fit_lgbm_es, _fit_xgb_es, _fit_cat_es
from scripts.run_views import _fit_full_predict_lgbm, _fit_full_predict_xgb, _fit_full_predict_cat
from scripts.run_phase12 import CHAMPION_PARAMS, CAT_PARAMS
from scripts.run_phase13b import SLOTS, fit_slot
from scripts.run_phase14 import XGB_BASE, CAT_BASE, fit_xgb, fit_cat
from scripts.run_sol_a import model_spec
from scripts.sol_aux_cache import probability_cache
from src.features.aux_distribution import signatures
from scripts.run_phase13 import RATINGS
from scripts.native_cat import default_cat_cols

PROTOCOL = 'sol_clean_classical_v1_strict_te_exact_refit'


def frozen_roles():
    members = json.loads((REPORTS/'finalist_v3_final.json').read_text(encoding='utf-8'))['members']
    index = store._load_index()
    aux = {model_spec(a)['member']: (a, model_spec(a)) for a in list(SLOTS)+['X0','X1','X2','C1']}
    roles = []
    for m in members:
        if m['family'] not in ('lgbm','xgb','cat'):
            continue
        role = {'member': m['exp_id'], 'family': m['family'], 'view': m['featureset'],
                'scheme': index[m['exp_id']]['fold_scheme'], 'seed': m.get('seed',1),
                'overrides': m.get('params',{}), 'aux_arm': None}
        if role['member'] in aux:
            arm, spec = aux[role['member']]
            role.update(aux_arm=arm, seed=spec['seed'], overrides=spec['params'])
        role['es_seed_policy'] = '1' if role['aux_arm'] else ('seed+fold' if role['member'].startswith(('prod5_','ogm_')) else 'seed')
        role['te_inner_seed_policy'] = '0' if role['member'].startswith(('prod5_','ogm_')) and not role['aux_arm'] else 'fold'
        roles.append(role)
    assert len(roles)==47 and sum(r['aux_arm'] is not None for r in roles)==10
    return roles


def resolved_params(role):
    family, seed = role['family'], role['seed']
    if family=='lgbm':
        p = dict(CHAMPION_PARAMS, extra_trees=False)
        if role['aux_arm']:
            p.update(CAT_PARAMS)
        p.update(role['overrides'])
        p.update(random_state=seed, bagging_seed=seed+1, feature_fraction_seed=seed+2)
    elif family=='xgb':
        p = dict(XGB_BASE, **role['overrides'], random_state=seed)
    else:
        p = dict(CAT_BASE, **role['overrides'], random_seed=seed)
        if role['aux_arm']:
            p['boosting_type']='Plain'
    return p


def add_expected_values(x, apply, names, fit_ids, val_ids, fold_hash, fold, scheme='primary'):
    pf,pv,cache=probability_cache(x,apply,names,fit_ids,val_ids,fold_hash,label=f'clean fold{fold}')
    positions=[names.index(c) for c in RATINGS]
    fresh_a,_=signatures(pf,x[:,positions],0)
    fresh_b,_=signatures(pv,apply[:,positions],0)
    gap=None
    if scheme=='primary':
        fit_path, val_path = REPORTS/f'p13b_auxfit_f{fold}.npy', REPORTS/f'p13b_auxval_f{fold}.npy'
        a,b = np.load(fit_path),np.load(val_path)
        assert a.shape==(len(x),13) and b.shape==(len(apply),13)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        gap=max(float(np.abs(a-fresh_a).max()),float(np.abs(b-fresh_b).max()))
        assert gap<=1e-10,'Historical auxiliary EV control failed; do not interpret downstream fits'
    else:
        assert scheme=='shadow', 'Only frozen shadow confirmation is supported'
        a,b=fresh_a,fresh_b
    return np.column_stack([x,a]),np.column_stack([apply,b]),{
        'fit_expected_values_sha256':arr_sha256(a),'validation_expected_values_sha256':arr_sha256(b),
        'recipe':'250 rounds, three rating-stratified inner folds, seed1; predicted rating excluded',
        'independent_ev_max_gap':gap,'upstream_fingerprint':cache['fingerprint'],
        'upstream_contract':cache['contract']}


def guarded_fit(root,contract,path,fingerprint,function):
    from src.models.resource_guard import inference_guard,ResourcePreflightError
    start=time.monotonic()
    try:
        with inference_guard(root,contract,min_available_gib=4):
            return function()
    except Exception as error:
        import torch
        resource=isinstance(error,(ResourcePreflightError,torch.cuda.OutOfMemoryError))
        save_json({'contract':contract,'fingerprint':fingerprint,'git':git_commit(),
            'status':'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE',
            'seconds':time.monotonic()-start,'error':repr(error)},path)
        raise


def complete_role(role,tr,te,y,ids,folds,root,reports,params,refit_test,sources,data_hash):
    ks=sorted(set(folds.tolist()))
    paths=[reports/f'{role["member"]}_f{k}.json' for k in ks]
    if not all(p.exists() for p in paths):
        if refit_test:
            raise RuntimeError('Every immutable fold is required before a test refit')
        return
    oof=np.full(len(y),np.nan,dtype='float32'); counts=[]; fold_records=[]
    for k,path in zip(ks,paths):
        r=json.loads(path.read_text(encoding='utf-8')); c=r['contract']
        assert c['role']==role and c['params']==params and c['source_sha256']==sources
        assert c['data_sha256']==data_hash and c['fold_sha256']==arr_sha256(folds)
        va=np.flatnonzero(folds==k)
        p=np.load(root/(path.stem+'.npy'))
        assert arr_sha256(p)==r['prediction_sha256'] and arr_sha256(ids[va])==c['validation_ids_sha256']
        assert np.array_equal(np.load(root/(path.stem+'_ids.npy')),ids[va])
        oof[va]=p; counts.append(r['n_trees']); fold_records.append(file_sha256(path))
    assert np.isfinite(oof).all()
    test=None
    if refit_test:
        test_path=root/f'{role["member"]}_test.npy'; report_path=reports/f'{role["member"]}_test.json'
        n_trees=int(np.median(counts))
        capacity={'n_trees':n_trees,'selected_fold_counts':counts,'fold_reports_sha256':fold_records}
        if report_path.exists():
            old=json.loads(report_path.read_text(encoding='utf-8'))
            if 'prediction_sha256' not in old:
                raise FileExistsError('Preserve the failed test attempt; select a new tag')
            assert old['contract']['capacity']==capacity and old['contract']['source_sha256']==sources
            test=np.load(test_path)
            assert arr_sha256(test)==old['prediction_sha256']
        else:
            start=time.monotonic()
            vb=ViewBuilder(tr,te,role['view']); vb.build_static()
            xf,xa,names=vb.assemble(np.arange(len(tr)),y,np.empty(0,dtype='int64'),
                np.arange(len(tr),len(tr)+len(te)),inner_seed=0)
            xt=xa['test']; del vb,xa; gc.collect()
            upstream=None
            if role['aux_arm']:
                pf,pt,cache=probability_cache(xf,xt,names,ids,te['id'].to_numpy(),arr_sha256(folds),label='full-label test')
                pos=[names.index(c) for c in RATINGS]
                a,_=signatures(pf,xf[:,pos],0); b,_=signatures(pt,xt[:,pos],0)
                xf,xt=np.column_stack([xf,a]),np.column_stack([xt,b])
                names=list(names)+[f'aux_ev_{j}' for j in range(13)]
                upstream={'fingerprint':cache['fingerprint'],'contract':cache['contract']}
                del pf,pt,a,b; gc.collect()
            contract={'protocol':PROTOCOL,'role':role,'params':params,'capacity':capacity,
                'source_sha256':sources,'data_sha256':data_hash,'fit_ids_sha256':arr_sha256(ids),
                'test_ids_sha256':arr_sha256(te['id'].to_numpy()),'train_rows':len(tr),'test_rows':len(te),
                'feature_fit_sha256':arr_sha256(xf),'feature_test_sha256':arr_sha256(xt),
                'feature_names':names,'upstream':upstream,'test_policy':'full-label fixed-count fit; no test labels or new OOF claim'}
            native=default_cat_cols(True) if role['aux_arm']=='C1' else []
            contract.update(native_categorical_columns=native,model_feature_count=len(names)+len(native))
            def fit_test():
                if role['aux_arm']=='C1':
                    from catboost import CatBoostClassifier
                    from scripts.native_cat import attach,cat_frame,default_cat_cols,cat_indices
                    cats=default_cat_cols(True)
                    f,cn=attach(xf,names,cat_frame(tr,cats,np.arange(len(tr))))
                    t,_=attach(xt,names,cat_frame(te,cats,np.arange(len(te))))
                    p=dict(params,iterations=n_trees)
                    model=CatBoostClassifier(**p)
                    model.fit(f,y,cat_features=cat_indices(f,cn),verbose=0)
                    prediction=model.predict_proba(t)[:,1]
                    del model,f,t
                    return prediction
                else:
                    function={'lgbm':_fit_full_predict_lgbm,'xgb':_fit_full_predict_xgb,'cat':_fit_full_predict_cat}[role['family']]
                    return function(xf,y,xt,params,role['seed'],n_trees)
            test=guarded_fit(root,contract,report_path,sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest(),fit_test)
            test=np.asarray(test,dtype='float32')
            assert test.shape==(len(te),) and np.isfinite(test).all() and ((test>=0)&(test<=1)).all()
            np.save(test_path,test); np.save(root/'test_ids.npy',te['id'].to_numpy())
            save_json({'contract':contract,'git':git_commit(),'prediction_sha256':arr_sha256(test),
                'seconds':time.monotonic()-start,'status':'CLEAN_TEST_REFIT_REQUIRES_FINALIST_SCORECARD'},report_path)
            del xf,xt; gc.collect()
    auc=float(roc_auc_score(y,oof))
    fa=[float(roc_auc_score(y[folds==k],oof[folds==k])) for k in ks]
    meta={'family':role['family'],'featureset':role['view'],'params':params,'seed':role['seed'],
        'auc':auc,'fold_aucs':fa,'training_protocol':PROTOCOL,'source_role':role['member'],
        'fold_reports_sha256':fold_records,'test_policy':'full-label fixed-count same configuration' if test is not None else 'pending',
        'ordered_train_ids_sha256':arr_sha256(ids),'ordered_test_ids_sha256':arr_sha256(te['id'].to_numpy())}
    store.save(f'{root.name}_{role["member"]}',oof,test,fold_scheme=role['scheme'],meta=meta)
    save_json({'role':role,'oof_auc':auc,'fold_aucs':fa,'prediction_sha256':arr_sha256(oof),
        'test_sha256':arr_sha256(test) if test is not None else None,'meta':meta},reports/f'{role["member"]}_complete.json')
    print(f'complete {role["member"]}: OOF{auc:.9f}; test={test is not None}',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--members',default='all')
    ap.add_argument('--folds',default='all')
    ap.add_argument('--families',default='lgbm,xgb,cat')
    ap.add_argument('--tag',default='sol_clean_classical')
    ap.add_argument('--refit-test',action='store_true')
    ap.add_argument('--confirmation-shadow',action='store_true',
        help='Confirm the frozen auxiliary10 recipe on immutable shadow folds; no tuning')
    args=ap.parse_args()
    roles=frozen_roles()
    root,reports=ARTIFACTS/args.tag,REPORTS/args.tag
    root.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    selected=set(args.members.split(','))
    chosen=[r for r in roles if (args.members=='all' or r['member'] in selected or
        (args.members=='aux10' and r['aux_arm'])) and r['family'] in args.families.split(',')]
    if args.members not in ('all','aux10'):
        assert selected.issubset({r['member'] for r in roles})
    if args.confirmation_shadow:
        assert len(chosen)==10 and all(r['aux_arm'] for r in chosen), 'Confirm all ten frozen auxiliary roles together'
        assert args.tag!='sol_clean_classical', 'Use a distinct shadow artifact tag'
        assert not args.refit_test, 'Final inference uses primary-selected capacity; shadow is confirmation only'
        for role in chosen:
            role.update(source_scheme=role['scheme'],scheme='shadow')
    plan={'protocol':PROTOCOL,'roles':roles,'role_count':47,'weights':'47 frozen equal-logit roles; no new-score filtering',
          'confirmation_shadow':args.confirmation_shadow,
          'test_policy':'separate full-label fixed-count refit, same estimator configuration; does not create a new OOF score'}
    plan_path=reports/'frozen_roles.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding='utf-8'))==plan
    else:
        save_json(plan,plan_path)
    tr,te=load_cached_parquet()
    y,ids=tr['satisfaction'].to_numpy(dtype='int8'),tr['id'].to_numpy()
    sources={p:file_sha256(p) for p in ('scripts/replay_sol_classical.py','scripts/run_views.py',
        'scripts/run_phase13.py','scripts/run_phase13b.py','scripts/run_phase14.py',
        'scripts/native_cat.py','scripts/run_sol_a.py','scripts/sol_aux_cache.py',
        'src/features/aux_distribution.py','src/features/view.py','src/features/s6e10.py')}
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    for role in chosen:
        folds=get_scheme(role['scheme'],y,ids).folds
        ks=sorted(set(folds.tolist())) if args.folds=='all' else list(map(int,args.folds.split(',')))
        params=resolved_params(role)
        for k in ks:
            assert k in folds
            fi,va=np.flatnonzero(folds!=k),np.flatnonzero(folds==k)
            seed=1 if role['es_seed_policy']=='1' else role['seed']+(k if role['es_seed_policy']=='seed+fold' else 0)
            itr,es=_inner_es_split(fi,y,seed)
            inner=k if role['te_inner_seed_policy']=='fold' else 0
            vb=ViewBuilder(tr,te,role['view']); vb.build_static()
            xf,xa,names=vb.assemble(fi,y,va,None,inner_seed=inner)
            xv=xa['val']; del vb,xa; gc.collect()
            aux=None
            if role['aux_arm']:
                xf,xv,aux=add_expected_values(xf,xv,names,ids[fi],ids[va],arr_sha256(folds),k,role['scheme'])
                names=list(names)+[f'aux_ev_{j}' for j in range(13)]
            contract={'protocol':PROTOCOL,'role':role,'fold':k,'params':params,'data_sha256':data_hash,
                'fold_sha256':arr_sha256(folds),'source_sha256':sources,
                'outer_fit_ids_sha256':arr_sha256(ids[fi]),'train_ids_sha256':arr_sha256(ids[itr]),
                'es_ids_sha256':arr_sha256(ids[es]),'validation_ids_sha256':arr_sha256(ids[va]),
                'feature_fit_sha256':arr_sha256(xf),'feature_val_sha256':arr_sha256(xv),
                'feature_names':names,'train_rows':len(itr),'es_rows':len(es),'validation_rows':len(va),
                'seed':role['seed'],'inner_es_seed':seed,'te_inner_seed':inner,'aux':aux,
                'test_policy':plan['test_policy'],'libraries':{p:version(p) for p in ('numpy','lightgbm','xgboost','catboost','scikit-learn')}}
            native=default_cat_cols(True) if role['aux_arm']=='C1' else []
            contract.update(native_categorical_columns=native,model_feature_count=len(names)+len(native))
            fp=sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
            stem=f'{role["member"]}_f{k}'; rp=reports/(stem+'.json'); pp=root/(stem+'.npy')
            if rp.exists():
                previous=json.loads(rp.read_text(encoding='utf-8'))
                if 'prediction_sha256' not in previous:
                    raise FileExistsError('Preserve the failed OOF attempt; select a new tag')
                assert previous['fingerprint']==fp and arr_sha256(np.load(pp))==previous['prediction_sha256']
                print(f'reuse {stem}',flush=True); del xf,xv; continue
            tl,el=np.searchsorted(fi,itr),np.searchsorted(fi,es)
            assert np.array_equal(fi[tl],itr) and np.array_equal(fi[el],es)
            start=time.monotonic()
            def fit_oof():
                if role['aux_arm']=='C1':
                    from scripts.native_cat import attach,cat_frame,default_cat_cols
                    cats=default_cat_cols(True)
                    f,cn=attach(xf[tl],names,cat_frame(tr,cats,itr))
                    e,_=attach(xf[el],names,cat_frame(tr,cats,es))
                    v,_=attach(xv,names,cat_frame(tr,cats,va))
                    pred,best=fit_cat(f,y[itr],v,role['seed'],role['overrides'],cn,e,y[es])
                    del f,e,v
                    return pred,best
                else:
                    function=(fit_slot if role['aux_arm'] else _fit_lgbm_es) if role['family']=='lgbm' else ((fit_xgb if role['aux_arm'] else _fit_xgb_es) if role['family']=='xgb' else _fit_cat_es)
                    return function(xf[tl],y[itr],xv,role['overrides'],role['seed'],xf[el],y[es]) if not role['aux_arm'] else function(xf[tl],y[itr],xv,role['seed'],role['overrides'],xf[el],y[es])
            pred,best=guarded_fit(root,contract,rp,fp,fit_oof)
            pred=np.asarray(pred,dtype='float32')
            assert pred.shape==(len(va),) and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
            np.save(pp,pred); np.save(root/(stem+'_ids.npy'),ids[va])
            record={'contract':contract,'fingerprint':fp,'git':git_commit(),'auc':float(roc_auc_score(y[va],pred)),
                'n_trees':int(best)+(role['family'] in ('xgb','cat')),'seconds':time.monotonic()-start,
                'prediction_sha256':arr_sha256(pred),'status':'CORRECTED_BASELINE_PARTIAL_REQUIRES_FULL_CV_AND_TEST'}
            save_json(record,rp)
            print(f'{stem}: AUC{record["auc"]:.9f}; {record["n_trees"]} trees; {record["seconds"]:.1f}s',flush=True)
            del xf,xv; gc.collect()
        complete_role(role,tr,te,y,ids,folds,root,reports,params,args.refit_test,sources,data_hash)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
