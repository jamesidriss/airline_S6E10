"""One frozen auxiliary role on one immutable block10 confirmation fold.

Reuse the repaired feature/model helpers without editing primary replay bytes.
This runner never reads historical primary-fold auxiliary expectation arrays.
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.features.view import ViewBuilder
from src.features.aux_distribution import signatures
from src.validation.folds import get_scheme
from scripts.replay_sol_classical import frozen_roles,resolved_params,PROTOCOL,guarded_fit,complete_role
from scripts.run_views import _inner_es_split
from scripts.run_phase13b import fit_slot
from scripts.run_phase14 import fit_xgb,fit_cat
from scripts.run_phase13 import RATINGS
from scripts.sol_aux_cache import probability_cache
from scripts.native_cat import attach,cat_frame,default_cat_cols


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--member',required=True)
    ap.add_argument('--fold',type=int,required=True)
    ap.add_argument('--tag',required=True)
    args=ap.parse_args()
    roles=[r for r in frozen_roles() if r['aux_arm']]
    assert len(roles)==10 and all(r['scheme']=='primary' for r in roles)
    for r in roles:
        r.update(source_scheme=r['scheme'],scheme='block10')
    selected=[r for r in roles if r['member']==args.member]
    assert len(selected)==1
    role=selected[0]; params=resolved_params(role)
    root,folder=ARTIFACTS/args.tag,REPORTS/args.tag
    root.mkdir(exist_ok=True); folder.mkdir(exist_ok=True)
    stem=f'{args.member}_f{args.fold}'; report_path=folder/(stem+'.json')
    if report_path.exists() or (root/(stem+'.npy')).exists():
        raise FileExistsError('Preserve existing block10 evidence; select a new tag')
    plan={'protocol':PROTOCOL,'roles':roles,'weights':'ten fixed equal-logit auxiliary roles',
        'purpose':'immutable block10 confirmation only; never tune or select members'}
    plan_path=folder/'frozen_roles.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding='utf-8'))==plan
    else:
        save_json(plan,plan_path)
    dependencies=('scripts/run_sol_aux_block10.py','scripts/replay_sol_classical.py','scripts/run_views.py',
        'scripts/run_phase12.py','scripts/run_phase13.py','scripts/run_phase13b.py','scripts/run_phase14.py',
        'scripts/native_cat.py','scripts/run_sol_a.py','scripts/sol_aux_cache.py',
        'src/features/aux_distribution.py','src/features/view.py','src/features/s6e10.py','src/models/resource_guard.py')
    sources={p:file_sha256(p) for p in dependencies}
    tr,te=load_cached_parquet(); y=tr.satisfaction.to_numpy(dtype='int8'); ids=tr.id.to_numpy()
    folds=get_scheme('block10',y,ids).folds
    assert set(folds)==set(range(10)) and args.fold in folds
    fi,va=np.flatnonzero(folds!=args.fold),np.flatnonzero(folds==args.fold)
    train,es=_inner_es_split(fi,y,1)
    assert np.array_equal(np.sort(np.concatenate([train,es])),fi)
    assert not np.intersect1d(ids[fi],ids[va]).size
    vb=ViewBuilder(tr,te,role['view']); vb.build_static()
    xf,apply,names=vb.assemble(fi,y,va,None,inner_seed=args.fold)
    xv=apply['val']; del vb,apply; gc.collect()
    pf,pv,cache=probability_cache(xf,xv,names,ids[fi],ids[va],arr_sha256(folds),label=f'block10 f{args.fold}')
    pos=[names.index(c) for c in RATINGS]
    a,_=signatures(pf,xf[:,pos],0); b,_=signatures(pv,xv[:,pos],0)
    xf,xv=np.column_stack([xf,a]),np.column_stack([xv,b])
    aux={'fit_expected_values_sha256':arr_sha256(a),'validation_expected_values_sha256':arr_sha256(b),
        'recipe':'250 rounds, three rating-stratified inner folds, seed1; predicted rating excluded',
        'independent_ev_max_gap':None,'upstream_fingerprint':cache['fingerprint'],'upstream_contract':cache['contract']}
    del pf,pv,a,b; gc.collect()
    names=list(names)+[f'aux_ev_{j}' for j in range(13)]
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    native=default_cat_cols(True) if role['aux_arm']=='C1' else []
    contract={'protocol':PROTOCOL,'role':role,'fold':args.fold,'params':params,'data_sha256':data_hash,
        'fold_sha256':arr_sha256(folds),'source_sha256':sources,'outer_fit_ids_sha256':arr_sha256(ids[fi]),
        'train_ids_sha256':arr_sha256(ids[train]),'es_ids_sha256':arr_sha256(ids[es]),
        'validation_ids_sha256':arr_sha256(ids[va]),'feature_fit_sha256':arr_sha256(xf),
        'feature_val_sha256':arr_sha256(xv),'feature_names':names,'train_rows':len(train),'es_rows':len(es),
        'validation_rows':len(va),'seed':role['seed'],'inner_es_seed':1,'te_inner_seed':args.fold,'aux':aux,
        'test_policy':'confirmation only; final inference retains primary-selected capacity',
        'libraries':{p:version(p) for p in ('numpy','lightgbm','xgboost','catboost','scikit-learn')},
        'native_categorical_columns':native,'model_feature_count':len(names)+len(native)}
    fingerprint=sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    tl,el=np.searchsorted(fi,train),np.searchsorted(fi,es)
    assert np.array_equal(fi[tl],train) and np.array_equal(fi[el],es)
    def fit(xf=xf,xv=xv):
        if native:
            f,cn=attach(xf[tl],names,cat_frame(tr,native,train))
            e,_=attach(xf[el],names,cat_frame(tr,native,es))
            v,_=attach(xv,names,cat_frame(tr,native,va))
            return fit_cat(f,y[train],v,role['seed'],role['overrides'],cn,e,y[es])
        function=fit_slot if role['family']=='lgbm' else fit_xgb
        return function(xf[tl],y[train],xv,role['seed'],role['overrides'],xf[el],y[es])
    start=time.monotonic()
    prediction,best=guarded_fit(root,contract,report_path,fingerprint,fit)
    prediction=np.asarray(prediction,dtype='float32')
    assert prediction.shape==(len(va),) and np.isfinite(prediction).all()
    assert ((prediction>=0)&(prediction<=1)).all()
    np.save(root/(stem+'.npy'),prediction); np.save(root/(stem+'_ids.npy'),ids[va])
    record={'contract':contract,'fingerprint':fingerprint,'git':git_commit(),
        'prediction_sha256':arr_sha256(prediction),'auc':float(roc_auc_score(y[va],prediction)),
        'n_trees':int(best)+(role['family'] in ('xgb','cat')),'seconds':time.monotonic()-start,
        'status':'FROZEN_BLOCK10_CONFIRMATION_PARTIAL'}
    save_json(record,report_path)
    print(f'{stem}: AUC{record["auc"]:.9f}; {record["seconds"]:.1f}s',flush=True)
    del fit,xf,xv; gc.collect()
    complete_role(role,tr,te,y,ids,folds,root,folder,params,False,sources,data_hash)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
