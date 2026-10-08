"""One matched, strict native39 treatment; preserve the fixed portfolio weights."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from datetime import datetime,timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.features.view import ViewBuilder
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.replay_sol_classical import frozen_roles,resolved_params,add_expected_values,guarded_fit
from scripts.evaluate_sol_foundation import foundation,classical
from scripts.run_views import _inner_es_split
from scripts.run_phase14 import fit_cat
from scripts.native_cat import attach,cat_frame,default_cat_cols
from scripts.sol_native_crosses import cross_frame


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--baseline-tag',default='sol_clean_aux10_primary')
    ap.add_argument('--tag',default='sol_native39')
    args=ap.parse_args()
    root,folder=ARTIFACTS/args.tag,REPORTS/args.tag
    root.mkdir(exist_ok=True);folder.mkdir(exist_ok=True)
    path=folder/f'cross_f{args.fold}.json'
    if path.exists() or (root/f'cross_f{args.fold}.npy').exists():
        raise FileExistsError('Preserve the existing categorical probe')
    roles=[r for r in frozen_roles() if r['aux_arm']];role=next(r for r in roles if r['aux_arm']=='C1')
    assert role['seed']==3 and role['scheme']=='primary' and role['view']=='full'
    tr,te=load_cached_parquet();y=tr.satisfaction.to_numpy(dtype='int8');ids=tr.id.to_numpy()
    folds=get_scheme('primary',y,ids).folds;assert args.fold in folds
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    fi,va=np.flatnonzero(folds!=args.fold),np.flatnonzero(folds==args.fold)
    train,es=_inner_es_split(fi,y,1);tl,el=np.searchsorted(fi,train),np.searchsorted(fi,es)
    aux,aux_proofs=classical(args.fold,args.baseline_tag,roles,ids,folds,y,data_hash)
    route,_,route_proof=foundation(args.fold,['sol_tabpfn35_route','sol_tabpfn35_route_reserve'],ids,folds,y,data_hash)
    control_path=REPORTS/args.baseline_tag/f'{role["member"]}_f{args.fold}.json'
    control=json.loads(control_path.read_text(encoding='utf-8'));cc=control['contract']
    assert cc['libraries']=={p:version(p) for p in cc['libraries']}
    baseline=np.load(ARTIFACTS/args.baseline_tag/f'{role["member"]}_f{args.fold}.npy')
    assert arr_sha256(baseline)==control['prediction_sha256']
    vb=ViewBuilder(tr,te,'full');vb.build_static()
    xf,apply,names=vb.assemble(fi,y,va,None,inner_seed=args.fold);xv=apply['val'];del vb,apply;gc.collect()
    xf,xv,aux_metadata=add_expected_values(xf,xv,names,ids[fi],ids[va],arr_sha256(folds),args.fold)
    names=list(names)+[f'aux_ev_{j}' for j in range(13)]
    assert arr_sha256(xf)==cc['feature_fit_sha256'] and arr_sha256(xv)==cc['feature_val_sha256']
    assert names==cc['feature_names'] and resolved_params(role)==cc['params']
    native=default_cat_cols(True)
    assert native==cc['native_categorical_columns']
    def frame(matrix,rows):
        cats=pd.concat([cat_frame(tr,native,rows),cross_frame(tr,rows)],axis=1)
        return attach(matrix,names,cats)
    f,cn=frame(xf[tl],train);e,_=frame(xf[el],es);v,_=frame(xv,va)
    assert len(cn)==len(native)+39
    assert list(f.columns)==list(e.columns)==list(v.columns)
    assert len(f.columns)==cc['model_feature_count']+39
    assert cn[-39:]==list(cross_frame(tr,va[:1]).columns)
    assert args.fold in range(5) and not np.intersect1d(train,va).size and not np.intersect1d(es,va).size
    cat_hash={n:arr_sha256(pd.util.hash_pandas_object(frame[cn],index=False).to_numpy()) for n,frame in [('train',f),('es',e),('validation',v)]}
    contract={'hypothesis':'explicit39 native rating/context categories, same C1 settings',
        'protocol_sha256':file_sha256('research/sol_native_cross_protocol.md'),'role':role,'params':resolved_params(role),
        'fold':args.fold,'fold_sha256':arr_sha256(folds),'data_sha256':data_hash,
        'outer_fit_ids_sha256':arr_sha256(ids[fi]),'train_ids_sha256':arr_sha256(ids[train]),
        'es_ids_sha256':arr_sha256(ids[es]),'validation_ids_sha256':arr_sha256(ids[va]),
        'numeric_fit_sha256':arr_sha256(xf),'numeric_validation_sha256':arr_sha256(xv),'categorical_hashes':cat_hash,
        'feature_names':list(f.columns),'native_columns':cn,'feature_provenance':'ix_*: deterministic label-free literal covariate tuples',
        'train_rows':len(train),'es_rows':len(es),'validation_rows':len(va),'seed':3,'inner_es_seed':1,
        'control_report_sha256':file_sha256(control_path),'control_prediction_sha256':arr_sha256(baseline),
        'upstream':aux_metadata,'route_proof':route_proof,'auxiliary_proofs':aux_proofs,
        'source_sha256':{p:file_sha256(p) for p in ('scripts/run_sol_native_crosses.py','scripts/sol_native_crosses.py',*cc['source_sha256'])},
        'library_versions':cc['libraries'],'test_policy':'no test inference in discovery; same full-label median-capacity recipe if admitted'}
    del xf,xv;gc.collect();start=time.monotonic()
    def fit():return fit_cat(f,y[train],v,3,role['overrides'],cn,e,y[es])
    fingerprint=sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    save_json({'contract':contract,'fingerprint':fingerprint,'status':'FITTING'},root/f'progress_f{args.fold}.json')
    prediction,best=guarded_fit(root,contract,path,fingerprint,fit)
    prediction=np.asarray(prediction,dtype='float32')
    assert prediction.shape==(len(va),) and np.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()
    original=(1/(1+np.exp(-(logit(route)+aux)/2))).astype('float32')
    candidate=(1/(1+np.exp(-((logit(route)+aux)/2+(logit(prediction)-logit(baseline))/20)))).astype('float32')
    np.save(root/f'cross_f{args.fold}.npy',prediction);np.save(root/f'ids_f{args.fold}.npy',ids[va])
    np.save(root/f'portfolio_f{args.fold}.npy',candidate)
    auc=float(roc_auc_score(y[va],prediction));delta=auc-float(roc_auc_score(y[va],baseline))
    gain=float(roc_auc_score(y[va],candidate)-roc_auc_score(y[va],original))
    r={'contract':contract,'fingerprint':fingerprint,'git':git_commit(),'auc':auc,'baseline_auc':control['auc'],
        'standalone_delta':delta,'fixed_portfolio_delta':gain,'n_trees':int(best)+1,'seconds':time.monotonic()-start,
        'prediction_sha256':arr_sha256(prediction),'portfolio_prediction_sha256':arr_sha256(candidate),
        'logit_correlation_vs_fixed_portfolio':float(np.corrcoef(logit(prediction),logit(original))[0,1]),
        'status':'POSITIVE_UNCONFIRMED' if delta>=5e-5 or gain>=1e-5 else 'NEGATIVE_OR_NULL_NO_PROMOTION'}
    save_json(r,path)
    save_json({'status':'COMPLETE','report_sha256':file_sha256(path)},root/f'progress_f{args.fold}.json')
    ledger=Path('experiments/ledger.jsonl');eid=f'{args.tag}_f{args.fold}'
    existing={r.get('exp_id',r.get('id')) for r in map(json.loads,ledger.read_text().splitlines())}
    assert eid not in existing
    with ledger.open('a',encoding='utf-8') as handle:
        handle.write(json.dumps({'exp_id':eid,'ts':datetime.now(timezone.utc).isoformat(),'git':git_commit(),
            'kind':'MATCHED_NATIVE_CATEGORICAL_INTERACTION','params':contract['params'],'seed':3,
            'data_hash':data_hash,'fold_hash':arr_sha256(folds),'contract':contract,'validation_auc':auc,
            'paired_fold_deltas':[gain],'standalone_paired_fold_delta':[delta],
            'corr_with_champion':r['logit_correlation_vs_fixed_portfolio'],'report_hash':file_sha256(path),
            'report':str(path),'verdict':r['status']})+'\n')
    print(json.dumps({k:r[k] for k in ('auc','standalone_delta','fixed_portfolio_delta','seconds','status')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
