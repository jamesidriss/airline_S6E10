"""Small isolated policy audit; its AUC is never the finalist's OOF AUC."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.models.resource_guard import inference_guard,ResourcePreflightError
from src.validation.folds import get_scheme
from scripts.run_sol_tabpfn import frames,create_model
from scripts.assemble_sol_foundation_test import verify_primary


def partitions(primary,shadow,ids):
    rng=np.random.default_rng(20261010)
    pool=np.flatnonzero(shadow!=0)
    context=np.sort(rng.choice(pool,100000,replace=False))
    query=np.sort(rng.choice(np.flatnonzero(shadow==0),20000,replace=False))
    fits={'all':context,**{str(k):context[primary[context]!=k] for k in range(5)}}
    assert all(not np.intersect1d(ids[fit],ids[query]).size for fit in fits.values())
    return fits,query


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-report',default='reports/sol_route_aux10_primary_final.json')
    ap.add_argument('--context',choices=['all','0','1','2','3','4'],required=True)
    ap.add_argument('--tag',default='sol_policy_pseudotest')
    args=ap.parse_args()
    root,folder=ARTIFACTS/args.tag,REPORTS/args.tag
    root.mkdir(exist_ok=True);folder.mkdir(exist_ok=True)
    output=folder/f'context_{args.context}.json'
    if output.exists():raise FileExistsError('Preserve existing pseudo-test audit')
    tr,te=load_cached_parquet();y=tr.satisfaction.to_numpy(dtype='int8');ids=tr.id.to_numpy()
    data={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    primary,_,folds=verify_primary(args.primary_report,ids,y,data)
    shadow=get_scheme('shadow',y,ids).folds
    fits,query=partitions(folds,shadow,ids);fit=fits[args.context]
    proof=primary['folds'][0]['foundation']
    reference_path=REPORTS/proof['tag']/'route_f0.json'
    assert file_sha256(reference_path)==proof['report_sha256']
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    dependencies={'scripts/run_sol_tabpfn.py':reference['source_sha256'],
        'src/models/windows_attention.py':reference['backend_source_sha256'],
        'src/models/pointwise_inference.py':reference['decoder_chunk_source_sha256'],
        'src/models/resource_guard.py':reference['resource_guard_source_sha256']}
    assert all(file_sha256(p)==h for p,h in dependencies.items())
    assert file_sha256(reference['params']['model_path'])==reference['checkpoint']['checkpoint_sha256']
    x,_,names,cats,maps=frames(tr,te,True)
    assert names==reference['feature_names'] and maps==reference['label_free_category_maps']
    assert cats==reference['params']['categorical_features_indices']
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register,verify_gpu_equivalence
    gate={'fp16':verify_gpu_equivalence(reuse_query_output=True),
          'bf16':verify_gpu_equivalence(torch.bfloat16,reuse_query_output=True)}
    register(reuse_query_output=True)
    contract={'git':git_commit(),'purpose':'Small leakage-safe inference-policy audit, not full OOF or deployment test AUC',
        'context':args.context,'outer_scheme':'shadow','outer_holdout':0,'train_rows':len(fit),'query_rows':len(query),
        'fit_ids_sha256':arr_sha256(ids[fit]),'query_ids_sha256':arr_sha256(ids[query]),
        'primary_fold_sha256':arr_sha256(folds),'shadow_fold_sha256':arr_sha256(shadow),
        'data_sha256':data,'seed':1201,'partition_seed':20261010,'params':reference['params'],
        'checkpoint':reference['checkpoint'],'reference_report_sha256':file_sha256(reference_path),
        'primary_report_sha256':file_sha256(args.primary_report),'policy_scope_sha256':file_sha256('research/sol_recovery_scope_20261008.json'),
        'feature_fit_sha256':arr_sha256(x[fit]),'feature_query_sha256':arr_sha256(x[query]),
        'source_sha256':{**dependencies,'scripts/run_sol_policy_pseudotest.py':file_sha256(__file__)},
        'backend_reference_gap':gate,'outer_query_labels_used_in_fit':False,'early_stopping':'none'}
    start=time.monotonic();model=None
    save_json({**contract,'status':'FITTING'},root/'progress.json')
    try:
        with inference_guard(root,contract,min_available_gib=4,max_seconds=1200):
            model=create_model(reference['params'],True,262144,1,None,True)
            model.fit(x[fit],y[fit])
        parts=[]
        with inference_guard(root,contract,min_available_gib=2,max_seconds=1200):
            for begin in range(0,len(query),1024):
                stop=min(begin+1024,len(query))
                parts.append(model.predict_proba(x[query[begin:stop]])[:,1])
                save_json({**contract,'status':'PREDICTING','completed_rows':stop},root/'progress.json')
        prediction=np.concatenate(parts).astype('float32')
        assert prediction.shape==(len(query),) and np.isfinite(prediction).all()
        assert ((prediction>=0)&(prediction<=1)).all()
        np.save(root/f'context_{args.context}.npy',prediction)
        np.save(root/f'query_ids_{args.context}.npy',ids[query])
        save_json({'contract':contract,'prediction_sha256':arr_sha256(prediction),
            'auc_on_fixed_small_outer_query':float(roc_auc_score(y[query],prediction)),
            'seconds':time.monotonic()-start,'status':'COMPLETE_SMALL_POLICY_AUDIT_NOT_FULL_OOF'},output)
        print(f'context{args.context}: {len(fit)} FIT rows; {len(query)} query rows; small audit complete',flush=True)
    except Exception as error:
        save_json({'contract':contract,'error':repr(error),'seconds':time.monotonic()-start,
            'status':'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if isinstance(error,(ResourcePreflightError,torch.cuda.OutOfMemoryError))
                else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'},output)
        raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
    return 0


if __name__=='__main__':raise SystemExit(main())
