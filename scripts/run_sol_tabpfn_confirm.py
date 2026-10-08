"""Independent frozen-recipe confirmation; no weight or parameter selection.

Every context target comes from the immutable confirmation FIT partition.
Primary runner and inference helpers stay unchanged and hash-verified.
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.models.resource_guard import inference_guard,ResourcePreflightError
from src.validation.folds import get_scheme
from scripts.run_sol_tabpfn import frames,create_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',default='reports/sol_tabpfn35_route/route_f0.json')
    ap.add_argument('--scheme',choices=['shadow','block10'],required=True)
    ap.add_argument('--fold',type=int,required=True)
    ap.add_argument('--tag',required=True)
    args=ap.parse_args()
    ref_path=Path(args.reference); ref=json.loads(ref_path.read_text(encoding='utf-8'))
    assert ref['prediction_sha256'] and ref['full_intended_population'] and not ref['timing_only']
    assert ref['arm']=='route' and ref['seed']==1201 and ref['params']['n_estimators']==1
    dependencies={'scripts/run_sol_tabpfn.py':ref['source_sha256'],
        'src/models/windows_attention.py':ref['backend_source_sha256'],
        'src/models/pointwise_inference.py':ref['decoder_chunk_source_sha256'],
        'src/models/resource_guard.py':ref['resource_guard_source_sha256']}
    assert all(file_sha256(p)==h for p,h in dependencies.items())
    assert version('tabpfn')==ref['library_version']
    assert ref['batch_size']==1024 and ref['precision']=='autocast'
    assert ref['icl_bf16'] and ref['windows_mqa_backend'] and ref['reuse_query_output']
    assert ref['decoder_inplace_gelu'] and ref['decoder_chunk_rows'] is None
    assert file_sha256(ref['params']['model_path'])==ref['checkpoint']['checkpoint_sha256']
    root,folder=ARTIFACTS/args.tag,REPORTS/args.tag
    root.mkdir(exist_ok=True); folder.mkdir(exist_ok=True)
    report_path=folder/f'route_f{args.fold}.json'
    if report_path.exists() or (root/f'route_f{args.fold}.npy').exists():
        raise FileExistsError('Preserve the existing confirmation result; use a new tag')
    torch.cuda.set_per_process_memory_fraction(ref['gpu_memory_fraction'])
    from src.models.windows_attention import register,verify_gpu_equivalence
    backend_gate={'fp16':verify_gpu_equivalence(reuse_query_output=True),
        'bf16':verify_gpu_equivalence(torch.bfloat16,reuse_query_output=True)}
    register(reuse_query_output=True)
    tr,te=load_cached_parquet(); y=tr.satisfaction.to_numpy(dtype='int8'); ids=tr.id.to_numpy()
    folds=get_scheme(args.scheme,y,ids).folds
    assert args.fold in folds
    fi,va=np.flatnonzero(folds!=args.fold),np.flatnonzero(folds==args.fold)
    assert len(fi)+len(va)==len(tr) and not np.intersect1d(ids[fi],ids[va]).size
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    assert data_hash==ref['data_sha256']
    x,_,names,cats,maps=frames(tr,te,True)
    assert names==ref['feature_names'] and maps==ref['label_free_category_maps']
    assert cats==ref['params']['categorical_features_indices']
    settings={k:ref[k] for k in ('batch_size','precision','icl_bf16','inference_chunk_cells',
        'inference_col_chunk_size','decoder_inplace_gelu','reuse_query_output')}
    contract={**settings,'git':git_commit(),'scheme':args.scheme,'fold':args.fold,'arm':'route',
        'model_family':ref['model_family'],'params':ref['params'],'seed':ref['seed'],
        'library_version':version('tabpfn'),'feature_names':names,'label_free_category_maps':maps,
        'train_rows':len(fi),'validation_rows':len(va),'full_intended_population':True,'timing_only':False,
        'fit_ids_sha256':arr_sha256(ids[fi]),'validation_ids_sha256':arr_sha256(ids[va]),
        'fold_sha256':arr_sha256(folds),'data_sha256':data_hash,'feature_fit_sha256':arr_sha256(x[fi]),
        'feature_val_sha256':arr_sha256(x[va]),'reference_report_sha256':file_sha256(ref_path),
        'source_sha256':{**dependencies,'scripts/run_sol_tabpfn_confirm.py':file_sha256(__file__)},
        'checkpoint':ref['checkpoint'],'backend_max_reference_gap':backend_gate,
        'early_stopping':'none; every context label belongs to confirmation FIT',
        'test_policy':ref['test_policy'],'purpose':'independent confirmation only; never tune on this scheme'}
    start=time.monotonic(); model=None
    save_json({**contract,'status':'FITTING'},root/'progress.json')
    try:
        torch.cuda.reset_peak_memory_stats()
        with inference_guard(root,contract,min_available_gib=4):
            model=create_model(ref['params'],ref['icl_bf16'],ref['inference_chunk_cells'],
                ref['inference_col_chunk_size'],None,ref['decoder_inplace_gelu'])
            model.fit(x[fi],y[fi])
        fit_seconds=time.monotonic()-start
        gc.collect(); torch.cuda.empty_cache()
        chunks=[]
        with inference_guard(root,contract,min_available_gib=2,max_seconds=max(1,2700-fit_seconds)):
            for begin in range(0,len(va),1024):
                stop=min(begin+1024,len(va))
                chunks.append(model.predict_proba(x[va[begin:stop]])[:,1])
                save_json({**contract,'status':'PREDICTING','completed_rows':stop,
                    'fit_seconds':fit_seconds,'seconds':time.monotonic()-start},root/'progress.json')
                if begin==0 or stop==len(va) or begin//1024%20==0:
                    print(f'{args.scheme} f{args.fold}: {stop}/{len(va)}',flush=True)
        prediction=np.concatenate(chunks).astype('float32')
        assert prediction.shape==(len(va),) and np.isfinite(prediction).all()
        assert ((prediction>=0)&(prediction<=1)).all()
        np.save(root/f'route_f{args.fold}.npy',prediction); np.save(root/f'ids_f{args.fold}.npy',ids[va])
        auc=float(roc_auc_score(y[va],prediction))
        save_json({**contract,'prediction_sha256':arr_sha256(prediction),'auc':auc,
            'fit_seconds':fit_seconds,'seconds':time.monotonic()-start,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'status':'FROZEN_INDEPENDENT_CONFIRMATION_PARTIAL'},report_path)
        print(f'{args.scheme} f{args.fold}: AUC{auc:.9f}; {time.monotonic()-start:.1f}s',flush=True)
    except Exception as error:
        resource=isinstance(error,(ResourcePreflightError,torch.cuda.OutOfMemoryError))
        save_json({**contract,'error':repr(error),'seconds':time.monotonic()-start,
            'status':'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'},report_path)
        raise
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
