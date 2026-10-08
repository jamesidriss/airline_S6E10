"""Frozen fallback test inference from one exact immutable primary FIT context."""
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.models.resource_guard import inference_guard,ResourcePreflightError
from scripts.run_sol_tabpfn import frames,create_model
from scripts.assemble_sol_foundation_test import verify_primary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-report',default='reports/sol_route_aux10_primary.json')
    ap.add_argument('--fold',type=int,required=True)
    ap.add_argument('--tag',default='sol_tabpfn35_cv_test')
    args=ap.parse_args()
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();test_ids=te.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    primary,_,folds=verify_primary(args.primary_report,ids,y,data_hash)
    assert args.fold in range(5)
    proof=next(r for r in primary['folds'] if r['fold']==args.fold)['foundation']
    ref_path=REPORTS/proof['tag']/f'route_f{args.fold}.json'
    assert file_sha256(ref_path)==proof['report_sha256']
    ref=json.loads(ref_path.read_text(encoding='utf-8'))
    dependencies={'scripts/run_sol_tabpfn.py':ref['source_sha256'],
        'src/models/windows_attention.py':ref['backend_source_sha256'],
        'src/models/pointwise_inference.py':ref['decoder_chunk_source_sha256'],
        'src/models/resource_guard.py':ref['resource_guard_source_sha256']}
    assert all(file_sha256(p)==h for p,h in dependencies.items())
    assert version('tabpfn')==ref['library_version'] and ref['seed']==1201 and ref['params']['n_estimators']==1
    assert file_sha256(ref['params']['model_path'])==ref['checkpoint']['checkpoint_sha256']
    assert ref['batch_size']==1024 and ref['icl_bf16'] and ref['decoder_inplace_gelu'] and ref['reuse_query_output']
    assert ref['decoder_chunk_rows'] is None and ref['windows_mqa_backend'] and ref['precision']=='autocast'
    fi=np.flatnonzero(folds!=args.fold)
    assert arr_sha256(ids[fi])==ref['fit_ids_sha256'] and len(fi)==ref['train_rows']
    assert not np.intersect1d(ids[fi],test_ids).size
    x,xt,names,cats,maps=frames(tr,te,True)
    assert names==ref['feature_names'] and maps==ref['label_free_category_maps'] and cats==ref['params']['categorical_features_indices']
    assert arr_sha256(x[fi])==ref['feature_fit_sha256']
    root,folder=ARTIFACTS/args.tag,REPORTS/args.tag
    root.mkdir(exist_ok=True);folder.mkdir(exist_ok=True)
    path=folder/f'test_f{args.fold}.json'
    if path.exists() or (root/f'test_f{args.fold}.npy').exists():raise FileExistsError('Preserve the existing context test fit')
    torch.cuda.set_per_process_memory_fraction(ref['gpu_memory_fraction'])
    from src.models.windows_attention import register,verify_gpu_equivalence
    gate={'fp16':verify_gpu_equivalence(reuse_query_output=True),
        'bf16':verify_gpu_equivalence(torch.bfloat16,reuse_query_output=True)}
    register(reuse_query_output=True)
    contract={'git':git_commit(),'fold':args.fold,'scheme':'primary','params':ref['params'],'seed':1201,
        'reference_report_sha256':file_sha256(ref_path),'primary_report_sha256':file_sha256(args.primary_report),
        'train_rows':len(fi),'test_rows':len(te),'fit_ids_sha256':arr_sha256(ids[fi]),'test_ids_sha256':arr_sha256(test_ids),
        'data_sha256':data_hash,'fold_sha256':arr_sha256(folds),'feature_fit_sha256':arr_sha256(x[fi]),
        'feature_test_sha256':arr_sha256(xt),'feature_names':names,'label_free_category_maps':maps,
        'source_sha256':{**dependencies,'scripts/run_sol_tabpfn_fold_test.py':file_sha256(__file__)},
        'library_versions':{p:version(p) for p in ('tabpfn','torch','numpy','scikit-learn')},
        'checkpoint':ref['checkpoint'],'backend_reference_gap':gate,'test_policy_sha256':file_sha256('research/sol_test_inference_contract.md'),
        'test_policy':'One of five exact primary FIT-context predictions; final test uses equal probability average',
        'entire_competition_training_context':False,'timing_only':False,'performance_verdict':'No new CV or test-label score'}
    start=time.monotonic();model=None
    save_json({**contract,'status':'FITTING'},root/'progress.json')
    try:
        torch.cuda.reset_peak_memory_stats()
        with inference_guard(root,contract,min_available_gib=4,max_seconds=2700):
            model=create_model(ref['params'],True,ref['inference_chunk_cells'],ref['inference_col_chunk_size'],None,True)
            model.fit(x[fi],y[fi])
        fit_seconds=time.monotonic()-start;gc.collect();torch.cuda.empty_cache();parts=[]
        with inference_guard(root,contract,min_available_gib=2,max_seconds=max(1,2700-fit_seconds)):
            for begin in range(0,len(te),1024):
                stop=min(begin+1024,len(te));parts.append(model.predict_proba(xt[begin:stop])[:,1])
                save_json({**contract,'status':'PREDICTING','completed_rows':stop,'fit_seconds':fit_seconds,
                    'seconds':time.monotonic()-start},root/'progress.json')
                if begin==0 or stop==len(te) or begin//1024%20==0:print(f'context f{args.fold}: {stop}/{len(te)}',flush=True)
        prediction=np.concatenate(parts).astype('float32')
        assert prediction.shape==(len(te),) and np.isfinite(prediction).all() and ((prediction>=0)&(prediction<=1)).all()
        np.save(root/f'test_f{args.fold}.npy',prediction);np.save(root/f'test_ids_f{args.fold}.npy',test_ids)
        save_json({'contract':contract,'test_prediction_sha256':arr_sha256(prediction),'fit_seconds':fit_seconds,
            'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'status':'COMPLETE_ONE_PRIMARY_CONTEXT_TEST_REQUIRES_ALL_FIVE'},path)
        print(f'context f{args.fold}: complete in{time.monotonic()-start:.1f}s; no AUC',flush=True)
    except Exception as error:
        resource=isinstance(error,(ResourcePreflightError,torch.cuda.OutOfMemoryError))
        save_json({'contract':contract,'seconds':time.monotonic()-start,'error':repr(error),
            'status':'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'},path)
        raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
