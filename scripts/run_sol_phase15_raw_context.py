"""Exact raw TabPFN context for independent shadow or five-context test inference."""
from __future__ import annotations
import argparse
import gc
import json
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.models.resource_guard import inference_guard, ResourcePreflightError
from src.validation.folds import get_scheme
from scripts.run_sol_tabpfn import frames, create_model
from scripts.evaluate_sol_phase15_raw import verify_raw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['shadow','test'],required=True)
    ap.add_argument('--fold',type=int,required=True);ap.add_argument('--tag',required=True)
    args=ap.parse_args();assert args.fold in range(5)
    if args.mode=='shadow':assert args.fold==2,'Only the declared untouched shadow fold'
    path=REPORTS/args.tag/f'{args.mode}_f{args.fold}.json';root=ARTIFACTS/args.tag
    if path.exists() or (root/f'{args.mode}_f{args.fold}.npy').exists():
        raise FileExistsError('Preserve completed or invalid contexts; choose a new tag')
    root.mkdir(exist_ok=True);path.parent.mkdir(exist_ok=True)
    source_path=REPORTS/'sol_tabpfn35_predict_guard/raw_f0.json';source=json.loads(source_path.read_text())
    tr,te=load_cached_parquet();ids=tr.id.to_numpy();test_ids=te.id.to_numpy();y=tr.satisfaction.to_numpy(dtype='int8')
    dh={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    assert dh==source['data_sha256']
    deps={'scripts/run_sol_tabpfn.py':source['source_sha256'],
          'src/models/windows_attention.py':source['backend_source_sha256'],
          'src/models/pointwise_inference.py':source['decoder_chunk_source_sha256'],
          'src/models/resource_guard.py':source['resource_guard_source_sha256']}
    assert all(file_sha256(p)==h for p,h in deps.items())
    assert version('tabpfn')==source['library_version'] and source['seed']==1201
    assert file_sha256(source['params']['model_path'])==source['checkpoint']['checkpoint_sha256']
    assert source['batch_size']==1024 and source['icl_bf16'] and source['decoder_inplace_gelu'] and source['reuse_query_output']
    assert source['decoder_chunk_rows'] is None and source['windows_mqa_backend'] and source['precision']=='autocast'
    scheme='shadow' if args.mode=='shadow' else 'primary';folds=get_scheme(scheme,y,ids).folds
    fi=np.flatnonzero(folds!=args.fold);va=np.flatnonzero(folds==args.fold)
    x,xt,names,cats,maps=frames(tr,te,False)
    assert names==source['feature_names'] and cats==source['params']['categorical_features_indices'] and maps==source['label_free_category_maps']
    if args.mode=='test':
        _,cv_proof=verify_raw(args.fold,'sol_tabpfn35_predict_guard' if args.fold==0 else 'sol_phase15_raw',tr,te,ids,y,folds,dh)
        cv=json.loads((REPORTS/cv_proof['tag']/f'raw_f{args.fold}.json').read_text())
        assert arr_sha256(x[fi])==cv['feature_fit_sha256'] and len(fi)==cv['train_rows']
        apply,apply_ids=xt,test_ids
    else:
        cv_proof=None;apply,apply_ids=x[va],ids[va]
    assert not np.intersect1d(ids[fi],apply_ids).size and len(np.unique(apply_ids))==len(apply_ids)
    deps.update({'scripts/run_sol_phase15_raw_context.py':file_sha256(__file__),
                 'scripts/evaluate_sol_phase15_raw.py':file_sha256('scripts/evaluate_sol_phase15_raw.py')})
    contract={'git':git_commit(),'mode':args.mode,'arm':'raw','scheme':scheme,'fold':args.fold,'seed':1201,
        'params':source['params'],'source_reference_sha256':file_sha256(source_path),'cv_proof':cv_proof,
        'train_rows':len(fi),'apply_rows':len(apply),'fit_ids_sha256':arr_sha256(ids[fi]),'apply_ids_sha256':arr_sha256(apply_ids),
        'feature_fit_sha256':arr_sha256(x[fi]),'feature_apply_sha256':arr_sha256(apply),'feature_names':names,
        'label_free_category_maps':maps,'feature_provenance':'raw21; label-free joint covariate category levels; no target transform',
        'checkpoint':source['checkpoint'],'data_sha256':dh,'fold_sha256':arr_sha256(folds),'source_sha256':deps,
        'library_versions':{p:version(p) for p in ('tabpfn','torch','numpy','scikit-learn')},
        'protocol_sha256':file_sha256('research/sol_phase15_raw_protocol.md'),
        'scope_sha256':file_sha256('research/sol_phase15_scope_20261008.json'),
        'early_stopping':'none; every context label from outer FIT only','test_policy':'Equal probability average of five exact primary FIT contexts; frozen route.375/raw.125/aux.5 logits'}
    torch.cuda.set_per_process_memory_fraction(source['gpu_memory_fraction'])
    from src.models.windows_attention import register, verify_gpu_equivalence
    gate={'fp16':verify_gpu_equivalence(reuse_query_output=True),'bf16':verify_gpu_equivalence(torch.bfloat16,reuse_query_output=True)}
    register(reuse_query_output=True);contract['backend_reference_gap']=gate
    model=None;start=time.monotonic();save_json({'contract':contract,'status':'FITTING'},root/'progress.json')
    try:
        if shutil.disk_usage(root).free < 20 * 1024**3:
            raise ResourcePreflightError('At least20 GiB disk reserve required before this model fit')
        torch.cuda.reset_peak_memory_stats()
        with inference_guard(root,contract,min_available_gib=4,max_seconds=2700):
            model=create_model(source['params'],True,source['inference_chunk_cells'],source['inference_col_chunk_size'],None,True)
            model.fit(x[fi],y[fi])
        fit_seconds=time.monotonic()-start;gc.collect();torch.cuda.empty_cache();parts=[]
        with inference_guard(root,contract,min_available_gib=2,max_seconds=max(1,2700-fit_seconds)):
            for begin in range(0,len(apply),1024):
                stop=min(begin+1024,len(apply));parts.append(model.predict_proba(apply[begin:stop])[:,1])
                save_json({'contract':contract,'status':'PREDICTING','completed_rows':stop,'fit_seconds':fit_seconds,
                           'seconds':time.monotonic()-start},root/'progress.json')
                if begin==0 or stop==len(apply) or begin//1024%20==0:
                    print(f'raw {args.mode} f{args.fold}: {stop}/{len(apply)}',flush=True)
        p=np.concatenate(parts).astype('float32')
        assert p.shape==(len(apply),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        np.save(root/f'{args.mode}_f{args.fold}.npy',p);np.save(root/f'ids_{args.mode}_f{args.fold}.npy',apply_ids)
        result={'contract':contract,'prediction_sha256':arr_sha256(p),'fit_seconds':fit_seconds,
                'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                'status':'COMPLETE_INDEPENDENT_SHADOW_FOLD' if args.mode=='shadow' else 'COMPLETE_RAW_TEST_CONTEXT_REQUIRES_ALL_FIVE'}
        if args.mode=='shadow':result['auc']=float(roc_auc_score(y[va],p))
        save_json(result,path);print(f'raw {args.mode} f{args.fold} complete in{result["seconds"]:.1f}s',flush=True)
    except Exception as error:
        resource=isinstance(error,(ResourcePreflightError,torch.cuda.OutOfMemoryError))
        save_json({'contract':contract,'error':repr(error),'seconds':time.monotonic()-start,
                  'status':'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'},path)
        raise
    finally:
        del model;gc.collect();torch.cuda.empty_cache()
    return 0


if __name__=='__main__':raise SystemExit(main())
