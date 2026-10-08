"""Whole-model query GELU reuse gate on the known100k control; no AUC."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,REPORTS,arr_sha256,file_sha256,git_commit,load_cached_parquet,save_json
from src.validation.folds import get_scheme
from src.models.resource_guard import inference_guard,ResourcePreflightError
from src.models.scaling_reuse import install_scaling_reuse
from src.models.query_activation_reuse import install_query_activation_reuse
from src.models.head_view_attention import register_head_views,HeadViewBackend
from scripts.run_sol_tabpfn import frames,create_model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reuse',action='store_true')
    ap.add_argument('--tag',required=True)
    args=ap.parse_args()
    reference_path=REPORTS/'sol_tabpfn35_gelu'/'timing_raw_100000.json'
    ref=json.loads(reference_path.read_text(encoding='utf-8'))
    assert ref['timing_only'] and ref['arm']=='raw' and ref['train_rows']==100000
    assert ref['seed']==1201 and ref['batch_size']==1024 and ref['decoder_chunk_rows'] is None
    assert file_sha256(ref['params']['model_path'])==ref['checkpoint']['checkpoint_sha256']
    for p,key in [('src/models/windows_attention.py','backend_source_sha256'),
        ('src/models/pointwise_inference.py','decoder_chunk_source_sha256'),
        ('src/models/resource_guard.py','resource_guard_source_sha256')]:
        assert file_sha256(p)==ref[key]
    root,folder=ARTIFACTS/args.tag,REPORTS/args.tag
    root.mkdir(exist_ok=True); folder.mkdir(exist_ok=True)
    path=folder/'probe.json'
    if path.exists() or (root/'probe.npy').exists():
        raise FileExistsError('Preserve the existing equivalence probe')
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register,verify_gpu_equivalence
    backend_gate={'fp16':verify_gpu_equivalence(reuse_query_output=True),
        'bf16':verify_gpu_equivalence(torch.bfloat16,reuse_query_output=True)}
    register(reuse_query_output=True)
    primitive=[]
    if args.reuse:
        from src.models.windows_attention import WindowsMQABackend
        for dtype in (torch.float16,torch.bfloat16):
            generator=torch.Generator(device='cuda').manual_seed(79)
            for heads in (1,16):
                q=torch.randn(1,1007,16,64,generator=generator,device='cuda',dtype=dtype)
                k=torch.randn(1,2009,heads,64,generator=generator,device='cuda',dtype=dtype)
                v=torch.randn_like(k);old_k,old_v=k.clone(),v.clone()
                with torch.no_grad():
                    expected=WindowsMQABackend(query_chunk_size=257,reuse_query_output=True).run(q.clone(),k,v)
                    actual=HeadViewBackend(query_chunk_size=257,reuse_query_output=True).run(q.clone(),k,v)
                assert torch.equal(k,old_k) and torch.equal(v,old_v)
                primitive.append({'dtype':str(dtype),'kv_heads':heads,'query_shape':list(q.shape),
                    'key_shape':list(k.shape),'maximum_output_gap':float((actual-expected).abs().max()),
                    'bit_equal':torch.equal(actual,expected),'all_keys_retained':True,'keys_values_unchanged':True})
        del q,k,v,old_k,old_v,expected,actual
        gc.collect();torch.cuda.empty_cache()
    tr,te=load_cached_parquet(); y=tr.satisfaction.to_numpy(dtype='int8'); ids=tr.id.to_numpy()
    folds=get_scheme('primary',y,ids).folds; fi=np.flatnonzero(folds!=0); va=np.flatnonzero(folds==0)[:1024]
    fit=np.sort(np.random.default_rng(1201).choice(fi,100000,replace=False))
    assert arr_sha256(ids[fit])==ref['fit_ids_sha256'] and arr_sha256(ids[va])==ref['validation_ids_sha256']
    x,_,names,cats,maps=frames(tr,te,False)
    assert names==ref['feature_names'] and maps==ref['label_free_category_maps']
    assert cats==ref['params']['categorical_features_indices']
    assert arr_sha256(x[fit])==ref['feature_fit_sha256'] and arr_sha256(x[va])==ref['feature_val_sha256']
    data_hash={s:file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    assert data_hash==ref['data_sha256']
    expected=np.load(ARTIFACTS/'sol_tabpfn35_gelu'/'probe_raw_100000.npy')
    assert arr_sha256(expected)==ref['probe_prediction_sha256']
    contract={'git':git_commit(),'timing_only':True,'train_rows':len(fit),'query_rows':len(va),
        'fit_ids_sha256':arr_sha256(ids[fit]),'query_ids_sha256':arr_sha256(ids[va]),
        'params':ref['params'],'reference_sha256':file_sha256(reference_path),'reuse_scaling':args.reuse,'head_views':args.reuse,'query_activation_reuse':args.reuse,
        'data_sha256':data_hash,'backend_gate':backend_gate,'head_view_primitive':primitive,'source_sha256':{p:file_sha256(p) for p in
            ('scripts/verify_sol_query_activation.py','src/models/query_activation_reuse.py','src/models/head_view_attention.py','src/models/scaling_reuse.py','scripts/run_sol_tabpfn.py',
             'src/models/windows_attention.py','src/models/pointwise_inference.py','src/models/resource_guard.py')},
        'performance_verdict':'none; fixed numerical reference only','maximum_allowed_probability_gap':2e-6}
    start=time.monotonic(); model=None
    try:
        torch.cuda.reset_peak_memory_stats()
        with inference_guard(root,contract,min_available_gib=4):
            model=create_model(ref['params'],True,262144,1,None,True)
            count=0
            if args.reuse:
                count=install_scaling_reuse(model.model_path.model)
                register_head_views()
                install_query_activation_reuse(model.model_path.model)
            model.fit(x[fit],y[fit])
        fit_seconds=time.monotonic()-start; gc.collect(); torch.cuda.empty_cache()
        with inference_guard(root,contract,min_available_gib=2):
            prediction=model.predict_proba(x[va])[:,1].astype('float32')
        assert prediction.shape==expected.shape and np.isfinite(prediction).all()
        np.save(root/'probe.npy',prediction); np.save(root/'query_ids.npy',ids[va])
        gap=float(np.abs(prediction-expected).max())
        save_json({'contract':contract,'fit_seconds':fit_seconds,'seconds':time.monotonic()-start,
            'patched_scaling_modules':count,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'prediction_sha256':arr_sha256(prediction),'reference_prediction_sha256':arr_sha256(expected),
            'maximum_probability_gap':gap,'status':'NUMERICAL_GATE_PASS' if gap<=2e-6 else 'NUMERICAL_GATE_REJECT'},path)
        print(f'max probability gap{gap:.12g}; peak{torch.cuda.max_memory_allocated()/2**30:.3f} GiB; no AUC',flush=True)
        return 0 if gap<=2e-6 else 2
    except Exception as error:
        resource=isinstance(error,(ResourcePreflightError,torch.cuda.OutOfMemoryError))
        save_json({'contract':contract,'error':repr(error),'seconds':time.monotonic()-start,
            'status':'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'},path)
        raise
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()


if __name__=='__main__':
    raise SystemExit(main())
