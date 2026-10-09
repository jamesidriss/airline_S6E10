"""Frozen numerical-only probe with exact failed SDPA tensor diagnostics."""
from __future__ import annotations
import argparse
import gc
import json
import platform
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
import numpy as np
import psutil
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json
from src.models.resource_guard import inference_guard
from scripts.run_sol_tabpfn import frames, create_model
from scripts.phase16_common import bank
from scripts.phase16_tabpfn import prepare, sequential_predict, release, config_hash, library_sources

SCOPE = Path('research/phase17_scope_20261009.json')


def tensor_description(t):
    return {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
            'device': str(t.device), 'contiguous': t.is_contiguous(), 'storage_offset': t.storage_offset()}


def capabilities(q, k, v):
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0., False, False)
    return {name: getattr(torch.backends.cuda, 'can_use_'+name+'_attention')(params, debug=True)
            for name in ('flash', 'efficient', 'cudnn')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=('diagnose', 'equivalence'), default='diagnose')
    args = ap.parse_args()
    path = Path('reports')/(args.tag+'.json'); root = ARTIFACTS/args.tag
    if path.exists() or root.exists(): raise FileExistsError('Preserve every prior probe')
    started = time.monotonic(); scope = json.loads(SCOPE.read_text())
    tr, te, y, folds, _, _, proof = bank()
    fi = np.sort(np.random.default_rng(1201).choice(np.flatnonzero(folds != 0),100000,replace=False))
    va = np.flatnonzero(folds == 0)[:1024]; ids = tr.id.to_numpy()
    assert not np.intersect1d(ids[fi], ids[va]).size
    old = json.loads(Path('reports/phase16_tabpfn_probe_v2_20261009.json').read_text())
    assert arr_sha256(ids[fi]) == old['fit_ids_sha256']
    assert arr_sha256(ids[va]) == old['apply_ids_sha256']
    x, _, names, cats, maps = frames(tr, te, True)
    ref = json.loads(Path('reports/sol_tabpfn35_route/route_f0.json').read_text())
    assert names == ref['feature_names'] and cats == ref['params']['categorical_features_indices']
    assert maps == ref['label_free_category_maps']
    root.mkdir(); np.save(root/'apply_ids.npy',ids[va]); np.save(root/'fit_ids.npy',ids[fi])
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register
    register(reuse_query_output=True)
    result = {'status':'RUNNING_NO_AUC', 'mode':args.mode, 'git':git_commit(),
        'scope_sha256':file_sha256(SCOPE),'bank':proof,'apply_ids_sha256':arr_sha256(ids[va]),
        'fit_ids_sha256':arr_sha256(ids[fi]),'feature_fit_sha256':arr_sha256(x[fi]),
        'feature_apply_sha256':arr_sha256(x[va]),'library_source_sha256':library_sources(),
        'source_sha256':{p:file_sha256(p) for p in (__file__,'scripts/phase16_tabpfn.py','src/models/windows_attention.py')},
        'environment':{'os':platform.platform(),'python':platform.python_version(),
            'versions':{p:version(p) for p in ('torch','tabpfn','numpy','scikit-learn')},
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'gpu':torch.cuda.get_device_name(),'compute_capability':list(torch.cuda.get_device_capability()),
            'free_gpu_bytes':torch.cuda.mem_get_info()[0],'host_available_bytes':psutil.virtual_memory().available,
            'disk_free_bytes':psutil.disk_usage('.').free,'flash_built':torch.backends.cuda.is_flash_attention_available()},
        'checkpoint':ref['checkpoint'],'probes':[],'no_apply_labels':True,'no_auc':True,
        'maximum_absolute_probability_gap':scope['maximum_absolute_probability_gap']}
    save_json(result,path)
    base = create_model(ref['params'],True,ref['inference_chunk_cells'],ref['inference_col_chunk_size'],None,True)
    kwargs = {**ref['params'],'model_path':base.model_path}
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def diagnostic_sdpa(q,k,v,*a,**kw):
        try: return original_sdpa(q,k,v,*a,**kw)
        except RuntimeError:
            result['failed_operator']={'q':tensor_description(q),'k':tensor_description(k),'v':tensor_description(v),
                'kwargs':{k:str(v) for k,v in kw.items()},'capabilities':capabilities(q,k,v),
                'free_gpu_bytes':torch.cuda.mem_get_info()[0], 'autocast':torch.is_autocast_enabled('cuda'),
                'backend_enabled':{'efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),
                    'flash':torch.backends.cuda.flash_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled(),
                    'cudnn':torch.backends.cuda.cudnn_sdp_enabled()}}
            save_json(result,path)
            # Capability probes alter one independent dimension at a time, no numerical model output.
            for label,qq,kk,vv in [('one_batch',q[:1],k[:1],v[:1]),
                                  ('one_query',q[...,:1,:].contiguous(),k,v),
                                  ('one_head',q[:,:1],k[:,:1],v[:,:1])]:
                result['failed_operator'][label]={'q':tensor_description(qq),'k':tensor_description(kk),
                    'capabilities':capabilities(qq,kk,vv)}
            save_json(result,path)
            raise

    try:
        with inference_guard(root,result,max_seconds=1200,min_available_gib=4):
            for n in (1,2):
                clf = type(base)(**{**kwargs,'n_estimators':n}); t=time.monotonic()
                row={'name':'P0' if n==1 else 'P1','n_estimators':n}
                torch.cuda.reset_peak_memory_stats()
                try:
                    with patch('torch.nn.functional.scaled_dot_product_attention',diagnostic_sdpa):
                        clf.fit(x[fi],y[fi]); row['fit_seconds']=time.monotonic()-t
                        save_json({**result,'current_fit':row},path)
                        p=clf.predict_proba(x[va]).astype('float32')
                    np.save(root/f'native{n}.npy',p)
                    row.update(status='PASS_NATIVE',prediction_sha256=arr_sha256(p),
                        configuration_hashes=[config_hash(c) for c in clf.ensemble_configs_])
                except Exception as e:
                    row.update(status='INVALID_TECHNICAL_BLOCKER',error=repr(e),traceback=traceback.format_exc())
                    print(row['traceback'],flush=True)
                finally:
                    row.update(seconds=time.monotonic()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    release(clf); del clf; gc.collect(); torch.cuda.empty_cache()
                    row['allocated_after_release']=torch.cuda.memory_allocated()
                    result['probes'].append(row); save_json(result,path)
            result['status']='DIAGNOSIS_CAPTURED' if 'failed_operator' in result else 'NATIVE2_EXECUTES_NEEDS_EQUIVALENCE'
    finally:
        result['seconds']=time.monotonic()-started; save_json(result,path)
    print(json.dumps({'status':result['status'],'probes':result['probes'],'failed_operator':result.get('failed_operator')}),flush=True)


if __name__=='__main__': main()
