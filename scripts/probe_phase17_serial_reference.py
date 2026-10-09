"""Official native2 with supported zero batching budget versus manual sequential2."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS,arr_sha256,file_sha256,git_commit,save_json
from src.models.resource_guard import inference_guard
from scripts.phase16_common import bank
from scripts.phase16_tabpfn import prepare,sequential_predict,release,config_hash,library_sources
from scripts.run_sol_tabpfn import frames,create_model
from scripts.phase17_contracts import equivalence


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); args=ap.parse_args()
    root=ARTIFACTS/args.tag; path=Path('reports')/(args.tag+'.json')
    if root.exists() or path.exists(): raise FileExistsError('Preserve prior experiments')
    scope_path=Path('research/phase17_scope_20261009.json'); scope=json.loads(scope_path.read_text())
    diagnosis_path=Path('reports/phase17_attention_diagnosis_20261009.json')
    diagnosis=json.loads(diagnosis_path.read_text()); d=diagnosis['failed_operator']
    assert d['q']['shape'][0]==1 and d['k']['shape'][0]==2 and d['one_batch']['capabilities']['efficient']
    tr,te,y,folds,_,_,proof=bank(); ids=tr.id.to_numpy()
    fi=np.sort(np.random.default_rng(1201).choice(np.flatnonzero(folds!=0),100000,replace=False))
    va=np.flatnonzero(folds==0)[:1024]; assert not np.intersect1d(ids[fi],ids[va]).size
    assert arr_sha256(ids[fi])==diagnosis['fit_ids_sha256'] and arr_sha256(ids[va])==diagnosis['apply_ids_sha256']
    x,_,names,cats,maps=frames(tr,te,True)
    ref=json.loads(Path('reports/sol_tabpfn35_route/route_f0.json').read_text())
    assert names==ref['feature_names'] and cats==ref['params']['categorical_features_indices'] and maps==ref['label_free_category_maps']
    root.mkdir(); np.save(root/'fit_ids.npy',ids[fi]); np.save(root/'apply_ids.npy',ids[va])
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register
    register(reuse_query_output=True)
    base=create_model(ref['params'],True,ref['inference_chunk_cells'],ref['inference_col_chunk_size'],None,True)
    kwargs={**ref['params'],'model_path':base.model_path,'n_estimators':2}
    result={'status':'RUNNING_NO_AUC','git':git_commit(),'scope_sha256':file_sha256(scope_path),
        'diagnosis_report_sha256':file_sha256(diagnosis_path),'bank':proof,
        'fit_ids_sha256':arr_sha256(ids[fi]),'apply_ids_sha256':arr_sha256(ids[va]),
        'library_source_sha256':library_sources(),'checkpoint':ref['checkpoint'],
        'source_sha256':{p:file_sha256(p) for p in (__file__,'scripts/phase17_contracts.py','scripts/phase16_tabpfn.py','src/models/windows_attention.py','scripts/probe_phase17_equivalence.py')},
        'native_memory_saving_mode':True,'native_estimator_row_budget':0,'sequential_memory_saving_mode':True,
        'maximum_absolute_probability_gap':scope['maximum_absolute_probability_gap'],'no_auc':True,
        'no_apply_labels':True,'observed_cached_chunk_factors':[]}
    save_json(result,path); started=time.monotonic(); clf=None
    from tabpfn.architectures.tabpfn_v3_5 import ICLTransformerBlock
    original=ICLTransformerBlock.forward

    def observed_forward(self,x,single_eval_pos,save_peak_memory_factor=None,**kw):
        if kw.get('cached_kv') is not None:
            row={'phase':result.get('current_phase'),'query_batch':x.shape[0],
                 'chunk_factor':save_peak_memory_factor}
            if row not in result['observed_cached_chunk_factors']: result['observed_cached_chunk_factors'].append(row)
        return original(self,x,single_eval_pos,save_peak_memory_factor,**kw)

    try:
        with inference_guard(root,result,max_seconds=1200,min_available_gib=4),patch.object(ICLTransformerBlock,'forward',observed_forward):
            from tabpfn.settings import settings
            settings.tabpfn.max_batched_estimator_rows=0
            clf=type(base)(**kwargs); result['current_phase']='native'
            t=time.monotonic(); torch.cuda.reset_peak_memory_stats()
            clf.fit(x[fi],y[fi]); result['native_fit_seconds']=time.monotonic()-t; save_json(result,path)
            result['native_cache_groups']=clf.executor_.cache_groups
            assert result['native_cache_groups']==[[0],[1]]
            native=clf.predict_proba(x[va]).astype('float32'); np.save(root/'native.npy',native)
            cfg=[config_hash(c) for c in clf.ensemble_configs_]
            result.update(native_config_hashes=cfg,native_prediction_sha256=arr_sha256(native),
                native_seconds=time.monotonic()-t,native_peak_gpu_bytes=torch.cuda.max_memory_allocated())
            release(clf); del clf; gc.collect(); torch.cuda.empty_cache(); clf=None; save_json(result,path)
            settings.tabpfn.max_batched_estimator_rows=None
            clf=type(base)(**kwargs); result['current_phase']='sequential'; prepared=prepare(clf,x[fi],y[fi])
            result['official_metadata']=prepared[-1]; save_json(result,path)
            def complete(i,values,stats):
                np.save(root/f'member{i}_raw_logits.npy',values); save_json(stats,root/f'member{i}_stats.json')
            sequential,raw,stats=sequential_predict(clf,prepared,x[va],estimator_complete=complete)
            np.save(root/'sequential.npy',sequential); np.save(root/'raw_logits.npy',raw)
            result['sequential_stats']=stats; result['sequential_prediction_sha256']=arr_sha256(sequential)
            check=equivalence(native,sequential,ids[va],ids[va],cfg,
                [m['config_sha256'] for m in stats['members']],scope['maximum_absolute_probability_gap'])
            result.update(check)
            prior=np.load(ARTIFACTS/'phase16_tabpfn_probe_v2'/'sequential.npy')
            result['sequential_phase16_bitexact']=bool(np.array_equal(prior,sequential))
            result['status']='PASS_REQUIRES_EXACT_FULL_B0' if check['passes'] and result['sequential_phase16_bitexact'] else 'INVALID_EQUIVALENCE_STOP'
    except Exception as e:
        result.update(status='INVALID_TECHNICAL_BLOCKER',error=repr(e),traceback=traceback.format_exc()); raise
    finally:
        if clf is not None: release(clf)
        result['seconds']=time.monotonic()-started; save_json(result,path)
    print(json.dumps({k:result.get(k) for k in ['status','maximum_absolute_probability_gap','native_peak_gpu_bytes','native_seconds','seconds','sequential_phase16_bitexact','observed_cached_chunk_factors']}),flush=True)


if __name__=='__main__': main()
