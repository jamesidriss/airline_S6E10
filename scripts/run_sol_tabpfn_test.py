"""Full-label TabPFN inference or a resource-only full-context timing probe.

Reuse the frozen primary recipe. No OOF score, test label, parameter search, or
public leaderboard value enters this runner.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.models.resource_guard import inference_guard, ResourcePreflightError
from scripts.run_sol_tabpfn import frames, create_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='reports/sol_tabpfn35_route/route_f0.json')
    ap.add_argument('--primary-report',default='reports/sol_route_aux10_primary.json')
    ap.add_argument('--tag', default='sol_tabpfn35_full_context_probe')
    ap.add_argument('--probe-only', action='store_true')
    ap.add_argument('--scaling-reuse',action='store_true',help='Use only after the complete-model numerical resource gate passes')
    ap.add_argument('--head-views',action='store_true',help='Use only after its complete-model numerical gate passes')
    ap.add_argument('--query-activation-reuse',action='store_true')
    ap.add_argument('--gpu-fraction', type=float, default=.85)
    args = ap.parse_args()
    allocator=os.environ.get('PYTORCH_ALLOC_CONF','')
    assert allocator in ('','expandable_segments:True'), 'Undeclared allocator policy'
    reference_path = Path(args.reference)
    ref = json.loads(reference_path.read_text(encoding='utf-8'))
    assert ref['full_intended_population'] and ref['prediction_sha256']
    assert ref['model_family'] == 'TabPFN-3.5' and ref['arm'] == 'route'
    assert ref['source_sha256'] == file_sha256('scripts/run_sol_tabpfn.py')
    assert ref['backend_source_sha256'] == file_sha256('src/models/windows_attention.py')
    assert ref['decoder_chunk_source_sha256'] == file_sha256('src/models/pointwise_inference.py')
    assert ref['resource_guard_source_sha256'] == file_sha256('src/models/resource_guard.py')
    assert ref['icl_bf16'] and ref['decoder_inplace_gelu'] and ref['reuse_query_output']
    assert ref['decoder_chunk_rows'] is None and ref['windows_mqa_backend']
    assert ref['precision']=='autocast' and ref['params']['inference_precision']=='autocast'
    assert ref['batch_size']==1024
    assert version('tabpfn')==ref['library_version']
    assert 0 < args.gpu_fraction <= 1
    numerical_gate=None
    if args.scaling_reuse:
        gate_path=REPORTS/'sol_scaling_reuse'/'probe.json'
        gate=json.loads(gate_path.read_text(encoding='utf-8'))
        assert gate['status']=='NUMERICAL_GATE_PASS' and gate['maximum_probability_gap']<=2e-6
        assert gate['contract']['reuse_scaling'] and gate['contract']['train_rows']==100000
        assert gate['contract']['params']['model_path']==ref['params']['model_path']
        assert all(file_sha256(p)==h for p,h in gate['contract']['source_sha256'].items())
        assert arr_sha256(np.load(ARTIFACTS/'sol_scaling_reuse'/'probe.npy'))==gate['prediction_sha256']
        numerical_gate={'report_sha256':file_sha256(gate_path),'maximum_probability_gap':gate['maximum_probability_gap']}
    head_gate=None
    if args.head_views:
        assert args.scaling_reuse
        gate_path=REPORTS/'sol_head_views_reuse'/'probe.json'
        gate=json.loads(gate_path.read_text(encoding='utf-8'))
        assert gate['status']=='NUMERICAL_GATE_PASS' and gate['maximum_probability_gap']<=2e-6
        assert gate['contract']['head_views'] and gate['contract']['reuse_scaling']
        assert gate['contract']['train_rows']==100000
        expected=dict(ref['params'],categorical_features_indices=gate['contract']['params']['categorical_features_indices'])
        assert gate['contract']['params']==expected
        assert all(file_sha256(p)==h for p,h in gate['contract']['source_sha256'].items())
        assert arr_sha256(np.load(ARTIFACTS/'sol_head_views_reuse'/'probe.npy'))==gate['prediction_sha256']
        head_gate={'report_sha256':file_sha256(gate_path),'maximum_probability_gap':gate['maximum_probability_gap']}
    query_gate=None
    if args.query_activation_reuse:
        assert args.head_views and args.scaling_reuse
        gate_path=REPORTS/'sol_query_activation_reuse'/'probe.json'
        gate=json.loads(gate_path.read_text(encoding='utf-8'))
        assert gate['status']=='NUMERICAL_GATE_PASS' and gate['maximum_probability_gap']<=2e-6
        assert gate['contract']['query_activation_reuse'] and gate['contract']['train_rows']==100000
        expected=dict(ref['params'],categorical_features_indices=gate['contract']['params']['categorical_features_indices'])
        assert gate['contract']['params']==expected
        assert all(file_sha256(p)==h for p,h in gate['contract']['source_sha256'].items())
        assert arr_sha256(np.load(ARTIFACTS/'sol_query_activation_reuse'/'probe.npy'))==gate['prediction_sha256']
        query_gate={'report_sha256':file_sha256(gate_path),'maximum_probability_gap':gate['maximum_probability_gap']}
    allocator_gate=None
    if allocator:
        assert args.query_activation_reuse and args.head_views and args.scaling_reuse
        gate_path=REPORTS/'sol_allocator_expandable'/'probe.json'
        gate=json.loads(gate_path.read_text(encoding='utf-8'))
        assert gate['status']=='NUMERICAL_GATE_PASS' and gate['maximum_probability_gap']<=2e-6
        assert gate['contract']['allocator_policy']==allocator and gate['contract']['train_rows']==100000
        expected=dict(ref['params'],categorical_features_indices=gate['contract']['params']['categorical_features_indices'])
        assert gate['contract']['params']==expected
        assert all(file_sha256(p)==h for p,h in gate['contract']['source_sha256'].items())
        assert arr_sha256(np.load(ARTIFACTS/'sol_allocator_expandable'/'probe.npy'))==gate['prediction_sha256']
        allocator_gate={'report_sha256':file_sha256(gate_path),'maximum_probability_gap':gate['maximum_probability_gap']}
    checkpoint = Path(ref['params']['model_path'])
    assert file_sha256(checkpoint) == ref['checkpoint']['checkpoint_sha256']
    root, reports = ARTIFACTS/args.tag, REPORTS/args.tag
    root.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    result = reports/('probe.json' if args.probe_only else 'test.json')
    if result.exists():
        raise FileExistsError(f'Preserve existing result: {result}')
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    from src.models.windows_attention import register, verify_gpu_equivalence
    backend_gate = {'fp16': verify_gpu_equivalence(reuse_query_output=True),
        'bf16': verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)}
    register(reuse_query_output=True)
    if args.head_views:
        from src.models.head_view_attention import register_head_views
        register_head_views()
    tr, te = load_cached_parquet()
    data_hash = {s: file_sha256(f'data/raw/{s}.csv') for s in ('train', 'test')}
    assert data_hash == ref['data_sha256']
    x, xt, names, cats, maps = frames(tr, te, True)
    assert names == ref['feature_names'] and cats == ref['params']['categorical_features_indices']
    assert maps == ref['label_free_category_maps']
    ids, test_ids = tr['id'].to_numpy(), te['id'].to_numpy()
    y = tr['satisfaction'].to_numpy(dtype='int8')
    assert not np.intersect1d(ids, test_ids).size
    primary_proof=None
    if not args.probe_only:
        from scripts.assemble_sol_foundation_test import verify_primary
        primary,_,_=verify_primary(args.primary_report,ids,y,data_hash)
        first=next(r for r in primary['folds'] if r['fold']==0)
        assert first['foundation']['report_sha256']==file_sha256(reference_path)
        primary_proof=file_sha256(args.primary_report)
    params = dict(ref['params'])
    count = min(1024, len(te)) if args.probe_only else len(te)
    contract = {'git': git_commit(), 'reference_report_sha256': file_sha256(reference_path),
        'primary_report_sha256':primary_proof,
        'library_versions':{p:version(p) for p in ('tabpfn','torch','numpy','scikit-learn','safetensors')},
        'model_family': 'TabPFN-3.5', 'arm': 'route', 'params': params, 'seed': ref['seed'],
        'train_rows': len(tr), 'test_rows': count, 'entire_competition_training_context': True,
        'fit_ids_sha256': arr_sha256(ids), 'test_ids_sha256': arr_sha256(test_ids[:count]),
        'feature_names': names, 'feature_fit_sha256': arr_sha256(x),
        'feature_test_sha256': arr_sha256(xt[:count]), 'data_sha256': data_hash,
        'source_sha256': {p: file_sha256(p) for p in ('scripts/run_sol_tabpfn_test.py',
            'scripts/run_sol_tabpfn.py', 'src/models/windows_attention.py',
            'src/models/pointwise_inference.py', 'src/models/resource_guard.py')},
        'scaling_reuse':args.scaling_reuse,'scaling_reuse_gate':numerical_gate,
        'head_views':args.head_views,'head_view_gate':head_gate,
        'query_activation_reuse':args.query_activation_reuse,'query_activation_gate':query_gate,
        'allocator_policy':allocator,'allocator_numerical_gate':allocator_gate,
        'backend_reference_gap': backend_gate, 'gpu_fraction': args.gpu_fraction,
        'inference_settings': {k: ref[k] for k in ('batch_size', 'icl_bf16', 'inference_chunk_cells',
            'inference_col_chunk_size', 'decoder_inplace_gelu', 'reuse_query_output')},
        'test_policy': 'entire training set as context; frozen pretrained model and representation',
        'timing_only': args.probe_only, 'performance_verdict': 'none; no OOF or test-label score'}
    if args.scaling_reuse:
        contract['source_sha256']['src/models/scaling_reuse.py']=file_sha256('src/models/scaling_reuse.py')
    if args.head_views:
        contract['source_sha256']['src/models/head_view_attention.py']=file_sha256('src/models/head_view_attention.py')
    if args.query_activation_reuse:
        contract['source_sha256']['src/models/query_activation_reuse.py']=file_sha256('src/models/query_activation_reuse.py')
    start = time.monotonic()
    model = None
    save_json({**contract, 'status': 'FITTING'}, root/'progress.json')
    try:
        if shutil.disk_usage(root).free < 20*1024**3:
            raise ResourcePreflightError('At least20 GiB disk reserve required')
        torch.cuda.reset_peak_memory_stats()
        with inference_guard(root, contract, min_available_gib=4, max_seconds=2700):
            model = create_model(params, ref['icl_bf16'], ref['inference_chunk_cells'],
                ref['inference_col_chunk_size'], None, ref['decoder_inplace_gelu'])
            if args.scaling_reuse:
                from src.models.scaling_reuse import install_scaling_reuse
                install_scaling_reuse(model.model_path.model)
            if args.query_activation_reuse:
                from src.models.query_activation_reuse import install_query_activation_reuse
                install_query_activation_reuse(model.model_path.model)
            model.fit(x, y)
        fit_seconds = time.monotonic()-start
        gc.collect(); torch.cuda.empty_cache()
        chunks = []
        with inference_guard(root, contract, min_available_gib=2, max_seconds=max(1, 2700-fit_seconds)):
            for begin in range(0, count, ref['batch_size']):
                stop = min(begin+ref['batch_size'], count)
                chunks.append(model.predict_proba(xt[begin:stop])[:,1])
                save_json({**contract, 'status': 'PREDICTING', 'completed_rows': stop,
                    'fit_seconds': fit_seconds, 'seconds': time.monotonic()-start}, root/'progress.json')
                if begin == 0 or stop == count or begin//ref['batch_size'] % 20 == 0:
                    print(f'full-context route: predicted {stop}/{count}', flush=True)
        prediction = np.concatenate(chunks).astype('float32')
        assert prediction.shape == (count,) and np.isfinite(prediction).all()
        assert ((prediction >= 0) & (prediction <= 1)).all()
        stem = 'probe' if args.probe_only else 'test'
        np.save(root/f'{stem}.npy', prediction); np.save(root/f'{stem}_ids.npy', test_ids[:count])
        save_json({'contract': contract,
            'probe_prediction_sha256' if args.probe_only else 'test_prediction_sha256': arr_sha256(prediction),
            'fit_seconds': fit_seconds, 'seconds': time.monotonic()-start,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
            'status': 'TIMING_ONLY' if args.probe_only else 'COMPLETE_TEST_REQUIRES_FULL_OOF_SCORECARD'}, result)
        print(f'{stem}: {count} predictions saved; {time.monotonic()-start:.1f}s; no performance score', flush=True)
    except Exception as error:
        resource = isinstance(error, (torch.cuda.OutOfMemoryError, ResourcePreflightError))
        save_json({'contract': contract, 'seconds': time.monotonic()-start, 'error': repr(error),
            'status': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'}, result)
        raise
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
