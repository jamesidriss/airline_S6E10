"""The single authorized official Fast checkpoint: timing probe or frozen primary fold."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json
from src.models.resource_guard import inference_guard, ResourcePreflightError
from scripts.run_sol_tabpfn import frames, create_model
from scripts.phase16_common import bank, SCOPE
from scripts.phase16_tabpfn import release, library_sources, config_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--fold', type=int, choices=range(5), default=0)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    root = ARTIFACTS/args.tag
    path = Path('reports')/(args.tag+f'_f{args.fold}.json')
    if path.exists() or root.exists():
        raise FileExistsError('Preserve previous experiment')
    policy_path = Path('research/phase16_fast_contract.json')
    policy = json.loads(policy_path.read_text())
    provenance_path = Path('research/raw/phase16_fast/predownload_provenance.json')
    download_path = Path('research/raw/phase16_fast/download_certificate.json')
    provenance = json.loads(provenance_path.read_text()); downloaded = json.loads(download_path.read_text())
    checkpoint = Path(downloaded['path'])
    assert file_sha256(checkpoint) == policy['checkpoint_sha256'] == provenance['lfs_sha256']
    assert provenance['revision'] == policy['revision'] == downloaded['revision']
    assert downloaded['predownload_provenance_sha256'] == file_sha256(provenance_path)
    assert provenance['license_sha256'] == file_sha256('research/raw/phase16_fast/LICENSE')
    assert provenance['rules'][0]['name'] == 'rules' and provenance['rules'][0]['content']
    tr, te, y, folds, _, _, proof = bank()
    ids = tr.id.to_numpy(); fi = np.flatnonzero(folds != args.fold); va = np.flatnonzero(folds == args.fold)
    ref_path = Path('reports/sol_tabpfn35_route/route_f0.json')
    ref = json.loads(ref_path.read_text())
    if args.timing:
        assert args.fold == 0
        fi = np.sort(np.random.default_rng(1201).choice(fi, 100000, replace=False)); va = va[:1024]
    else:
        control = json.loads(Path('reports/phase16_B0_f0.json').read_text())
        assert control['status'] == 'EXACT_B0_HASH_MATCH'
        assert control['prediction_sha256'] == ref['prediction_sha256']
        probe = json.loads(Path('reports/phase16_fast_probe_f0.json').read_text())
        assert probe['status'] == 'COMPLETE_TIMING_ONLY' and probe['contract']['policy_sha256'] == file_sha256(policy_path)
    assert len(fi) <= 559709 and not np.intersect1d(ids[fi], ids[va]).size
    x, _, names, cats, maps = frames(tr, te, True)
    assert names == ref['feature_names'] and cats == ref['params']['categorical_features_indices']
    assert maps == ref['label_free_category_maps']
    params = {**ref['params'], 'model_path': str(checkpoint.resolve())}
    root.mkdir(); (root/'fit').mkdir(); (root/'predict').mkdir()
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register, verify_gpu_equivalence
    gaps = {'fp16': verify_gpu_equivalence(reuse_query_output=True),
            'bf16': verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)}
    register(reuse_query_output=True)
    contract = {'git': git_commit(), 'fold': args.fold, 'scheme': 'primary', 'params': params,
        'seed': 1201, 'train_rows': len(fi), 'apply_rows': len(va), 'fit_ids_sha256': arr_sha256(ids[fi]),
        'apply_ids_sha256': arr_sha256(ids[va]), 'bank': proof,
        'feature_fit_sha256': arr_sha256(x[fi]), 'feature_apply_sha256': arr_sha256(x[va]),
        'feature_names': names, 'label_free_category_maps': maps, 'timing_only': args.timing,
        'reference_report_sha256': file_sha256(ref_path), 'scope_sha256': file_sha256(SCOPE),
        'policy_sha256': file_sha256(policy_path), 'checkpoint': downloaded,
        'provenance_sha256': file_sha256(provenance_path), 'download_certificate_sha256': file_sha256(download_path),
        'source_sha256': {p: file_sha256(p) for p in ('scripts/phase16_tabpfn.py', 'scripts/run_phase16_fast.py',
            'scripts/phase16_common.py', 'scripts/run_sol_tabpfn.py', 'src/models/windows_attention.py',
            'src/models/pointwise_inference.py', 'src/models/resource_guard.py')},
        'library_versions': {p: version(p) for p in ('tabpfn', 'torch', 'numpy', 'scikit-learn')},
        'library_source_sha256': library_sources(), 'backend_reference_gaps': gaps,
        'outer_labels_used_for_fit_or_configuration': False, 'entire_training_context': False}
    start = time.monotonic(); clf = None; result = {'contract': contract}
    save_json({**contract, 'status': 'FITTING'}, root/'progress.json')
    try:
        clf = create_model(params, True, ref['inference_chunk_cells'], ref['inference_col_chunk_size'], None, True)
        torch.cuda.reset_peak_memory_stats()
        with inference_guard(root/'fit', contract, max_seconds=2700, min_available_gib=4):
            clf.fit(x[fi], y[fi])
        fit_seconds = time.monotonic()-start
        gc.collect(); torch.cuda.empty_cache(); parts = []
        with inference_guard(root/'predict', contract, max_seconds=max(1, 2700-fit_seconds), min_available_gib=2):
            for begin in range(0, len(va), 1024):
                stop = min(begin+1024, len(va)); parts.append(clf.predict_proba(x[va[begin:stop]])[:, 1])
                save_json({'status': 'PREDICTING', 'completed_rows': stop, 'fit_seconds': fit_seconds,
                    'seconds': time.monotonic()-start}, root/'progress.json')
                if begin == 0 or stop == len(va) or begin//1024 % 20 == 0:
                    print(f'Fast f{args.fold}: {stop}/{len(va)}', flush=True)
        p = np.concatenate(parts).astype('float32')
        assert p.shape == (len(va),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        np.save(root/'prediction.npy', p); np.save(root/'apply_ids.npy', ids[va])
        result.update(status='COMPLETE_TIMING_ONLY' if args.timing else 'COMPLETE_FROZEN_PRIMARY_FOLD',
            prediction_sha256=arr_sha256(p), fit_seconds=fit_seconds, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            configuration_hashes=[config_hash(c) for c in clf.ensemble_configs_])
        if not args.timing:
            result['auc'] = float(roc_auc_score(y[va], p))
    except Exception as error:
        resource = isinstance(error, (ResourcePreflightError, torch.cuda.OutOfMemoryError))
        result.update(status='INVALID_RESOURCE_NO_MODELLING_VERDICT' if resource else 'INVALID_IMPLEMENTATION_NO_MODELLING_VERDICT', error=repr(error))
        raise
    finally:
        if clf is not None:
            release(clf)
        result['seconds'] = time.monotonic()-start; save_json(result, path)
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'fit_seconds', 'peak_gpu_bytes') if k in result}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
