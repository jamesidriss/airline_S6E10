"""Frozen full-context B0/native1 and B1/official2 sequential-cache primary CV."""
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
from scripts.phase16_tabpfn import prepare, sequential_predict, release, library_sources, config_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--estimators', type=int, choices=(1, 2, 4), required=True)
    ap.add_argument('--fold', type=int, choices=range(5), default=0)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    assert args.estimators != 1 or args.fold == 0, 'B0 is the one required exact control'
    root = ARTIFACTS/args.tag
    path = Path('reports')/(args.tag+f'_f{args.fold}.json')
    if path.exists() or root.exists():
        raise FileExistsError('Preserve existing experiment and arrays')
    probe_path = Path('reports/phase16_tabpfn_probe_v2_20261009.json')
    probe = json.loads(probe_path.read_text())
    if args.estimators > 1:
        assert probe['status'] == 'PASS_REQUIRES_EXACT_FULL_B0'
    assert probe['library_source_sha256'] == library_sources()
    assert probe['source_sha256']['scripts/phase16_tabpfn.py'] == file_sha256('scripts/phase16_tabpfn.py')
    if args.estimators > 1:
        control = json.loads(Path('reports/phase16_B0_f0.json').read_text())
        assert control['status'] == 'EXACT_B0_HASH_MATCH'
    if args.estimators == 4:
        admission = json.loads(Path('reports/phase16_B1_f0_evaluation.json').read_text())
        assert admission['discovery_pass'] and admission['portfolio_delta'] >= 3e-5
    tr, te, y, folds, _, _, proof = bank()
    ref_tags = ['sol_tabpfn35_route', 'sol_tabpfn35_route_reserve', 'sol_tabpfn35_route_reserve',
                'sol_tabpfn35_route_recovery', 'sol_tabpfn35_route_recovery']
    ref_path = Path('reports')/ref_tags[args.fold]/f'route_f{args.fold}.json'
    ref = json.loads(ref_path.read_text())
    fi, va = np.flatnonzero(folds != args.fold), np.flatnonzero(folds == args.fold)
    x, _, names, cats, maps = frames(tr, te, True)
    ids = tr.id.to_numpy()
    assert len(fi) <= 559709 and len(fi) == ref['train_rows']
    assert arr_sha256(ids[fi]) == ref['fit_ids_sha256']
    assert arr_sha256(x[fi]) == ref['feature_fit_sha256']
    assert not np.intersect1d(ids[fi], ids[va]).size
    assert names == ref['feature_names'] and cats == ref['params']['categorical_features_indices']
    assert maps == ref['label_free_category_maps']
    params = {**ref['params'], 'n_estimators': args.estimators}
    root.mkdir()
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register, verify_gpu_equivalence
    gaps = {'fp16': verify_gpu_equivalence(reuse_query_output=True),
            'bf16': verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)}
    register(reuse_query_output=True)
    contract = {'git': git_commit(), 'fold': args.fold, 'scheme': 'primary', 'params': params,
        'seed': 1201, 'train_rows': len(fi), 'apply_rows': len(va), 'fit_ids_sha256': arr_sha256(ids[fi]),
        'apply_ids_sha256': arr_sha256(ids[va]), 'bank': proof,
        'feature_fit_sha256': arr_sha256(x[fi]), 'feature_apply_sha256': arr_sha256(x[va]),
        'feature_names': names, 'label_free_category_maps': maps,
        'reference_report_sha256': file_sha256(ref_path), 'scope_sha256': file_sha256(SCOPE),
        'probe_report_sha256': file_sha256(probe_path), 'checkpoint': ref['checkpoint'],
        'source_sha256': {p: file_sha256(p) for p in ('scripts/phase16_tabpfn.py',
            'scripts/run_phase16_tabpfn.py', 'scripts/phase16_common.py')},
        'library_versions': {p: version(p) for p in ('tabpfn', 'torch', 'numpy', 'scikit-learn')},
        'library_source_sha256': library_sources(), 'backend_reference_gaps': gaps,
        'test_policy': 'Five exact primary FIT-context predictions, native estimator aggregation then equal probability context average',
        'outer_labels_used_for_fit_or_configuration': False, 'entire_training_context': False}
    start = time.monotonic(); clf = None; result = {'contract': contract}
    save_json({**contract, 'status': 'FITTING'}, root/'progress.json')
    try:
        with inference_guard(root, contract, max_seconds=4500, min_available_gib=4):
            clf = create_model(params, True, ref['inference_chunk_cells'],
                               ref['inference_col_chunk_size'], None, True)
            if args.estimators == 1:
                (root/'native_fit').mkdir(); (root/'native_predict').mkdir()
                torch.cuda.reset_peak_memory_stats()
                with inference_guard(root/'native_fit', contract, max_seconds=2700, min_available_gib=4):
                    clf.fit(x[fi], y[fi])
                fit_seconds = time.monotonic()-start
                gc.collect(); torch.cuda.empty_cache(); parts = []
                with inference_guard(root/'native_predict', contract,
                                     max_seconds=max(1, 2700-fit_seconds), min_available_gib=2):
                    for begin in range(0, len(va), 1024):
                        stop = min(begin+1024, len(va))
                        parts.append(clf.predict_proba(x[va[begin:stop]])[:, 1])
                        save_json({'status': 'PREDICTING', 'completed_rows': stop,
                            'fit_seconds': fit_seconds, 'seconds': time.monotonic()-start}, root/'progress.json')
                        if begin == 0 or stop == len(va) or begin//1024 % 20 == 0:
                            print(f'B0: {stop}/{len(va)}', flush=True)
                p = np.concatenate(parts).astype('float32')
                result.update(fit_seconds=fit_seconds, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                    configuration_hashes=[config_hash(c) for c in clf.ensemble_configs_])
                result['status'] = 'EXACT_B0_HASH_MATCH' if arr_sha256(p) == ref['prediction_sha256'] else 'INVALID_CONTROL_HASH_MISMATCH_STOP'
            else:
                prepared = prepare(clf, x[fi], y[fi])
                save_json({'contract': contract, 'metadata': prepared[-1]}, root/'configuration.json')
                estimator_started = {}

                def context(i, phase):
                    folder = root/f'member_{i}_{phase}'; folder.mkdir(exist_ok=True)
                    if phase == 'fit':
                        estimator_started[i] = time.monotonic()
                    remaining = 2700-(time.monotonic()-estimator_started[i])
                    return inference_guard(folder, contract, max_seconds=max(1, remaining),
                                           min_available_gib=4 if phase == 'fit' else 2)

                def progress(i, stop, fit_seconds, elapsed):
                    save_json({'status': 'PREDICTING', 'member': i, 'completed_rows': stop,
                        'fit_seconds': fit_seconds, 'member_seconds': elapsed,
                        'seconds': time.monotonic()-start}, root/'progress.json')
                    if stop == len(va) or stop <= 1024 or stop//1024 % 20 == 0:
                        print(f'B{args.estimators}: member{i} {stop}/{len(va)}', flush=True)

                def complete(i, values, stats):
                    np.save(root/f'member_{i}_raw_logits.npy', values)
                    save_json(stats, root/f'member_{i}_stats.json')

                probabilities, _, metadata = sequential_predict(clf, prepared, x[va],
                    context=context, progress=progress, estimator_complete=complete)
                p = probabilities[:, 1]
                result.update(metadata=metadata, status='COMPLETE_FROZEN_PRIMARY_FOLD')
            assert p.shape == (len(va),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            np.save(root/'prediction.npy', p); np.save(root/'apply_ids.npy', ids[va])
            result.update(prediction_sha256=arr_sha256(p), auc=float(roc_auc_score(y[va], p)))
    except Exception as error:
        resource = isinstance(error, (ResourcePreflightError, torch.cuda.OutOfMemoryError))
        result.update(status='INVALID_RESOURCE_NO_MODELLING_VERDICT' if resource else 'INVALID_IMPLEMENTATION_NO_MODELLING_VERDICT', error=repr(error))
        raise
    finally:
        if clf is not None:
            release(clf)
        result['seconds'] = time.monotonic()-start
        save_json(result, path)
    print(json.dumps({k: result[k] for k in ('status', 'auc', 'seconds', 'prediction_sha256')}), flush=True)
    return 1 if result['status'].startswith('INVALID') else 0


if __name__ == '__main__':
    raise SystemExit(main())
