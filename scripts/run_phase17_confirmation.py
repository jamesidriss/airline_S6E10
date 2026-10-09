"""Gated shadow confirmation or test prediction using the frozen official members."""
from __future__ import annotations
import argparse
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
from src.validation.folds import get_scheme
from scripts.phase16_common import bank
from scripts.phase16_tabpfn import prepare, sequential_predict, release, library_sources
from scripts.run_sol_tabpfn import frames, create_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('shadow', 'test'), required=True)
    ap.add_argument('--primary-tag', required=True)
    ap.add_argument('--fold', type=int, choices=range(5), required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    assert args.mode != 'shadow' or args.fold in (2, 3), 'Predeclared confirmation folds only'
    primary_path = Path('reports')/(args.primary_tag+'_primary.json')
    primary = json.loads(primary_path.read_text())
    assert primary['status'] == 'PRIMARY_PASS_REQUIRES_INDEPENDENT_CONFIRMATION_AND_TEST'
    assert all(file_sha256(s) == h for s, h in primary['source_sha256'].items())
    scope_path = Path('research/phase17_scope_20261009.json')
    assert primary['scope_sha256'] == file_sha256(scope_path)
    count = primary['internal_estimators']
    assert count in (2, 4)
    confirmation_path = Path('reports')/(args.primary_tag+'_shadow_confirmation.json')
    if args.mode == 'test':
        confirmation = json.loads(confirmation_path.read_text())
        assert confirmation['status'] == 'MATCHED_SHADOW_CONFIRMATION_PASS'
        assert confirmation['primary_report_sha256'] == file_sha256(primary_path)
        assert all(file_sha256(s) == h for s, h in confirmation['source_sha256'].items())
    probe_path = Path('reports')/('phase17_internal2_serial_equivalence_20261009.json' if count == 2
                                 else 'phase17_internal4_equivalence_20261009.json')
    probe = json.loads(probe_path.read_text())
    assert probe['status'] == 'PASS_REQUIRES_EXACT_FULL_B0'
    assert probe['maximum_absolute_probability_gap'] <= 2e-6
    assert probe['library_source_sha256'] == library_sources()
    assert all(file_sha256(s) == h for s, h in probe['source_sha256'].items())
    control = json.loads(Path('reports/phase17_B0_f0.json').read_text())
    assert control['status'] == 'EXACT_B0_HASH_MATCH'
    tr, te, y, primary_folds, _, _, proof = bank()
    assert primary['bank'] == proof
    ids, test_ids = tr.id.to_numpy(), te.id.to_numpy()
    scheme = 'shadow' if args.mode == 'shadow' else 'primary'
    folds = get_scheme(scheme, y, ids).folds
    if scheme == 'primary':
        assert np.array_equal(folds, primary_folds)
    fi = np.flatnonzero(folds != args.fold)
    va = np.flatnonzero(folds == args.fold)
    applied_ids = ids[va] if args.mode == 'shadow' else test_ids
    assert not np.intersect1d(ids[fi], applied_ids).size
    x, xt, names, cats, maps = frames(tr, te, True)
    applied_x = x[va] if args.mode == 'shadow' else xt
    ref_path = Path('reports/sol_tabpfn35_route/route_f0.json')
    ref = json.loads(ref_path.read_text())
    assert names == ref['feature_names'] and cats == ref['params']['categorical_features_indices']
    assert maps == ref['label_free_category_maps']
    dependencies = {'scripts/run_sol_tabpfn.py': ref['source_sha256'],
                    'src/models/windows_attention.py': ref['backend_source_sha256'],
                    'src/models/pointwise_inference.py': ref['decoder_chunk_source_sha256'],
                    'src/models/resource_guard.py': ref['resource_guard_source_sha256']}
    assert all(file_sha256(s) == h for s, h in dependencies.items())
    assert file_sha256(ref['params']['model_path']) == ref['checkpoint']['checkpoint_sha256']
    params = {**ref['params'], 'n_estimators': count}
    primary_fold_path = Path('reports')/(args.primary_tag+f'_f{args.fold}.json')
    primary_fold = json.loads(primary_fold_path.read_text())
    assert primary_fold['contract']['params'] == params
    assert all(file_sha256(s) == h for s, h in primary_fold['contract']['source_sha256'].items())
    if args.mode == 'test':
        assert primary_fold['contract']['fit_ids_sha256'] == arr_sha256(ids[fi])
        assert primary_fold['contract']['feature_fit_sha256'] == arr_sha256(x[fi])
        assert primary_fold['contract']['train_rows'] == len(fi)
    root = ARTIFACTS/args.tag/f'f{args.fold}'
    path = Path('reports')/(args.tag+f'_f{args.fold}.json')
    if root.exists() or path.exists():
        raise FileExistsError('Preserve the existing context and member contributions')
    root.mkdir(parents=True)
    np.save(root/'fit_ids.npy', ids[fi]); np.save(root/'apply_ids.npy', applied_ids)
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register, verify_gpu_equivalence
    gaps = {'fp16': verify_gpu_equivalence(reuse_query_output=True),
            'bf16': verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)}
    register(reuse_query_output=True)
    contract = {'git': git_commit(), 'mode': args.mode, 'scheme': scheme, 'fold': args.fold,
                'params': params, 'seed': 1201, 'train_rows': len(fi), 'apply_rows': len(applied_ids),
                'fit_ids_sha256': arr_sha256(ids[fi]), 'apply_ids_sha256': arr_sha256(applied_ids),
                'fold_sha256': arr_sha256(folds), 'bank': proof,
                'feature_fit_sha256': arr_sha256(x[fi]), 'feature_apply_sha256': arr_sha256(applied_x),
                'feature_names': names, 'label_free_category_maps': maps, 'checkpoint': ref['checkpoint'],
                'scope_sha256': file_sha256(scope_path), 'primary_report_sha256': file_sha256(primary_path),
                'primary_fold_report_sha256': file_sha256(primary_fold_path),
                'probe_report_sha256': file_sha256(probe_path),
                'confirmation_report_sha256': file_sha256(confirmation_path) if args.mode == 'test' else None,
                'source_sha256': {**dependencies, **{s: file_sha256(s) for s in
                    ('scripts/run_phase17_confirmation.py', 'scripts/phase16_tabpfn.py', 'scripts/phase16_common.py')}},
                'library_versions': {s: version(s) for s in ('tabpfn', 'torch', 'numpy', 'scikit-learn')},
                'library_source_sha256': library_sources(), 'backend_reference_gaps': gaps,
                'outer_labels_used_for_fit_or_configuration': False, 'entire_training_context': False,
                'test_policy': primary_fold['contract']['test_policy'], 'no_test_labels': True}
    result = {'contract': contract}; started = time.monotonic(); clf = None
    save_json({**contract, 'status': 'FITTING'}, root/'progress.json')
    try:
        with inference_guard(root, contract, max_seconds=count*2700, min_available_gib=4):
            clf = create_model(params, True, ref['inference_chunk_cells'], ref['inference_col_chunk_size'], None, True)
            prepared = prepare(clf, x[fi], y[fi]); expected = probe['official_metadata']
            for field in ('softmax_temperature', 'average_before_softmax'):
                assert prepared[-1][field] == expected[field]
            for field in ('config_sha256', 'pipeline_seed'):
                assert [m[field] for m in prepared[-1]['members']] == [m[field] for m in expected['members']]
            save_json({'contract': contract, 'metadata': prepared[-1]}, root/'configuration.json')
            member_started = {}

            def context(i, phase):
                folder = root/f'member_{i}_{phase}'; folder.mkdir(exist_ok=True)
                if phase == 'fit':
                    member_started[i] = time.monotonic()
                    save_json({'status': 'FITTING', 'member': i, 'seconds': time.monotonic()-started}, root/'progress.json')
                remaining = 2700-(time.monotonic()-member_started[i])
                return inference_guard(folder, contract, max_seconds=max(1, remaining),
                                       min_available_gib=4 if phase == 'fit' else 2)

            def progress(i, stop, fit_seconds, elapsed):
                save_json({'status': 'PREDICTING', 'member': i, 'completed_rows': stop,
                           'fit_seconds': fit_seconds, 'member_seconds': elapsed,
                           'seconds': time.monotonic()-started}, root/'progress.json')
                if stop == len(applied_ids) or stop <= 1024 or stop//1024 % 40 == 0:
                    print(f'{args.mode} f{args.fold} member{i}: {stop}/{len(applied_ids)}', flush=True)

            def complete(i, values, stats):
                np.save(root/f'member_{i}_raw_logits.npy', values)
                save_json(stats, root/f'member_{i}_stats.json')

            probabilities, _, metadata = sequential_predict(clf, prepared, applied_x, context=context,
                progress=progress, estimator_complete=complete)
            p = probabilities[:, 1]
            assert p.shape == (len(applied_ids),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            np.save(root/'prediction.npy', p)
            result.update(metadata=metadata, prediction_sha256=arr_sha256(p),
                          status='COMPLETE_FROZEN_SHADOW_FOLD' if args.mode == 'shadow' else 'COMPLETE_FROZEN_PRIMARY_CONTEXT_TEST')
            if args.mode == 'shadow':
                result['auc'] = float(roc_auc_score(y[va], p))
    except Exception as error:
        resource = isinstance(error, (ResourcePreflightError, torch.cuda.OutOfMemoryError))
        result.update(status='INVALID_RESOURCE_NO_MODELLING_VERDICT' if resource else 'INVALID_IMPLEMENTATION_NO_MODELLING_VERDICT', error=repr(error))
        raise
    finally:
        if clf is not None:
            release(clf)
        result['seconds'] = time.monotonic()-started; save_json(result, path)
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'prediction_sha256')}), flush=True)


if __name__ == '__main__':
    main()
