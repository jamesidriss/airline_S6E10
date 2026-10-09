"""Continue an interrupted context under a new tag after exact provenance checks."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json
from src.models.resource_guard import inference_guard
from src.validation.folds import get_scheme
from scripts.phase16_common import bank
from scripts.phase16_tabpfn import prepare, release, library_sources
from scripts.phase17_resume import completed_prefix, resume_predict
from scripts.run_sol_tabpfn import frames, create_model


def assert_no_other_workers():
    import os
    import psutil
    # Windows venv launchers and the command shell are ancestors of this worker,
    # and can carry its exact script arguments without running a second fit.
    own_chain = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    producers = {'run_phase17_tabpfn.py', 'run_phase17_confirmation.py', 'resume_phase17_context.py'}
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.info['pid'] in own_chain:
            continue
        if any(Path(token).name in producers for token in (proc.info['cmdline'] or [])):
            try:
                same_workspace = Path(proc.cwd()).resolve() == Path.cwd().resolve()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                same_workspace = False
            assert not same_workspace, 'Another Phase17 worker is active; do not duplicate a live job'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--origin', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--fold', type=int, choices=range(5), required=True)
    args = ap.parse_args()
    assert args.tag != args.origin and all(Path(t).name == t for t in (args.origin, args.tag))
    assert_no_other_workers()
    origin = ARTIFACTS/args.origin/f'f{args.fold}'; config_path = origin/'configuration.json'
    previous = json.loads(config_path.read_text()); old = previous['contract']; count = old['params']['n_estimators']
    assert count in (2, 4) and old['fold'] == args.fold and old['seed'] == old['params']['random_state'] == 1201
    assert old['params']['memory_saving_mode'] is True and not old['entire_training_context']
    assert not old['outer_labels_used_for_fit_or_configuration']
    assert all(file_sha256(s) == h for s, h in old['source_sha256'].items())
    assert old['library_source_sha256'] == library_sources()
    assert old['scope_sha256'] == file_sha256('research/phase17_scope_20261009.json')
    tr, te, y, primary_folds, _, _, proof = bank(); assert old['bank'] == proof
    ids = tr.id.to_numpy(); mode = old.get('mode', 'primary'); scheme = old['scheme']
    folds = primary_folds if scheme == 'primary' else get_scheme('shadow', y, ids).folds
    fi, va = np.flatnonzero(folds != args.fold), np.flatnonzero(folds == args.fold)
    applied_ids = te.id.to_numpy() if mode == 'test' else ids[va]
    x, xt, names, cats, maps = frames(tr, te, True); applied = xt if mode == 'test' else x[va]
    assert old['fit_ids_sha256'] == arr_sha256(ids[fi]) and old['apply_ids_sha256'] == arr_sha256(applied_ids)
    assert old['train_rows'] == len(fi) and old['apply_rows'] == len(applied_ids)
    assert old['feature_fit_sha256'] == arr_sha256(x[fi]) and old['feature_apply_sha256'] == arr_sha256(applied)
    assert old['feature_names'] == names and old['label_free_category_maps'] == maps
    assert old['params']['categorical_features_indices'] == cats
    assert np.array_equal(np.load(origin/'fit_ids.npy'), ids[fi]) and np.array_equal(np.load(origin/'apply_ids.npy'), applied_ids)
    assert not np.intersect1d(ids[fi], applied_ids).size
    prefix = completed_prefix(origin, count, len(applied_ids)); assert prefix, 'Nothing complete to reuse'
    if mode in ('shadow', 'test'):
        reports = list(Path('reports').glob('*_primary.json'))
        matched = [p for p in reports if file_sha256(p) == old['primary_report_sha256']]
        assert len(matched) == 1 and json.loads(matched[0].read_text())['status'] == 'PRIMARY_PASS_REQUIRES_INDEPENDENT_CONFIRMATION_AND_TEST'
        if mode == 'test':
            matched = [p for p in Path('reports').glob('*_shadow_confirmation.json') if file_sha256(p) == old['confirmation_report_sha256']]
            assert len(matched) == 1 and json.loads(matched[0].read_text())['status'] == 'MATCHED_SHADOW_CONFIRMATION_PASS'
    probe_path = Path('reports')/('phase17_internal2_serial_equivalence_20261009.json' if count == 2 else 'phase17_internal4_equivalence_20261009.json')
    probe = json.loads(probe_path.read_text())
    assert probe['status'] == 'PASS_REQUIRES_EXACT_FULL_B0' and probe['maximum_absolute_probability_gap'] <= 2e-6
    assert probe['library_source_sha256'] == library_sources()
    assert json.loads(Path('reports/phase17_B0_f0.json').read_text())['status'] == 'EXACT_B0_HASH_MATCH'
    assert file_sha256(old['params']['model_path']) == old['checkpoint']['checkpoint_sha256']
    root = ARTIFACTS/args.tag/f'f{args.fold}'; output = Path('reports')/(args.tag+f'_f{args.fold}.json')
    if root.exists() or output.exists():
        raise FileExistsError('Use a new resume tag')
    root.mkdir(parents=True); np.save(root/'fit_ids.npy', ids[fi]); np.save(root/'apply_ids.npy', applied_ids)
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register, verify_gpu_equivalence
    gaps = {'fp16': verify_gpu_equivalence(reuse_query_output=True), 'bf16': verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)}
    register(reuse_query_output=True)
    ref = json.loads(Path('reports/sol_tabpfn35_route/route_f0.json').read_text())
    c = {**old, 'git': git_commit(), 'resume_origin_tag': args.origin, 'resume_configuration_sha256': file_sha256(config_path),
         'reused_members': len(prefix), 'own_probe_report_sha256': file_sha256(probe_path), 'backend_reference_gaps': gaps,
         'source_sha256': {**old['source_sha256'], **{s: file_sha256(s) for s in ('scripts/resume_phase17_context.py', 'scripts/phase17_resume.py')}}}
    start = time.monotonic(); result = {'contract': c}; clf = None
    try:
        with inference_guard(root, c, max_seconds=max(300, (count-len(prefix))*2700), min_available_gib=4):
            clf = create_model(c['params'], True, ref['inference_chunk_cells'], ref['inference_col_chunk_size'], None, True)
            prepared = prepare(clf, x[fi], y[fi])
            for field in ('config_sha256', 'pipeline_seed', 'prepared_x_sha256', 'prepared_y_sha256'):
                assert [m[field] for m in prepared[-1]['members']] == [m[field] for m in previous['metadata']['members']]
            assert prepared[-1]['members'] == previous['metadata']['members']
            save_json({'contract': c, 'metadata': prepared[-1]}, root/'configuration.json'); member_start = {}
            def context(i, phase):
                folder = root/f'member_{i}_{phase}'; folder.mkdir(exist_ok=True)
                if phase == 'fit': member_start[i] = time.monotonic()
                return inference_guard(folder, c, max_seconds=max(1, 2700-(time.monotonic()-member_start[i])), min_available_gib=4 if phase == 'fit' else 2)
            def progress(i, stop, fit_seconds, elapsed):
                save_json({'status': 'PREDICTING', 'member': i, 'completed_rows': stop, 'fit_seconds': fit_seconds,
                           'member_seconds': elapsed, 'seconds': time.monotonic()-start}, root/'progress.json')
                if stop <= 1024 or stop == len(applied_ids) or stop//1024 % 40 == 0:
                    print(f'resume member{i}: {stop}/{len(applied_ids)}', flush=True)
            def complete(i, values, stats):
                np.save(root/f'member_{i}_raw_logits.npy', values); save_json(stats, root/f'member_{i}_stats.json')
            probs, metadata = resume_predict(clf, prepared, applied, prefix, context, progress, complete)
            p = probs[:, 1]; np.save(root/'prediction.npy', p)
            result.update(metadata=metadata, prediction_sha256=arr_sha256(p),
                status='COMPLETE_FROZEN_PRIMARY_CONTEXT_TEST' if mode == 'test' else 'COMPLETE_FROZEN_SHADOW_FOLD' if mode == 'shadow' else 'COMPLETE_FROZEN_PRIMARY_FOLD')
            if mode != 'test': result['auc'] = float(roc_auc_score(y[va], p))
    except Exception as error:
        result.update(status='INVALID_RESUME_NO_MODELLING_VERDICT', error=repr(error)); raise
    finally:
        if clf is not None: release(clf)
        result['seconds'] = time.monotonic()-start
        result['reused_member_seconds'] = sum(s['total_seconds'] for _, s in prefix)
        if 'metadata' in result:
            result['total_producing_member_seconds'] = sum(m['total_seconds'] for m in result['metadata']['members'])
        save_json(result, output)
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'prediction_sha256')}))


if __name__ == '__main__':
    main()
