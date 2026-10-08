"""Conditional route+13 expected-rating TabPFN experiment; timing before discovery."""
from __future__ import annotations
import argparse
import gc
import json
import shutil
import sys
import time
from hashlib import sha256
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.models.resource_guard import inference_guard, ResourcePreflightError
from src.validation.compare import logit
from scripts.assemble_sol_foundation_test import verify_primary
from scripts.run_sol_tabpfn import frames, create_model
from scripts.sol_phase15_auxpfn import expected_values
from scripts.score_sol_foundation import correlations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    assert args.fold in range(5) and (not args.timing or args.fold == 0)
    raw_report = json.loads((REPORTS / 'sol_phase15_raw_primary.json').read_text())
    assert raw_report['scope'] == 'full primary OOF'
    audit = json.loads((REPORTS / 'sol_phase15_zero_training_20261008.json').read_text())
    # The scope trigger is also recorded by a separate decision before launch.
    decision_path = REPORTS / 'sol_phase15_third_branch_decision.json'
    decision = json.loads(decision_path.read_text())
    assert decision['execute'] == 'route_plus13_auxiliary_EV_TabPFN'
    assert decision['raw_report_sha256'] == file_sha256(REPORTS / 'sol_phase15_raw_primary.json')
    assert decision['audit_report_sha256'] == file_sha256(REPORTS / 'sol_phase15_zero_training_20261008.json')
    assert decision['classical_replay_trigger_pass'] is False
    assert not any(audit['comparisons'][name]['pooled_delta_vs_v6'] >= 3e-5 and
                   audit['comparisons'][name]['positive_folds'] >= 4
                   for name in ('legacy_classical_10', 'legacy_classical_20'))
    if raw_report['admission_gate']['admit'] and raw_report['selected_pooled_delta_vs_v6'] >= 3e-5:
        shadow_path = Path(decision['failed_independent_report'])
        shadow = json.loads(shadow_path.read_text())
        assert file_sha256(shadow_path) == decision['failed_independent_report_sha256']
        assert shadow['mode'] == 'shadow' and shadow['delta_vs_v6'] < 1e-5
        assert shadow['verdict'] == 'INDEPENDENT_CHECK_FAILED_NO_PERFORMANCE_SUBMISSION'
    root, folder = ARTIFACTS / args.tag, REPORTS / args.tag
    stem = 'timing_f0' if args.timing else f'auxpfn_f{args.fold}'
    path = folder / (stem + '.json')
    if path.exists() or (root / (stem + '.npy')).exists():
        raise FileExistsError('Preserve completed or invalid attempts')
    if not args.timing:
        probe_path = REPORTS / 'sol_phase15_auxpfn_probe/timing_f0.json'
        probe = json.loads(probe_path.read_text())
        assert probe['status'] == 'TIMING_ONLY_NO_PERFORMANCE_MEASUREMENT'
        assert all(file_sha256(p) == h for p, h in probe['contract']['source_sha256'].items())
    root.mkdir(exist_ok=True); folder.mkdir(exist_ok=True)
    source_path = REPORTS / 'sol_tabpfn35_route/route_f0.json'
    source = json.loads(source_path.read_text())
    deps = {'scripts/run_sol_tabpfn.py': source['source_sha256'],
            'src/models/windows_attention.py': source['backend_source_sha256'],
            'src/models/pointwise_inference.py': source['decoder_chunk_source_sha256'],
            'src/models/resource_guard.py': source['resource_guard_source_sha256']}
    assert all(file_sha256(p) == h for p, h in deps.items())
    assert version('tabpfn') == source['library_version'] and source['seed'] == 1201
    assert source['batch_size'] == 1024 and source['icl_bf16'] and source['decoder_inplace_gelu'] and source['reuse_query_output']
    assert source['decoder_chunk_rows'] is None and source['windows_mqa_backend'] and source['precision'] == 'autocast'
    assert file_sha256(source['params']['model_path']) == source['checkpoint']['checkpoint_sha256']
    tr, te = load_cached_parquet()
    ids, y = tr.id.to_numpy(), tr.satisfaction.to_numpy(dtype='int8')
    dh = {s: file_sha256(f'data/raw/{s}.csv') for s in ('train', 'test')}
    _, vectors, folds = verify_primary('reports/sol_route_aux10_primary_final.json', ids, y, dh)
    fi, va = np.flatnonzero(folds != args.fold), np.flatnonzero(folds == args.fold)
    x, _, names, cats, maps = frames(tr, te, True)
    assert len(names) == 22 and names == source['feature_names'] and maps == source['label_free_category_maps']
    assert cats == source['params']['categorical_features_indices']
    control_path = REPORTS / 'sol_clean_aux10_primary' / f'z3_cat_d8_s2_f{args.fold}.json'
    control = json.loads(control_path.read_text())
    cc = control['contract']
    assert sha256(json.dumps(cc, sort_keys=True).encode()).hexdigest() == control['fingerprint']
    assert cc['role']['aux_arm'] == 'C1' and cc['role']['view'] == 'full' and cc['seed'] == 3
    assert cc['inner_es_seed'] == 1 and cc['fold'] == args.fold
    assert cc['outer_fit_ids_sha256'] == arr_sha256(ids[fi]) and cc['validation_ids_sha256'] == arr_sha256(ids[va])
    assert all(file_sha256(p) == h for p, h in cc['source_sha256'].items())
    assert cc['libraries'] == {p: version(p) for p in cc['libraries']}
    assert control['contract']['data_sha256'] == dh
    assert control['contract']['fold_sha256'] == arr_sha256(folds)
    a, b, aux = expected_values(control['contract']['aux'], x[fi], x[va], names, ids[fi], ids[va], arr_sha256(folds))
    xf, xv = np.column_stack([x[fi], a]), np.column_stack([x[va], b])
    del x, a, b; gc.collect()
    names += [f'aux_ev_{j}' for j in range(13)]
    assert xf.shape == (len(fi), 35) and xv.shape == (len(va), 35)
    assert np.isfinite(xf).all() and np.isfinite(xv).all()
    fitted_ids, applied_ids, fit_y = ids[fi], ids[va], y[fi]
    if args.timing:
        keep = np.sort(np.random.default_rng(1201).choice(len(fi), 100000, replace=False))
        xf, xv, fit_y = xf[keep], xv[:1024], fit_y[keep]
        fitted_ids, applied_ids = fitted_ids[keep], applied_ids[:1024]
    assert not np.intersect1d(fitted_ids, applied_ids).size
    deps.update({p: file_sha256(p) for p in ('scripts/run_sol_phase15_auxpfn.py', 'scripts/sol_phase15_auxpfn.py', 'scripts/run_phase13.py')})
    contract = {'utc': datetime.now(timezone.utc).isoformat(), 'git': git_commit(), 'arm': 'route_plus13_auxEV',
                'scheme': 'primary', 'fold': args.fold, 'timing_only': args.timing,
                'train_rows': len(xf), 'apply_rows': len(xv), 'seed': 1201, 'params': source['params'],
                'fit_ids_sha256': arr_sha256(fitted_ids), 'apply_ids_sha256': arr_sha256(applied_ids),
                'feature_fit_sha256': arr_sha256(xf), 'feature_apply_sha256': arr_sha256(xv),
                'feature_names': names, 'categorical_indices': cats, 'label_free_category_maps': maps,
                'auxiliary': aux, 'control_report_sha256': file_sha256(control_path),
                'route_reference_sha256': file_sha256(source_path), 'checkpoint': source['checkpoint'],
                'data_sha256': dh, 'fold_sha256': arr_sha256(folds), 'source_sha256': deps,
                'library_versions': {p: version(p) for p in ('tabpfn', 'torch', 'numpy', 'scikit-learn')},
                'protocol_sha256': file_sha256('research/sol_phase15_auxpfn_protocol.md'),
                'scope_sha256': file_sha256('research/sol_phase15_scope_20261008.json'),
                'decision_sha256': file_sha256(decision_path), 'weights': {'new_tabpfn': .125, 'v6': .875},
                'early_stopping': 'none; classifier context labels are outer FIT only',
                'test_policy_if_admitted': 'five exact primary FIT contexts, equal probability average, frozen method logits'}
    torch.cuda.set_per_process_memory_fraction(.85)
    from src.models.windows_attention import register, verify_gpu_equivalence
    contract['backend_reference_gap'] = {'fp16': verify_gpu_equivalence(reuse_query_output=True),
                                        'bf16': verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)}
    register(reuse_query_output=True)
    model = None; start = time.monotonic()
    save_json({'contract': contract, 'status': 'FITTING'}, root / 'progress.json')
    try:
        if shutil.disk_usage(root).free < 20 * 1024**3:
            raise ResourcePreflightError('At least20 GiB disk reserve required before model fit')
        torch.cuda.reset_peak_memory_stats()
        with inference_guard(root, contract, min_available_gib=4, max_seconds=2700):
            model = create_model(source['params'], True, source['inference_chunk_cells'], source['inference_col_chunk_size'], None, True)
            model.fit(xf, fit_y)
        fit_seconds = time.monotonic()-start; gc.collect(); torch.cuda.empty_cache(); parts = []
        with inference_guard(root, contract, min_available_gib=2, max_seconds=max(1, 2700-fit_seconds)):
            for begin in range(0, len(xv), 1024):
                stop = min(begin+1024, len(xv)); parts.append(model.predict_proba(xv[begin:stop])[:, 1])
                save_json({'contract': contract, 'status': 'PREDICTING', 'completed_rows': stop,
                           'fit_seconds': fit_seconds, 'seconds': time.monotonic()-start}, root / 'progress.json')
                if begin == 0 or begin//1024 % 20 == 0 or stop == len(xv):
                    print(f'auxpfn f{args.fold}: {stop}/{len(xv)}', flush=True)
        p = np.concatenate(parts).astype('float32')
        assert p.shape == (len(xv),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        np.save(root / (stem+'.npy'), p); np.save(root / (stem+'_ids.npy'), applied_ids)
        result = {'contract': contract, 'fit_seconds': fit_seconds, 'seconds': time.monotonic()-start,
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'prediction_sha256': arr_sha256(p)}
        if args.timing:
            result['status'] = 'TIMING_ONLY_NO_PERFORMANCE_MEASUREMENT'
        else:
            candidate = expit(.125*logit(p)+.875*logit(vectors['candidate'][va])).astype('float32')
            np.save(root / f'portfolio_f{args.fold}.npy', candidate)
            standalone = float(roc_auc_score(y[va], p)-roc_auc_score(y[va], vectors['route'][va]))
            gain = float(roc_auc_score(y[va], candidate)-roc_auc_score(y[va], vectors['candidate'][va]))
            result.update(auc=float(roc_auc_score(y[va], p)), standalone_gain_vs_route=standalone,
                          candidate_auc=float(roc_auc_score(y[va], candidate)), v6_gain=gain,
                          portfolio_prediction_sha256=arr_sha256(candidate), correlation_vs_v6=correlations(candidate, vectors['candidate'][va]),
                          status='DISCOVERY_PASS_UNCONFIRMED' if standalone >= 5e-5 or gain >= 1e-5 else 'DISCOVERY_FAIL_STOP')
        save_json(result, path)
        with Path('experiments/ledger.jsonl').open('a', encoding='utf-8') as ledger:
            ledger.write(json.dumps({'exp_id': f'{args.tag}_f{args.fold}', 'ts': contract['utc'],
                                     'kind': 'TIMING_NO_PERFORMANCE_MODEL' if args.timing else 'CONDITIONAL_AUXILIARY_TABPFN',
                                     'report': str(path), 'report_sha256': file_sha256(path),
                                     'paired_fold_deltas': [] if args.timing else [result['v6_gain']],
                                     'corr_with_champion': result.get('correlation_vs_v6'), 'verdict': result['status']})+'\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('contract', 'correlation_vs_v6')}))
    except Exception as error:
        resource = isinstance(error, (ResourcePreflightError, torch.cuda.OutOfMemoryError))
        save_json({'contract': contract, 'error': repr(error), 'seconds': time.monotonic()-start,
                   'status': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'}, path)
        with Path('experiments/ledger.jsonl').open('a', encoding='utf-8') as ledger:
            ledger.write(json.dumps({'exp_id': f'{args.tag}_f{args.fold}', 'ts': contract['utc'],
                                     'kind': 'INVALID_ATTEMPT_NO_PERFORMANCE_EVIDENCE', 'report': str(path),
                                     'report_sha256': file_sha256(path), 'paired_fold_deltas': [], 'corr_with_champion': None,
                                     'verdict': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE'})+'\n')
        raise
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
