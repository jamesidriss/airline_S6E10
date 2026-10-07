"""Own TabPFN 3.5 full-context OOF: timing first, raw/route paired ablation.

No fitting, early stopping, target encoding, or parameter selection uses an
outer-validation label. The timing subset is a resource probe, never CV evidence.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.features.view import RAW21
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.audit_sol_state import reconstruct_v5


def frames(tr, te, route):
    cov = pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)
    maps = {}
    meta = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for col in meta:
        levels = sorted(cov[col].unique().tolist())
        maps[col] = levels
        cov[col] = pd.Categorical(cov[col], categories=levels).codes
    cats = [RAW21.index(c) for c in meta]
    if route:
        cats += [RAW21.index('Flight Distance')]
        cov['Flight Distance coarse10'] = cov['Flight Distance'] // 10
        cats += [len(RAW21)]
    cov = cov.fillna(-1)
    assert len(cov) == len(tr) + len(te) and 'satisfaction' not in cov and 'id' not in cov
    return cov.iloc[:len(tr)].to_numpy(dtype='float32'), cov.iloc[len(tr):].to_numpy(dtype='float32'), list(cov.columns), cats, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--folds', default='0')
    ap.add_argument('--arms', default='raw,route')
    ap.add_argument('--probe-rows', type=int, default=100000)
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--precision', choices=['autocast', 'fp16', 'bf16'], default='autocast')
    ap.add_argument('--windows-mqa', action='store_true')
    ap.add_argument('--memory-saving', choices=['auto', 'on'], default='auto')
    ap.add_argument('--icl-bf16', action='store_true')
    ap.add_argument('--chunk-cells', type=int)
    ap.add_argument('--col-chunk', type=int)
    ap.add_argument('--tag', default='sol_tabpfn35')
    args = ap.parse_args()
    from tabpfn import TabPFNClassifier
    import tabpfn
    backend_gate = None
    if args.windows_mqa:
        from src.models.windows_attention import register, verify_gpu_equivalence
        backend_gate = verify_gpu_equivalence()
        if args.icl_bf16:
            backend_gate = {'fp16': backend_gate, 'bf16': verify_gpu_equivalence(torch.bfloat16)}
        register()
    checkpoint = ARTIFACTS / 'tabpfn35' / 'tabpfn-v3.5-20260909.safetensors'
    provenance = json.loads(Path('research/raw/sol_tabpfn_provenance.json').read_text(encoding='utf-8'))
    assert file_sha256(checkpoint) == provenance['checkpoint_sha256']
    tr, te = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    _, champ, _ = reconstruct_v5(y, folds)
    root, records = ARTIFACTS / args.tag, REPORTS / args.tag
    root.mkdir(exist_ok=True)
    records.mkdir(exist_ok=True)
    for k in map(int, args.folds.split(',')):
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert len(fi) == len(y) - len(va) and not np.intersect1d(ids[fi], ids[va]).size
        for arm in args.arms.split(','):
            X, _, names, cats, maps = frames(tr, te, arm == 'route')
            fit_idx = fi
            eval_idx = va
            if args.timing:
                # Independent fixed, label-agnostic subset; report NO AUC.
                fit_idx = np.sort(np.random.default_rng(1201).choice(fi, min(args.probe_rows, len(fi)), replace=False))
                eval_idx = va[:min(args.batch_size, 1024)]
            path = records / (f'timing_{arm}_{len(fit_idx)}.json' if args.timing else f'{arm}_f{k}.json')
            if path.exists():
                raise RuntimeError(f'Preserve existing result: {path}')
            params = dict(model_path=str(checkpoint.resolve()), n_estimators=1, auto_scale_n_estimators=False,
                          random_state=1201, device='cuda', ignore_pretraining_limits=True,
                          fit_mode='fit_with_cache', kv_cache_precision='int8', keep_cache_on_device=True,
                          inference_precision={'autocast': 'autocast', 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.precision], categorical_features_indices=cats,
                          memory_saving_mode=True if args.memory_saving == 'on' else 'auto', n_preprocessing_jobs=1, tuning_config=None)
            contract = {'git': git_commit(), 'arm': arm, 'fold': k, 'model_family': 'TabPFN-3.5',
                        'library_version': tabpfn.__version__, 'params': params, 'seed': 1201,
                        'train_rows': len(fit_idx), 'validation_rows': len(eval_idx), 'full_intended_population': not args.timing,
                        'fit_ids_sha256': arr_sha256(ids[fit_idx]), 'validation_ids_sha256': arr_sha256(ids[eval_idx]),
                        'fold_sha256': arr_sha256(folds), 'feature_names': names, 'label_free_category_maps': maps,
                        'feature_fit_sha256': arr_sha256(X[fit_idx]), 'feature_val_sha256': arr_sha256(X[eval_idx]),
                        'data_sha256': {s: file_sha256(f'data/raw/{s}.csv') for s in ('train','test')},
                        'source_sha256': file_sha256(__file__), 'checkpoint': provenance,
                        'early_stopping': 'none; pretrained inference uses every outer-fit target and no evaluation targets',
                        'test_policy': 'same representation; entire competition training set as context; no new gradient fitting',
                        'predeclared_ensemble_geometry': 'append one equal-logit member to the 59-member legacy-v5 reference',
                        'hardware': 'RTX 5070 Ti 16GB; 32GB RAM', 'timing_only': args.timing,
                        'batch_size': args.batch_size, 'precision': args.precision,
                        'icl_bf16': args.icl_bf16,
                        'inference_chunk_cells': args.chunk_cells, 'inference_col_chunk_size': args.col_chunk,
                        'windows_mqa_backend': args.windows_mqa, 'backend_max_reference_gap': backend_gate,
                        'backend_source_sha256': file_sha256('src/models/windows_attention.py') if args.windows_mqa else None,
                        'resource_guard_source_sha256': file_sha256('src/models/resource_guard.py'),
                        'flash_attention_compiled': torch.backends.cuda.is_flash_attention_available()}
            save_json({**contract, 'status': 'FITTING'}, root / 'progress.json')
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            if shutil.disk_usage(root).free < 20 * 1024**3:
                raise RuntimeError('At least 20 GiB disk reserve required before this model fit')
            model = None
            try:
                print(f'{arm} f{k}: fit {len(fit_idx)} context rows; {len(names)} columns', flush=True)
                from src.models.resource_guard import inference_guard
                guard = inference_guard(root, contract) if not args.timing else nullcontext()
                with guard:
                    # Check the reserve before allocating the checkpoint, rather
                    # than demanding the same reserve again after loading it.
                    model = create_model(params, args.icl_bf16, args.chunk_cells, args.col_chunk)
                    model.fit(X[fit_idx], y[fit_idx])
                fit_seconds = time.monotonic() - start
                print(f'{arm}: fitted in {fit_seconds:.1f}s; peak GPU {torch.cuda.max_memory_allocated()/2**30:.3f} GiB', flush=True)
                chunks = []
                pred_start = time.monotonic()
                prediction_guard = inference_guard(root, contract, max_seconds=max(1,2700-fit_seconds), min_available_gib=4) if not args.timing else nullcontext()
                with prediction_guard:
                    for begin in range(0, len(eval_idx), args.batch_size):
                        stop = min(begin + args.batch_size, len(eval_idx))
                        chunks.append(model.predict_proba(X[eval_idx[begin:stop]])[:, 1])
                        save_json({**contract, 'status': 'PREDICTING', 'completed_rows': stop,
                                   'fit_seconds': fit_seconds, 'seconds': time.monotonic() - start}, root / 'progress.json')
                        print(f'{arm}: predicted {stop}/{len(eval_idx)}', flush=True)
                pred = np.concatenate(chunks).astype('float32')
                assert pred.shape == (len(eval_idx),) and np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
                rec = {**contract, 'fit_seconds': fit_seconds, 'predict_seconds': time.monotonic()-pred_start,
                       'seconds': time.monotonic()-start, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                       'status': 'TIMING_ONLY' if args.timing else 'POSITIVE_UNCONFIRMED'}
                if not args.timing:
                    np.save(root / f'{arm}_f{k}.npy', pred)
                    np.save(root / f'ids_f{k}.npy', ids[va])
                    blended = (59 * champ[va] + logit(pred)) / 60
                    rec.update(auc=float(roc_auc_score(y[va], pred)), prediction_sha256=arr_sha256(pred),
                               logit_corr_vs_v5=float(np.corrcoef(logit(pred), champ[va])[0,1]),
                               v5_equal_member_delta=float(roc_auc_score(y[va], blended)-roc_auc_score(y[va], champ[va])))
                    print(f'{arm} AUC {rec["auc"]:.9f}; v5 addition {rec["v5_equal_member_delta"]:+.9f}; corr {rec["logit_corr_vs_v5"]:.6f}', flush=True)
                save_json(rec, path)
            except torch.cuda.OutOfMemoryError as exc:
                save_json({**contract, 'status': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE', 'error': str(exc),
                           'seconds': time.monotonic()-start}, path)
                raise
            except Exception as exc:
                save_json({**contract, 'status': 'INVALID_IMPLEMENTATION_NOT_NEGATIVE', 'error': repr(exc),
                           'seconds': time.monotonic()-start}, path)
                raise
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            del X
    return 0


def create_model(params, icl_bf16=False, chunk_cells=None, col_chunk=None):
    from tabpfn import TabPFNClassifier
    if not icl_bf16 and chunk_cells is None and col_chunk is None:
        return TabPFNClassifier(**params)
    # Keep categorical/fingerprint preprocessing at fp32. Only the officially
    # supported ICL blocks and residual stream switch to bf16; no gradient fit.
    from tabpfn.base import ModelSpecs, initialize_tabpfn_model
    models, configs, _, inference = initialize_tabpfn_model(params['model_path'], 'classifier',
                n_estimators_override=1, devices=[torch.device('cpu')])
    if icl_bf16:
        models[0].enable_icl_bf16()
    if chunk_cells is not None:
        assert chunk_cells > 0
        models[0].inference_chunk_cells = chunk_cells
    if col_chunk is not None:
        assert col_chunk > 0
        models[0].inference_col_chunk_size = col_chunk
    spec = ModelSpecs(models[0], configs[0], inference)
    return TabPFNClassifier(**dict(params, model_path=spec))


if __name__ == '__main__':
    raise SystemExit(main())
