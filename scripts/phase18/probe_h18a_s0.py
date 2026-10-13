"""H18-A stage S0 — bounded memory/timing probe at 90% coverage context.

Pre-registered in configs/phase18/h18a_coverage_prereg.json. Makes NO
performance claim: it fits TabPFN once on the 629,671-row block10-fold-0
complement context (n_estimators=1, otherwise the v7 frozen config) and
records peak VRAM + fit seconds + a bounded 10k-row test-row prediction
timing (test rows carry no labels, so no AUC is even computable).

Abort contract: peak VRAM > 16.6 GB decimal or any CUDA OOM => status
RESOURCE_BOUND_H18A_CLOSED; do NOT shrink params/precision to fit — that
would be a different hypothesis.

Allocator note: the v7 runner used torch.cuda.set_per_process_memory_fraction(.85)
= 14.53 GB, which would false-OOM before hardware does at 90% coverage. This
probe raises only that allocator ceiling to 0.95 (16.23 GB). The ceiling is an
OS-level limit, not a numerics change: memory_saving_mode, kv_cache_precision,
inference_precision and chunking stay byte-identical to the certified config.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json  # noqa: E402
from src.models.resource_guard import inference_guard, ResourcePreflightError  # noqa: E402
from scripts.run_sol_tabpfn import frames, create_model  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402

OUT = ROOT / "reports" / "phase18"
VRAM_ABORT_BYTES = int(16.6e9)
CONTEXT_ROWS = 629_671
EVAL_FOLD = 0
PRED_TIMING_ROWS = 10_000
SEED = 1201  # frozen v7 random_state (corrected 2026-10-14; see prereg)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "h18a_s0_probe.json"
    if out_path.exists():
        raise FileExistsError("Preserve the existing probe record")

    tr, te, y, _, _, _, proof = bank()
    block10 = np.load(ROOT / "data/cache/folds/block10_10f_seed20261010.npy")
    assert block10.shape == (len(tr),)
    ctx = np.flatnonzero(block10 != EVAL_FOLD)
    ev = np.flatnonzero(block10 == EVAL_FOLD)
    assert len(ctx) == CONTEXT_ROWS, f"context rows {len(ctx)} != {CONTEXT_ROWS}"
    assert not np.intersect1d(ctx, ev).size

    x, xt, names, cats, maps = frames(tr, te, True)
    assert cats == [13, 14, 15, 16, 18, 21] and len(names) == 22

    params = {
        "model_path": str(ROOT / "artifacts/tabpfn35/tabpfn-v3.5-20260909.safetensors"),
        "n_estimators": 1,
        "auto_scale_n_estimators": False,
        "random_state": SEED,
        "device": "cuda",
        "ignore_pretraining_limits": True,
        "fit_mode": "fit_with_cache",
        "kv_cache_precision": "int8",
        "keep_cache_on_device": True,
        "inference_precision": "autocast",
        "categorical_features_indices": cats,
        "memory_saving_mode": True,
        "n_preprocessing_jobs": 1,
        "tuning_config": None,
    }
    contract = {
        "hypothesis_id": "H18-A",
        "stage": "S0_probe_memory_timing_only",
        "git": git_commit(),
        "prereg_sha256": file_sha256(ROOT / "configs/phase18/h18a_coverage_prereg.json"),
        "context_rows": int(len(ctx)),
        "eval_rows_not_used": int(len(ev)),
        "block10_sha256": arr_sha256(block10),
        "context_ids_sha256": arr_sha256(tr.id.to_numpy()[ctx]),
        "feature_context_sha256": arr_sha256(x[ctx]),
        "params": params,
        "allocator_ceiling_fraction": 0.95,
        "allocator_note": "raised from 0.85 only; not a numerics change",
        "bank": proof,
        "performance_claim": "NONE — memory and wall-time only; test rows have no labels",
    }
    result = dict(contract)

    work = ARTIFACTS / "phase18" / "h18a_s0_probe"
    work.mkdir(parents=True, exist_ok=True)
    save_json(contract, work / "contract.json")

    from src.models.windows_attention import register  # noqa: E402

    register(reuse_query_output=True)
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.cuda.reset_peak_memory_stats()

    start = time.monotonic()
    clf = None
    try:
        with inference_guard(work, contract, max_seconds=3000, min_available_gib=4):
            clf = create_model(params, True, 262144, 1, None, True)
            clf.fit(x[ctx], y[ctx])
            fit_seconds = time.monotonic() - start
            peak_fit = int(torch.cuda.max_memory_allocated())
            gc.collect(); torch.cuda.empty_cache()
            # bounded timing-only prediction on PRED_TIMING_ROWS unlabeled test rows
            pred_start = time.monotonic()
            with inference_guard(work / "predict", contract, max_seconds=900, min_available_gib=2):
                _ = clf.predict_proba(xt[:PRED_TIMING_ROWS])[:, 1]
            pred_seconds = time.monotonic() - pred_start
            peak_total = int(torch.cuda.max_memory_allocated())
        result.update(
            status=("RESOURCE_BOUND_H18A_CLOSED" if max(peak_fit, peak_total) > VRAM_ABORT_BYTES
                    else "S0_PROBE_PASS_FEASIBLE"),
            fit_seconds=fit_seconds,
            peak_gpu_bytes_fit=peak_fit,
            peak_gpu_bytes_total=peak_total,
            vram_abort_bytes=VRAM_ABORT_BYTES,
            predict_timing_rows=PRED_TIMING_ROWS,
            predict_seconds=pred_seconds,
            projected_full_test_predict_seconds=pred_seconds / PRED_TIMING_ROWS * 299_844,
            projection_note="linear in query rows at fixed context; first-chunk overhead included",
            status_note=("90% context does not fit in VRAM under the frozen config; "
                         "coverage axis closed as RESOURCE_BOUND per prereg stop condition"
                         if max(peak_fit, peak_total) > VRAM_ABORT_BYTES else
                         "90% context fits; advance to S1 matched discovery"),
        )
    except (ResourcePreflightError, torch.cuda.OutOfMemoryError) as error:
        result.update(status="RESOURCE_BOUND_H18A_CLOSED", error=repr(error),
                      status_note="OOM or guard refusal at 90% coverage; "
                                  "coverage axis closed per prereg stop condition; no param shrink")
        save_json(result, out_path)
        raise
    finally:
        del clf
        gc.collect(); torch.cuda.empty_cache()
        save_json(result, out_path)
        print(json.dumps({k: result[k] for k in
                          ("status", "fit_seconds", "peak_gpu_bytes_fit", "peak_gpu_bytes_total",
                           "predict_seconds") if k in result}, indent=2))


if __name__ == "__main__":
    main()
