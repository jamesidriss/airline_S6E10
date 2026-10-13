"""H18-A stage S1 — one matched arm on a block10 eval fold.

Pre-registered in configs/phase18/h18a_coverage_prereg.json. Both arms share
every frozen v7 parameter (n_estimators=2, random_state=1201, int8 KV cache,
autocast, memory_saving_mode, chunk cells 262144/1) and the identical eval
rows; ONLY the context differs:

  control   : fixed-seed (1801) 559,708-row subsample of the non-eval rows
  treatment : all 629,671 non-eval rows (90% coverage)

One arm per invocation (<=1 heavy GPU job). Exact member decomposition via the
certified prepare/sequential_predict path; member config hashes must equal the
frozen phase17 probe metadata (data-independent for fixed params -- proven by
the certified OOF and test contexts passing the same assertion at two context
sizes).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json  # noqa: E402
from src.models.resource_guard import inference_guard, ResourcePreflightError  # noqa: E402
from scripts.run_sol_tabpfn import frames, create_model  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402
from scripts.phase16_tabpfn import prepare, sequential_predict, release, library_sources  # noqa: E402

OUT = ROOT / "reports" / "phase18"
CONTROL_ROWS = 559_708
TREATMENT_ROWS = 629_671
SUBSAMPLE_SEED = 1801
MODEL_SEED = 1201
ALLOCATOR_FRACTION = 0.95  # ceiling raised from 0.85; allocator limit, not numerics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("control", "treatment"), required=True)
    ap.add_argument("--fold", type=int, choices=range(10), required=True,
                    help="block10 eval fold")
    args = ap.parse_args()
    tag = f"h18a_s1_f{args.fold}_{args.arm}"
    out_path = OUT / f"{tag}.json"
    root = ARTIFACTS / "phase18" / tag
    if out_path.exists() or root.exists():
        raise FileExistsError("Preserve existing experiment and arrays")
    root.mkdir(parents=True)

    probe = json.loads((OUT / "h18a_s0_probe.json").read_text(encoding="utf-8"))
    if probe["status"] != "S0_PROBE_PASS_FEASIBLE":
        raise SystemExit(f"S0 probe status {probe['status']} does not authorize S1")

    tr, te, y, _, _, _, proof = bank()
    block10 = np.load(ROOT / "data/cache/folds/block10_10f_seed20261010.npy")
    ids = tr.id.to_numpy()
    non_eval = np.flatnonzero(block10 != args.fold)
    eval_rows = np.flatnonzero(block10 == args.fold)
    if args.arm == "control":
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        ctx = np.sort(rng.choice(non_eval, size=CONTROL_ROWS, replace=False))
        expected_rows = CONTROL_ROWS
    else:
        ctx = non_eval
        expected_rows = TREATMENT_ROWS
    assert len(ctx) == expected_rows
    assert not np.intersect1d(ctx, eval_rows).size

    x, _, names, cats, maps = frames(tr, te, True)
    assert cats == [13, 14, 15, 16, 18, 21] and len(names) == 22
    params = {
        "model_path": str(ROOT / "artifacts/tabpfn35/tabpfn-v3.5-20260909.safetensors"),
        "n_estimators": 2, "auto_scale_n_estimators": False,
        "random_state": MODEL_SEED, "device": "cuda",
        "ignore_pretraining_limits": True, "fit_mode": "fit_with_cache",
        "kv_cache_precision": "int8", "keep_cache_on_device": True,
        "inference_precision": "autocast",
        "categorical_features_indices": cats,
        "memory_saving_mode": True, "n_preprocessing_jobs": 1, "tuning_config": None,
    }
    frozen = json.loads((ROOT / "reports/phase17_internal2_serial_equivalence_20261009.json").read_text())
    assert frozen["status"] == "PASS_REQUIRES_EXACT_FULL_B0"
    assert frozen["library_source_sha256"] == library_sources()

    contract = {
        "hypothesis_id": "H18-A", "stage": "S1_matched_discovery_arm",
        "arm": args.arm, "eval_fold_block10": args.fold,
        "git": git_commit(),
        "prereg_sha256": file_sha256(ROOT / "configs/phase18/h18a_coverage_prereg.json"),
        "s0_probe_sha256": file_sha256(OUT / "h18a_s0_probe.json"),
        "context_rows": int(len(ctx)), "eval_rows": int(len(eval_rows)),
        "context_ids_sha256": arr_sha256(ids[ctx]), "eval_ids_sha256": arr_sha256(ids[eval_rows]),
        "feature_context_sha256": arr_sha256(x[ctx]), "feature_eval_sha256": arr_sha256(x[eval_rows]),
        "control_subsample_seed": SUBSAMPLE_SEED, "model_seed": MODEL_SEED,
        "params": params, "allocator_ceiling_fraction": ALLOCATOR_FRACTION,
        "bank": proof, "library_source_sha256": library_sources(),
        "only_variable": "context size; both arms share eval rows, params, seeds, estimator count",
    }
    save_json({**contract, "status": "FITTING"}, root / "progress.json")
    np.save(root / "context_ids.npy", ids[ctx])
    np.save(root / "eval_ids.npy", ids[eval_rows])

    from src.models.windows_attention import register  # noqa: E402

    register(reuse_query_output=True)
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_FRACTION)
    start = time.monotonic()
    clf = None
    result = dict(contract)
    try:
        with inference_guard(root, contract, max_seconds=3600, min_available_gib=4):
            clf = create_model(params, True, 262144, 1, None, True)
            torch.cuda.reset_peak_memory_stats()
            prepared = prepare(clf, x[ctx], y[ctx])
            metadata = prepared[-1]
            expected = frozen["official_metadata"]["members"]
            assert [m["config_sha256"] for m in metadata["members"]] == \
                [m["config_sha256"] for m in expected], "member configs diverged from frozen probe"
            assert [m["pipeline_seed"] for m in metadata["members"]] == \
                [m["pipeline_seed"] for m in expected], "pipeline seeds diverged"
            assert metadata["softmax_temperature"] == frozen["official_metadata"]["softmax_temperature"]
            assert metadata["average_before_softmax"] == frozen["official_metadata"]["average_before_softmax"]
            save_json({"contract": contract, "metadata": metadata}, root / "configuration.json")
            fit_seconds = time.monotonic() - start
            peak_fit = int(torch.cuda.max_memory_allocated())
            gc.collect(); torch.cuda.empty_cache()
            estimator_started = {}

            def context(i, phase):
                folder = root / f"member_{i}_{phase}"
                folder.mkdir(exist_ok=True)
                if phase == "fit":
                    estimator_started[i] = time.monotonic()
                remaining = 2700 - (time.monotonic() - estimator_started[i])
                return inference_guard(folder, contract, max_seconds=max(1, remaining),
                                       min_available_gib=4 if phase == "fit" else 2)

            def progress(i, stop, fit_s, elapsed):
                save_json({"status": "PREDICTING", "member": i, "completed_rows": stop,
                           "seconds": time.monotonic() - start}, root / "progress.json")

            def complete(i, values, stats):
                np.save(root / f"member_{i}_raw_logits.npy", values)
                save_json(stats, root / f"member_{i}_stats.json")

            probabilities, _, md = sequential_predict(
                clf, prepared, x[eval_rows],
                context=context, progress=progress, estimator_complete=complete)
            p = probabilities[:, 1].astype("float32")
        assert p.shape == (len(eval_rows),) and np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        np.save(root / "prediction.npy", p)
        result.update(
            status="COMPLETE", fit_seconds=fit_seconds,
            peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
            prediction_sha256=arr_sha256(p),
            auc=float(roc_auc_score(y[eval_rows], p)),
            estimator_metadata=md,
        )
    except (ResourcePreflightError, torch.cuda.OutOfMemoryError) as error:
        result.update(status="INVALID_RESOURCE_NO_MODELLING_VERDICT", error=repr(error))
        save_json(result, out_path)
        raise
    except Exception as error:
        result.update(status="INVALID_IMPLEMENTATION_NO_MODELLING_VERDICT", error=repr(error))
        save_json(result, out_path)
        raise
    finally:
        if clf is not None:
            try:
                release(clf)
            except Exception:  # noqa: BLE001
                pass
        del clf
        gc.collect(); torch.cuda.empty_cache()
        save_json(result, out_path)
        print(json.dumps({k: result[k] for k in ("status", "arm", "context_rows",
                                                 "auc", "fit_seconds",
                                                 "peak_gpu_bytes") if k in result}, indent=2))


if __name__ == "__main__":
    main()
