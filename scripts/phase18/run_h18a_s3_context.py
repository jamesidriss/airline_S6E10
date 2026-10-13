"""H18-A stage S3 — one 90%-coverage context of the 10-fold route OOF.

Context = all block10-fold-k complement rows (629,671 = 90% coverage); eval =
the fold-k rows (69,964). This is the same geometry as the S1 treatment arm;
for block10 fold 0 the completed S1 treatment artifacts are REUSED after full
hash verification (no GPU). Frozen v7 params, n_estimators=2, seed 1201 —
identical to S1/S2, no param change after those stages.

Launched ONLY after the preregistered S3-launch rule fires (S1+S2 mean matched
route delta >= 5.6e-5). One context per invocation (<=1 heavy GPU job).
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
CONTEXT_ROWS = 629_671
MODEL_SEED = 1201
ALLOCATOR_FRACTION = 0.95


def reuse_s1_treatment(fold: int) -> None:
    """Fold 0: alias the certified S1 treatment prediction; refuse on any drift."""
    s1_root = ARTIFACTS / "phase18" / f"h18a_s1_f{fold}_treatment"
    s1_report = json.loads((OUT / f"h18a_s1_f{fold}_treatment.json").read_text(encoding="utf-8"))
    assert s1_report["status"] == "COMPLETE" and s1_report["arm"] == "treatment"
    assert s1_report["context_rows"] == CONTEXT_ROWS and s1_report["params"]["n_estimators"] == 2
    assert s1_report["model_seed"] == MODEL_SEED
    p = np.load(s1_root / "prediction.npy")
    assert arr_sha256(p) == s1_report["prediction_sha256"]

    tr, te, y, _, _, _, _ = bank()
    block10 = np.load(ROOT / "data/cache/folds/block10_10f_seed20261010.npy")
    ids = tr.id.to_numpy()
    non_eval = np.flatnonzero(block10 != fold)
    ev = np.flatnonzero(block10 == fold)
    assert len(non_eval) == CONTEXT_ROWS
    assert arr_sha256(ids[non_eval]) == s1_report["context_ids_sha256"], "context drift vs S1"
    assert arr_sha256(ids[ev]) == s1_report["eval_ids_sha256"], "eval drift vs S1"

    root = ARTIFACTS / "phase18" / f"h18a_s3_f{fold}"
    if root.exists() or (OUT / f"h18a_s3_f{fold}.json").exists():
        raise FileExistsError("Preserve existing context")
    root.mkdir(parents=True)
    np.save(root / "prediction.npy", p)
    np.save(root / "eval_ids.npy", ids[ev])
    result = {
        "hypothesis_id": "H18-A", "stage": "S3_context", "context_fold_block10": fold,
        "status": "COMPLETE_REUSED_S1_TREATMENT",
        "git": git_commit(),
        "context_rows": CONTEXT_ROWS, "eval_rows": int(len(ev)),
        "auc": float(roc_auc_score(y[ev], p)),
        "prediction_sha256": arr_sha256(p),
        "context_ids_sha256": arr_sha256(ids[non_eval]), "eval_ids_sha256": arr_sha256(ids[ev]),
        "reuse_source_report": f"reports/phase18/h18a_s1_f{fold}_treatment.json",
        "reuse_source_report_sha256": file_sha256(OUT / f"h18a_s1_f{fold}_treatment.json"),
        "gpu_burned": 0.0,
    }
    save_json(result, OUT / f"h18a_s3_f{fold}.json")
    print(json.dumps(result, indent=2))


def run_context(fold: int) -> None:
    out_path = OUT / f"h18a_s3_f{fold}.json"
    root = ARTIFACTS / "phase18" / f"h18a_s3_f{fold}"
    if out_path.exists() or root.exists():
        raise FileExistsError("Preserve existing context")
    probe = json.loads((OUT / "h18a_s0_probe.json").read_text(encoding="utf-8"))
    if probe["status"] != "S0_PROBE_PASS_FEASIBLE":
        raise SystemExit(f"S0 probe status {probe['status']} does not authorize S3")
    s02 = [json.loads((OUT / f"h18a_s1_f{k}_comparison.json").read_text(encoding="utf-8"))
           for k in (0, 1)]
    mean_route_delta = float(np.mean([c["delta"] for c in s02]))
    if mean_route_delta < 5.6e-5:
        raise SystemExit(f"preregistered S3-launch rule failed: S1+S2 mean route delta "
                         f"{mean_route_delta:.3e} < 5.6e-5")

    tr, te, y, _, _, _, proof = bank()
    block10 = np.load(ROOT / "data/cache/folds/block10_10f_seed20261010.npy")
    ids = tr.id.to_numpy()
    ctx = np.flatnonzero(block10 != fold)
    ev = np.flatnonzero(block10 == fold)
    assert len(ctx) == CONTEXT_ROWS and not np.intersect1d(ctx, ev).size
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
        "hypothesis_id": "H18-A", "stage": "S3_context", "context_fold_block10": fold,
        "git": git_commit(),
        "prereg_sha256": file_sha256(ROOT / "configs/phase18/h18a_coverage_prereg.json"),
        "s0_probe_sha256": file_sha256(OUT / "h18a_s0_probe.json"),
        "s1_s2_mean_route_delta": mean_route_delta,
        "s3_launch_rule_passed": mean_route_delta >= 5.6e-5,
        "context_rows": int(len(ctx)), "eval_rows": int(len(ev)),
        "context_ids_sha256": arr_sha256(ids[ctx]), "eval_ids_sha256": arr_sha256(ids[ev]),
        "feature_context_sha256": arr_sha256(x[ctx]), "feature_eval_sha256": arr_sha256(x[ev]),
        "model_seed": MODEL_SEED, "params": params,
        "allocator_ceiling_fraction": ALLOCATOR_FRACTION,
        "bank": proof, "library_source_sha256": library_sources(),
    }
    root.mkdir(parents=True)
    save_json({**contract, "status": "FITTING"}, root / "progress.json")
    np.save(root / "eval_ids.npy", ids[ev])

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
                clf, prepared, x[ev], context=context, progress=progress, estimator_complete=complete)
            p = probabilities[:, 1].astype("float32")
        assert p.shape == (len(ev),) and np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        np.save(root / "prediction.npy", p)
        result.update(
            status="COMPLETE", fit_seconds=fit_seconds,
            peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
            prediction_sha256=arr_sha256(p),
            auc=float(roc_auc_score(y[ev], p)),
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
        print(json.dumps({k: result[k] for k in ("status", "context_fold_block10", "auc",
                                                 "fit_seconds", "peak_gpu_bytes") if k in result},
                         indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, choices=range(10), required=True)
    args = ap.parse_args()
    if args.fold == 0:
        reuse_s1_treatment(0)
    else:
        run_context(args.fold)


if __name__ == "__main__":
    main()
