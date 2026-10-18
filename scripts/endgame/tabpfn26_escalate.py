"""Champion-B escalation: full 5-fold OOF + certified test inference for the
TabPFN-2.6 route member under the MATCHED champion-route protocol.

Runs ONLY if the fold-0 screen's frozen rule returns PROMOTE_TO_5FOLD
(reports/tabpfn26_screen_f0.json -> frozen_decision.decision).  The screen's
prereg (4ff41f64) governs; this script refuses to run otherwise, so a promotion
cannot happen after the fact.

Modes
  --mode oof    folds 1-4 (+ fold 0 if not already screened): writes
                artifacts/endgame/tabpfn26_oof/f{k}/prediction.npy and assembles
                a full 699,635-row OOF vector
  --mode test   fold-wise test inference (299,844 rows), fold k models use
                FIT rows = folds != k exactly as the OOF arm; writes
                artifacts/endgame/tabpfn26_test/f{k}/prediction.npy and
                assembles the mean-over-folds test vector

Usage:
  ./.venv/Scripts/python.exe scripts/endgame/tabpfn26_escalate.py --mode oof
  ./.venv/Scripts/python.exe scripts/endgame/tabpfn26_escalate.py --mode test
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import (ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit,  # noqa: E402
                        load_cached_parquet, save_json)
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_sol_tabpfn import create_model, frames  # noqa: E402

SCREEN_PREREG_SHA = "4ff41f646f4b166b7342b55eeaa2d49cc1b239b9eeb941b7471e0c8425b33631"
SCREEN_REPORT = "reports/tabpfn26_screen_f0.json"
CKPT = ROOT / "artifacts/tabpfn26/tabpfn-v2.6-classifier-v2.6_default.ckpt"
CKPT_SHA = "0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a"
CHAMPION = "v7_sol_tabpfn_internal2_aux10"
BATCH = 1024


def matched_params(cats):
    return dict(model_path=str(CKPT.resolve()), n_estimators=1, auto_scale_n_estimators=False,
                random_state=1201, device="cuda", ignore_pretraining_limits=True,
                fit_mode="fit_with_cache", kv_cache_precision="int8", keep_cache_on_device=True,
                inference_precision="autocast", categorical_features_indices=cats,
                memory_saving_mode=True, n_preprocessing_jobs=1, tuning_config=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oof", "test"), required=True)
    ap.add_argument("--folds", default=None, help="comma list; default = all five")
    args = ap.parse_args()

    # hard gate: the screen must have promoted, and its prereg must be the pinned one
    sha = hashlib.sha256((ROOT / "configs/phase20/endgame_tabpfn26_screen_prereg.json").read_bytes()).hexdigest()
    assert sha == SCREEN_PREREG_SHA, "screen prereg changed after the fact"
    screen = json.loads((ROOT / SCREEN_REPORT).read_text(encoding="utf-8"))
    decision = screen["frozen_decision"]["decision"]
    if decision != "PROMOTE_TO_5FOLD":
        raise SystemExit(f"REFUSING escalation: screen decision is {decision!r}, not PROMOTE_TO_5FOLD")
    assert file_sha256(CKPT) == CKPT_SHA

    torch.cuda.set_per_process_memory_fraction(0.85)
    from src.models.windows_attention import register, verify_gpu_equivalence

    verify_gpu_equivalence(reuse_query_output=True)
    verify_gpu_equivalence(torch.bfloat16, reuse_query_output=True)
    register(reuse_query_output=True)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    Xtr, Xte, names, cats, maps = frames(tr, te, True)

    fold_list = ([int(f) for f in args.folds.split(",")] if args.folds else list(range(5)))
    base = ROOT / ("artifacts/endgame/tabpfn26_oof" if args.mode == "oof" else "artifacts/endgame/tabpfn26_test")
    report_path = REPORTS / f"tabpfn26_{args.mode}.json"
    if report_path.exists():
        raise SystemExit(f"Preserve existing result: {report_path}")

    results = {}
    for k in fold_list:
        out = base / f"f{k}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "prediction.npy").exists():
            print(f"fold {k}: already present, skipping (idempotent)")
            continue
        fi = np.flatnonzero(folds != k)
        assert len(fi) == len(y) - int((folds == k).sum())
        if args.mode == "oof":
            va = np.flatnonzero(folds == k)
            X, yv = Xtr, y
        else:
            va = np.arange(len(te))          # test rows; no labels exist
            X, yv = Xte, None
        contract = {"git": git_commit(), "mode": args.mode, "fold": k, "model_family": "TabPFN-2.6",
                    "checkpoint_sha256": CKPT_SHA, "params": matched_params(cats), "seed": 1201,
                    "fit_rows": int(len(fi)), "apply_rows": int(len(va)),
                    "fit_ids_sha256": arr_sha256(ids[fi]), "fold_sha256": arr_sha256(folds),
                    "screen_decision": decision, "batch": BATCH,
                    "utc": datetime.now(timezone.utc).isoformat()}
        save_json({**contract, "status": "FITTING"}, out / "progress.json")
        start = time.monotonic()
        model = create_model(matched_params(cats), True, 262144, None, None, True)
        torch.cuda.reset_peak_memory_stats()
        model.fit(X[fi], y[fi])
        fit_seconds = time.monotonic() - start
        gc.collect(); torch.cuda.empty_cache()
        parts = []
        for begin in range(0, len(va), BATCH):
            stop = min(begin + BATCH, len(va))
            parts.append(model.predict_proba(X[va[begin:stop]])[:, 1])
            if stop == len(va) or stop <= BATCH or stop // BATCH % 40 == 0:
                save_json({**contract, "status": "PREDICTING", "completed_rows": stop,
                           "fit_seconds": fit_seconds, "seconds": time.monotonic() - start},
                          out / "progress.json")
                print(f"  fold {k}: {stop}/{len(va)} ({time.monotonic()-start:.0f}s)", flush=True)
        p = np.concatenate(parts).astype("float32")
        assert p.shape == (len(va),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        np.save(out / "prediction.npy", p)
        np.save(out / "apply_ids.npy", ids[va] if args.mode == "oof" else te["id"].to_numpy()[va])
        rec = {"seconds": time.monotonic() - start, "fit_seconds": fit_seconds,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
               "prediction_sha256": arr_sha256(p)}
        if args.mode == "oof":
            rec["auc"] = float(roc_auc_score(y[va], p))
        results[str(k)] = rec
        save_json(rec, out / "stats.json")
        del model
        gc.collect(); torch.cuda.empty_cache()
        print(f"fold {k} done: {rec['seconds']:.0f}s auc={rec.get('auc')}", flush=True)

    # ---- assemble ----
    assembled = None
    if args.mode == "oof":
        vec = np.zeros(len(y), dtype=np.float32)
        seen = np.zeros(len(y), dtype=bool)
        for k in range(5):
            f = base / f"f{k}" / "prediction.npy"
            if not f.exists():
                continue
            ids_k = np.load(base / f"f{k}" / "apply_ids.npy")
            p = np.load(f)
            pos = {v: i for i, v in enumerate(ids.tolist())}
            idx = np.array([pos[v] for v in ids_k.tolist()])
            assert not seen[idx].any(), f"fold {k} overlaps an earlier fold"
            vec[idx] = p
            seen[idx] = True
        assembled = {"kind": "oof", "covered": int(seen.sum()), "total": len(y),
                     "oof_sha256": arr_sha256(vec) if seen.all() else None,
                     "oof_auc": float(roc_auc_score(y, vec)) if seen.all() else None}
        if seen.all():
            np.save(base / "oof_full.npy", vec)
    else:
        vecs = [np.load(base / f"f{k}" / "prediction.npy") for k in range(5) if (base / f"f{k}" / "prediction.npy").exists()]
        if len(vecs) == 5:
            mean = np.mean(np.stack(vecs), axis=0).astype(np.float32)
            np.save(base / "test_mean.npy", mean)
            assembled = {"kind": "test", "folds": 5, "policy": "mean of per-fold probabilities "
                          "(matches the certified v6/v7 route test policy)", "n_rows": int(len(mean)),
                          "test_sha256": arr_sha256(mean), "mean": float(mean.mean())}
    save_json({"mode": args.mode, "folds": fold_list, "per_fold": results, "assembled": assembled,
               "screen_decision": decision, "utc": datetime.now(timezone.utc).isoformat()}, report_path)
    print(f"-> {report_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())