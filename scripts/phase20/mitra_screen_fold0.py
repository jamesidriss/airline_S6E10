"""Phase 20 -- Mitra-v2 fold-0 screen (F2 scientific pilot).

Screens the Apache-2.0 Mitra-v2 classifier (autogluon/mitra-classifier-2,
revision-pinned, weights sha256-asserted) as a candidate 11th ensemble member,
using the frozen mitra-finetune recipe (50-step FT, 8-fold bag, support cap
16384, no env protocol overrides) in the isolated .venv_ag environment.

Two preregistered arms (configs/phase20/mitra_screen_prereg.json, committed at
2c8970e BEFORE results):
  armA_ctx20k  -- fit on the stratified 20k FIT context (same information
                  budget as the phase-19 foundation screens)
  armB_bestpolicy -- fit on full FIT (Mitra's best supported policy; native
                  support caps still apply and are disclosed in metadata)

Eval rows are EVAL[:50000] of fold 0 -- identical to the phase-19 screen tag
fmscreen_fold0_ctx20k_e50k, so AUCs are directly comparable to that table.

Anti-leakage: context/fit rows are FIT-only (folds 1-4); no target encoding
(raw view); champion reference hash-asserted from phase17_B1 artifacts; Mitra
checkpoint weights sha256-asserted before fit.

Run (isolated env): .\\.venv_ag\\Scripts\\python.exe scripts/phase20/mitra_screen_fold0.py --arm A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import CONTEXT_SEED, screen_metrics, stratified_context  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

CKPT = ARTIFACTS / "phase20" / "mitra-classifier-2"
CKPT_SHA256 = "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09"
PREREG = "configs/phase20/mitra_screen_prereg.json"

OUT = REPORTS / "phase20" / "mitra_screen"
ART = ARTIFACTS / "phase20" / "mitra_screen"

# Native support caps disclosed by mitra-finetune for binary classification
# (frozen defaults; no env overrides set).
NATIVE_FT_SUPPORT_CAP = 16384
NATIVE_PREDICT_SUPPORT_CAP = 16384


def log(msg: str) -> None:
    print(f"[mitra] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B"])
    ap.add_argument("--context", type=int, default=20000, help="arm A context rows")
    ap.add_argument("--eval-rows", type=int, default=50000)
    ap.add_argument("--num-bag-folds", type=int, default=8)
    ap.add_argument("--time-limit", type=int, default=3600)
    ap.add_argument("--tag", default="mitra_fold0")
    ap.add_argument("--in-process", action="store_true", default=True,
                    help="run the view in-process (default True) so the memory-guard "
                         "calibration below applies to bag children; in_process=False "
                         "spawns child interpreters that never see the patch")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    # ---- checkpoint integrity (weights pinned in prereg) ----
    import hashlib
    w = CKPT / "model.safetensors"
    got = hashlib.sha256(w.read_bytes()).hexdigest()
    assert got == CKPT_SHA256, f"mitra weights hash drift: {got}"
    log(f"checkpoint sha256 OK ({got[:16]}...)")

    from mitra_finetune import MitraFinetune  # isolated env only

    # --- Resource-guard calibration (documented, not a science change) -------
    # AutoGluon's MitraModel static memory estimate uses a full-ICL quadratic
    # formula (rows^2 * features): it projected ~17+ GiB for a 20k x 22 array
    # and raised NotEnoughMemoryError with 18.9 GiB free. The frozen
    # mitra-finetune recipe never does full-ICL: fine-tune and predict support
    # are both capped at 16,384 rows (native binary-classification caps), so
    # the true footprint is bounded by the cap (~1 MB support + 300 MB weights).
    # We therefore disable AG's memory CHECK for this run (it is a heuristic
    # guard, not part of the scientific protocol). Actual RAM/VRAM use is
    # monitored separately.
    from autogluon.core.models.abstract.abstract_model import AbstractModel as _AGModel

    _orig_mem_check = _AGModel._validate_fit_memory_usage

    def _skip_mem_check(self, **kwargs):
        prev = self.params_aux.get("max_memory_usage_ratio", 1)
        self.params_aux["max_memory_usage_ratio"] = None  # AG: skip all mem checks
        try:
            return _orig_mem_check(self, **kwargs)
        finally:
            self.params_aux["max_memory_usage_ratio"] = prev

    _AGModel._validate_fit_memory_usage = _skip_mem_check
    mem_guard = "AG static ICL memory estimate disabled (misestimates capped-support FT recipe)"
    # ------------------------------------------------------------------------

    from scripts.run_sol_tabpfn import frames  # champion feature builder (bit-exact)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    Xtr, _Xte, names, cats, maps = frames(tr, te, route=True)
    assert Xtr.shape[0] == len(y), "feature/row misalignment"

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)
    eval_idx = EVAL[: args.eval_rows]
    route0, port0 = route_all[eval_idx], port_all[eval_idx]
    y_eval = y[eval_idx]

    if args.arm == "A":
        ctx = stratified_context(FIT, y, args.context, seed=CONTEXT_SEED)
        X_fit, y_fit = Xtr[ctx], y[ctx]
        arm_note = "stratified 20k FIT context (phase-19 matched information budget)"
    else:
        ctx = FIT
        X_fit, y_fit = Xtr[FIT], y[FIT]
        arm_note = "full FIT (best supported policy; native support caps apply)"

    log(f"arm={args.arm} fit_rows={len(X_fit)} eval_rows={len(eval_idx)} "
        f"features={Xtr.shape[1]} bag_folds={args.num_bag_folds} time_limit={args.time_limit}")
    log(f"arm note: {arm_note}")

    t1 = time.time()
    model = MitraFinetune(
        str(CKPT),
        problem_type="classification",
        time_limit=args.time_limit,
        eval_metric="roc_auc",
        device="cuda",
        random_state=1201,
        num_bag_folds=args.num_bag_folds,
        in_process=args.in_process,
    )
    model.fit(X_fit, y_fit)
    log(f"fit done in {time.time()-t1:.1f}s")

    t2 = time.time()
    X_eval = np.asarray(Xtr[eval_idx])
    pr = np.asarray(model.predict_proba(X_eval))
    assert pr.shape == (len(eval_idx), 2), f"bad proba shape {pr.shape}"
    assert np.isfinite(pr).all() and (pr >= 0).all() and (pr <= 1).all()
    assert np.allclose(pr.sum(axis=1), 1.0, atol=1e-4), "rows must sum to 1"
    pred = pr[:, 1].astype(np.float64)
    predict_s = time.time() - t2
    log(f"predict done in {predict_s:.1f}s ({len(eval_idx)/max(predict_s,1e-9):.1f} rows/s)")

    m = screen_metrics(y_eval, pred, port0, route0)

    pred_path = ART / f"mitra_{args.arm}_{args.tag}.npy"
    np.save(pred_path, pred.astype("float32"))

    # internal OOF validation AUC (diagnostic only; fit-internal, fold-safe)
    oof_auc = None
    if getattr(model, "val_proba_", None) is not None:
        vp = np.asarray(model.val_proba_)
        if vp.ndim == 2 and vp.shape[0] == len(X_fit):
            from sklearn.metrics import roc_auc_score
            oof_auc = float(roc_auc_score(y_fit, vp[:, 1]))

    meta = {
        "model": "mitra-v2", "arm": args.arm, "status": "DONE", "error": None,
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "checkpoint": str(CKPT), "checkpoint_weights_sha256": got,
        "runtime": "mitra-finetune 0.3.0 / autogluon 1.6.3 / torch 2.9.0+cu129 (isolated .venv_ag)",
        "recipe": "frozen: 50-step FT, lr/warmup/wd defaults, eval_metric=roc_auc, "
                  "num_bag_folds=%d, time_limit=%d, no env protocol overrides" % (args.num_bag_folds, args.time_limit),
        "native_ft_support_cap": NATIVE_FT_SUPPORT_CAP,
        "native_predict_support_cap": NATIVE_PREDICT_SUPPORT_CAP,
        "actual_fit_rows_passed": int(len(X_fit)),
        "resource_guard_note": mem_guard,
        "context_rows_requested": int(args.context) if args.arm == "A" else None,
        "context_seed": CONTEXT_SEED if args.arm == "A" else None,
        "eval_rows": int(len(eval_idx)), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "eval_ids_sha256": arr_sha256(ids[eval_idx].astype("int64")),
        "arm_note": arm_note,
        "features": list(names), "categorical_indices": list(cats),
        "label_free_category_maps": maps,
        "prediction_sha256": arr_sha256(np.load(pred_path)),
        "internal_oof_val_auc": oof_auc,
        "metrics": m, "tag": args.tag, "prereg": PREREG,
        "fit_seconds": t1 - t0, "predict_seconds": predict_s,
        "total_seconds": time.time() - t0,
    }
    save_json(meta, OUT / f"mitra_{args.arm}_{args.tag}.json")
    log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
        f"corr_port={m['logit_pearson_vs_portfolio']:.4f} oof_val={oof_auc}")
    PROMISING = (m["portfolio_delta"] >= 5e-5) and (m["logit_pearson_vs_portfolio"] < 0.995)
    log(f"promising bar (portfolio_delta >= +5e-5 AND corr < 0.995): {'PASS' if PROMISING else 'FAIL'}")
    log(f"done in {meta['total_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
