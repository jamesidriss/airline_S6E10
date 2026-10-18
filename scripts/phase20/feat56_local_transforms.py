"""P20-FEAT56-LOCAL-TRANSFORMS (item 56) -- prereg configs/phase20/feat56_local_transforms_prereg.json.

Context-sensitive numeric transforms: empirical-CDF (quantile) transform of the
4 continuous covariates computed WITHIN each (Customer Type x Class) segment on
outer-FIT rows only, then applied to both FIT and EVAL rows.

Label-free (no satisfaction label used anywhere in the transform), and fitted on
outer-FIT covariates only -- strictly stronger than the transductive allowance.

Frozen decision rule (prereg text, applied literally):
  PROMISING iff fold-0 port_delta >= +5e-5 AND corr_vs_portfolio < 0.995,
  port_delta = AUC(equal-logit blend(candidate, route_ref)) - AUC(route_ref)
  on fold-0 rows; corr = Spearman(candidate, route_ref) on fold-0 rows.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat56_local_transforms.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat56_local_transforms"
PRED_OUT = ROOT / "artifacts" / "phase20" / "feat56_local_transforms"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG = ROOT / "configs" / "phase20" / "feat56_local_transforms_prereg.json"

NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
SEG_COLS = ["Customer Type", "Class"]
ALL_COV = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
    "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
    "Gender", "Customer Type", "Type of Travel", "Class",
]
assert len(ALL_COV) == 21
PORTFOLIO_REF = "sol_route_aux10_primary_final"


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def raw21(tr) -> np.ndarray:
    n = len(tr)
    X = np.zeros((n, 21), dtype=np.float32)
    for j, c in enumerate(ALL_COV):
        col = tr[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            X[:, j] = col.astype("category").cat.codes.astype(np.float32)
        else:
            X[:, j] = col.astype(np.float32)
    for j in range(21):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad])
    return X


def seg_codes(tr) -> np.ndarray:
    """Joint segment code: Customer Type x Class (<= 6 segments)."""
    ct = tr["Customer Type"].astype("category").cat.codes.to_numpy()
    cl = tr["Class"].astype("category").cat.codes.to_numpy()
    return (ct * 8 + cl).astype(np.int64)


def local_ecdf_transform(tr, fit_m: np.ndarray) -> np.ndarray:
    """(n, 4) within-segment ECDF scores; source rows = outer-FIT only."""
    n = len(tr)
    out = np.zeros((n, len(NUM_COLS)), dtype=np.float32)
    segs = seg_codes(tr)
    for j, c in enumerate(NUM_COLS):
        vals = tr[c].to_numpy(dtype=np.float64)
        med = np.nanmedian(vals[fit_m])
        vals = np.where(np.isfinite(vals), vals, med)
        for s in np.unique(segs[fit_m]):
            src = fit_m & (segs == s)
            tgt = segs == s
            svals = vals[src]
            order = np.sort(svals)
            # map via ECDF: fraction of FIT-in-segment values <= v
            out[tgt, j] = np.searchsorted(order, vals[tgt], side="right").astype(np.float32) / len(order)
        # segments unseen in FIT (should not happen) fall back to global FIT ECDF
        unseen = np.array([not (fit_m & (segs == s)).any() for s in segs])
        if unseen.any():
            order = np.sort(vals[fit_m])
            out[unseen, j] = np.searchsorted(order, vals[unseen], side="right").astype(np.float32) / len(order)
    return out


def lgbm_eval(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, float, int]:
    from lightgbm import LGBMClassifier
    eval_m = folds == 0
    fit_m = ~eval_m
    rng = np.random.default_rng(1)
    yfit = y[fit_m]
    val_parts = []
    for cls in (0, 1):
        ci = np.where(yfit == cls)[0]
        nv = max(1, int(round(0.1 * len(ci))))
        val_parts.append(rng.choice(ci, nv, replace=False))
    inner_val = np.concatenate(val_parts)
    inner_train = np.setdiff1d(np.arange(fit_m.sum()), inner_val)
    X_fit = X[fit_m]
    m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **{
        "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
        "n_jobs": 8, "random_state": 1})
    m.fit(X_fit[inner_train], y[fit_m][inner_train],
          eval_set=[(X_fit[inner_val], y[fit_m][inner_val])], eval_metric="auc")
    pred = m.predict_proba(X[eval_m])[:, 1]
    return pred, float(roc_auc_score(y[eval_m], pred)), int(m.best_iteration_)


def main() -> int:
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)
    fit_m = folds != 0
    eval_m = ~fit_m

    X_base = raw21(tr)
    local = local_ecdf_transform(tr, fit_m)
    X_treat = np.column_stack([X_base, local]).astype(np.float32)

    pred_c, auc_c, it_c = lgbm_eval(X_base, y, folds)
    pred_t, auc_t, it_t = lgbm_eval(X_treat, y, folds)

    port = load_oof(PORTFOLIO_REF)[eval_m]
    y_e = y[eval_m]
    auc_port = float(roc_auc_score(y_e, port))
    blend = (logit(pred_t) + logit(port)) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    port_delta = auc_blend - auc_port
    corr = float(spearmanr(pred_t, port).statistic)
    promising = bool(port_delta >= 5e-5 and corr < 0.995)

    report = {
        "status": "DONE",
        "prereg": "configs/phase20/feat56_local_transforms_prereg.json",
        "prereg_sha256": sha,
        "method": "within-(Customer Type x Class) ECDF transform of 4 continuous covariates, "
                  "fitted on outer-FIT rows only (label-free); added to 21raw.",
        "metrics": {
            "control_fold0_auc": round(auc_c, 6),
            "treatment_fold0_auc": round(auc_t, 6),
            "augmentation_delta_descriptive_only": round(auc_t - auc_c, 8),
            "portfolio_ref": PORTFOLIO_REF,
            "portfolio_fold0_auc": round(auc_port, 6),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "equal_logit_blend_fold0_auc": round(auc_blend, 6),
            "port_delta": round(port_delta, 8),
            "promising_frozen_rule": promising,
            "best_iter_control": it_c,
            "best_iter_treatment": it_t,
            "eval_rows": int(eval_m.sum()),
        },
        "eval_rows": int(eval_m.sum()),
        "n_local_features": len(NUM_COLS),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat56_local_transforms.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    np.save(PRED_OUT / "fold0_candidate.npy", pred_t.astype(np.float32))

    m = report["metrics"]
    print(f"control={m['control_fold0_auc']} treatment={m['treatment_fold0_auc']} "
          f"aug_delta={m['augmentation_delta_descriptive_only']:+.2e} (descriptive)")
    print(f"port_delta={m['port_delta']:+.3e} (need >= +5e-5)  corr={m['candidate_spearman_vs_portfolio']} "
          f"(need < 0.995)  -> PROMISING={promising}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
