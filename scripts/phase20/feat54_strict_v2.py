"""P20-FEAT54-STRICT-V2 -- prereg 4d200f7d..., frozen before results.

Strict fold-safe re-measurement of smoothed categorical encodings:
  * EVAL (fold-0) encodings: statistics from outer-FIT (folds 1-4) ONLY.
  * FIT (folds 1-4) encodings: inner cross-fit WITHIN outer-FIT (each FIT row
    gets stats from the other three FIT folds). No fold-0 label anywhere in
    any encoding; no row sees its own label.
  * ES: inner 10% carved from FIT only, seed 1; ES labels used only as the
    early-stopping metric, never encoded into features.
Frozen decision rule (applied literally):
  PROMISING iff port_delta >= +5e-5 AND corr_vs_portfolio < 0.995,
  port_delta = AUC(equal-logit blend(candidate, route_ref)) - AUC(route_ref)
  measured on fold-0 rows; corr = Spearman(candidate, route_ref) on fold-0.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat54_strict_v2.py
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

OUT = ROOT / "reports" / "phase20" / "feat54_smoothed_categorical"
PRED_OUT = ROOT / "artifacts" / "phase20" / "feat54_strict_v2"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG = ROOT / "configs" / "phase20" / "feat54_strict_prereg.json"

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
ALL_COV = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
    "Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
    "Gender", "Customer Type", "Type of Travel", "Class",
]
assert len(ALL_COV) == 21
ALPHA = 10.0
PORTFOLIO_REF = "sol_route_aux10_primary_final"

EXPECTED_SHA = "4d200f7db408b058c490505e32b488d3f8e4502f342d508db73b1f7d80894fd6"


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


def smoothed_from(fit_idx: np.ndarray, y: np.ndarray, codes: np.ndarray,
                  eval_idx: np.ndarray, global_mean: float | None = None):
    """Bayesian-smoothed target encoding for rows in eval_idx using ONLY fit_idx labels."""
    fit_codes = codes[fit_idx]
    fit_y = y[fit_idx]
    if global_mean is None:
        global_mean = float(fit_y.mean())
    uniq, counts = np.unique(fit_codes, return_counts=True)
    means = np.array([fit_y[fit_codes == c].mean() for c in uniq])
    w = counts / (counts + ALPHA)
    smoothed = w * means + (1.0 - w) * global_mean
    lookup = dict(zip(uniq.tolist(), smoothed.tolist()))
    vals = np.array([lookup.get(c, global_mean) for c in codes[eval_idx]], dtype=np.float32)
    return vals, global_mean


def es_split(y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Global boolean mask of inner-ES rows: stratified 10% of outer-FIT, seed 1.

    ES labels are NEVER used in any encoding source; they are only the
    early-stopping metric. Control and treatment share this exact split.
    """
    fit_m = folds != 0
    rng = np.random.default_rng(1)
    yfit = y[fit_m]
    parts = []
    for cls in (0, 1):
        ci = np.where(yfit == cls)[0]
        nv = max(1, int(round(0.1 * len(ci))))
        parts.append(rng.choice(ci, nv, replace=False))
    es_local = np.concatenate(parts)
    es_global = np.where(fit_m)[0][es_local]
    mask = np.zeros(len(y), dtype=bool)
    mask[es_global] = True
    return mask


def build_encodings(y: np.ndarray, folds: np.ndarray, es_mask: np.ndarray) -> np.ndarray:
    """(n, 4) strictly fold-safe smoothed encodings.

    Source-label rules (per prereg 4d200f7d):
      * no fold-0 label ever used,
      * no ES label ever used as a source (ES = early-stopping metric only),
      * no row's own inner fold contributes to that row's encoding.
    """
    n = len(y)
    out = np.zeros((n, len(CAT_COLS)), dtype=np.float32)
    eval_m = folds == 0
    fit_m = ~eval_m
    src_m = fit_m & ~es_mask          # allowed source rows: outer-FIT, non-ES
    eval_idx = np.where(eval_m)[0]
    codes = {c: y_codes[c] for c in CAT_COLS}
    for j, c in enumerate(CAT_COLS):
        # EVAL rows: stats from outer-FIT (minus ES) only.
        vals, _ = smoothed_from(np.where(src_m)[0], y, codes[c], eval_idx)
        out[eval_idx, j] = vals
        # FIT rows: cross-fit within outer-FIT, own inner fold excluded, ES excluded.
        for inner in (1, 2, 3, 4):
            row_m = fit_m & (folds == inner)
            row_idx = np.where(row_m)[0]
            src_idx = np.where(src_m & (folds != inner))[0]
            vals_i, _ = smoothed_from(src_idx, y, codes[c], row_idx)
            out[row_idx, j] = vals_i
    return out


def lgbm_eval(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
              es_mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    from lightgbm import LGBMClassifier
    eval_m = folds == 0
    fit_m = ~eval_m
    fit_global = np.where(fit_m)[0]
    inner_val = np.where(es_mask[fit_m])[0]      # ES positions within FIT
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
    assert sha == EXPECTED_SHA, f"prereg sha mismatch {sha} != {EXPECTED_SHA}"

    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)

    global y_codes
    y_codes = {}
    for c in CAT_COLS:
        col = tr[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            y_codes[c] = col.astype("category").cat.codes.to_numpy()
        else:
            y_codes[c] = col.to_numpy().astype(np.int64)

    X_base = raw21(tr)
    es_mask = es_split(y, folds)
    enc = build_encodings(y, folds, es_mask)
    X_treat = np.column_stack([X_base, enc]).astype(np.float32)

    pred_c, auc_c, it_c = lgbm_eval(X_base, y, folds, es_mask)
    pred_t, auc_t, it_t = lgbm_eval(X_treat, y, folds, es_mask)

    eval_m = folds == 0
    y_e = y[eval_m]
    port = load_oof(PORTFOLIO_REF)[eval_m]

    cand = pred_t
    auc_port = float(roc_auc_score(y_e, port))
    blend = (logit(cand) + logit(port)) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    port_delta = auc_blend - auc_port
    corr = float(spearmanr(cand, port).statistic)

    promising = bool(port_delta >= 5e-5 and corr < 0.995)

    report = {
        "status": "DONE",
        "prereg": "configs/phase20/feat54_strict_prereg.json",
        "prereg_sha256": sha,
        "supersedes": {
            "report": "reports/phase20/feat54_smoothed_categorical/feat54_smoothed_categorical.json",
            "note": "v1 INVALID_PROTOCOL (see correction_20261018_d in v1 report); preserved.",
        },
        "protocol": "strict outer-FIT-only EVAL encodings + inner cross-fit within outer-FIT "
                    "for FIT rows; ES labels used only as early-stopping metric",
        "metrics": {
            "control_fold0_auc": round(auc_c, 6),
            "treatment_fold0_auc": round(auc_t, 6),
            "augmentation_delta_descriptive_only": round(auc_t - auc_c, 8),
            "portfolio_ref": PORTFOLIO_REF,
            "portfolio_fold0_auc": round(auc_port, 6),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "equal_logit_blend_fold0_auc": round(auc_blend, 6),
            "port_delta": round(port_delta, 8),
            "port_delta_threshold": 5e-5,
            "corr_threshold": 0.995,
            "promising_frozen_rule": promising,
            "best_iter_control": it_c,
            "best_iter_treatment": it_t,
            "eval_rows": int(eval_m.sum()),
        },
        "eval_rows": int(eval_m.sum()),
        "n_smooth_features": len(CAT_COLS),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat54_strict_v2.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    np.save(PRED_OUT / "fold0_candidate.npy", cand.astype(np.float32))

    m = report["metrics"]
    print(f"control={m['control_fold0_auc']} treatment={m['treatment_fold0_auc']} "
          f"aug_delta={m['augmentation_delta_descriptive_only']:+.2e} (descriptive)")
    print(f"port_delta={m['port_delta']:+.3e} (need >= +5e-5)  corr={m['candidate_spearman_vs_portfolio']} "
          f"(need < 0.995)  -> PROMISING={promising}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
