"""P20-FEAT49-CONDITIONAL-ORDINAL-EMBED (item 49) -- prereg
configs/phase20/feat49_conditional_ordinal_embed_prereg.json.

Adaptive (James-Stein) shrinkage target encoding of each of the 13 service
ratings CONDITIONED on the traveler context cell (Customer Type x Type of
Travel x Class): key = (rating value, context cell), one encoding column per
rating (13 added). Strict FEAT58 protocol: EVAL encodings from outer-FIT only;
FIT encodings inner cross-fit within outer-FIT; ES labels excluded from every
encoding source. Control vs treatment share identical LGBM hyperparameters.

Frozen decision rule (prereg text, applied literally):
  PROMISING iff fold-0 port_delta >= +5e-5 AND corr_vs_portfolio < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat49_conditional_ordinal_embed.py
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

from src.common import TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat49_conditional_ordinal_embed"
PRED_OUT = ROOT / "artifacts" / "phase20" / "feat49_conditional_ordinal_embed"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG = ROOT / "configs" / "phase20" / "feat49_conditional_ordinal_embed_prereg.json"

CONTEXT_COLS = ["Customer Type", "Type of Travel", "Class"]
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
assert len(RATING_COLS) == 13
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


def context_cell(tr) -> np.ndarray:
    key = np.zeros(len(tr), dtype=np.int64)
    for c in CONTEXT_COLS:
        key = key * 8 + tr[c].astype("category").cat.codes.to_numpy().astype(np.int64)
    return key


def rating_values(tr) -> np.ndarray:
    """(n, 13) integer rating values; NaN-safe (median-imputed per column)."""
    n = len(tr)
    V = np.zeros((n, len(RATING_COLS)), dtype=np.int64)
    for j, c in enumerate(RATING_COLS):
        col = tr[c].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad])
        V[:, j] = np.round(col).astype(np.int64)
    return V


def adaptive_encode(y: np.ndarray, keys: np.ndarray, src_idx: np.ndarray,
                    tgt_idx: np.ndarray, lam: float = 40.0) -> np.ndarray:
    """James-Stein-style adaptive shrinkage: w = n/(n+lam)."""
    src_k = keys[src_idx]
    src_y = y[src_idx]
    global_mean = float(src_y.mean())
    uniq, counts = np.unique(src_k, return_counts=True)
    means = np.array([src_y[src_k == k].mean() for k in uniq])
    w = counts / (counts + lam)
    enc = w * means + (1.0 - w) * global_mean
    lookup = dict(zip(uniq.tolist(), enc.tolist()))
    return np.array([lookup.get(k, global_mean) for k in keys[tgt_idx]], dtype=np.float32)


def es_split(y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    fit_m = folds != 0
    rng = np.random.default_rng(1)
    yfit = y[fit_m]
    parts = []
    for cls in (0, 1):
        ci = np.where(yfit == cls)[0]
        nv = max(1, int(round(0.1 * len(ci))))
        parts.append(rng.choice(ci, nv, replace=False))
    mask = np.zeros(len(y), dtype=bool)
    mask[np.where(fit_m)[0][np.concatenate(parts)]] = True
    return mask


def build_encodings(y: np.ndarray, folds: np.ndarray, es_mask: np.ndarray,
                    cell: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """(n, 13) fold-safe conditional encodings; protocol identical to FEAT58."""
    n, ncol = vals.shape
    out = np.zeros((n, ncol), dtype=np.float32)
    eval_m = folds == 0
    fit_m = ~eval_m
    src_m = fit_m & ~es_mask
    for j in range(ncol):
        # key = rating value * n_cells + context cell (unique per value+cell)
        keys = vals[:, j].astype(np.int64) * 1024 + cell
        src_idx = np.where(src_m)[0]
        # EVAL rows: outer-FIT (minus ES) statistics only.
        out[eval_m, j] = adaptive_encode(y, keys, src_idx, np.where(eval_m)[0])
        # FIT rows: cross-fit within outer-FIT, own inner fold excluded, ES excluded.
        for inner in (1, 2, 3, 4):
            row_m = fit_m & (folds == inner)
            row_idx = np.where(row_m)[0]
            s_idx = np.where(src_m & (folds != inner))[0]
            out[row_idx, j] = adaptive_encode(y, keys, s_idx, row_idx)
    return out


def lgbm_eval(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
              es_mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    from lightgbm import LGBMClassifier
    eval_m = folds == 0
    fit_m = ~eval_m
    inner_val = np.where(es_mask[fit_m])[0]
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
    eval_m = folds == 0

    cell = context_cell(tr)
    vals = rating_values(tr)
    n_cells = int(len(np.unique(cell)))
    es_mask = es_split(y, folds)
    enc = build_encodings(y, folds, es_mask, cell, vals)

    X_base = raw21(tr)
    X_treat = np.column_stack([X_base, enc]).astype(np.float32)

    pred_c, auc_c, it_c = lgbm_eval(X_base, y, folds, es_mask)
    pred_t, auc_t, it_t = lgbm_eval(X_treat, y, folds, es_mask)

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
        "prereg": "configs/phase20/feat49_conditional_ordinal_embed_prereg.json",
        "prereg_sha256": sha,
        "method": "adaptive-shrinkage target encoding (lam=40) of each of the 13 "
                  "service ratings conditioned on the context cell (Customer Type x "
                  "Type of Travel x Class); key = rating value x cell; strict "
                  "outer-FIT-only EVAL stats + inner cross-fit within outer-FIT, ES "
                  "labels excluded; added to 21raw (13 columns).",
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
        "n_context_cells": n_cells,
        "n_rating_columns": len(RATING_COLS),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat49_conditional_ordinal_embed.json"
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
