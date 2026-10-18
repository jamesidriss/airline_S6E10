"""P20-FEAT58-RARE-COMBINATION (item 58) -- prereg configs/phase20/feat58_rare_combination_prereg.json.

Adaptive (count-based) shrinkage target encoding of 4-way categorical
combinations (Gender x Customer Type x Type of Travel x Class, <= 48 cells).
Strict protocol identical to FEAT54 v2: EVAL encodings from outer-FIT only;
FIT encodings inner cross-fit within outer-FIT; ES labels excluded from all
encoding sources.

Frozen decision rule (prereg text, applied literally):
  PROMISING iff fold-0 port_delta >= +5e-5 AND corr_vs_portfolio < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat58_rare_combination.py
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

OUT = ROOT / "reports" / "phase20" / "feat58_rare_combination"
PRED_OUT = ROOT / "artifacts" / "phase20" / "feat58_rare_combination"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG = ROOT / "configs" / "phase20" / "feat58_rare_combination_prereg.json"

COMBO_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
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


def combo_key(tr) -> np.ndarray:
    key = np.zeros(len(tr), dtype=np.int64)
    for c in COMBO_COLS:
        key = key * 8 + tr[c].astype("category").cat.codes.to_numpy().astype(np.int64)
    return key


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


def build_encoding(y: np.ndarray, folds: np.ndarray, es_mask: np.ndarray,
                   keys: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float32)
    eval_m = folds == 0
    fit_m = ~eval_m
    src_m = fit_m & ~es_mask
    src_idx = np.where(src_m)[0]
    # EVAL rows: outer-FIT (minus ES) statistics only.
    out[eval_m] = adaptive_encode(y, keys, src_idx, np.where(eval_m)[0])
    # FIT rows: cross-fit within outer-FIT, own inner fold excluded, ES excluded.
    for inner in (1, 2, 3, 4):
        row_m = fit_m & (folds == inner)
        row_idx = np.where(row_m)[0]
        s_idx = np.where(src_m & (folds != inner))[0]
        out[row_idx] = adaptive_encode(y, keys, s_idx, row_idx)
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

    keys = combo_key(tr)
    es_mask = es_split(y, folds)
    enc = build_encoding(y, folds, es_mask, keys)

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
        "prereg": "configs/phase20/feat58_rare_combination_prereg.json",
        "prereg_sha256": sha,
        "method": "adaptive-shrinkage target encoding (lam=40) of 4-way categorical "
                  "combination keys, strict outer-FIT-only EVAL stats + inner cross-fit "
                  "within outer-FIT, ES labels excluded from sources; added to 21raw.",
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
        "n_combo_cells": int(len(np.unique(keys))),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat58_rare_combination.json"
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
