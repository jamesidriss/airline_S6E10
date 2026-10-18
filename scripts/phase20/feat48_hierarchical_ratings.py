"""P20-FEAT48-HIERARCHICAL-Ratings (item 48) -- prereg 84795f40, frozen before results.

Hierarchical structure: multi-output HistGB regressor predicting all 13 service
ratings from the other 20 covariates. The multi-output model captures shared
structure across ratings (correlated errors, common feature importance). Cross-fit
5-fold: train on folds 1-4, predict fold-0. Add 13 predictions as features to
rich view, LGBM fold-0 screen.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat48_hierarchical_ratings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat48_hierarchical_ratings"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
NUMERIC_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
ALL_COV = RATING_COLS + NUMERIC_COLS + CAT_COLS
assert len(ALL_COV) == 21

LGBM_PARAMS = {
    "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
    "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
    "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
    "n_jobs": 8, "random_state": 1,
}

CHAMPION_REF = 0.9618529228157235


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_design(tr, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_raw [n,21] numeric-encoded, X_hier [n,13] cross-fit multi-output predictions)."""
    n = len(tr)
    X_raw = np.zeros((n, 21), dtype=np.float32)
    for j, c in enumerate(ALL_COV):
        col = tr[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            X_raw[:, j] = col.astype("category").cat.codes.astype(np.float32)
        else:
            X_raw[:, j] = col.astype(np.float32)
    # median-impute
    for j in range(21):
        col = X_raw[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad])

    # cross-fit multi-output HistGB: predict all 13 ratings from other 20 covariates
    X_hier = np.zeros((n, 13), dtype=np.float32)
    for k in range(5):
        eval_m = folds == k
        fit_m = ~eval_m
        rating_idx = [ALL_COV.index(c) for c in RATING_COLS]
        Y = X_raw[fit_m][:, rating_idx].astype(np.float64)
        # features: other 8 non-rating covariates (4 numeric + 4 cat)
        non_rating = [c for c in ALL_COV if c not in RATING_COLS]
        other_idx = [ALL_COV.index(c) for c in non_rating]
        X_other = X_raw[fit_m][:, other_idx].astype(np.float64)

        # Multi-output regressor
        mor = MultiOutputRegressor(HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=200, max_depth=6, random_state=1, verbose=0))
        mor.fit(X_other, Y)

        # Predict on eval fold
        X_eval_other = X_raw[eval_m][:, other_idx].astype(np.float64)
        preds = mor.predict(X_eval_other)
        X_hier[eval_m, :] = preds
    return X_raw, X_hier


def lgbm_fold0(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, float]:
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
    m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **LGBM_PARAMS)
    m.fit(X_fit[inner_train], yfit[inner_train],
          eval_set=[(X_fit[inner_val], yfit[inner_val])], eval_metric="auc")
    pred = m.predict_proba(X[eval_m])[:, 1]
    return pred, float(roc_auc_score(y[eval_m], pred))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)

    X_raw, X_hier = build_design(tr, folds)
    X_aug = np.column_stack([X_raw, X_hier]).astype(np.float32)

    eval_m = folds == 0
    y_e = y[eval_m]

    pred_base, auc_base = lgbm_fold0(X_raw, y, folds)
    pred_aug, auc_aug = lgbm_fold0(X_aug, y, folds)

    delta = auc_aug - auc_base
    promising = delta >= 5e-5

    # decorrelation with champion proxy
    from scipy.stats import spearmanr
    from src.submission.store import load_oof
    champ_ref = load_oof("sol_route_aux10_primary_final")[eval_m]
    spear = float(spearmanr(pred_aug, champ_ref).statistic)
    blend = (logit(pred_aug) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    blend_delta = auc_blend - auc_aug
    pred_hier_only, auc_hier_only = lgbm_fold0(X_hier, y, folds)

    report = {
        "prereg": "configs/phase20/feat48_hierarchical_ratings_prereg.json",
        "prereg_sha256": "84795f40f0fa7031e3eb04627b6b36780ec5c5be3e7d156c5aef986ac461a356",
        "method": "cross-fit multi-output HistGB (13 ratings from 8 non-rating covariates), 5-fold",
        "fold0_results": {
            "base_21raw_auc": round(auc_base, 6),
            "augmented_21raw_13hier_auc": round(auc_aug, 6),
            "augmentation_delta": round(delta, 8),
            "hier_only_auc": round(auc_hier_only, 6),
            "spearman_aug_vs_champion": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta_vs_aug": round(blend_delta, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(auc_aug - CHAMPION_REF, 8),
            "promising": bool(promising),
            "decision_rule": "PROMISING if augmentation_delta >= +5e-5",
        },
        "n_rating_cols": len(RATING_COLS),
        "n_models": 5,
    }
    path = OUT / "feat48_hierarchical_ratings.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: base(21raw)={f0['base_21raw_auc']} "
          f"aug(+13hier)={f0['augmented_21raw_13hier_auc']} "
          f"delta={f0['augmentation_delta']:+.2e}")
    print(f"hier_only={f0['hier_only_auc']} "
          f"spearman_aug_vs_champ={f0['spearman_aug_vs_champion']} "
          f"blend_delta={f0['blend_delta_vs_aug']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} "
          f"promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())