"""P20-FEAT57-LOWRANK-INTERACTIONS (item 57) -- prereg e815517d, frozen before results.

Low-rank pairwise interactions: cross-fit SVD of 13 ratings → 8 latent dims.
Captures rating × rating synergy. Cross-fit SVD: fit on folds 1-4, project fold-0.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat57_lowrank_interactions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import svd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat57_lowrank_interactions"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
ALL_COV = RATING_COLS + ["Age", "Flight Distance", "Departure Delay in Minutes",
                         "Arrival Delay in Minutes", "Gender", "Customer Type",
                         "Type of Travel", "Class"]
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


def build_design(tr, folds: np.ndarray, rank: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_raw [n,21], X_svd [n, rank] cross-fit SVD latent dims)."""
    n = len(tr)
    X_raw = np.zeros((n, 21), dtype=np.float32)
    for j, c in enumerate(ALL_COV):
        col = tr[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            X_raw[:, j] = col.astype("category").cat.codes.astype(np.float32)
        else:
            X_raw[:, j] = col.astype(np.float32)
    for j in range(21):
        col = X_raw[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad])

    rating_idx = [ALL_COV.index(c) for c in RATING_COLS]
    X_r = X_raw[:, rating_idx].astype(np.float64)  # (n, 13)

    X_svd = np.zeros((n, rank), dtype=np.float32)
    for k in range(5):
        eval_m = np.load(FOLDS_PATH) == k
        fit_m = ~eval_m
        R_fit = X_r[fit_m]
        R_eval = X_r[eval_m]
        # Center
        R_fit_mean = R_fit.mean(axis=0)
        R_fit_centered = R_fit - R_fit_mean
        R_eval_centered = R_eval - R_fit_mean
        # SVD on fit
        U, s, Vt = svd(R_fit_centered, full_matrices=False)
        # Project eval: R_eval_centered @ Vt^T[:, :rank]
        X_svd[eval_m] = (R_eval_centered @ Vt[:rank].T).astype(np.float32)
    return X_raw, X_svd


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
    m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **{
        "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
        "n_jobs": 8, "random_state": 1})
    m.fit(X_fit[inner_train], y[fit_m][inner_train],
          eval_set=[(X_fit[inner_val], y[fit_m][inner_val])], eval_metric="auc")
    pred = m.predict_proba(X[eval_m])[:, 1]
    return pred, float(roc_auc_score(y[eval_m], pred))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)

    X_raw, X_svd = build_design(tr, folds, rank=8)
    X_aug = np.column_stack([X_raw, X_svd]).astype(np.float32)

    eval_m = np.load(FOLDS_PATH) == 0
    y_e = ds.train[TARGET].to_numpy()[eval_m]

    pred_base, auc_base = lgbm_fold0(X_raw, y, folds)
    pred_aug, auc_aug = lgbm_fold0(X_aug, y, folds)

    delta = auc_aug - auc_base
    promising = delta >= 5e-5

    spear = float(spearmanr(pred_aug, load_oof("sol_route_aux10_primary_final")[eval_m]).statistic)
    blend = (logit(pred_aug) + logit(load_oof("sol_route_aux10_primary_final")[eval_m])) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    blend_delta = auc_blend - auc_aug
    pred_svd_only, auc_svd_only = lgbm_fold0(X_svd, y, folds)

    report = {
        "prereg": "configs/phase20/feat57_lowrank_interactions_prereg.json",
        "prereg_sha256": "e815517dd97b9df2aea328cacc1c257a1a7ae5802f8f12a6ea5e9bf1df8a76cf",
        "method": "Cross-fit SVD (rank 8) of 13 ratings, 5-fold. 8 latent dims added to 21raw.",
        "fold0_results": {
            "base_21raw_auc": round(auc_base, 6),
            "augmented_21raw_8svd_auc": round(auc_aug, 6),
            "augmentation_delta": round(delta, 8),
            "svd_only_auc": round(auc_svd_only, 6),
            "spearman_aug_vs_champion": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta_vs_aug": round(blend_delta, 8),
            "champion_ref": 0.961853,
            "port_delta_vs_champion": round(auc_aug - 0.961853, 8),
            "promising": bool(promising),
            "decision_rule": "PROMISING if augmentation_delta >= +5e-5",
        },
        "rank": 8,
    }
    path = OUT / "feat57_lowrank_interactions.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: base(21raw)={f0['base_21raw_auc']} "
          f"aug(+8svd)={f0['augmented_21raw_8svd_auc']} "
          f"delta={f0['augmentation_delta']:+.2e}")
    print(f"svd_only={f0['svd_only_auc']} "
          f"spearman_aug_vs_champ={f0['spearman_aug_vs_champion']} "
          f"blend_delta={f0['blend_delta_vs_aug']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} "
          f"promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())