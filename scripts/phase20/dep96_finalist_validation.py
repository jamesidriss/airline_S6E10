"""P20-DEP96-FINALIST-VALIDATION (item 96) -- prereg acd3986c, frozen before results.

Block10 (10-fold, seed 20261010) fold scheme generation + champion validation.
Infrastructure for finalist B validation. CPU-only LGBM 10-fold CV.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/dep96_finalist_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "dep96_finalist_validation"

LGBM_PARAMS = {
    "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
    "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
    "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
    "n_jobs": 8, "random_state": 1,
}

CHAMPION_PRIMARY_OOF = 0.9618529228157235


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr, te = ds.train, ds.test
    y = tr[TARGET].to_numpy()

    # Build champion's rich view
    from src.features.view import ViewBuilder
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    X = vb.static_tr  # (699635, 285)

    # Generate block10 folds (StratifiedKFold 10, seed 20261010)
    block10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=20261010)
    block10_folds = np.full(len(y), -1, dtype=np.int8)
    for k, (_, va) in enumerate(block10.split(np.zeros(len(y)), y)):
        block10_folds[va] = k

    # Save block10 folds to cache
    cache_dir = ROOT / "data" / "cache" / "folds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "block10_10f_seed20261010.npy", block10_folds)

    # 10-fold CV on rich view
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    oof = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []

    for k in range(10):
        fit_m = block10_folds != k
        val_m = block10_folds == k

        # inner 10% ES from FIT
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
              eval_set=[(X_fit[inner_val], yfit[inner_val])], eval_metric="auc")
        oof[val_m] = m.predict_proba(X[val_m])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[val_m], oof[val_m])))

    auc_10f = float(roc_auc_score(y, oof))
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    # Compare with primary 5-fold
    delta_vs_primary = auc_10f - CHAMPION_PRIMARY_OOF

    # Correlation with primary 5-fold OOF (if available)
    # Load primary OOF from store if available
    primary_oof = None
    try:
        from src.submission.store import load_oof
        primary_oof = load_oof("prod5_lgbm_full_primary")
        spearman = float(spearmanr(oof, primary_oof).statistic)
        pearson = float(np.corrcoef(oof, primary_oof)[0, 1])
    except Exception:
        spearman = pearson = None

    report = {
        "prereg": "configs/phase20/dep96_finalist_validation_prereg.json",
        "prereg_sha256": "acd3986c07101d5c43c7164b72ff9893e7197917f7ce13b3fc517560e9fe592c",
        "block10": {
            "n_splits": 10,
            "seed": 20261010,
            "fold_counts": [int((block10_folds == k).sum()) for k in range(10)],
        },
        "champion_rich_view": "prod5_lgbm_full_primary recipe (xt_xt_f10) on 285-feature full view",
        "block10_oof_auc": round(auc_10f, 6),
        "champion_primary_oof": CHAMPION_PRIMARY_OOF,
        "delta_vs_primary": round(auc_10f - CHAMPION_PRIMARY_OOF, 8),
        "per_fold_aucs": [round(v, 6) for v in fold_aucs],
        "mean_fold_auc": round(float(mean_auc), 6),
        "std_fold_auc": round(float(std_auc), 6),
        "correlation_with_primary": {
            "spearman": round(spearman, 6) if spearman else None,
            "pearson": round(pearson, 6) if pearson else None,
            "note": "primary OOF from store if available",
        },
        "artifact": {
            "block10_folds": "data/cache/folds/block10_10f_seed20261010.npy",
            "oof_predictions": "reports/phase20/dep96_finalist_validation/block10_oof.npy",
        },
    }
    path = OUT / "dep96_finalist_validation.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    # Save block10 OOF
    np.save(OUT / "block10_oof.npy", oof.astype(np.float32))

    print(f"Block10 10-fold OOF AUC: {auc_10f:.6f} vs primary {CHAMPION_PRIMARY_OOF:.6f}")
    print(f"delta: {delta_vs_primary:+.8f}")
    print(f"per-fold: {[f'{v:.6f}' for v in fold_aucs]}")
    print(f"mean/std: {mean_auc:.6f}/{std_auc:.6f}")
    if spearman is not None:
        print(f"spearman vs primary: {spearman:.6f}, pearson: {pearson:.6f}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    print(f"block10 folds saved -> data/cache/folds/block10_10f_seed20261010.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())