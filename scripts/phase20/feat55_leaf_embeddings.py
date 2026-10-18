"""P20-FEAT55-LEAF-EMBEDDINGS (item 55) -- prereg 7dc3beb4, frozen before results.

LGBM leaf embeddings: train LGBM (xt_xt_f10 recipe) on rich view folds 1-4,
extract leaf indices for each tree as dense features, feed into second family
(LogReg + shallow MLP). Cross-fit 5-fold.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat55_leaf_embeddings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat55_leaf_embeddings"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

# Champion LGBM xt_xt_f10 params
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


def build_rich_view(tr, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_rich_train, y, folds) where X_rich is the champion's 285-feature view."""
    # For fold-0 screen, we use the pre-computed rich view from the ViewBuilder
    # Simpler: use the static rich view from the cached features
    from src.features.view import ViewBuilder
    from src.common import load_raw
    ds = load_raw()
    te = ds.test
    vb = ViewBuilder(ds.train, te, "full")
    vb.build_static()
    X_rich = vb.static_tr  # (699635, 285)
    return X_rich, tr[TARGET].to_numpy(), np.load(FOLDS_PATH)


def get_leaf_indices(model, X):
    """Extract leaf indices for all trees. Returns (n_samples, n_trees) int array."""
    # LightGBM: model.booster_.predict(X, pred_leaf=True) returns (n_samples, n_trees)
    return model.booster_.predict(X, pred_leaf=True)


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
        val_parts.append(np.random.default_rng(1).choice(ci, nv, replace=False))
    inner_val = np.concatenate(val_parts)
    inner_train = np.setdiff1d(np.arange(fit_m.sum()), inner_val)
    X_fit = X[fit_m]
    m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **LGBM_PARAMS)
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

    # Build rich view
    from src.features.view import ViewBuilder
    from src.common import load_raw as load_raw2
    ds2 = load_raw2()
    te = ds2.test
    vb = ViewBuilder(ds2.train, te, "full")
    vb.build_static()
    X_rich = vb.static_tr  # (699635, 285)

    fit_m = folds > 0
    eval_m = folds == 0
    y_fit = y[fit_m]
    y_eval = y[eval_m]

    # Train LGBM on FIT (folds 1-4)
    X_fit_rich = X_rich[fit_m]
    X_eval_rich = X_rich[eval_m]

    # Inner ES split
    rng = np.random.default_rng(1)
    val_parts = []
    for cls in (0, 1):
        ci = np.where(y_fit == cls)[0]
        nv = max(1, int(round(0.1 * len(ci))))
        val_parts.append(rng.choice(ci, nv, replace=False))
    inner_val = np.concatenate(val_parts)
    inner_train = np.setdiff1d(np.arange(fit_m.sum()), inner_val)

    lgbm = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **LGBM_PARAMS)
    lgbm.fit(X_fit_rich[inner_train], y_fit[inner_train],
             eval_set=[(X_fit_rich[inner_val], y_fit[inner_val])], eval_metric="auc")

    # Extract leaf indices for eval fold
    leaf_eval = get_leaf_indices(lgbm, X_eval_rich)  # (n_eval, n_trees)
    leaf_fit = get_leaf_indices(lgbm, X_fit_rich)    # (n_fit, n_trees)

    y_fit_sub = y_fit
    # Second family: LogisticRegression on leaf indices (one-hot encoded or integer)
    # Use integer leaf indices with StandardScaler + LogisticRegression
    scaler = StandardScaler()
    leaf_fit_scaled = scaler.fit_transform(leaf_fit.astype(np.float32))
    leaf_eval_scaled = scaler.transform(leaf_eval.astype(np.float32))

    logreg = LogisticRegression(C=1.0, max_iter=2000, random_state=1, n_jobs=8)
    logreg.fit(leaf_fit_scaled, y_fit)
    pred_logreg = logreg.predict_proba(leaf_eval_scaled)[:, 1]
    auc_logreg = float(roc_auc_score(y_eval, pred_logreg))

    # Second family: shallow MLP on leaf indices
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                        alpha=1e-3, max_iter=200, early_stopping=True,
                        validation_fraction=0.1, random_state=1)
    mlp.fit(leaf_fit_scaled, y_fit)
    pred_mlp = mlp.predict_proba(leaf_eval_scaled)[:, 1]
    auc_mlp = float(roc_auc_score(y_eval, pred_mlp))

    # Base LGBM on rich view (for comparison)
    pred_base, auc_base = lgbm_fold0(X_rich, ds2.train[TARGET].to_numpy(), np.load(FOLDS_PATH))

    # Best of second family
    best_second = max(auc_logreg, auc_mlp)
    best_second_name = "logreg" if auc_logreg > auc_mlp else "mlp"

    # Blend with champion proxy
    from scipy.stats import spearmanr
    from src.submission.store import load_oof
    champ_ref = load_oof("sol_route_aux10_primary_final")[eval_m]
    best_pred = pred_logreg if auc_logreg > auc_mlp else pred_mlp
    spear = float(spearmanr(best_pred, champ_ref).statistic)
    blend = (logit(best_pred) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(ds.train[TARGET].to_numpy()[eval_m], blend))
    blend_delta = auc_blend - best_second

    report = {
        "prereg": "configs/phase20/feat55_leaf_embeddings_prereg.json",
        "prereg_sha256": "7dc3beb48a3069de4bcd845de2a70292c5805c8420ba723e6f9963cb16509456",
        "method": "LGBM leaf embeddings (xt_xt_f10) -> LogReg/MLP second family, cross-fit",
        "fold0_results": {
            "base_rich_lgbm_auc": round(auc_base, 6),
            "leaf_logreg_auc": round(auc_logreg, 6),
            "leaf_mlp_auc": round(auc_mlp, 6),
            "best_second_family": best_second_name,
            "best_second_auc": round(best_second, 6),
            "delta_vs_base": round(best_second - auc_base, 8),
            "spearman_vs_champion_proxy": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta_vs_second": round(blend_delta, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(best_second - CHAMPION_REF, 8),
            "promising": best_second - auc_base >= 5e-5,
        },
        "n_trees": 6000,  # LGBM early stopping determines actual
    }
    path = OUT / "feat55_leaf_embeddings.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: base_rich={f0['base_rich_lgbm_auc']} "
          f"leaf_logreg={f0['leaf_logreg_auc']} leaf_mlp={f0['leaf_mlp_auc']}")
    print(f"best_second={f0['best_second_family']}={f0['best_second_auc']} "
          f"delta={f0['delta_vs_base']:+.2e}")
    print(f"spearman={f0['spearman_vs_champion_proxy']} blend_delta={f0['blend_delta_vs_second']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())