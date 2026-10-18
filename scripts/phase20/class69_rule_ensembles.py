"""P20-CLASS69-RULE-ENSEMBLES (item 69) -- prereg 65dfca82, frozen before results.

Conditional rule ensembles: shallow trees on random feature subspaces,
rule extraction, weighted voting. Cross-fit 5-fold on rich view.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/class69_rule_ensembles.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "class69_rule_ensembles"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

CHAMPION_REF = 0.9618529228157235


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)

    # Build champion's rich view
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

    X_fit = X_rich[fit_m]
    X_eval = X_rich[eval_m]

    # Train ensemble of shallow trees on random feature subspaces
    n_trees = 50
    max_depth = 3
    subspace_frac = 0.3  # 30% of features per tree
    n_features = X_rich.shape[1]
    subspace_size = max(1, int(n_features * subspace_frac))

    trees = []
    tree_weights = []
    rng = np.random.default_rng(1)

    for i in range(n_trees):
        # Random feature subspace
        feat_idx = rng.choice(n_features, subspace_size, replace=False)
        X_fit_sub = X_fit[:, feat_idx]

        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
        tree.fit(X_fit_sub, y_fit)

        # Weight by AUC on inner validation (10% of fit)
        rng_inner = np.random.default_rng(i)
        val_parts = []
        for cls in (0, 1):
            ci = np.where(y_fit == cls)[0]
            nv = max(1, int(round(0.1 * len(ci))))
            val_parts.append(rng_inner.choice(ci, nv, replace=False))
        inner_val = np.concatenate(val_parts)
        inner_train = np.setdiff1d(np.arange(len(y_fit)), inner_val)

        # Quick AUC on inner val
        tree.fit(X_fit_sub[inner_train], y_fit[inner_train])
        pred_val = tree.predict_proba(X_fit_sub[inner_val])[:, 1]
        weight = roc_auc_score(y_fit[inner_val], pred_val)

        # Retrain on full fit
        tree.fit(X_fit_sub, y_fit)

        trees.append((tree, feat_idx))
        tree_weights.append(weight)

    # Normalize weights
    tree_weights = np.array(tree_weights)
    tree_weights = tree_weights / tree_weights.sum()

    # Predict on eval
    pred_ensemble = np.zeros(len(X_eval))
    for (tree, feat_idx), w in zip(trees, tree_weights):
        pred_ensemble += w * tree.predict_proba(X_eval[:, feat_idx])[:, 1]

    auc = float(roc_auc_score(y[eval_m], pred_ensemble))

    # Decorrelation with champion proxy
    from src.submission.store import load_oof
    champ_ref = load_oof("sol_route_aux10_primary_final")[folds == 0]
    spear = float(spearmanr(pred_ensemble, champ_ref).statistic)
    blend = (logit(pred_ensemble) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(y[eval_m], blend))
    blend_delta = auc_blend - auc

    report = {
        "prereg": "configs/phase20/class69_rule_ensembles_prereg.json",
        "prereg_sha256": "65dfca8266ccf58ef62b53924788494c46ee75dbea63ae14eb69d79d581a59f4",
        "method": "Conditional rule ensembles: 50 shallow DecisionTrees (max_depth=3) on random 30% feature subspaces, weighted by inner-val AUC. Cross-fit 5-fold on rich view.",
        "fold0_results": {
            "auc": round(auc, 6),
            "spearman_vs_champion_proxy": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta": round(blend_delta, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(auc - CHAMPION_REF, 8),
            "promising": auc - CHAMPION_REF >= 5e-5,
        },
        "n_trees": 50,
        "max_depth": 3,
        "subspace_frac": 0.3,
    }
    path = OUT / "class69_rule_ensembles.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: RuleEnsemble AUC={f0['auc']} spearman={f0['spearman_vs_champion_proxy']} blend_delta={f0['blend_delta']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())