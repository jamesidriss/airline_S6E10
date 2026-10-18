"""P20-CLASS62-GAM-SPLINES (item 62) -- prereg e9ec6d74, frozen before results.

GAM with spline basis: sklearn SplineTransformer (degree=3, n_knots=10) on
285-feature rich view + LogisticRegression. Cross-fit 5-fold.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/class62_gam_splines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "class62_gam_splines"
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
    X = vb.static_tr  # (699635, 285)

    fit_m = folds > 0
    eval_m = folds == 0
    y_fit = y[fit_m]
    y_eval = y[eval_m]

    X_fit = X[fit_m]
    X_eval = X[eval_m]

    # SplineTransformer: univariate splines for each feature
    # degree=3, n_knots=10, include_bias=False
    spline = SplineTransformer(degree=3, n_knots=10, include_bias=False)
    X_fit_spline = spline.fit_transform(X_fit)
    X_eval_spline = spline.transform(X_eval)

    # LogisticRegression on spline features
    logreg = LogisticRegression(C=1.0, max_iter=2000, random_state=1, n_jobs=8)
    logreg.fit(X_fit_spline, y_fit)
    pred = logreg.predict_proba(X_eval_spline)[:, 1]
    auc = float(roc_auc_score(y_eval, pred))

    # Decorrelation with champion proxy
    champ_ref = load_oof("sol_route_aux10_primary_final")[eval_m]
    spear = float(spearmanr(pred, champ_ref).statistic)
    blend = (logit(pred) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(y_eval, blend))
    blend_delta = auc_blend - auc

    report = {
        "prereg": "configs/phase20/class62_gam_splines_prereg.json",
        "prereg_sha256": "e9ec6d74fe322086211bae373e9fe8f8912079438335fd497b07891e66c10ca8",
        "method": "GAM: SplineTransformer (degree=3, n_knots=10) -> LogisticRegression(C=1.0), cross-fit 5-fold",
        "fold0_results": {
            "auc": round(auc, 6),
            "spearman_vs_champion_proxy": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta": round(blend_delta, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(auc - CHAMPION_REF, 8),
            "promising": auc - 0.961853 >= 5e-5,
        },
        "n_spline_features": X_fit_spline.shape[1],
    }
    path = OUT / "class62_gam_splines.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: GAM spline AUC={f0['auc']} "
          f"spearman={f0['spearman_vs_champion_proxy']} "
          f"blend_delta={f0['blend_delta']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} "
          f"promising={f0['promising']}")
    print(f"n_spline_features={report['n_spline_features']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())