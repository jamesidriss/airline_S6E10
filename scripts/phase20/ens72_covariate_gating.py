"""P20-ENS72-COVARIATE-GATING (item 72) -- prereg ae9c07e5, frozen before results.

Cross-fit covariate gate: learn when to trust the route reference vs the rich
blend, using ONLY the 21 raw covariates (label-free features) + FIT-fold labels
(cross-fitting). F2 fold-0 screen on saved OOF arrays.

Gate: per-row target = "route is correct" (route on correct side of 0.5).
Features: 21 raw covariates standardized on FIT-train only. L2 logistic (C=1.0).
Cross-fit: train on folds 1-4 OOF rows, apply to fold-0 rows.
Blend: w = P(route correct | covariates); final = w*route + (1-w)*rich.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/ens72_covariate_gating.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "ens72_covariate_gating"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

ROUTE = "sol_route_aux10_primary_final"
RICH = "blend_v3_final"


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    cov_df = tr.drop(columns=[ID_COL, TARGET]).copy()
    # ordinal-encode categoricals (label-free): Gender, Customer Type, Type of Travel, Class
    import pandas as pd
    cat_cols = [c for c in cov_df.columns if not pd.api.types.is_numeric_dtype(cov_df[c])]
    for c in cat_cols:
        cov_df[c] = cov_df[c].astype("category").cat.codes.astype(np.float64)
    cov = cov_df.to_numpy(dtype=np.float64)
    cov_names = cov_df.columns.tolist()
    assert len(cov_names) == 21, f"expected 21 covariates, got {len(cov_names)}"
    folds = np.load(FOLDS_PATH)

    # ---- fold-0 eval: train gate on folds 1-4, apply to fold 0 ----
    fit_mask = folds > 0          # folds 1-4 = FIT for fold-0 eval
    eval_mask = folds == 0        # fold 0 = eval

    # median-impute NaN (e.g. Arrival Delay) -- FIT-train only, fold-safe
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    cov[fit_mask] = imp.fit_transform(cov[fit_mask])
    cov[eval_mask] = imp.transform(cov[eval_mask])

    route = load_oof(ROUTE)
    rich = load_oof(RICH)

    # gate target: route is correct (on correct side of 0.5)
    route_correct = ((route >= 0.5) == (y == 1)).astype(np.int64)

    scaler = StandardScaler().fit(cov[fit_mask])  # FIT-train only
    X_fit = scaler.transform(cov[fit_mask])
    X_eval = scaler.transform(cov[eval_mask])

    gate = LogisticRegression(C=1.0, max_iter=2000)
    gate.fit(X_fit, route_correct[fit_mask])

    # P(route correct | covariates) on fold 0
    w = gate.predict_proba(X_eval)[:, 1]
    route_e = route[eval_mask]
    rich_e = rich[eval_mask]
    y_e = y[eval_mask]

    gated = w * route_e + (1.0 - w) * rich_e
    equal = 0.5 * route_e + 0.5 * rich_e

    auc_route = float(roc_auc_score(y_e, route_e))
    auc_rich = float(roc_auc_score(y_e, rich_e))
    auc_equal = float(roc_auc_score(y_e, equal))
    auc_gated = float(roc_auc_score(y_e, gated))

    # also: gated blend on ALL rows (for reference, using a gate trained on folds 1-4
    # applied everywhere -- but that is NOT the fold-0 screen; report it separately)
    w_all = gate.predict_proba(scaler.transform(cov))[:, 1]
    gated_all = w_all * route + (1.0 - w_all) * rich
    # NOTE: gated_all uses fold-0 rows in training -> NOT a valid OOF; only the
    # fold-0 eval above is the F2 screen. We do NOT report gated_all AUC as a result.

    promising = auc_gated >= auc_route + 1e-4

    report = {
        "prereg": "configs/phase20/ens72_covariate_gating_prereg.json",
        "prereg_sha256": "ae9c07e5d05dad9ab2e5d7224e1e98b5db1edcab1a0063a70c2f3b19f06e58be",
        "candidates": {"route": ROUTE, "rich": RICH},
        "gate": {
            "features": "21 raw covariates (StandardScaler on folds 1-4 only)",
            "model": "LogisticRegression(C=1.0, L2)",
            "target": "route_correct (route on correct side of 0.5)",
            "cross_fit": "train folds 1-4, apply fold-0",
        },
        "fold0_results": {
            "route_auc": round(auc_route, 6),
            "rich_auc": round(auc_rich, 6),
            "equal_blend_auc": round(auc_equal, 6),
            "gated_blend_auc": round(auc_gated, 6),
            "gated_minus_route": round(auc_gated - auc_route, 8),
            "gated_minus_equal": round(auc_gated - auc_equal, 8),
            "promising": bool(promising),
            "decision_rule": "PROMISING if gated >= route + 1e-4; NEGATIVE if gated < equal",
        },
        "fold0_eval_rows": int(eval_mask.sum()),
        "fold0_route_accuracy": round(float(route_correct[eval_mask].mean()), 6),
    }
    path = OUT / "ens72_covariate_gating.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    f0 = report["fold0_results"]
    print(f"fold-0: route={f0['route_auc']} rich={f0['rich_auc']} "
          f"equal={f0['equal_blend_auc']} gated={f0['gated_blend_auc']}")
    print(f"gated-route={f0['gated_minus_route']:+.2e} "
          f"gated-equal={f0['gated_minus_equal']:+.2e} "
          f"promising={f0['promising']}")
    print(f"fold-0 route accuracy={report['fold0_route_accuracy']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
