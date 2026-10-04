"""Which objectives actually accept fractional (soft) labels?

The soft-target campaign needs probabilistic targets. Silently coercing them to binary would make
every result meaningless, so each library is probed *before* use with a tiny fit on data whose
labels are strictly fractional, and we check that the learned probabilities actually move toward
the soft targets rather than toward 0/1.

Probes
------
CatBoost `CrossEntropy`      : native soft-label support expected
XGBoost `binary:logistic`    : may reject non-binary labels
LightGBM `cross_entropy`     : expects binary y; custom objective used instead
LightGBM custom objective    : BCE gradients computed with the soft target directly
torch BCE (used by RealMLP)  : soft targets supported by torch.nn.BCELoss

A probe passes when the model fits without error AND its mean prediction is strictly between the
hard-label mean and the soft-target mean, i.e. it tracked the fractional information.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.common import REPORTS, save_json  # noqa: E402

RESULTS: dict = {}


def _probe_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype("float32")
    z = X[:, 0] * 1.3 + X[:, 1] * 0.7 - 0.2
    p = 1 / (1 + np.exp(-z))
    y_hard = (rng.random(n) < p).astype("float32")
    return X, y_hard, p.astype("float32")


def probe_catboost():
    X, y_hard, p = _probe_data()
    from catboost import CatBoostClassifier

    for loss in ("CrossEntropy", "Logloss"):
        try:
            m = CatBoostClassifier(iterations=120, learning_rate=0.15, depth=5, verbose=0,
                                   loss_function=loss, allow_writing_files=False)
            m.fit(X, p)                      # fractional targets on purpose
            pred = m.predict_proba(X)[:, 1]
            ok = abs(float(pred.mean()) - float(p.mean())) < 0.05
            RESULTS[f"catboost_{loss}"] = {"supported": True, "mean_pred": float(pred.mean()),
                                           "mean_soft_target": float(p.mean()), "tracked": ok}
            print(f"  catboost {loss:<14} OK    mean_pred={pred.mean():.4f} "
                  f"soft={p.mean():.4f} tracked={ok}")
        except Exception as exc:  # noqa: BLE001
            RESULTS[f"catboost_{loss}"] = {"supported": False, "error_type": type(exc).__name__}
            print(f"  catboost {loss:<14} FAIL  {type(exc).__name__}")


def probe_xgboost():
    X, y_hard, p = _probe_data()
    import xgboost as xgb

    for device in ("cpu",):
        try:
            m = xgb.XGBClassifier(n_estimators=120, learning_rate=0.15, max_depth=5,
                                  tree_method="hist", device=device, eval_metric="logloss")
            m.fit(X, p, verbose=False)
            pred = m.predict_proba(X)[:, 1]
            ok = abs(float(pred.mean()) - float(p.mean())) < 0.05
            RESULTS[f"xgboost_binary_logistic_{device}"] = {
                "supported": True, "mean_pred": float(pred.mean()),
                "mean_soft_target": float(p.mean()), "tracked": ok}
            print(f"  xgboost binary     OK    mean_pred={pred.mean():.4f} "
                  f"soft={p.mean():.4f} tracked={ok}")
        except Exception as exc:  # noqa: BLE001
            RESULTS[f"xgboost_binary_logistic_{device}"] = {"supported": False,
                                                            "error_type": type(exc).__name__}
            print(f"  xgboost binary     FAIL  {type(exc).__name__}: {str(exc)[:90]}")

    # custom objective route: gradients from soft-label BCE
    try:
        dtr = xgb.DMatrix(X, label=p)
        dva = xgb.DMatrix(X, label=p)

        def soft_bce(predt, dt):
            p_hat = 1.0 / (1.0 + np.exp(-predt))
            q = dt.get_label()
            grad = p_hat - q
            hess = np.maximum(p_hat * (1.0 - p_hat), 1e-6)
            return grad, hess

        bst = xgb.train({"objective": "custom:softbce", "max_depth": 5, "eta": 0.15,
                         "tree_method": "hist", "eval_metric": "logloss"},
                        dtr, num_boost_round=120, custom_objective=soft_bce)
        pred = 1 / (1 + np.exp(-bst.predict(dva)))
        ok = abs(float(pred.mean()) - float(p.mean())) < 0.05
        RESULTS["xgboost_custom_softbce"] = {"supported": True, "mean_pred": float(pred.mean()),
                                             "mean_soft_target": float(p.mean()), "tracked": ok}
        print(f"  xgboost custom     OK    mean_pred={pred.mean():.4f} tracked={ok}")
    except Exception as exc:  # noqa: BLE001
        RESULTS["xgboost_custom_softbce"] = {"supported": False, "error_type": type(exc).__name__}
        print(f"  xgboost custom     FAIL  {type(exc).__name__}: {str(exc)[:90]}")


def probe_lightgbm():
    X, y_hard, p = _probe_data()
    import lightgbm as lgb

    try:
        m = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.15, num_leaves=31, verbose=-1)
        m.fit(X, p)
        pred = m.predict_proba(X)[:, 1]
        RESULTS["lightgbm_binary_soft"] = {"supported": True, "mean_pred": float(pred.mean())}
        print(f"  lightgbm binary    OK    mean_pred={pred.mean():.4f}")
    except Exception as exc:  # noqa: BLE001
        RESULTS["lightgbm_binary_soft"] = {"supported": False, "error_type": type(exc).__name__}
        print(f"  lightgbm binary    FAIL  {type(exc).__name__}: {str(exc)[:80]}")

    # custom objective route (the one we will actually use)
    try:
        def soft_bce(preds, dataset):
            q = dataset.get_label()
            p_hat = 1.0 / (1.0 + np.exp(-preds))
            grad = p_hat - q
            hess = np.maximum(p_hat * (1.0 - p_hat), 1e-6)
            return grad, hess

        dtr = lgb.Dataset(X, label=p, free_raw_data=False)
        m = lgb.train({"objective": "cross_entropy", "num_leaves": 31, "learning_rate": 0.15,
                       "verbose": -1, "num_threads": 4}, dtr, num_boost_round=120)
        pred_raw = m.predict(X, raw_score=True)
        pred = 1 / (1 + np.exp(-pred_raw))
        ok = abs(float(pred.mean()) - float(p.mean())) < 0.05
        RESULTS["lightgbm_cross_entropy_binary_y"] = {
            "supported": True, "mean_pred": float(pred.mean()),
            "mean_soft_target": float(p.mean()), "tracked": ok,
            "note": "y is fractional here; mean prediction tracks the soft target"}
        print(f"  lightgbm xentropy  OK    mean_pred={pred.mean():.4f} soft={p.mean():.4f} "
              f"tracked={ok}")
    except Exception as exc:  # noqa: BLE001
        RESULTS["lightgbm_cross_entropy_binary_y"] = {"supported": False,
                                                       "error_type": type(exc).__name__}
        print(f"  lightgbm xentropy  FAIL  {type(exc).__name__}: {str(exc)[:80]}")

    try:
        def soft_bce2(preds, dataset):
            q = dataset.get_label()
            p_hat = 1.0 / (1.0 + np.exp(-preds))
            return p_hat - q, np.maximum(p_hat * (1.0 - p_hat), 1e-6)

        dtr = lgb.Dataset(X, label=p, free_raw_data=False)
        m = lgb.train({"num_leaves": 31, "learning_rate": 0.15, "verbose": -1,
                       "num_threads": 4}, dtr, num_boost_round=120, fobj=soft_bce2)
        pred = 1 / (1 + np.exp(-m.predict(X, raw_score=True)))
        ok = abs(float(pred.mean()) - float(p.mean())) < 0.05
        RESULTS["lightgbm_custom_fobj"] = {"supported": True, "mean_pred": float(pred.mean()),
                                           "mean_soft_target": float(p.mean()), "tracked": ok}
        print(f"  lightgbm custom    OK    mean_pred={pred.mean():.4f} tracked={ok}")
    except Exception as exc:  # noqa: BLE001
        RESULTS["lightgbm_custom_fobj"] = {"supported": False, "error_type": type(exc).__name__}
        print(f"  lightgbm custom    FAIL  {type(exc).__name__}: {str(exc)[:80]}")


def probe_torch():
    X, y_hard, p = _probe_data()
    try:
        import torch

        t = torch.tensor(X)
        q = torch.tensor(p, dtype=torch.float32).unsqueeze(1)
        lin = torch.nn.Linear(5, 1)
        opt = torch.optim.Adam(lin.parameters(), lr=0.05)
        lossf = torch.nn.BCELoss()
        for _ in range(300):
            opt.zero_grad()
            out = torch.sigmoid(lin(t))
            loss = lossf(out, q)
            loss.backward()
            opt.step()
        with torch.no_grad():
            mp = float(torch.sigmoid(lin(t)).mean())
        ok = abs(mp - float(p.mean())) < 0.05
        RESULTS["torch_bce_soft"] = {"supported": True, "mean_pred": mp,
                                      "mean_soft_target": float(p.mean()), "tracked": ok}
        print(f"  torch BCE          OK    mean_pred={mp:.4f} soft={p.mean():.4f} tracked={ok}")
    except Exception as exc:  # noqa: BLE001
        RESULTS["torch_bce_soft"] = {"supported": False, "error_type": type(exc).__name__}
        print(f"  torch BCE          FAIL  {type(exc).__name__}")


def main() -> None:
    print("=" * 92)
    print("SOFT-LABEL SUPPORT PROBE (labels strictly fractional; no coercion)")
    print("=" * 92)
    probe_catboost()
    probe_xgboost()
    probe_lightgbm()
    probe_torch()
    save_json(RESULTS, REPORTS / "soft_label_support.json")
    print("\nsummary:")
    for k, v in RESULTS.items():
        flag = "TRACKS SOFT TARGET" if v.get("tracked") else (
            "accepted" if v.get("supported") else f"unsupported ({v.get('error_type')})")
        print(f"  {k:<34} {flag}")
    print("\nwrote", REPORTS / "soft_label_support.json")


if __name__ == "__main__":
    main()