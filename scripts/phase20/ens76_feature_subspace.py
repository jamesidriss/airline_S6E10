"""P20-ENS76-FEATURE-SUBSPACE-ENSEMBLE -- prereg 1ea664f4, frozen before results.

5 LGBM members, each on a deterministic 50% random subspace of the 237-feature
static full view (subspace seeds 11/22/33/44/55); equal-logit blend as candidate.
Outer-FIT training with identical inner 10% ES split (seed 1); fold-0 eval.

Frozen decision rule: port_delta >= +5e-5 AND corr_vs_portfolio < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/ens76_feature_subspace.py
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
from src.features.view import ViewBuilder  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "ens76_feature_subspace"
ART = ROOT / "artifacts" / "phase20" / "ens76_feature_subspace"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"
PREREG_SHA = "1ea664f4d246c773b6ccd074dce967546d4590eb62e5c39a80541fd825f3dfcf"
PORTFOLIO_REF = "sol_route_aux10_primary_final"
SUBSEEDS = [11, 22, 33, 44, 55]


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def lgbm_member(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, float]:
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
    ds = load_raw()
    tr, te = ds.train, ds.test
    y = tr[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)
    eval_m = folds == 0
    y_e = y[eval_m]

    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    X = np.asarray(vb.static_tr, dtype=np.float32)
    n_feat = X.shape[1]
    n_sub = n_feat // 2

    member_preds = []
    member_aucs = []
    sub_infos = []
    for s in SUBSEEDS:
        rng = np.random.default_rng(s)
        cols = np.sort(rng.choice(n_feat, n_sub, replace=False))
        pred, auc = lgbm_member(X[:, cols], y, folds)
        member_preds.append(pred)
        member_aucs.append(round(auc, 6))
        sub_infos.append({"subseed": s, "n_features": int(n_sub),
                          "first_cols": cols[:8].tolist()})

    blend = np.mean([logit(p) for p in member_preds], axis=0)
    y_pred = np.empty(len(y), dtype=np.float32)
    y_pred[eval_m] = np.exp(blend) / (1 + np.exp(blend))  # sigmoid; AUC invariant
    auc_blend = float(roc_auc_score(y_e, blend))

    port = load_oof(PORTFOLIO_REF)[eval_m]
    auc_port = float(roc_auc_score(y_e, port))
    pd76 = auc_blend - auc_port
    corr = float(spearmanr(blend, port).statistic)
    prom = bool(pd76 >= 5e-5 and corr < 0.995)

    rep = {
        "status": "DONE",
        "prereg": "configs/phase20/ens76_feature_subspace_prereg.json",
        "prereg_sha256": PREREG_SHA,
        "metrics": {
            "member_aucs": member_aucs,
            "mean_member_auc": round(float(np.mean(member_aucs)), 6),
            "blend_fold0_auc": round(auc_blend, 6),
            "blend_gain_vs_mean_member": round(auc_blend - float(np.mean(member_aucs)), 8),
            "portfolio_fold0_auc": round(auc_port, 6),
            "port_delta": round(pd76, 8),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "promising_frozen_rule": prom,
            "eval_rows": int(eval_m.sum()),
        },
        "eval_rows": int(eval_m.sum()),
        "subspaces": sub_infos,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ens76_feature_subspace.json").write_text(
        json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "fold0_blend.npy", blend.astype(np.float32))

    m = rep["metrics"]
    print(f"members={m['member_aucs']} mean={m['mean_member_auc']} "
          f"blend={m['blend_fold0_auc']} (+{m['blend_gain_vs_mean_member']:+.2e})")
    print(f"port_delta={m['port_delta']:+.3e} (need >= +5e-5)  corr={m['candidate_spearman_vs_portfolio']} "
          f"-> PROMISING={prom}")
    print(f"report -> {(OUT / 'ens76_feature_subspace.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
