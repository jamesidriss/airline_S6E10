"""P20-FEAT51-RESIDUAL-RATINGS (item 51) -- prereg a1013332, frozen before results.

Cross-fit residual service-rating features. For each of the 13 service-rating
columns, a HistGB teacher predicts that rating from the OTHER 20 covariates,
trained on the other 4 folds (cross-fitting). residual = observed - teacher_pred.
This captures rating-pattern INCONSISTENCY (a signal the absolute ratings lack).

Fold-0 screen: LGBM on 21 raw (base) vs 21 raw + 13 residuals, fold-0 eval.
The residual is fold-safe: the teacher never sees the row it predicts, and no
satisfaction label enters the residual.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat51_residual_ratings.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat51_residual_ratings"
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
CHAMPION_REF = 0.961853  # campaign_state champion_a.oof (immutable)


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_design(tr, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_raw [n,21] numeric-encoded, X_resid [n,13] cross-fit residuals)."""
    n = len(tr)
    # numeric-encode the 21 covariates (ordinal for categoricals, label-free)
    X_raw = np.zeros((n, 21), dtype=np.float32)
    for j, c in enumerate(ALL_COV):
        col = tr[c]
        if col.dtype == object or str(col.dtype).startswith("str"):
            X_raw[:, j] = col.astype("category").cat.codes.astype(np.float32)
        else:
            X_raw[:, j] = col.astype(np.float32)
    # median-impute (Arrival Delay has NaN)
    for j in range(21):
        col = X_raw[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad])

    # cross-fit residuals: for each rating j, teacher trained on other 4 folds
    # predicts rating j from the other 20 covariates, applied to the held-out fold.
    X_resid = np.zeros((n, 13), dtype=np.float32)
    from sklearn.ensemble import HistGradientBoostingRegressor
    for j, rc in enumerate(RATING_COLS):
        obs = X_raw[:, ALL_COV.index(rc)].astype(np.float64)
        others = [k for k in range(21) if k != ALL_COV.index(rc)]
        Xo = X_raw[:, others].astype(np.float64)
        for k in range(5):
            eval_m = folds == k
            fit_m = ~eval_m
            teacher = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=200, max_depth=6,
                random_state=1, verbose=0)
            teacher.fit(Xo[fit_m], obs[fit_m])
            pred = teacher.predict(Xo[eval_m])
            X_resid[eval_m, j] = obs[eval_m] - pred
    return X_raw, X_resid


def lgbm_fold0(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, float]:
    """Train LGBM on folds 1-4, eval on fold-0. Returns (fold-0 preds, AUC)."""
    from lightgbm import LGBMClassifier
    eval_m = folds == 0
    fit_m = ~eval_m
    # inner 10% ES split carved from FIT only
    rng = np.random.default_rng(1)
    yfit = y[fit_m]
    val_parts = []
    for cls in (0, 1):
        ci = np.where((yfit == cls))[0]
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

    X_raw, X_resid = build_design(tr, folds)
    X_aug = np.column_stack([X_raw, X_resid]).astype(np.float32)

    eval_m = folds == 0
    y_e = y[eval_m]

    pred_base, auc_base = lgbm_fold0(X_raw, y, folds)
    pred_aug, auc_aug = lgbm_fold0(X_aug, y, folds)

    # blend of base+aug isn't meaningful; report the augmentation delta
    delta = auc_aug - auc_base
    promising = delta >= 5e-5

    # KEY decorrelation test: is the residual signal already captured by the champion?
    # Spearman of augmented preds vs the route+aux10 reference (champion proxy) on fold-0.
    from scipy.stats import spearmanr
    from src.submission.store import load_oof
    champ_ref = load_oof("sol_route_aux10_primary_final")[eval_m]
    spear = float(spearmanr(pred_aug, champ_ref).statistic)
    # also: equal-logit blend of augmented-LGBM with the champion proxy
    blend = (logit(pred_aug) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    blend_delta = auc_blend - auc_aug
    # residual self-signal: do the 13 residuals alone predict satisfaction?
    pred_resid_only, auc_resid_only = lgbm_fold0(X_resid, y, folds)

    report = {
        "prereg": "configs/phase20/feat51_residual_ratings_prereg.json",
        "prereg_sha256": "a1013332af66f003af42a8db7aea15a69cdd02c199ac05360121d45b9a266eb0",
        "method": "cross-fit HistGB residual (13 service ratings, teacher from other 20 covariates, 5-fold)",
        "fold0_results": {
            "base_21raw_auc": round(auc_base, 6),
            "augmented_21raw_13resid_auc": round(auc_aug, 6),
            "augmentation_delta": round(delta, 8),
            "resid_only_auc": round(auc_resid_only, 6),
            "champion_ref_proxy": "sol_route_aux10_primary_final (0.961766)",
            "spearman_aug_vs_champion": round(spear, 6),
            "equal_blend_auc": round(auc_blend, 6),
            "blend_delta_vs_aug": round(blend_delta, 8),
            "champion_ref": CHAMPION_REF,
            "port_delta_vs_champion": round(auc_aug - CHAMPION_REF, 8),
            "promising": bool(promising),
            "decision_rule": "PROMISING if augmentation_delta >= +5e-5",
        },
        "n_rating_cols": len(RATING_COLS),
        "n_teachers": len(RATING_COLS) * 5,
    }
    path = OUT / "feat51_residual_ratings.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    # persist the cross-fit residuals for future champion integration (aux feature)
    art = ROOT / "artifacts" / "phase20" / "feat51_residual_ratings"
    art.mkdir(parents=True, exist_ok=True)
    np.save(art / "resid_crossfit_13.npy", X_resid.astype(np.float32))
    np.save(art / "base21_raw_encoded.npy", X_raw.astype(np.float32))
    np.save(art / "fold0_pred_aug.npy", pred_aug.astype(np.float32))
    print(f"artifacts -> {art.relative_to(ROOT).as_posix()}/")

    f0 = report["fold0_results"]
    print(f"fold-0: base(21raw)={f0['base_21raw_auc']} "
          f"aug(+13resid)={f0['augmented_21raw_13resid_auc']} "
          f"delta={f0['augmentation_delta']:+.2e}")
    print(f"resid_only={f0['resid_only_auc']} "
          f"spearman_aug_vs_champ={f0['spearman_aug_vs_champion']} "
          f"blend_delta={f0['blend_delta_vs_aug']:+.2e}")
    print(f"port_delta vs champion={f0['port_delta_vs_champion']:+.2e} "
          f"promising={f0['promising']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
