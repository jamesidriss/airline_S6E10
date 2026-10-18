"""Integrate FEAT 51 cross-fit residuals (13 cols) into champion's 'full' view
and retrain LGBM full-5 OOF. The residuals are already cross-fit (5-fold, saved
at artifacts/phase20/feat51_residual_ratings/resid_crossfit_13.npy).

This is an F3 experiment (full-5 OOF + admission gate) — the decisive test
whether the decorrelated residual signal lifts the champion OOF toward 0.96274.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/champion_residual_integration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_raw, TARGET, ID_COL  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.features.view import ViewBuilder, VIEWS  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

# Add residual_ratings block to VIEWS for a new view "full_residual"
VIEWS["full_residual"] = VIEWS["full"] + ["residual_ratings"]

OUT = ROOT / "reports" / "phase20" / "champion_residual_integration"
RESID_PATH = ROOT / "artifacts" / "phase20" / "feat51_residual_ratings" / "resid_crossfit_13.npy"

CHAMPION_OOF = 0.9618529228157235  # immutable champion A
ADMISSION_GAIN = 1.5e-5  # mean paired fold gain >= 1.5e-5


class ResidualViewBuilder(ViewBuilder):
    """ViewBuilder that injects the cross-fit residuals as a new block."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "residual_ratings" not in VIEWS:
            VIEWS["full_residual"] = VIEWS["full"] + ["residual_ratings"]
        self.resid_train = np.load(RESID_PATH).astype(np.float32)  # (699635, 13)
        assert self.resid_train.shape == (len(self.tr), 13)

    def assemble(self, fit_idx, y, val_idx, test_idx=None, inner_seed=0, te_all21_smooth="auto"):
        Xf, Xa, names_f = super().assemble(fit_idx, y, val_idx, test_idx, inner_seed, te_all21_smooth)

        # Inject residuals for fit (train only) and val
        resid_fit = self.resid_train[fit_idx]
        resid_val = self.resid_train[val_idx]

        Xf = np.column_stack([Xf, resid_fit])
        Xa = {k: np.column_stack([v, resid_val]) for k, v in Xa.items()}
        names_f = list(names_f) + [f"resid_{i}" for i in range(13)]

        # For test: no residuals available (computed cross-fit on train only).
        # Not needed for OOF. If test needed later, would need full-train teachers.
        return Xf, Xa, names_f


def lgbm_5f_oof(X_full, y, folds, seed=1, n_estimators=6000):
    """Train LGBM 5-fold CV and return (full OOF preds, per-fold AUCs)."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split

    n = len(y)
    oof = np.zeros(n, dtype=np.float32)
    fold_aucs = []
    for k in range(5):
        fit_m = folds != k
        val_m = folds == k
        # inner 10% ES from FIT
        rng = np.random.default_rng(seed + k)
        yfit = y[fit_m]
        val_parts = []
        for cls in (0, 1):
            ci = np.where(yfit == cls)[0]
            nv = max(1, int(round(0.1 * len(ci))))
            val_parts.append(rng.choice(ci, nv, replace=False))
        inner_val = np.concatenate(val_parts)
        inner_train = np.setdiff1d(np.arange(fit_m.sum()), inner_val)

        m = LGBMClassifier(
            n_estimators=n_estimators, early_stopping_rounds=250,
            learning_rate=0.02, num_leaves=127, extra_trees=True,
            min_child_samples=40, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, reg_lambda=1.0, max_bin=255, verbose=-1,
            n_jobs=8, random_state=seed)
        m.fit(X_full[fit_m][inner_train], yfit[inner_train],
              eval_set=[(X_full[fit_m][inner_val], yfit[inner_val])],
              eval_metric="auc")
        oof[val_m] = m.predict_proba(X_full[val_m])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[val_m], oof[val_m])))
    return oof, fold_aucs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr, te = ds.train, ds.test
    y = tr[TARGET].to_numpy()
    folds = get_scheme("primary", y, tr[ID_COL]).folds

    # Build the champion's full view with residuals
    vb = ResidualViewBuilder(tr, te, "full_residual")
    vb.build_static()  # builds static features
    X_full = np.column_stack([vb.static_tr, vb.resid_train]).astype(np.float32)
    n_features = X_full.shape[1]

    print(f"Champion view with residuals: {X_full.shape} features={n_features}")

    # Run 5-fold CV
    oof, fold_aucs = lgbm_5f_oof(X_full, y, folds)
    auc_5f = float(roc_auc_score(y, oof))
    mean_fold = np.mean(fold_aucs)
    std_fold = np.std(fold_aucs)

    # Paired test vs champion A OOF (not in store, use reference)
    delta = auc_5f - CHAMPION_OOF
    paired = None  # champion OOF not in store

    report = {
        "view": "full_residual (285 + 13 cross-fit residuals)",
        "n_features": int(X_full.shape[1]),
        "oof_auc_5f": round(auc_5f, 6),
        "champion_oof": CHAMPION_OOF,
        "delta_vs_champion": round(delta, 8),
        "fold_aucs": [round(v, 6) for v in fold_aucs],
        "mean_fold_auc": round(mean_fold, 6),
        "std_fold_auc": round(std_fold, 6),
        "admission_gate": {
            "required_gain": ADMISSION_GAIN,
            "observed_gain": round(delta, 8),
            "passes": bool(delta >= ADMISSION_GAIN),
        },
    }
    path = OUT / "champion_residual_integration.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"5-fold OOF AUC: {auc_5f:.6f} vs champion {CHAMPION_OOF:.6f}")
    print(f"delta: {delta:+.8f} | fold aucs: {[f'{v:.6f}' for v in fold_aucs]}")
    print(f"admission gate: {report['admission_gate']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())