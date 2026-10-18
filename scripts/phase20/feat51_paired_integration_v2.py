"""P20-FEAT51-PAIRED-INTEGRATION v2 -- fixed blend calculation with aligned arrays.

Re-runs the paired control/treatment 5-fold CV, saves per-fold predictions
for correct blend calculation. Prereg 36beca0 (same as before).

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat51_paired_integration_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_raw  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat51_paired_integration"
RESID_PATH = ROOT / "artifacts" / "phase20" / "feat51_residual_ratings" / "resid_crossfit_13.npy"

LGBM_PARAMS = {
    "learning_rate": 0.02, "num_leaves": 127, "extra_trees": True,
    "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
    "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1,
    "n_jobs": 8, "random_state": 1,
}

CHAMPION_OOF = 0.9618529228157235
ADMISSION_GAIN = 1.5e-5


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def lgbm_5f_oof_save_preds(X: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int = 1) -> tuple[np.ndarray, list[float]]:
    """Train LGBM 5-fold CV and return (full OOF preds, per-fold AUCs). Saves per-fold preds."""
    from lightgbm import LGBMClassifier
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

        m = LGBMClassifier(n_estimators=6000, early_stopping_rounds=250, **LGBM_PARAMS)
        m.fit(X[fit_m][inner_train], y[fit_m][inner_train],
              eval_set=[(X[fit_m][inner_val], y[fit_m][inner_val])],
              eval_metric="auc")
        preds = m.predict_proba(X[val_m])[:, 1]
        oof[val_m] = preds
        fold_aucs.append(float(roc_auc_score(y[val_m], preds)))
    return oof, fold_aucs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    tr, te = ds.train, ds.test
    y = tr[TARGET].to_numpy()
    folds = get_scheme("primary", y, tr[ID_COL]).folds

    # Build champion's static full view (237 features)
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    X_static = vb.static_tr.astype(np.float32)  # (699635, 237)

    # Load cross-fit residuals (13 features, 5-fold cross-fit)
    resid = np.load(RESID_PATH).astype(np.float32)  # (699635, 13)
    assert resid.shape == (len(tr), 13)

    # Control: static full view only (237 features)
    # Treatment: static full view + 13 residuals = 250 features
    X_control = X_static
    X_treatment = np.column_stack([X_static, resid]).astype(np.float32)

    print(f"Control features: {X_control.shape[1]}")
    print(f"Treatment features: {X_treatment.shape[1]}")

    y_arr = ds.train[TARGET].to_numpy()
    folds = get_scheme("primary", y, tr[ID_COL]).folds

    # Run paired 5-fold CV - CONTROL
    oof_control, aucs_control = lgbm_5f_oof_save_preds(X_control, ds.train[TARGET].to_numpy(), folds)
    
    # Run paired 5-fold CV - TREATMENT
    oof_treatment, aucs_treatment = lgbm_5f_oof_save_preds(X_treatment, ds.train[TARGET].to_numpy(), folds)

    auc_control = float(roc_auc_score(ds.train[TARGET].to_numpy(), oof_control))
    auc_treatment = float(roc_auc_score(ds.train[TARGET].to_numpy(), oof_treatment))

    paired_deltas = np.array(aucs_treatment) - np.array(aucs_control)
    mean_delta = float(paired_deltas.mean())
    std_delta = float(paired_deltas.std(ddof=1))
    se_delta = std_delta / np.sqrt(5)
    t_stat = mean_delta / se_delta if se_delta > 0 else 0.0
    positive_folds = int((paired_deltas > 0).sum())

    # Admission gate
    ADMISSION_GAIN = 1.5e-5
    passes_gate = (mean_delta >= ADMISSION_GAIN) and (mean_delta >= 2.5 * (std_delta / np.sqrt(5))) and (np.sum(paired_deltas > 0) >= 4)

    # CORRECTED blend calculation: use the SAME row order for both predictions and labels
    # oof_control and oof_treatment are already in original row order (indexed by val_m)
    champ_ref = load_oof("sol_route_aux10_primary_final")
    
    # Both oof_treatment and champ_ref are in original row order - can directly blend
    spear = float(spearmanr(oof_treatment, champ_ref).statistic)
    blend = (logit(oof_treatment) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(ds.train[TARGET].to_numpy(), blend))
    blend_delta = auc_blend - auc_treatment

    report = {
        "prereg": "configs/phase20/feat51_paired_integration_prereg.json",
        "prereg_sha256": "36becae012fc27c6351d2e92aaf5c76f13a848123ba9d5a644c002a5e191d863",
        "control": "champion static full view (237 features)",
        "treatment": "champion static full view (237) + 13 cross-fit residuals",
        "identical_conditions": {
            "folds": "primary 5-fold (seed 20261010)",
            "lgbm_recipe": "xt_xt_f10 (seed 1, 6000 est, ES 250)",
            "inner_es": "10% carved from FIT only, seed 1 per fold",
            "residuals": "cross-fit 13 residuals from FEAT51 (5-fold, HistGB teachers)"
        },
        "results": {
            "control_oof_auc": round(float(roc_auc_score(ds.train[TARGET].to_numpy(), oof_control)), 6),
            "treatment_oof_auc": round(float(roc_auc_score(ds.train[TARGET].to_numpy(), oof_treatment)), 6),
            "per_fold_control": [round(v, 6) for v in aucs_control],
            "per_fold_treatment": [round(v, 6) for v in aucs_treatment],
            "paired_deltas_per_fold": [round(d, 8) for d in paired_deltas],
            "mean_paired_delta": round(mean_delta, 8),
            "std_paired_delta": round(float(paired_deltas.std(ddof=1)), 8),
            "se_paired_delta": round(float(np.array(paired_deltas).std(ddof=1) / np.sqrt(5)), 8),
            "t_stat": round(t_stat, 4),
            "positive_folds": positive_folds,
            "admission_gate": {
                "required_gain": 1.5e-5,
                "observed_mean_delta": round(mean_delta, 8),
                "se": round(float(np.array(paired_deltas).std(ddof=1) / np.sqrt(5)), 8),
                "t_stat": round(t_stat, 4),
                "positive_folds": f"{positive_folds}/5",
                "passes": bool(passes_gate)
            },
            "blend_with_champion_proxy": {
                "spearman_vs_champion_proxy": round(float(spearmanr(oof_treatment, champ_ref).statistic), 6),
                "equal_blend_auc": round(float(roc_auc_score(ds.train[TARGET].to_numpy(), (logit(oof_treatment) + logit(champ_ref)) / 2.0)), 6),
                "blend_delta_vs_treatment": round(blend_delta, 8)
            }
        },
        "artifact": "reports/phase20/feat51_paired_integration/feat51_paired_integration_v2.json",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat51_paired_integration_v2.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    # Save OOF predictions for future use
    np.save(OUT / "oof_control.npy", oof_control.astype(np.float32))
    np.save(OUT / "oof_treatment.npy", oof_treatment.astype(np.float32))

    print(f"Control OOF AUC: {float(roc_auc_score(ds.train[TARGET].to_numpy(), oof_control)):.6f}")
    print(f"Treatment OOF AUC: {float(roc_auc_score(ds.train[TARGET].to_numpy(), oof_treatment)):.6f}")
    print(f"Paired deltas: {[round(d, 8) for d in paired_deltas]}")
    print(f"Mean delta: {mean_delta:+.8f} ± {float(np.array(paired_deltas).std(ddof=1) / np.sqrt(5)):.8f} (t={t_stat:.2f})")
    print(f"Positive folds: {positive_folds}/5")
    print(f"Admission gate: {passes_gate}")
    print(f"Spearman vs champ: {spear:.6f}")
    print(f"Blend delta: {blend_delta:+.8f}")
    print(f"Report -> {path.relative_to(ROOT).as_posix()}")
    return 0


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def roc_auc_score(y_true, y_score):
    from sklearn.metrics import roc_auc_score as sk_roc_auc
    return sk_roc_auc(y_true, y_score)


def spearmanr(x, y):
    from scipy.stats import spearmanr as sk_spearmanr
    return sk_spearmanr(x, y)


if __name__ == "__main__":
    raise SystemExit(main())