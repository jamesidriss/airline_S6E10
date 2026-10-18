"""P20-FEAT48-F3 -- full paired 5-fold completion of the hierarchical-rating screen.

Prereg df88f6f2 (configs/phase20/feat48_hierarchical_f3_prereg.json), extends the
fold-0 screen prereg 84795f40.  Protocol mirrors feat51_paired_integration_v2.py
(prereg 36beca0) verbatim except the host view (21raw, row 48's own screen host)
and the feature block (+13 cross-fit multi-output HistGB hierarchical predictions,
fold-safe: each fold's features predicted by teachers trained on the other 4 folds).

Fold-0 parity: control/treatment must reproduce the screen (0.953423/0.953817,
tol 5e-6) or the batch is voided before any gate decision.

Admission gate (AGENTS.md, FEAT51-F3 application): mean paired fold delta
>= +1.5e-5 AND mean >= 2.5 * paired SE AND positive in >= 4 of 5 folds.
Descriptive metrics (spearman/blend vs route_ref) never decide.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat48_hierarchical_f3.py
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
from src.submission.store import load_oof  # noqa: E402
from scripts.phase20.feat48_hierarchical_ratings import LGBM_PARAMS, build_design  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat48_hierarchical_ratings"
PREREG = "configs/phase20/feat48_hierarchical_f3_prereg.json"
PREREG_SHA = "df88f6f22d30d5322dd4287783d764412c44d3f3e5c66f4709d0e599b65a91a8"  # pinned pre-result
SCREEN_F0 = {"control": 0.953423, "treatment": 0.953817}
PARITY_TOL = 5e-6
ADMISSION_GAIN = 1.5e-5
ROUTE_REF = "sol_route_aux10_primary_final"


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def lgbm_5f_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int = 1) -> tuple[np.ndarray, list[float]]:
    """Paired-CV LGBM OOF; verbatim protocol from feat51_paired_integration_v2 (prereg 36beca0)."""
    from lightgbm import LGBMClassifier
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)
    fold_aucs = []
    for k in range(5):
        fit_m = folds != k
        val_m = folds == k
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
        oof[val_m] = m.predict_proba(X[val_m])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[val_m], oof[val_m])))
        print(f"  fold {k}: control-arm fit done, auc={fold_aucs[-1]:.6f}", flush=True)
    return oof, fold_aucs


def main() -> int:
    import hashlib
    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"

    ds = load_raw()
    tr = ds.train
    y = tr[TARGET].to_numpy()
    folds = get_scheme("primary", y, tr[ID_COL].to_numpy()).folds

    print("[f3] building design (cross-fit multi-output HistGB, 5 folds x 13 ratings)...", flush=True)
    X_raw, X_hier = build_design(tr, folds)
    X_aug = np.column_stack([X_raw, X_hier]).astype(np.float32)
    print(f"[f3] design ready: raw={X_raw.shape} aug={X_aug.shape}", flush=True)

    oof_ctrl, auc_ctrl = lgbm_5f_oof(X_raw, y, folds)
    oof_treat, auc_treat = lgbm_5f_oof(X_aug, y, folds)

    # ---- fold-0 parity with the screen (void on mismatch) ----
    pc, pt = auc_ctrl[0], auc_treat[0]
    parity_ok = (abs(pc - SCREEN_F0["control"]) <= PARITY_TOL) and (abs(pt - SCREEN_F0["treatment"]) <= PARITY_TOL)
    if not parity_ok:
        rep = {
            "prereg": PREREG, "prereg_sha256": sha, "status": "PARITY_FAIL_VOID",
            "fold0_expected": SCREEN_F0, "fold0_observed": {"control": pc, "treatment": pt},
            "tolerance": PARITY_TOL,
        }
        path = OUT / "feat48_hierarchical_f3.json"
        path.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(f"[f3] PARITY FAIL: control {pc:.6f} vs {SCREEN_F0['control']}, "
              f"treatment {pt:.6f} vs {SCREEN_F0['treatment']} -> batch void", flush=True)
        return 2
    print(f"[f3] fold-0 parity OK: control {pc:.6f} == {SCREEN_F0['control']}, "
          f"treatment {pt:.6f} == {SCREEN_F0['treatment']}", flush=True)

    paired = np.array(auc_treat) - np.array(auc_ctrl)
    mean_delta = float(paired.mean())
    se_delta = float(paired.std(ddof=1) / np.sqrt(5))
    t_stat = mean_delta / se_delta if se_delta > 0 else 0.0
    pos_folds = int((paired > 0).sum())
    passes = bool(mean_delta >= ADMISSION_GAIN and mean_delta >= 2.5 * se_delta and pos_folds >= 4)

    oof_ctrl_auc = float(roc_auc_score(y, oof_ctrl))
    oof_treat_auc = float(roc_auc_score(y, oof_treat))
    champ_ref = load_oof(ROUTE_REF)
    sp = float(spearmanr(oof_treat, champ_ref).statistic)
    blend = (logit(oof_treat) + logit(champ_ref)) / 2.0
    auc_blend = float(roc_auc_score(y, blend))

    report = {
        "prereg": PREREG,
        "prereg_sha256": sha,
        "status": "COMPLETE_F3",
        "protocol": "FEAT51-F3 mirror (prereg 36beca0); host=21raw; treatment=+13 cross-fit hierarchical",
        "parity": {"fold0_control": pc, "fold0_treatment": pt, "tol": PARITY_TOL, "ok": True},
        "results": {
            "control_oof_auc": round(oof_ctrl_auc, 8),
            "treatment_oof_auc": round(oof_treat_auc, 8),
            "per_fold_control": [round(v, 6) for v in auc_ctrl],
            "per_fold_treatment": [round(v, 6) for v in auc_treat],
            "paired_deltas_per_fold": [round(d, 8) for d in paired],
            "mean_paired_delta": round(mean_delta, 8),
            "se_paired_delta": round(se_delta, 8),
            "t_stat": round(t_stat, 4),
            "positive_folds": pos_folds,
            "admission_gate": {
                "required_gain": ADMISSION_GAIN,
                "passes": passes,
            },
            "descriptive_only": {
                "spearman_vs_route_ref": round(sp, 6),
                "equal_logit_blend_auc": round(auc_blend, 8),
                "blend_delta_vs_treatment": round(auc_blend - oof_treat_auc, 8),
            },
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat48_hierarchical_f3.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    np.save(OUT / "f3_oof_control.npy", oof_ctrl.astype(np.float32))
    np.save(OUT / "f3_oof_treatment.npy", oof_treat.astype(np.float32))

    print(f"[f3] control OOF {oof_ctrl_auc:.6f}  treatment OOF {oof_treat_auc:.6f}")
    print(f"[f3] paired deltas {[round(d, 8) for d in paired]}")
    print(f"[f3] mean {mean_delta:+.8f} se {se_delta:.8f} t={t_stat:.2f} pos={pos_folds}/5")
    print(f"[f3] ADMISSION GATE: {passes}")
    print(f"[f3] spearman {sp:.6f} blend {auc_blend:.6f} (descriptive)")
    print(f"[f3] report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
