"""P20-FEAT48-RICH-F3 -- the +13 hierarchical rating block on the rich host.

Prereg fee4482d (configs/phase20/feat48_rich_f3_prereg.json).  Control =
champion static-full view (237); treatment = static-full + 13 cross-fit
hierarchical predictions (250).  Protocol verbatim from FEAT51-F3 (prereg
36beca0) via the shared lgbm_5f_oof; parity gate: control fold-0 must
reproduce FEAT51's control arm 0.960953 (tol 5e-6) or the batch is voided.
If the admission gate passes, the ENS-stage stack admission test (frozen
equal-logit vs champion v7) runs immediately in the same process.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/feat48_rich_f3.py
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

from src.common import ID_COL, TARGET, arr_sha256, load_raw  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission.store import load_oof  # noqa: E402
from scripts.phase20.feat48_hierarchical_ratings import build_design  # noqa: E402
from scripts.phase20.feat48_hierarchical_f3 import lgbm_5f_oof, logit  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "feat48_hierarchical_ratings"
HIER_ART = ROOT / "artifacts" / "phase20" / "feat48_hierarchical_ratings" / "hier_crossfit_13.npy"
PREREG = "configs/phase20/feat48_rich_f3_prereg.json"
PREREG_SHA = "fee4482d5c570c3958448eaac149ae8e8cc82acb3a2088c91b2d341a23aa3747"  # pinned pre-result
PARITY_F0 = 0.960953  # FEAT51-F3 control arm fold-0 (identical recipe/view/folds)
PARITY_TOL = 5e-6
ADMISSION_GAIN = 1.5e-5
CHAMPION = "v7_sol_tabpfn_internal2_aux10"


def main() -> int:
    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"

    ds = load_raw()
    tr, te = ds.train, ds.test
    y = tr[TARGET].to_numpy()
    folds = get_scheme("primary", y, tr[ID_COL].to_numpy()).folds

    print("[rich] building static full view (237)...", flush=True)
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    X_static = vb.static_tr.astype(np.float32)
    assert X_static.shape[1] == 237, X_static.shape

    print("[rich] building X_hier (cross-fit multi-output HistGB, 5x13)...", flush=True)
    _raw, X_hier = build_design(tr, folds)
    HIER_ART.parent.mkdir(parents=True, exist_ok=True)
    np.save(HIER_ART, X_hier.astype(np.float32))
    hier_sha = arr_sha256(X_hier.astype(np.float32))
    print(f"[rich] X_hier saved -> {HIER_ART.relative_to(ROOT).as_posix()} sha256={hier_sha[:16]}", flush=True)

    X_treat = np.column_stack([X_static, X_hier]).astype(np.float32)
    print(f"[rich] control={X_static.shape} treatment={X_treat.shape}", flush=True)

    oof_ctrl, auc_ctrl = lgbm_5f_oof(X_static, y, folds)
    print("[rich] control arm complete", flush=True)
    oof_treat, auc_treat = lgbm_5f_oof(X_treat, y, folds)
    print("[rich] treatment arm complete", flush=True)

    # ---- parity with FEAT51-F3 control arm (void on mismatch) ----
    pc = auc_ctrl[0]
    parity_ok = abs(pc - PARITY_F0) <= PARITY_TOL
    if not parity_ok:
        rep = {
            "prereg": PREREG, "prereg_sha256": sha, "status": "PARITY_FAIL_VOID",
            "expected_fold0_control": PARITY_F0, "observed_fold0_control": pc, "tol": PARITY_TOL,
        }
        path = OUT / "feat48_rich_f3.json"
        path.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(f"[rich] PARITY FAIL: fold-0 control {pc:.6f} vs {PARITY_F0} -> batch void", flush=True)
        return 2
    print(f"[rich] fold-0 parity OK: control {pc:.6f} == {PARITY_F0}", flush=True)

    paired = np.array(auc_treat) - np.array(auc_ctrl)
    mean_delta = float(paired.mean())
    se_delta = float(paired.std(ddof=1) / np.sqrt(5))
    t_stat = mean_delta / se_delta if se_delta > 0 else 0.0
    pos_folds = int((paired > 0).sum())
    passes = bool(mean_delta >= ADMISSION_GAIN and mean_delta >= 2.5 * se_delta and pos_folds >= 4)

    oof_ctrl_auc = float(roc_auc_score(y, oof_ctrl))
    oof_treat_auc = float(roc_auc_score(y, oof_treat))

    report = {
        "prereg": PREREG,
        "prereg_sha256": sha,
        "status": "COMPLETE_F3_RICH",
        "parity": {"fold0_control": pc, "expected": PARITY_F0, "tol": PARITY_TOL, "ok": True},
        "hier_artifact": {"path": HIER_ART.relative_to(ROOT).as_posix(), "arr_sha256": hier_sha},
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
            "admission_gate": {"required_gain": ADMISSION_GAIN, "passes": passes},
        },
    }

    # ---- ENS-stage stack admission runs immediately if the feature gate passes ----
    if passes:
        champ = load_oof(CHAMPION)
        champ_fold, blend_fold = [], []
        for k in range(5):
            m = folds == k
            blend = (0.5 * (logit(champ[m]) + logit(oof_treat[m]))).astype(np.float32)
            champ_fold.append(float(roc_auc_score(y[m], champ[m])))
            blend_fold.append(float(roc_auc_score(y[m], blend)))
        s_paired = np.array(blend_fold) - np.array(champ_fold)
        s_mean = float(s_paired.mean())
        s_se = float(s_paired.std(ddof=1) / np.sqrt(5))
        s_t = s_mean / s_se if s_se > 0 else 0.0
        s_pos = int((s_paired > 0).sum())
        s_pass = bool(s_mean >= ADMISSION_GAIN and s_mean >= 2.5 * s_se and s_pos >= 4)
        report["stack_admission"] = {
            "champion": CHAMPION,
            "champion_fold_auc": [round(v, 6) for v in champ_fold],
            "blend_fold_auc": [round(v, 6) for v in blend_fold],
            "paired_deltas": [round(d, 8) for d in s_paired],
            "mean": round(s_mean, 8), "se": round(s_se, 8), "t": round(s_t, 4),
            "positive_folds": s_pos, "passes": s_pass,
            "spearman_vs_champion": round(float(spearmanr(champ, oof_treat).statistic), 6),
        }
        print(f"[rich] STACK ADMISSION: mean {s_mean:+.3e} t={s_t:.2f} pos={s_pos}/5 -> {s_pass}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat48_rich_f3.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    np.save(OUT / "rich_f3_oof_control.npy", oof_ctrl.astype(np.float32))
    np.save(OUT / "rich_f3_oof_treatment.npy", oof_treat.astype(np.float32))

    print(f"[rich] control OOF {oof_ctrl_auc:.6f}  treatment OOF {oof_treat_auc:.6f}")
    print(f"[rich] paired {[round(d, 8) for d in paired]}")
    print(f"[rich] mean {mean_delta:+.8f} se {se_delta:.8f} t={t_stat:.2f} pos={pos_folds}/5")
    print(f"[rich] FEATURE GATE: {passes}")
    print(f"[rich] report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
