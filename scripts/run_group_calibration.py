"""Nested meta-validation of GROUP-CONDITIONAL score calibration on the finalist OOF.

Why this can move AUC at all
----------------------------
A GLOBAL monotone recalibration cannot change ROC-AUC: ROC-AUC is a function of the induced
score ORDER only, and any strictly increasing map preserves it. That makes G0 a mandatory control
-- it must reproduce the base AUC to floating-point precision, and if it does not, the harness is
broken and every other arm is uninterpretable.

A GROUP-CONDITIONAL map is not a global monotone map. Shifting one group's scores relative to
another changes the global ordering, so it CAN change global AUC. The question is whether the
current finalist has a stable, exploitable cross-group rank bias.

This differs from the already-rejected local-reliability gate, which changed MODEL WEIGHTS by
score region. Here the model predictions are untouched; only a low-dimensional monotone-in-logit
recalibration with per-group offsets is applied.

Arms (predeclared)
------------------
  G0  global Platt                a*base + b                    -- control, AUC must not move
  G1  base + group intercepts      [base, 1_g]                   -- cross-group rank alignment
  G2  G1 + per-group slope         [base, 1_g, base*1_g]
  G3  shared piecewise-linear base + group intercepts  (GAM-lite)

Protocol
--------
Meta folds are the immutable `primary` folds. For meta fold k the calibrator is fitted on the
OOF rows of the other four folds only and applied to fold k. Fit and evaluation rows are disjoint
by construction, so no group shift is ever fitted and evaluated on the same rows.

Caveat, stated plainly: a fold-j OOF prediction comes from a model that saw fold k's labels. So the
calibrator's handful of coefficients are influenced, very weakly, by rows they are later applied
to. This is the ordinary OOF-stacking coupling shared by every second-level layer; with ~560k
fitting rows and <40 parameters it is far below the noise floor, and it is identical across all
arms including G0, so the comparison is fair.

Usage: python scripts/run_group_calibration.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

BASE_MEMBER = "blend_v3_final"
GROUPS = {
    "Class": "Class",
    "TypeOfTravel": "Type of Travel",
    "CustomerType": "Customer Type",
    "Gender": "Gender",
}
# Crossing every group would give 3*3*4*2 = 72 cells; far too many for the row counts, and the
# prompt explicitly warns against high-capacity grouping. Each arm uses ONE grouping at a time,
# plus a low-order union of all four (8+6+4+2 levels -> max(k)-coded, 16 columns, one column per
# distinct value, so intercepts sum rather than chain).
ARMS = [
    ("G0_global", None, "intercept"),
    ("G1_class", ["Class"], "intercept"),
    ("G1_travel", ["TypeOfTravel"], "intercept"),
    ("G1_custtype", ["CustomerType"], "intercept"),
    ("G1_gender", ["Gender"], "intercept"),
    ("G1_union", list(GROUPS), "intercept"),
    ("G2_class", ["Class"], "slope"),
    ("G2_union", list(GROUPS), "slope"),
    ("G3_class", ["Class"], "spline"),
]


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(base, gcols, mode):
    """Return the design matrix for one arm. Column 0 is always the base logit.

    `gcols` is a list of one-hot blocks; each block may have several columns, and a group is
    added column-wise so that additive intercepts for a k-level factor give k-1 free offsets
    (the last level is the reference).
    """
    n = len(base)
    if mode == "intercept" and not gcols:
        return np.ones((n, 1))
    D = [base.reshape(n, 1)]
    if mode == "slope":
        for g in gcols:
            for j in range(g.shape[1]):
                D.append(g[:, j].reshape(n, 1))
        for g in gcols:
            for j in range(g.shape[1]):
                D.append((base * g[:, j]).reshape(n, 1))
        return np.column_stack(D)
    if mode == "intercept":
        for g in gcols:
            for j in range(g.shape[1]):
                D.append(g[:, j].reshape(n, 1))
        return np.column_stack(D)
    if mode == "spline":
        # Shared piecewise-linear shape on the base logit, then group intercepts. Knots are
        # quantiles of the base SCORE, which carries no label information, so computing them on
        # all rows cannot leak the target of any row.
        for k in np.quantile(base, [0.10, 0.25, 0.50, 0.75, 0.90]):
            D.append(np.maximum(base - k, 0.0).reshape(n, 1))
        for g in gcols:
            for j in range(g.shape[1]):
                D.append(g[:, j].reshape(n, 1))
        return np.column_stack(D)
    raise ValueError(mode)


def onehot(series):
    v = series.to_numpy()
    u = np.unique(v)
    cols = [(v == uu).astype("float64") for uu in u[1:]]
    if not cols:
        return np.zeros((len(v), 0))
    return np.column_stack(cols)


def main() -> None:
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr["id"]).folds

    base = logit(store.load_oof(BASE_MEMBER))
    base_auc = float(roc_auc_score(y, base))
    print(f"base = {BASE_MEMBER}   OOF AUC = {base_auc:.6f}\n")

    G = {name: onehot(tr[GROUPS[name]]) for name in GROUPS}

    out = {"base_member": BASE_MEMBER, "base_oof_auc": base_auc, "arms": []}
    print("=" * 104)
    print(f"{'arm':<14}{'cols':>6}{'AUC':>11}{'delta_e5':>11}{'folds_pos':>12}"
          f"{'per-fold delta (e-5)'}")
    print("=" * 104)

    for arm, gnames, mode in ARMS:
        gcols = []
        for g in (gnames or []):
            gcols.append(G[g])
        per_fold, transformed_all = [], np.full(len(base), np.nan)
        for k in sorted(set(folds.tolist())):
            tr_m = np.where(folds != k)[0]
            va_m = np.where(folds == k)[0]
            Xtr = design(base[tr_m], gcols, mode)
            Xva = design(base[va_m], gcols, mode)
            lr = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
            lr.fit(Xtr, y[tr_m])
            z = lr.decision_function(Xva)
            if np.std(z) <= 0 or lr.coef_[0][0] <= 0:
                per_fold.append(float("nan"))
                continue
            transformed_all[va_m] = z
            per_fold.append(roc_auc_score(y[va_m], z) - roc_auc_score(y[va_m], base[va_m]))
        ok = [d for d in per_fold if np.isfinite(d)]
        mean_d = float(np.mean(ok)) if ok else float("nan")
        auc_full = float(roc_auc_score(y, transformed_all)) if np.isfinite(
            transformed_all).all() else float("nan")
        ncols = design(base[:2], gcols, mode).shape[1]
        pos = sum(1 for d in ok if d > 0)
        pf = " ".join(f"{d*1e5:+.1f}" for d in per_fold)
        print(f"{arm:<14}{ncols:>6}{auc_full:>11.6f}{mean_d*1e5:>+11.1f}{pos:>7}/{len(ok):<4} {pf}")
        out["arms"].append({
            "arm": arm, "groups": gnames, "mode": mode, "n_columns": ncols,
            "auc": auc_full, "mean_delta": mean_d, "delta_e5": mean_d * 1e5,
            "per_fold_delta": per_fold, "folds_positive": pos, "n_folds": len(ok),
            "meets_promotion_rule": bool(len(ok) == 5 and pos == 5 and mean_d >= 3e-5),
        })

    print("=" * 104)
    g0 = next(a for a in out["arms"] if a["arm"] == "G0_global")
    tol = abs(g0["mean_delta"])
    print(f"\n  CONTROL CHECK (G0): |mean delta| = {tol:.3e}. "
          f"{'PASS' if tol < 1e-12 else 'FAIL -- harness is broken, other arms uninterpretable'}")
    best = max((a for a in out["arms"] if a["arm"] != "G0_global"), key=lambda a: a["mean_delta"])
    print(f"  best non-control arm: {best['arm']}  {best['delta_e5']:+.1f}e-5  "
          f"positive in {best['folds_positive']}/{best['n_folds']} folds")
    if best["mean_delta"] >= 3e-5 and best["folds_positive"] == best["n_folds"]:
        print(f"  VERDICT: {best['arm']} MEETS the predeclared rule (mean >= +3e-5, same sign "
              f"across folds). Escalate.")
    else:
        print("  VERDICT: no arm meets the predeclared rule (mean >= +3e-5 and same sign across "
              "all folds). CLOSE the group-calibration axis.")

    out["control_check_pass"] = bool(tol < 1e-12)
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()[:12]
    save_json(out, REPORTS / "group_calibration.json")
    print("wrote", REPORTS / "group_calibration.json")


if __name__ == "__main__":
    main()
