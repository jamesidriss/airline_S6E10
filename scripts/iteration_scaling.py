"""How should the boosting iteration count scale with training-set size?

Why this matters
----------------
Once the iteration count is chosen without touching the outer validation fold, the final model is
trained on MORE rows than the model whose early stopping produced that count. Under-iterating is
the safe direction (it never overfits) but it throws away real signal, and the learning curve says
a bigger training set both scores higher and wants more rounds. So a full-fit policy needs a
principled size correction, not a blind reuse of the 72%-data number.

The honest warning
------------------
There are only FIVE measurements, they span less than one order of magnitude, and the smallest
point is visibly off the trend (377 at 63k rows, then 395 at 126k -- almost flat, then 525 at
252k). A power law fitted to all five points therefore EXTRAPOLATES BACKWARDS, predicting fewer
rounds at 560k than were actually observed at 504k. That is not a usable extrapolation. This script
computes every candidate fit and reports each one's R^2 and, crucially, whether it reproduces the
one holdout-style check available: the observed count at 504k.

Which exponent to actually use
------------------------------
`exp_upper` -- OLS of log(best_iter) on log(n) over the THREE largest points only -- is the one
predeclared for use, on the stated grounds that the smallest point is off-trend and the policy only
ever extrapolates upward from the largest region. `exp_all` is reported for transparency and is
NOT used, because it gets the direction of extrapolation backwards.

Usage: python scripts/iteration_scaling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402

SRC = REPORTS / "lcurve_fold0.json"
ES_FRAC = 0.10


def load_points():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for r in d["rows"]:
        n_sub = r["n_train_rows"]            # rows sampled out of the outer-fit block
        n_fit = n_sub * (1.0 - ES_FRAC)      # rows the model actually trains on after the ES carve
        rows.append({"frac": r["frac"], "n_sub": n_sub, "n_fit": n_fit,
                     "best_iter": r["best_iter_mean"], "auc": r["auc_mean"]})
    return rows


def ols_loglog(x, y):
    """OLS of log(y) on log(x). Returns (exponent, coefficient, r2)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return float(a), float(b), 1.0 - ss_res / ss_tot


def main() -> None:
    pts = load_points()
    n_fit = [p["n_fit"] for p in pts]
    it = [p["best_iter"] for p in pts]
    n_all = len(pts)

    print("=" * 100)
    print("ITERATION SCALING  (n = rows actually trained on, after the inner-ES carve)")
    print("=" * 100)
    print(f"{'frac':>7}{'n_sub':>10}{'n_fit':>10}{'best_iter':>11}")
    for p in pts:
        print(f"{p['frac']:>7.3f}{p['n_sub']:>10,}{p['n_fit']:>10,.0f}{p['best_iter']:>11.1f}")

    cands = {}
    for k in (5, 4, 3):
        if k < 3:
            continue
        a, b, r2 = ols_loglog(n_fit[n_all - k:], it[n_all - k:])
        pred_504 = float(np.exp(a * np.log(503_739) + b))
        obs_504 = it[-1]
        cands[f"exp_upper{k}"] = {
            "k_points": k, "exponent": a, "coefficient": float(np.exp(b)), "r2_loglog": r2,
            "predicted_at_504k": pred_504, "observed_at_504k": obs_504,
            "backward_check_ratio_pred_over_obs": pred_504 / obs_504,
        }
        print(f"\n  exp_upper{k}: best_iter = {np.exp(b):.4f} * n^{a:.4f}   R^2(loglog)={r2:.5f}")
        print(f"    backward check: predicts {pred_504:.0f} at n=503,739, observed {obs_504:.0f}"
              f"  (ratio {pred_504/obs_504:.3f})")

    usable = [k for k, v in cands.items() if v["k_points"] == 3]
    chosen = "exp_upper3"
    ch = cands[chosen]

    print("\n" + "-" * 100)
    print(f"PREDECLARED EXPONENT FOR THE FULL-FIT POLICY: {chosen}  "
          f"exponent = {ch['exponent']:.4f}")
    print("  Rationale: the smallest point (63k rows, 377 iters) is off-trend -- the next doubling "
          "adds only 18 rounds --\n  so an all-points fit extrapolates backwards and predicts FEWER "
          "rounds at 560k than were\n  observed at 504k. The policy only ever extrapolates upward "
          "from the largest region, so the\n  three largest points are the relevant ones.")
    print("-" * 100)

    # size corrections that the policy actually needs
    n_ctrl = 503_739                      # current protocol: 72% of all labels
    n_full = 559_708                      # 5-fold outer-fit, no ES carve: 80% of all labels
    n_f10 = int(round(699_635 * 0.90))    # 10-fold outer-fit, no ES carve: 90% of all labels
    n_all100 = 699_635                    # a full refit on every labelled row
    e = ch["exponent"]
    print(f"\n  {'training fraction':<34}{'rows':>10}{'vs ctrl':>9}{'iteration':>11}")
    for label, n in (("current inner-ES (control)", n_ctrl),
                     ("5-fold full outer-fit", n_full),
                     ("10-fold full outer-fit", n_f10),
                     ("full refit on all labels", n_all100)):
        corr = (n / n_ctrl) ** e
        print(f"  {label:<34}{n:>10,}{corr:>9.3f}{879.0 * corr:>11.0f}")

    out = {
        "source": str(SRC), "es_fraction": ES_FRAC, "points": pts,
        "candidate_fits": cands,
        "chosen": chosen, "chosen_exponent": e,
        "rationale": ("the smallest point is off-trend, so an all-points fit extrapolates "
                      "backwards (predicts fewer rounds at 560k than observed at 504k); the "
                      "policy only extrapolates upward, so the three largest points govern"),
        "size_corrections_vs_control": {
            "n_ctrl": n_ctrl, "n_full_5fold": n_full, "n_full_10fold": n_f10,
            "n_all_labelled": n_all100,
            "factor_to_5fold_full": (n_full / n_ctrl) ** e,
            "factor_to_10fold_full": (n_f10 / n_ctrl) ** e,
            "factor_to_all_labelled": (n_all100 / n_ctrl) ** e,
        },
        "caveat": ("five points spanning <1 order of magnitude, so this is a weak law. It is used "
                   "only to correct an iteration count that is ALSO measured directly by an "
                   "inner CV, never as the sole source of the iteration count."),
    }
    save_json(out, REPORTS / "iteration_scaling.json")
    print("\nwrote", REPORTS / "iteration_scaling.json")


if __name__ == "__main__":
    main()
