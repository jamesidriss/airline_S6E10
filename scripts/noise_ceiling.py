"""Estimate the irreducible-noise (Bayes) ceiling of S6E10.

If the generator drew y ~ Bernoulli(p(x)) from a function p, then two rows with identical
feature vectors must have the same p. Comparing the observed within-group label variance with
the binomial expectation therefore bounds how much signal any model can extract.

We group by progressively richer keys and report:
  n_groups, mean group size, observed positive rate, binomial-vs-observed heterogeneity
  (a chi-square style dispersion) and the AUC attainable from the group label rate alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, TARGET, load_cached_parquet, save_json, REPORTS  # noqa: E402
from src.features.s6e10 import META4, NUMS, SURVEY13  # noqa: E402

KEYS = {
    "S13": SURVEY13,
    "S13+M4": SURVEY13 + META4,
    "S13+M4+age": SURVEY13 + META4 + ["Age"],
    "S13+M4+age+fd": SURVEY13 + META4 + ["Age", "Flight Distance"],
    "all21": SURVEY13 + META4 + NUMS,
    "all21_minus_arrdelay": SURVEY13 + META4 + ["Age", "Flight Distance", "Departure Delay in Minutes"],
    "FD": ["Flight Distance"],
}


def main() -> None:
    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    rep = {}
    print(f"{'key':<24} {'n_groups':>9} {'dup_rows%':>10} {'mean_size':>10} {'auc(group)':>11} {'dispersion':>11}")
    for name, cols in KEYS.items():
        h = pd.util.hash_pandas_object(tr[cols].astype(object), index=False)
        df = pd.DataFrame({"h": h.values, "y": y})
        g = df.groupby("h")["y"].agg(["size", "mean", "sum"])
        n_groups = len(g)
        dup = float(g.loc[g["size"] > 1, "size"].sum() / len(df))
        # score = group mean; AUC of the group-rate score over all rows
        score = df["h"].map(g["mean"]).to_numpy()
        a = roc_auc_score(y, score)
        # dispersion: sum over groups of (n_g * p_g * (1-p_g) * (k_g - 1)^2 / (n_g - 1)) style
        # simpler: Pearson chi2 statistic vs binomial expectation
        exp_p = y.mean()
        chi2 = float((((g["sum"] - g["size"] * exp_p) ** 2) / (g["size"] * exp_p * (1 - exp_p))).sum())
        dof = n_groups - 1
        ratio = chi2 / dof if dof > 0 else float("nan")
        rep[name] = {"n_groups": n_groups, "dup_row_frac": round(dup, 5),
                     "mean_group_size": round(float(g["size"].mean()), 4),
                     "auc_of_group_rate": round(float(a), 6),
                     "chi2_over_dof": round(ratio, 4)}
        print(f"{name:<24} {n_groups:>9} {dup*100:>9.3f}% {g['size'].mean():>10.3f} "
              f"{a:>11.6f} {ratio:>11.4f}")

    # ---- how much AUC is left on the table?  Compare to the champion model.
    from src.submission import store

    try:
        oof = store.load_oof("view_lgbm_full_primary").astype("float64")
        print(f"\nchampion lgbm/full OOF AUC = {roc_auc_score(y, oof):.6f}")
        for name, cols in KEYS.items():
            h = pd.util.hash_pandas_object(tr[cols].astype(object), index=False)
            df = pd.DataFrame({"h": h.values, "y": y})
            g = df.groupby("h")["y"].agg(["mean"])
            grp = df["h"].map(g["mean"]).to_numpy()
            rep[name]["blend_with_champion_rank"] = None
        save_json(rep, REPORTS / "noise_ceiling.json")
        print("wrote", REPORTS / "noise_ceiling.json")
    except Exception as exc:  # noqa: BLE001
        print("champion load failed:", exc)
        save_json(rep, REPORTS / "noise_ceiling.json")


if __name__ == "__main__":
    main()