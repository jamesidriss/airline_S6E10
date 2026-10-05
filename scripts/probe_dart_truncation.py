"""Does `predict(num_iteration=r)` on a long DART fit equal a fresh r-round DART fit?

Why this decides the compute bill
---------------------------------
The honest fixed-round protocol needs an inner-validation curve of AUC against round count, from
which the round is selected. There are two ways to get it:

  refit     train a separate model at each snapshot round. Rigorous, but with six snapshots and a
            long DART fit this costs several hours per fold.
  snapshot  train ONE model to the largest round, then call `predict(num_iteration=r)`. Cheap --
            one fit total -- but only valid if truncating an N-round DART model reproduces what
            training an r-round DART model would have produced.

That equivalence is not something to assume. DART randomly drops previously added trees and
renormalises the survivors, and both the dropout sequence and the normalisation are functions of
the full round schedule, so an r-round model is not obviously the r-round prefix of an N-round one.
LightGBM 3.x/4.x also applies different shrinkage bookkeeping depending on whether the model was
built with `num_iteration` set, so the two can genuinely differ.

This probe measures the gap at probe scale, where a full refit sweep is affordable. If the gap is
far below the +1.5e-5 admission gate, snapshot selection is sound and the compute collapses from
hours to minutes. If it is not, the harness keeps refit mode and we pay the compute.

GBDT with extra_trees is measured on the same question as a control: GBDT's boosting is strictly
additive, so truncation is exact there by construction, and if the harness reports a large GBDT gap
too then the measurement itself is wrong.

Usage: python scripts/probe_dart_truncation.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import RAW21  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

ROUNDS = [200, 400, 700, 1100, 1600]
GATE = 1.5e-5          # the ensemble admission gate; truncation error must sit far below it

BASE = {"objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
        "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8}

ARMS = {
    "gbdt_xt": {"extra_trees": True},
    "dart005_noxt": {"boosting_type": "dart", "drop_rate": 0.05, "skip_drop": 0.5},
    "dart010_noxt": {"boosting_type": "dart", "drop_rate": 0.10, "skip_drop": 0.5},
}


def prep(seed=0):
    tr, _te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y_int, tr[ID_COL]).folds
    fit = np.where(folds != 0)[0]
    rng = np.random.default_rng(seed)
    sub = rng.choice(fit, size=90_000, replace=False)
    pos, neg = sub[y_int[sub] == 1], sub[y_int[sub] == 0]
    iv = np.unique(np.concatenate([rng.choice(pos, 12_000, replace=False),
                                    rng.choice(neg, 18_000, replace=False)]))
    itr = np.setdiff1d(sub, iv)
    cols = list(RAW21)
    X = np.empty((len(tr), len(cols)), dtype="float32")
    for j, c in enumerate(cols):
        s = tr[c]
        X[:, j] = (s.to_numpy(dtype="float32") if pd.api.types.is_numeric_dtype(s)
                   else pd.factorize(s.astype(str), sort=True)[0].astype("float32"))
    X = np.nan_to_num(X, nan=-999.0)
    return X, y_int.astype("float64"), itr, iv


def fit(X, y, params, rounds):
    import lightgbm as lgb
    p = dict(BASE)
    p.update(params)
    p.update(random_state=7, bagging_seed=8, feature_fraction_seed=9)
    return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=int(rounds))


def main() -> None:
    X, y, itr, iv = prep()
    print(f"truncation probe: inner-train={len(itr):,}  inner-val={len(iv):,}  "
          f"features={X.shape[1]}  rounds={ROUNDS}")
    print(f"same seed for every fit, so the comparison is paired.\n")

    out = {"rounds": ROUNDS, "gate_e5": GATE * 1e5, "arms": {}}

    for name, extra in ARMS.items():
        print(f"{'='*100}\n{name}   {json.dumps(extra)}\n{'='*100}")
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            long_m = fit(X[itr], y[itr], extra, max(ROUNDS))
            warn = sorted({str(x.message)[:90] for x in w})
        t_long = time.time() - t0

        rows, worst = [], 0.0
        for r in ROUNDS:
            trunc = long_m.predict(X[iv], num_iteration=r)
            fresh = fit(X[itr], y[itr], extra, r)
            a_tr, a_fr = (float(roc_auc_score(y[iv], trunc)),
                          float(roc_auc_score(y[iv], fresh.predict(X[iv]))))
            gap = a_tr - a_fr
            worst = max(worst, abs(gap))
            rows.append({"round": r, "truncated": a_tr, "refit": a_fr, "gap_e5": gap * 1e5})
            print(f"  round {r:>5}   truncated={a_tr:.6f}   refit={a_fr:.6f}   "
                  f"gap={gap*1e5:+8.2f}e-5")
        # does the argmax land on the same round under either curve?
        b_tr = max(rows, key=lambda z: z["truncated"])["round"]
        b_fr = max(rows, key=lambda z: z["refit"])["round"]
        sel_tr = min(z["round"] for z in rows if z["truncated"] >= max(v["truncated"] for v in rows) - 1e-12)
        sel_fr = min(z["round"] for z in rows if z["refit"] >= max(v["refit"] for v in rows) - 1e-12)
        verdict = ("SNAPSHOT SAFE" if worst < GATE / 3 else
                   "BORDERLINE" if worst < GATE else "SNAPSHOT UNSAFE -> use refit mode")
        print(f"  worst |gap| = {worst*1e5:.2f}e-5   argmax round: truncated={b_tr} refit={b_fr}   "
              f"selected: truncated={sel_tr} refit={sel_fr}")
        print(f"  long fit {max(ROUNDS)} rounds took {t_long:.0f}s; warnings={warn}")
        print(f"  VERDICT: {verdict}   (gate {GATE*1e5:.1f}e-5)\n")
        out["arms"][name] = {"extra_params": extra, "rows": rows, "worst_gap_e5": worst * 1e5,
                             "argmax_truncated": b_tr, "argmax_refit": b_fr,
                             "selected_truncated": sel_tr, "selected_refit": sel_fr,
                             "seconds_long_fit": round(t_long, 1), "warnings": warn,
                             "verdict": verdict}

    dart_worst = max(out["arms"][k]["worst_gap_e5"] for k in out["arms"] if k.startswith("dart"))
    gbdt_worst = out["arms"]["gbdt_xt"]["worst_gap_e5"]
    print("=" * 100)
    print(f"SUMMARY   DART worst truncation error = {dart_worst:.2f}e-5   "
          f"GBDT control = {gbdt_worst:.2f}e-5   gate = {GATE*1e5:.1f}e-5")
    print("If the GBDT control is large the measurement is broken, because GBDT boosting is strictly")
    print("additive and truncation must be exact there.")
    out["summary"] = {"dart_worst_gap_e5": dart_worst, "gbdt_worst_gap_e5": gbdt_worst,
                      "snapshot_mode_safe_for_dart": bool(dart_worst < GATE / 3)}
    save_json(out, REPORTS / "dart_truncation.json")
    print("wrote", REPORTS / "dart_truncation.json")


if __name__ == "__main__":
    main()
