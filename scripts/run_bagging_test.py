"""Section 11 -- is tree-level row bagging discarding signal we already paid for?

This is NOT generic hyperparameter tuning. It is a direct mechanistic test with a stated hypothesis,
run because our own measurements say the scarce resource is unique training signal per leaf.

Hypothesis
----------
The champion LightGBM runs with `subsample=0.8, subsample_freq=1`, so every tree is grown on a
random 80% of the rows. The campaign already found that `extra_trees=True` -- random thresholds on
randomly chosen features -- is worth +2.3e-4, but only on a rich view, precisely because a split has
to survive both random feature selection AND random threshold selection to be used. If that
double-randomisation is already supplying enough regularisation, then row bagging is a second,
redundant source of randomness that throws away 20% of the real labels on every single tree.

Predictions if the hypothesis is right
--------------------------------------
  subsample 0.8 -> 0.9 : every tree sees 11% more rows, capacity unchanged. Small positive.
  subsample 0.8 -> 1.0 : every tree sees 25% more rows, no bagging regularisation at all.
                         `colsample_bytree=0.8` and `extra_trees` remain, so the model is not
                         undefended; but if row bagging was doing real work this should turn flat
                         or negative.

If 1.0 hurts, the hypothesis is refuted and row bagging is earning its keep. Stop there; do not go
sweeping colsample or num_leaves, which would just be the generic zoo this campaign closed.

Protocol
--------
Identical to the standard protocol in every respect except the one parameter under test: same view,
same folds, same inner-ES split, same seed, same early stopping. So the comparison is paired and the
control is the `subsample=0.8` arm, computed in the same process rather than copied from another
run.

Usage: python scripts/run_bagging_test.py --folds 0,1 --values 0.8,0.9,1.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

BASE = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 127,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample_freq": 1,
        "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True, "verbose": -1, "n_jobs": 8}
ROUNDS = 6000


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--values", default="0.8,0.9,1.0")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    import lightgbm as lgb

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds
    vals = [float(v) for v in args.values.split(",")]
    klist = [int(x) for x in args.folds.split(",")]

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    fin = store.load_oof("blend_v3_final").astype("float64")

    results = []
    for k in klist:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit, y_int, val, None, inner_seed=k)
        Xv = Xa["val"]
        # _inner_es_split returns (train_rows, es_rows) as GLOBAL row indices, but Xf is indexed by
        # position WITHIN `fit`. There are therefore TWO index spaces in play and both must be
        # carried: the LOCAL one to slice the assembled feature matrix, and the GLOBAL one to slice
        # the label vector. Conflating them trains on correctly-shaped but mismatched
        # feature/label pairs, which shows up only as a nonsense AUC -- an earlier version of this
        # file did exactly that and reported -586e-5 as if it were a bagging result.
        # The (train, es) return order also matters: the FIRST value is the training rows.
        tr_rows_global, es_rows_global = _inner_es_split(fit, y_int, args.seed + k)
        pos = {int(v): i for i, v in enumerate(fit)}
        tr_local = np.array([pos[int(v)] for v in tr_rows_global])
        es_local = np.array([pos[int(v)] for v in es_rows_global])
        assert len(tr_local) + len(es_local) == len(fit) and not (set(tr_local) & set(es_local))
        assert 0.05 < len(es_local) / len(fit) < 0.15, (
            f"ES holdout is {len(es_local)}/{len(fit)} = {len(es_local)/len(fit):.1%}, expected "
            f"~10%; the (train, es) return order or the index mapping is wrong")
        assert set(tr_rows_global.tolist()).isdisjoint(val.tolist()), "fit/eval overlap"
        print(f"\n{'='*92}\nfold {k}   rows={len(fit):,}   features={len(names)}"
              f"   fit={len(tr_local):,} ({len(tr_local)/len(y):.1%} of labels)\n{'='*92}", flush=True)

        ctrl_auc, preds = None, {}
        for v in vals:
            p = dict(BASE)
            p.update(subsample=v, n_estimators=ROUNDS, random_state=args.seed + k,
                     bagging_seed=args.seed + k + 1, feature_fraction_seed=args.seed + k + 2)
            set_seed(args.seed + k)
            t0 = time.time()
            ds = lgb.Dataset(Xf[tr_local], label=y[tr_rows_global])
            dv = lgb.Dataset(Xf[es_local], label=y[es_rows_global], reference=ds)
            m = lgb.train(p, ds, num_boost_round=ROUNDS, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            it = int(m.best_iteration or ROUNDS)
            pr = m.predict(Xv, num_iteration=it)
            auc = float(roc_auc_score(y[val], pr))
            preds[v] = pr
            if v == vals[0]:
                ctrl_auc = auc
            d = auc - ctrl_auc
            bg = {}
            for w in (0.05, 0.10, 0.20, 0.30, 0.50):
                mix = w * logit(pr) + (1 - w) * logit(fin[val])
                bg[str(w)] = float(roc_auc_score(y[val], mix)) - float(
                    roc_auc_score(y[val], fin[val]))
            print(f"  subsample={v:<5} iter={it:>5}  AUC={auc:.6f}  delta={d*1e5:+6.1f}e-5  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            print(f"      vs subsample={vals[0]}: spearman={spearman(pr, preds[vals[0]]):.5f} | "
                  f"vs finalist: logit={corr(logit(pr), logit(fin[val])):.5f} "
                  f"spearman={spearman(pr, fin[val]):.5f}")
            print(f"      finalist blend gains: "
                  f"{', '.join(f'w={a}:{b*1e5:+.1f}' for a, b in bg.items())}")
            results.append({"fold": k, "subsample": v, "auc": auc, "iter": it,
                            "delta_vs_control": d, "delta_vs_control_e5": d * 1e5,
                            "seconds": round(time.time() - t0, 1),
                            "finalist_blend_gains": bg,
                            "spearman_vs_control": spearman(pr, preds[vals[0]])})

    print("\n" + "=" * 92)
    print(f"{'subsample':>10}" + "".join(f"{'fold '+str(k):>12}" for k in klist)
          + f"{'mean':>11}{'pos':>7}")
    summary = {}
    for v in vals:
        ds = [r["delta_vs_control"] for r in results if r["subsample"] == v]
        m = float(np.mean(ds))
        pos = sum(1 for d in ds if d > 0)
        summary[str(v)] = {"mean_delta": m, "mean_delta_e5": m * 1e5, "folds_positive": pos,
                           "per_fold_e5": [d * 1e5 for d in ds]}
        print(f"{v:>10}" + "".join(f"{d*1e5:>+11.1f}e" for d in ds)
              + f"{m*1e5:>+10.1f}e{pos:>5}/{len(ds)}")

    top = max((v for v in vals if v != vals[0]), key=lambda v: summary[str(v)]["mean_delta"])
    print(f"\n  best bagging fraction = {top}  ({summary[str(top)]['mean_delta_e5']:+.1f}e-5 mean, "
          f"positive {summary[str(top)]['folds_positive']}/{len(klist)})")
    st = summary[str(top)]
    if st["folds_positive"] == len(klist) and st["mean_delta"] >= 5e-5:
        print("  VERDICT: PROMOTE -- both folds positive and mean >= +5e-5")
    elif st["folds_positive"] == len(klist):
        print("  VERDICT: positive in every fold but below +5e-5. The mechanism has a real but "
              "small effect;\n           hold rather than escalate, and do NOT sweep colsample or "
              "num_leaves on the strength of it.")
    else:
        print("  VERDICT: REJECT -- row bagging is earning its keep; more rows per tree does not "
              "help\n           monotonically, so extra_trees and subsample are not redundant.")

    save_json({"results": results, "summary": summary, "control_value": vals[0],
               "values_tested": vals, "view": args.view, "scheme": args.scheme,
               "folds": klist, "seed": args.seed,
               "hypothesis": ("extra_trees already supplies enough split randomisation that "
                              "subsample=0.8 discards 20% of real labels per tree redundantly"),
               "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip()[:12]},
              REPORTS / "bagging_test.json")
    print("\nwrote", REPORTS / "bagging_test.json")


if __name__ == "__main__":
    main()
