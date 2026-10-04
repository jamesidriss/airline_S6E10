"""Learning-curve diagnostic: are we bias-limited, variance-limited, or representation-limited?

Why this is the highest-value cheap experiment left
---------------------------------------------------
Two orthogonal mechanisms have now been tried and honestly failed to add ensemble value (TabR, and
the k-NN target-encoding block), and the existing GBDT/MLP families are saturated. Before choosing
the next direction it is worth spending ten minutes to learn WHICH KIND of limit we are under,
because the three answers imply completely different next moves:

  * still climbing steeply at 100% data  -> variance-limited. More averaging, more seeds, bagging,
    and stronger regularisation should all pay. Cheap, incremental, low ceiling.
  * flat well before 100% data           -> bias/representation-limited. No amount of averaging or
    extra trees helps; only genuinely new signal or a different hypothesis class can move it.
  * flat at 100% but noisy                -> variance-limited in a way more data would fix.

This is NOT a Bayes ceiling and is not claimed to be one. It is a directional diagnostic whose only
purpose is to prioritise what to try next.

Design
------
Champion configuration (LightGBM + extra_trees on the `full` view, the configuration AGENTS.md records
as our largest single-model lever) is retrained on a stratified SUBSAMPLE of the outer-FIT rows at
25 / 50 / 75 / 100%, scored on the SAME untouched outer-validation rows, and repeated over several
seeds so model variance can be separated from sampling noise.

Fold discipline: the subsample is drawn only from the outer-FIT rows; the early-stopping holdout is
carved from the subsample itself; the outer-validation rows are never sampled, scored once, and never
influence anything.

Usage:
  python scripts/learning_curve.py --fold 0 --fractions 0.25,0.5,0.75,1.0 --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

PARAMS = {"objective": "binary", "n_estimators": 6000, "learning_rate": 0.02, "num_leaves": 127,
          "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
          "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True, "verbose": -1, "n_jobs": 8}


def subsample_fit(fit_idx, y, frac, seed):
    """Stratified subsample of the FIT rows. Returns None for frac == 1.0."""
    if frac >= 1.0:
        return fit_idx
    rng = np.random.default_rng(seed)
    pos, neg = fit_idx[y[fit_idx] == 1], fit_idx[y[fit_idx] == 0]
    n = max(200, int(len(fit_idx) * frac))
    npos = int(round(n * len(pos) / len(fit_idx)))
    take = np.concatenate([rng.choice(pos, npos, replace=False),
                           rng.choice(neg, n - npos, replace=False)])
    return np.sort(take)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fractions", default="0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="lcurve")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    folds = get_scheme(args.folds, y_int, tr[ID_COL]).folds
    fit = np.where(folds != args.fold)[0]
    val = np.where(folds == args.fold)[0]

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    Xf, Xout, names = vb.assemble(fit, y_int, val, None, inner_seed=args.fold)
    Xv = Xout["val"]
    print(f"[cache] view={args.view} n_features={len(names)}  fit={len(fit)}  val={len(val)}",
          flush=True)

    import lightgbm as lgb

    fracs = [float(x) for x in args.fractions.split(",")]
    rows = []
    for frac in fracs:
        aucs, iters, secs = [], [], []
        for s in range(args.seeds):
            seed = 100 * (s + 1) + args.fold
            set_seed(seed)
            sub = subsample_fit(fit, y_int, frac, seed)
            inner_tr, inner_es = _inner_es_split(sub, y_int, seed)
            # Xf is assembled over ALL outer-FIT rows and is indexed by position WITHIN `fit`, not
            # within the subsample. Mapping positions through `sub` silently reads the wrong rows --
            # and it only looked correct at frac=1.0, where sub == fit and the two coincide. That
            # produced AUCs of 0.47-0.54, i.e. anti-correlated, which is what exposed it.
            pos = {int(v): i for i, v in enumerate(fit)}
            tr_local = np.array([pos[int(v)] for v in inner_tr])
            es_local = np.array([pos[int(v)] for v in inner_es])
            p = dict(PARAMS)
            p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
            t0 = time.time()
            ds = lgb.Dataset(Xf[tr_local], label=y[inner_tr])
            dv = lgb.Dataset(Xf[es_local], label=y[inner_es], reference=ds)
            m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            pred = m.predict(Xv, num_iteration=m.best_iteration)
            a = float(roc_auc_score(y[val], pred))
            aucs.append(a)
            iters.append(int(m.best_iteration))
            secs.append(time.time() - t0)
        rows.append({"frac": frac, "n_train_rows": int(len(sub)) if frac < 1 else len(fit),
                     "n_seeds": args.seeds, "auc_mean": float(np.mean(aucs)),
                     "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                     "aucs": [round(a, 6) for a in aucs],
                     "best_iter_mean": float(np.mean(iters)),
                     "seconds_mean": round(float(np.mean(secs)), 1)})
        print(f"  frac={frac:<6} n={rows[-1]['n_train_rows']:<7} "
              f"AUC={rows[-1]['auc_mean']:.6f} +/- {rows[-1]['auc_std']:.6f}  "
              f"iter={rows[-1]['best_iter_mean']:.0f}  {rows[-1]['seconds_mean']}s", flush=True)

    # ---- shape read-out ----
    full = [r for r in rows if r["frac"] >= 1.0][0]
    half = [r for r in rows if abs(r["frac"] - 0.5) < 1e-9]
    gain_50_100 = (full["auc_mean"] - half[0]["auc_mean"]) if half else None
    span = max(r["auc_mean"] for r in rows) - min(r["auc_mean"] for r in rows)
    print("\n=== interpretation ===")
    print(f"  AUC range across the whole curve : {span:.6f}")
    if gain_50_100 is not None:
        print(f"  gain from 50% -> 100% of the data: {gain_50_100:+.6f} "
              f"({gain_50_100*1e5:+.1f}e-5)")
    # A doubling of data typically buys a few e-5 if variance-limited; if the last doubling buys
    # almost nothing while the curve was still moving earlier, we are approaching the asymptote.
    verdict = _classify(rows, gain_50_100, span)
    for line in verdict:
        print(f"  {line}")

    out = {"fold": int(args.fold), "scheme": args.folds, "view": args.view,
           "model": "lgbm extra_trees (champion single-model configuration)",
           "rows": rows, "gain_50_to_100": gain_50_100, "auc_span": span,
           "interpretation": verdict,
           "protocol": ("subsample drawn only from outer-FIT rows, stratified; early-stopping "
                        "holdout carved from the subsample; outer-validation rows fixed, never "
                        "sampled, scored once per seed; seeds differ only in model/bagging/feature "
                        "randomness")}
    save_json(out, REPORTS / f"{args.tag}_fold{args.fold}.json")
    print("\nwrote", REPORTS / f"{args.tag}_fold{args.fold}.json")


def _classify(rows, gain_50_100, span):
    lines = []
    by = {round(r["frac"], 4): r for r in rows}
    if gain_50_100 is None:
        return ["insufficient fractions to classify"]
    lines.append(f"last data doubling (50%->100%) buys {gain_50_100*1e5:+.1f}e-5")
    # per-doubling gains
    fr = sorted(by)
    for a, b in zip(fr[:-1], fr[1:]):
        if b / a < 1.5:
            continue
        lines.append(f"  {a:.3g} -> {b:.3g}: {(by[b]['auc_mean']-by[a]['auc_mean'])*1e5:+.1f}e-5")
    if gain_50_100 > 6e-5:
        lines.append("VERDICT: still clearly climbing -> VARIANCE-limited. More averaging, more "
                     "seeds and bagging should pay, and are cheap.")
    elif gain_50_100 > 1.5e-5:
        lines.append("VERDICT: mildly still climbing -> partly variance-limited. Modest headroom "
                     "from averaging/regularisation; the bulk of the gap needs new signal.")
    else:
        lines.append("VERDICT: essentially flat before 100% -> REPRESENTATION-limited. Neither "
                     "more data nor more averaging will move the ranking; only genuinely new signal "
                     "or a different hypothesis class can.")
    return lines


if __name__ == "__main__":
    main()
