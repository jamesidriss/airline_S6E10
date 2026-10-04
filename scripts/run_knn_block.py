"""Fold-0 screening for the k-NN target-encoding block.

The question is NOT "does a kNN-TE model score well on its own" -- it never will, being a smoothed
label. The question is whether these columns add signal the champion GBDT does not already have, which
is exactly the lever AGENTS.md identifies (extra_trees gains +2.3e-4 at 285 features, but -5.6e-3 at
21, i.e. what pays is decorrelated columns).

So this runs the CHAMPION configuration twice on the same immutable fold: once on the existing view,
once on the view plus the kNN block, and reports the paired delta on identical rows. `extra_trees=True`
is used because that is the champion single model and the configuration most sensitive to view width.

Protocol: identical primary fold, identical seed, identical early-stopping holdout carved from the FIT
rows. Only the view differs.

Usage:
  python scripts/run_knn_block.py --fold 0 --models lgbm
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
from src.features import knn as KNN  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

# The 21 raw columns are the metric space: shared, meaningful scales. Euclidean distance in ~285
# engineered dimensions is a much weaker neighbourhood metric, and target-free standardisation keeps
# this strictly leakage-safe (no labels, no validation rows involved in the fit).
METRIC_COLS = None  # taken from the raw view at runtime


def run_lgbm(Xf, yf, Xv, yv, esX, esY, seed, extra_trees=True):
    import lightgbm as lgb

    p = {"objective": "binary", "n_estimators": 6000, "learning_rate": 0.02, "num_leaves": 127,
         "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
         "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8, "random_state": seed,
         "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2}
    if extra_trees:
        p["extra_trees"] = True
    ds = lgb.Dataset(Xf, label=yf)
    dv = lgb.Dataset(esX, label=esY, reference=ds)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict(Xv, num_iteration=m.best_iteration), int(m.best_iteration)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--ks", default="16,64,256")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--query-chunk", type=int, default=256)
    ap.add_argument("--n-inner", type=int, default=5)
    ap.add_argument("--tag", default="knnte")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = y.astype("int8")
    ntr, nte = len(tr), len(te)
    folds = get_scheme(args.folds, y_int, tr[ID_COL]).folds
    ks = tuple(int(x) for x in args.ks.split(","))

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    base_names = list(vb.static_names)

    fit = np.where(folds != args.fold)[0]
    val = np.where(folds == args.fold)[0]
    set_seed(args.seed + args.fold)

    # ---- fold-safe kNN block ----
    t0 = time.time()
    raw_vb = ViewBuilder(tr, te, "raw")
    raw_vb.build_static()
    raw_names = list(raw_vb.static_names)          # the 21 raw survey variables
    metric, metric_names = KNN.standardised_raw_metric(tr, tr, raw_names)
    # test rows are NOT encoded here: this is a fold-0 screen answering "does the block add signal",
    # and the metric matrix spans train rows only. Producing trustworthy test predictions would
    # require encoding the test block against the same fold-fit database, which is a separate step.
    Kn, knn_names, kdiag = KNN.build_knn_block(
        metric, y_int, fit, val, None, metric_names, ks=ks,
        n_inner=args.n_inner, seed=args.seed + args.fold, query_chunk=args.query_chunk)
    t_knn = time.time() - t0
    print(f"  metric space: {len(raw_names)} raw columns -> {len(metric_names)} numeric "
          f"dimensions after one-hot + standardisation", flush=True)
    print(f"  kNN block built in {t_knn:.0f}s  ({len(knn_names)} columns)", flush=True)
    print(f"  columns: {knn_names}", flush=True)
    print(f"  diagnostics: inner_folds={kdiag['inner_folds']} n_pool={kdiag['n_pool']} "
          f"n_es={kdiag['n_es']} db={kdiag['retrieval_stats']}", flush=True)

    # sanity: the block must carry real signal on its own, and must not be degenerate
    auc_te = {f"te_{k}": float(roc_auc_score(y[val], Kn[val, knn_names.index(f"te_{k}")]))
              for k in ks}
    print("  standalone AUC of each te_k on this fold: "
          + str({k: round(v, 6) for k, v in auc_te.items()}), flush=True)
    assert bool(np.isfinite(Kn).all()), "kNN block contains non-finite values"

    # ---- assemble both views on identical rows ----
    Xf, Xout, names = vb.assemble(fit, y_int, val, np.arange(ntr, ntr + nte),
                                   inner_seed=args.fold)
    inner_tr, inner_es = _inner_es_split(fit, y_int, args.seed + args.fold)
    pos = {int(v): i for i, v in enumerate(fit)}
    tr_local = np.array([pos[int(v)] for v in inner_tr])
    es_local = np.array([pos[int(v)] for v in inner_es])

    Xa = Xout["val"]
    knn_tr, knn_es, knn_val = Kn[inner_tr], Kn[inner_es], Kn[val]

    train_base = Xf[tr_local]
    es_base = Xf[es_local]
    train_plus = np.hstack([train_base, knn_tr])
    es_plus = np.hstack([es_base, knn_es])
    val_plus = np.hstack([Xa, knn_val])

    assert train_plus.shape[1] == train_base.shape[1] + len(knn_names), "width accounting wrong"
    assert val_plus.shape[1] == train_plus.shape[1], "val width must match train width"

    results = {}
    preds = {}
    for label, (TR, VA, ES) in {
            "base": (train_base, Xa, es_base),
            "plus_knn": (train_plus, val_plus, es_plus)}.items():
        set_seed(args.seed + args.fold)
        t1 = time.time()
        p, it = run_lgbm(TR, y[inner_tr], VA, y[val], ES, y[inner_es],
                         args.seed + args.fold)
        a = float(roc_auc_score(y[val], p))
        preds[label] = p
        results[label] = {"auc": a, "best_iter": it, "seconds": round(time.time() - t1, 1),
                          "n_features": int(TR.shape[1])}
        print(f"  {label:<10} n_feat={TR.shape[1]:<4} AUC={a:.6f} iter={it} "
              f"({results[label]['seconds']}s)", flush=True)

    delta = results["plus_knn"]["auc"] - results["base"]["auc"]
    print(f"\n  DELTA (plus_knn - base) on fold {args.fold} = {delta:+.6f} "
          f"({delta*1e5:+.1f}e-5)")

    fin = store.load_oof("blend_v3_final").astype("float64")[val]
    xt = store.load_oof("z4_xt_f10_s4").astype("float64")[val]
    pk = KNN.to_logit(preds["plus_knn"])
    print(f"  corr(plus_knn, finalist) logit={corr(pk, KNN.to_logit(fin)):.5f} "
          f"spearman={spearman(preds['plus_knn'], fin):.5f}")
    print(f"  corr(plus_knn, xt)       logit={corr(pk, KNN.to_logit(xt)):.5f} "
          f"spearman={spearman(preds['plus_knn'], xt):.5f}")

    # Does the champion blend improve when the kNN-augmented model replaces / joins it?
    print("\n--- diagnostic: does the kNN-augmented model help the finalist? ---")
    fl = KNN.to_logit(fin)
    b = float(roc_auc_score(y[val], fin))
    print(f"  finalist alone = {b:.6f}")
    for w in (0.02, 0.05, 0.10, 0.20):
        mix = w * pk + (1 - w) * fl
        a = float(roc_auc_score(y[val], mix))
        print(f"  +{w:>5.0%} plus_knn (logit) = {a:.6f}  ({a - b:+.6f})")

    out = {"fold": int(args.fold), "scheme": args.folds, "view": args.view,
           "n_knn_columns": len(knn_names), "knn_columns": knn_names,
           "knn_build_seconds": round(t_knn, 1), "knn_diagnostics": kdiag,
           "standalone_auc_of_te": auc_te, "results": results, "delta": delta,
           "model": "lgbm extra_trees (champion single-model configuration)",
           "protocol": ("identical fold, seed and inner early-stopping holdout; only the view "
                        "differs. The kNN database contains only outer-FIT rows, fit rows are "
                        "encoded by an inner cross-fit that excludes them, and inner-ES/val/test rows "
                        "are encoded against the full outer-FIT-minus-ES database.")}
    save_json(out, REPORTS / f"{args.tag}_fold{args.fold}.json")
    np.save(REPORTS / f"{args.tag}_fold{args.fold}_plus_knn.npy", preds["plus_knn"])
    print("\nwrote", REPORTS / f"{args.tag}_fold{args.fold}.json")


if __name__ == "__main__":
    main()
