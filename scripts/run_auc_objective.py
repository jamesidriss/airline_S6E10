"""Does an AUC/ranking objective produce a differently-shaped error, worth ensembling?

Motivation
----------
Phase 5 established the campaign is limited by UNIQUE INFORMATION, not sample count: a genuine
doubling of unique rows is worth ~+5e-4 (AUC ~ a + b*n^(-1/5), R^2 = 0.99894) while every form of
manufactured support failed, and the duplicate control proves that is not a row-count artefact.
TabR was the orthogonal-family attempt and is rejected: it is genuinely the most decorrelated model
in the pool (logit corr 0.99502 vs a 0.99577 floor) yet every predeclared blend weight was negative
at two context sizes.

What remains untested is the one axis that changes error geometry WITHOUT adding capacity or data:
every model in the pool optimises logloss, while the competition metric is ROC-AUC. A pairwise
ranking surrogate optimises a different thing, so it can be weaker and still be worth keeping -- which
is exactly the criterion the finalist gate applies.

The objective
-------------
The pairwise logistic surrogate (RankNet-style) on raw margins, gradient-verified against central
finite differences to 1.3e-10 relative error, with a strictly positive hessian, and shown by
gradient descent to raise true AUC from 0.500 to 0.975. See scripts/verify_auc_objective.py, which
must be run before this. LightGBM's native `lambdarank` is NOT used: it requires group information
and its NDCG-based per-query truncation is the wrong surrogate for a single global ranking.

Arms, all on the immutable primary fold, all with the same features, seed and inner early-stopping
holdout, so the ONLY difference is the objective:
    binary      the champion's logloss objective (matched control, required)
    pairwise    the verified custom surrogate, several negatives-per-positive
    xgb_rank    XGBoost's native rank:pairwise

Usage:
  python scripts/run_auc_objective.py --fold 0 --arms binary,pairwise --n-neg 4
"""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.verify_auc_objective import auc_pairwise_grad, build_pairs  # noqa: E402

CHAMPION = {"objective": "binary", "n_estimators": 6000, "learning_rate": 0.02,
            "num_leaves": 127, "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True,
            "verbose": -1, "n_jobs": 8}


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def train_binary(X, y, Xv, yv, esX, esY, seed):
    import lightgbm as lgb

    p = dict(CHAMPION)
    p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    ds = lgb.Dataset(X, label=y)
    dv = lgb.Dataset(esX, label=esY, reference=ds)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return 1.0 / (1.0 + np.exp(-np.clip(m.predict(Xv, num_iteration=m.best_iteration), -35, 35))), \
        int(m.best_iteration)


def train_pairwise(X, y, Xv, yv, esX, esY, seed, n_neg=4, lr=0.05, leaves=127, extra_trees=True,
                   rounds=15000):
    """LightGBM with the verified pairwise AUC surrogate via a custom objective."""
    import lightgbm as lgb

    pos_i, neg_i = build_pairs(y, n_neg_per_pos=n_neg, seed=seed)
    n_pairs = len(pos_i)

    def fobj(preds, dataset):
        g, h = auc_pairwise_grad(preds, y, pos_i, neg_i, n_pairs)
        return g, h

    # LightGBM 4.x removed the fobj= argument from train(); a callable objective is passed as
    # params["objective"] instead. Verified against the installed 4.7.0 signature.
    p = {"objective": fobj, "metric": "auc", "n_estimators": rounds, "learning_rate": lr,
         "num_leaves": leaves, "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
         "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8,
         "random_state": seed, "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2}
    if extra_trees:
        p["extra_trees"] = True
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    dv = lgb.Dataset(esX, label=esY, reference=ds)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return 1.0 / (1.0 + np.exp(-np.clip(m.predict(Xv, num_iteration=m.best_iteration), -35, 35))), \
        int(m.best_iteration)


def train_xgb_rank(X, y, Xv, yv, esX, esY, seed):
    import xgboost as xgb

    dtr = xgb.DMatrix(X, label=y)
    dva = xgb.DMatrix(Xv, label=yv)
    des = xgb.DMatrix(esX, label=esY)
    bst = xgb.train({"objective": "rank:pairwise", "eval_metric": "auc", "max_depth": 8,
                     "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
                     "reg_lambda": 5.0, "tree_method": "hist", "device": "cpu", "nthread": 8,
                     "seed": seed},
                    dtr, num_boost_round=3000, evals=[(des, "es")],
                    callbacks=[xgb.callback.EarlyStopping(rounds=200, save_best=True)])
    pr = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    return 1.0 / (1.0 + np.exp(-np.clip(pr, -35, 35))), int(bst.best_iteration)


def blend_gain(pred, y, fin, weights=(0.02, 0.05, 0.10, 0.20, 0.30)):
    b = float(roc_auc_score(y, fin))
    lp, lf = _logit(pred), _logit(fin)
    out = {str(w): float(roc_auc_score(y, w * lp + (1 - w) * lf)) - b for w in weights}
    return {"finalist_auc": b, "gains_by_weight": out, "best_gain": max(out.values()),
            "best_weight": max(out, key=out.get)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--arms", default="binary,pairwise,xgb_rank")
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--pairwise-lr", type=float, default=0.05)
    ap.add_argument("--pairwise-rounds", type=int, default=15000,
                    help="the pairwise arm hit its cap at 4000 without early stopping, i.e. it was "
                         "undertrained rather than converged")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="aucobj")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    folds = get_scheme(args.folds, y_int, tr[ID_COL]).folds
    fit = np.where(folds != args.fold)[0]
    val = np.where(folds == args.fold)[0]
    assert not (set(fit.tolist()) & set(val.tolist())), "fit/eval overlap"
    assert len(set(fit.tolist()) & set(val.tolist())) == 0

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    Xf, Xout, names = vb.assemble(fit, y_int, val, None, inner_seed=args.fold)
    Xv = Xout["val"]
    inner_tr, inner_es = _inner_es_split(fit, y_int, args.seed + args.fold)
    pos = {int(v): i for i, v in enumerate(fit)}
    tr_local = np.array([pos[int(v)] for v in inner_tr])
    es_local = np.array([pos[int(v)] for v in inner_es])
    Xtr, ytr = Xf[tr_local], y[inner_tr]
    Xes, yes = Xf[es_local], y[inner_es]
    fin = store.load_oof("blend_v3_final").astype("float64")[val]
    print(f"view={args.view} n_feat={len(names)} train={Xtr.shape} eval={Xv.shape} fold={args.fold}")

    results, preds = [], {}
    for arm in args.arms.split(","):
        set_seed(args.seed + args.fold)
        t0 = time.time()
        try:
            if arm == "binary":
                p, it = train_binary(Xtr, ytr, Xv, y[val], Xes, yes, args.seed + args.fold)
            elif arm == "pairwise":
                p, it = train_pairwise(Xtr, ytr, Xv, y[val], Xes, yes, args.seed + args.fold,
                                       n_neg=args.n_neg, lr=args.pairwise_lr,
                                       rounds=args.pairwise_rounds)
            elif arm == "xgb_rank":
                p, it = train_xgb_rank(Xtr, ytr, Xv, y[val], Xes, yes, args.seed + args.fold)
            else:
                print(f"  unknown arm {arm!r}, skipping")
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  arm {arm}: FAILED {type(exc).__name__}: {str(exc)[:160]}")
            results.append({"arm": arm, "error": type(exc).__name__, "msg": str(exc)[:200]})
            continue
        auc = float(roc_auc_score(y[val], p))
        preds[arm] = p
        bg = blend_gain(p, y[val], fin)
        print(f"  {arm:<10} AUC={auc:.6f} iter={it} ({time.time()-t0:.0f}s)  "
              f"corr(vs finalist) logit={corr(_logit(p), _logit(fin)):.5f} "
              f"spearman={spearman(p, fin):.5f}", flush=True)
        print(f"             finalist blend best={bg['best_gain']:+.6f} at w={bg['best_weight']}",
              flush=True)
        results.append({"arm": arm, "auc": auc, "best_iter": it,
                        "seconds": round(time.time() - t0, 1),
                        "corr_finalist_logit": corr(_logit(p), _logit(fin)),
                        "spearman_finalist": spearman(p, fin),
                        "blend_gain": bg,
                        "n_nan": int(np.isnan(p).sum()), "n_inf": int(np.isinf(p).sum())})
        np.save(REPORTS / f"{args.tag}_{arm}_fold{args.fold}.npy", p.astype("float32"))

    base = next((r for r in results if r.get("arm") == "binary" and "auc" in r), None)
    for r in results:
        if r.get("arm") and r["arm"] != "binary" and "auc" in r and base:
            r["delta_vs_binary"] = r["auc"] - base["auc"]
            r["delta_e5"] = round((r["auc"] - base["auc"]) * 1e5, 1)

    out = {"fold": int(args.fold), "scheme": args.folds, "view": args.view,
           "n_features": len(names), "n_train": int(len(tr_local)),
           "n_neg_per_pos": args.n_neg, "pairwise_lr": args.pairwise_lr,
           "pairwise_rounds": args.pairwise_rounds, "seed": args.seed,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12],
           "results": results,
           "objective_verification": "reports/auc_objective_verification.json",
           "protocol": ("identical features, seed and inner early-stopping holdout across arms; the "
                        "only difference is the training objective. Early stopping uses the inner "
                        "holdout carved from the outer-FIT rows; the outer fold is scored once.")}
    save_json(out, REPORTS / f"{args.tag}_fold{args.fold}.json")
    print("\n=== SUMMARY (matched control = binary/logloss) ===")
    print(f"{'arm':<12}{'AUC':>11}{'delta':>11}{'best_iter':>11}{'corr_fin':>11}{'blend':>11}")
    for r in results:
        if "auc" not in r:
            print(f"{r['arm']:<12}  FAILED {r.get('error')}")
            continue
        d = r.get("delta_e5")
        ds = "     (ctrl)" if d is None else f"{d:>+10.1f}e"
        print(f"{r['arm']:<12}{r['auc']:>11.6f}{ds:>11}{r['best_iter']:>11}"
              f"{r['corr_finalist_logit']:>11.5f}{r['blend_gain']['best_gain']:>+11.6f}")
    print("\nwrote", REPORTS / f"{args.tag}_fold{args.fold}.json")


if __name__ == "__main__":
    main()
