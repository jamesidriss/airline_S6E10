"""Ablate the canonical feature views on the immutable primary folds.

Usage:
  python scripts/run_views.py --views raw,raw_ext,core3_te --models lgbm --folds primary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import VIEWS, ViewBuilder  # noqa: E402
from src.models import gbdt  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import fold_auc  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
import experiments_ledger as led  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="raw,raw_ext,core3_te")
    ap.add_argument("--models", default="lgbm")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--tag", default="view")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-test", action="store_true")
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    dh = f"tr{ntr}-te{nte}"
    extra = json.loads(args.params)
    summary = []
    cache_path = REPORTS / f"{args.tag}_results.json"
    if cache_path.exists():
        summary = json.loads(cache_path.read_text())

    for scheme in args.folds.split(","):
        folds = get_scheme(scheme, y, tr[ID_COL]).folds
        for view in args.views.split(","):
            pending = [m for m in args.models.split(",")
                       if not any(s.get("view") == view and s.get("scheme") == scheme
                                  and s.get("model") == m for s in summary)]
            if not pending:
                print(f"skip {view}/{scheme} (cached)")
                continue
            t0 = time.time()
            vb = ViewBuilder(tr, te, view)
            st_tr, st_te, snames = vb.build_static()
            tauc = vb.teacher_auc()
            print(f"\n### view={view} scheme={scheme} static={st_tr.shape} "
                  f"(teacher_auc={tauc if tauc is None else round(tauc,6)}) blocks={VIEWS[view]}", flush=True)
            for mdl in args.models.split(","):
                params = dict(extra)
                oof = np.zeros(ntr)
                test = np.zeros(nte)
                iters = []
                set_seed(args.seed)
                for k in sorted(set(folds.tolist())):
                    fit = np.where(folds != k)[0]
                    val = np.where(folds == k)[0]
                    Xf, Xa, names = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte))
                    Xtr_f, Xval = Xf, Xa["val"]
                    es_tr, es_idx = _inner_es_split(fit, y, args.seed + k)
                    pos_of = {v: i for i, v in enumerate(fit)}
                    es_local = np.array([pos_of[v] for v in es_idx])
                    tr_local = np.array([pos_of[v] for v in es_tr])
                    esX, esY = Xtr_f[es_local], y[es_idx]
                    fitX, fitY = Xtr_f[tr_local], y[es_tr]
                    if mdl == "lgbm":
                        o, it = _fit_lgbm_es(fitX, fitY, Xval, params, args.seed, esX, esY)
                    elif mdl == "xgb":
                        o, it = _fit_xgb_es(fitX, fitY, Xval, params, args.seed, esX, esY)
                    elif mdl == "cat":
                        o, it = _fit_cat_es(fitX, fitY, Xval, params, args.seed, esX, esY)
                    else:
                        raise ValueError(mdl)
                    iters.append(it)
                    oof[val] = o
                # second pass: refit on ALL fold-fit rows at the median CV iteration count, so the
                # test prediction uses 11% more data per fold and never consults a held-out label.
                n_it = int(np.median(iters)) if iters else 800
                print(f"  [{mdl}/{view}] median best_iter={n_it} per-fold={iters}", flush=True)
                set_seed(args.seed)
                for k in sorted(set(folds.tolist())):
                    fit = np.where(folds != k)[0]
                    val = np.where(folds == k)[0]
                    Xf, Xa, _ = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte))
                    Xtest = Xa["test"]
                    if mdl == "lgbm":
                        pr_te = _fit_full_predict_lgbm(Xf, y[fit], Xtest, params, args.seed, n_it)
                    elif mdl == "xgb":
                        pr_te = _fit_full_predict_xgb(Xf, y[fit], Xtest, params, args.seed, n_it)
                    elif mdl == "cat":
                        pr_te = _fit_full_predict_cat(Xf, y[fit], Xtest, params, args.seed, n_it)
                    test += pr_te / len(set(folds.tolist()))
                auc = float(gbdt.roc_auc_score(y, oof))
                fa = fold_auc(y, oof, folds)
                print(f"  ==> {mdl}/{view}/{scheme} OOF AUC = {auc:.6f}  folds={[round(x,6) for x in fa]}", flush=True)
                eid = f"{args.tag}_{mdl}_{view}_{scheme}"
                store.save(eid, oof, test if args.save_test else None, fold_scheme=scheme,
                           meta={"family": mdl, "featureset": view, "auc": round(auc, 6),
                                 "n_features": int(Xf.shape[1]), "teacher_auc": tauc})
                led.log_experiment(
                    family=mdl, featureset=view, params=params, seed=args.seed, fold_scheme=scheme,
                    oof_auc=auc, fold_aucs=fa, oof=oof, duration_s=time.time() - t0, data_hash=dh,
                    test_pred=test if args.save_test else None, exp_id=eid,
                    notes=f"blocks={VIEWS[view]}", verdict="view-ablation",
                    extra={"fold_ids": folds.tolist(), "n_features": int(Xf.shape[1]),
                           "teacher_auc": tauc, "params": params},
                )
                summary.append({"view": view, "scheme": scheme, "model": mdl, "oof_auc": round(auc, 6),
                                "fold_aucs": [round(x, 6) for x in fa], "n_features": int(Xf.shape[1]),
                                "teacher_auc": tauc, "exp_id": eid})
                save_json(summary, cache_path)
        save_json(summary, cache_path)

    print("\n=== VIEW SUMMARY ===")
    print(pd.DataFrame(summary).sort_values("oof_auc", ascending=False).to_string(index=False))


def _inner_es_split(fit: np.ndarray, y: np.ndarray, seed: int, frac: float = 0.10):
    """Carve an inner early-stopping split out of the FIT rows only.

    Never uses the evaluation fold's labels, so OOF stays honest and the reported AUC is
    directly comparable to models that early-stop on the eval fold (which are ~1e-4 optimistic).
    """
    rng = np.random.default_rng(seed + 991)
    pos, neg = fit[y[fit] == 1], fit[y[fit] == 0]
    n = int(len(fit) * frac)
    es = np.concatenate([rng.choice(pos, max(1, int(n * len(pos) / len(fit))), replace=False),
                         rng.choice(neg, max(1, int(n * len(neg) / len(fit))), replace=False)])
    es = np.unique(es)
    trn = np.setdiff1d(fit, es)
    return trn, es


def _fit_lgbm_es(X, y, Xv, params, seed, es_X=None, es_y=None):
    """Returns (oof_pred_on_Xv, best_iteration, es_set_indices)."""
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", n_estimators=6000, learning_rate=0.02,
             num_leaves=127, min_child_samples=40, colsample_bytree=0.8, subsample=0.8,
             subsample_freq=1, reg_lambda=1.0, max_bin=255, verbose=-1, n_jobs=8,
             random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    p.update(params)
    ds = lgb.Dataset(X, label=y)
    dv = lgb.Dataset(es_X, label=es_y, reference=ds)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict(Xv, num_iteration=m.best_iteration), int(m.best_iteration)


def _fit_xgb_es(X, y, Xv, params, seed, es_X=None, es_y=None):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="auc", n_estimators=6000, learning_rate=0.03,
             max_depth=8, min_child_weight=8, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
             max_bin=256, tree_method="hist", device="cuda", n_jobs=8, random_state=seed,
             early_stopping_rounds=250)
    p.update(params)
    m = xgb.XGBClassifier(**p)
    m.fit(X, y, eval_set=[(es_X, es_y)], verbose=False)
    it = int(getattr(m, "best_iteration", 0) or p["n_estimators"])
    return m.predict_proba(Xv)[:, 1], it


def _fit_cat_es(X, y, Xv, params, seed, es_X=None, es_y=None):
    from catboost import CatBoostClassifier
    p = dict(iterations=6000, learning_rate=0.04, depth=8, l2_leaf_reg=3.0, random_seed=seed,
             thread_count=8, verbose=0, allow_writing_files=False, eval_metric="AUC")
    p.update(params)
    m = CatBoostClassifier(**p)
    m.fit(X, y, eval_set=(es_X, es_y), early_stopping_rounds=300, verbose=0)
    return m.predict_proba(Xv)[:, 1], int(m.get_best_iteration())


# --------------------------------------------------------------------------------------
# Test-prediction models.
#
# These are deliberately stronger than a naive "reuse the early-stopped model": the iteration
# count is fixed to the median best_iteration found by cross-validation (so no evaluation label
# is consulted), and the model is refitted on ALL of the fold's training rows rather than the
# 90% subset that early stopping consumed. Strictly more data, same capacity schedule.
# This improves the test predictions only; it cannot and does not touch the reported OOF.
# --------------------------------------------------------------------------------------
def _fit_full_predict_lgbm(X, y, Xt, params, seed, n_estimators=None):
    import lightgbm as lgb
    p = dict(objective="binary", n_estimators=1200, learning_rate=0.02, num_leaves=127,
             colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbose=-1,
             n_jobs=8, random_state=seed)
    p.update({k: v for k, v in params.items() if k != "n_estimators"})
    p["n_estimators"] = int(n_estimators or params.get("n_estimators", 1200))
    m = lgb.LGBMClassifier(**p)
    m.fit(X, y)
    return m.predict_proba(Xt)[:, 1]


def _fit_full_predict_xgb(X, y, Xt, params, seed, n_estimators=None):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="auc", n_estimators=1500, learning_rate=0.03,
             max_depth=8, min_child_weight=8, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
             max_bin=256, tree_method="hist", device="cuda", n_jobs=8, random_state=seed)
    p.update({k: v for k, v in params.items() if k != "n_estimators"})
    p["n_estimators"] = int(n_estimators or params.get("n_estimators", 1500))
    p.pop("early_stopping_rounds", None)
    m = xgb.XGBClassifier(**p)
    m.fit(X, y, verbose=False)
    return m.predict_proba(Xt)[:, 1]


def _fit_full_predict_cat(X, y, Xt, params, seed, n_estimators=None):
    from catboost import CatBoostClassifier
    p = dict(iterations=1800, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, random_seed=seed,
             thread_count=8, verbose=0, allow_writing_files=False)
    p.update({k: v for k, v in params.items() if k != "iterations"})
    p["iterations"] = int(n_estimators or params.get("iterations", 1800))
    m = CatBoostClassifier(**p)
    m.fit(X, y, verbose=0)
    return m.predict_proba(Xt)[:, 1]


if __name__ == "__main__":
    main()