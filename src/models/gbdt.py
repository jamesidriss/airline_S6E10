"""Model wrappers. Each returns (oof, test_pred, fold_aucs, fit_seconds).

Every wrapper trains K models (one per fold) so the test prediction is always a
reproducible full-fold ensemble.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

N_JOBS = 8


def _inner_fit_es(fit, labels, seed):
    train, es = train_test_split(fit, test_size=.1, random_state=seed,
                                 stratify=np.asarray(labels)[fit])
    assert not np.intersect1d(train, es).size
    return train, es


def _fold_auc(y, oof, folds):
    return [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in sorted(set(folds.tolist()))]


def lgbm(Xtr, ytr, Xte, folds, params=None, seed=1, feat_names=None, callbacks=None):
    import lightgbm as lgb

    p = dict(
        objective="binary", n_estimators=4000, learning_rate=0.03, num_leaves=63,
        min_child_samples=40, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        reg_lambda=1.0, max_bin=255, n_jobs=N_JOBS, verbose=-1, random_state=seed,
    )
    if params:
        p.update(params)
    oof = np.zeros(len(ytr), dtype="float64")
    test = np.zeros(len(Xte), dtype="float64")
    t0 = time.time()
    for k in sorted(set(folds.tolist())):
        a = np.where(folds != k)[0]
        b = np.where(folds == k)[0]
        train, es = _inner_fit_es(a, ytr, seed)
        ds = lgb.Dataset(Xtr[train], label=ytr[train], feature_name=feat_names, free_raw_data=False)
        dv = lgb.Dataset(Xtr[es], label=ytr[es], reference=ds, feature_name=feat_names, free_raw_data=False)
        m = lgb.train(
            {**p, "metric": "auc", "seed": seed, "bagging_seed": seed + 1, "feature_fraction_seed": seed + 2},
            ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
            callbacks=[lgb.early_stopping(200, verbose=False)] + (callbacks or []),
        )
        oof[b] = m.predict(Xtr[b], num_iteration=m.best_iteration)
        test += m.predict(Xte, num_iteration=m.best_iteration) / len(set(folds.tolist()))
        print(f"    lgbm fold{k} best_iter={m.best_iteration} auc={roc_auc_score(ytr[b], oof[b]):.6f}", flush=True)
    return oof, test, _fold_auc(ytr, oof, folds), time.time() - t0


def xgboost(Xtr, ytr, Xte, folds, params=None, seed=1, device="cuda"):
    import xgboost as xgb

    p = dict(
        objective="binary:logistic", eval_metric="auc", n_estimators=4000, learning_rate=0.03,
        max_depth=8, min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.0, max_bin=256, tree_method="hist",
        device=device, n_jobs=N_JOBS, random_state=seed,
    )
    if params:
        p.update(params)
    oof = np.zeros(len(ytr), dtype="float64")
    test = np.zeros(len(Xte), dtype="float64")
    t0 = time.time()
    for k in sorted(set(folds.tolist())):
        a = np.where(folds != k)[0]
        b = np.where(folds == k)[0]
        train, es = _inner_fit_es(a, ytr, seed)
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr[train], ytr[train], eval_set=[(Xtr[es], ytr[es])], verbose=False)
        best = getattr(m, "best_iteration", None)
        oof[b] = m.predict_proba(Xtr[b])[:, 1]
        test += m.predict_proba(Xte)[:, 1] / len(set(folds.tolist()))
        print(f"    xgb fold{k} best_iter={best} auc={roc_auc_score(ytr[b], oof[b]):.6f}", flush=True)
    return oof, test, _fold_auc(ytr, oof, folds), time.time() - t0


def catboost(Xtr, ytr, Xte, folds, params=None, seed=1, cat_idx=None):
    from catboost import CatBoostClassifier, Pool

    p = dict(
        iterations=4000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, loss_function="Logloss",
        eval_metric="AUC", random_seed=seed, thread_count=N_JOBS, verbose=0,
        allow_writing_files=False,
    )
    if params:
        p.update(params)
    oof = np.zeros(len(ytr), dtype="float64")
    test = np.zeros(len(Xte), dtype="float64")
    t0 = time.time()
    for k in sorted(set(folds.tolist())):
        a = np.where(folds != k)[0]
        b = np.where(folds == k)[0]
        train, es = _inner_fit_es(a, ytr, seed)
        ptr = Pool(Xtr[train], ytr[train], cat_features=cat_idx)
        pes = Pool(Xtr[es], ytr[es], cat_features=cat_idx)
        pva = Pool(Xtr[b], ytr[b], cat_features=cat_idx)
        m = CatBoostClassifier(**p)
        m.fit(ptr, eval_set=pes, early_stopping_rounds=200, verbose=0)
        oof[b] = m.predict_proba(pva)[:, 1]
        test += m.predict_proba(Pool(Xte, cat_features=cat_idx))[:, 1] / len(set(folds.tolist()))
        print(f"    cat fold{k} best_iter={m.get_best_iteration()} auc={roc_auc_score(ytr[b], oof[b]):.6f}", flush=True)
    return oof, test, _fold_auc(ytr, oof, folds), time.time() - t0


def etr(Xtr, ytr, Xte, folds, params=None, seed=1):
    from sklearn.ensemble import ExtraTreesClassifier

    p = dict(n_estimators=600, max_features=0.5, min_samples_leaf=4, n_jobs=N_JOBS,
             random_state=seed, criterion="entropy")
    if params:
        p.update(params)
    oof = np.zeros(len(ytr), dtype="float64")
    test = np.zeros(len(Xte), dtype="float64")
    t0 = time.time()
    for k in sorted(set(folds.tolist())):
        a = np.where(folds != k)[0]
        b = np.where(folds == k)[0]
        m = ExtraTreesClassifier(**p)
        m.fit(Xtr[a], ytr[a])
        oof[b] = m.predict_proba(Xtr[b])[:, 1]
        test += m.predict_proba(Xte)[:, 1] / len(set(folds.tolist()))
    return oof, test, _fold_auc(ytr, oof, folds), time.time() - t0


def histgb(Xtr, ytr, Xte, folds, params=None, seed=1):
    from sklearn.ensemble import HistGradientBoostingClassifier

    p = dict(max_iter=800, learning_rate=0.06, max_leaf_nodes=63, l2_regularization=1.0,
              early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=seed)
    if params:
        p.update(params)
    oof = np.zeros(len(ytr), dtype="float64")
    test = np.zeros(len(Xte), dtype="float64")
    t0 = time.time()
    for k in sorted(set(folds.tolist())):
        a = np.where(folds != k)[0]
        b = np.where(folds == k)[0]
        m = HistGradientBoostingClassifier(**p)
        m.fit(Xtr[a], ytr[a])
        oof[b] = m.predict_proba(Xtr[b])[:, 1]
        test += m.predict_proba(Xte)[:, 1] / len(set(folds.tolist()))
    return oof, test, _fold_auc(ytr, oof, folds), time.time() - t0
