"""Phase 3: residual boosting via base_margin, with explicit raw-score semantics.

The previous attempt at this was invalid, not negative: it initialised LightGBM with a logit but
predicted without the matching initialisation, so the two score spaces disagreed and it returned
AUC 0.163. This module does it properly and, crucially, proves it with a zero-round identity test
before any result is trusted.

Semantics
---------
For ``objective='binary:logistic'`` XGBoost's ``base_margin`` is the **raw score**, i.e. the logit,
not a probability. So:
    margin = log(p / (1 - p))            (safe-clipped)
    predict = sigmoid(margin + sum_of_trees)

Identity test (run first, every time):
    train for ZERO boosting rounds on a dataset carrying ``base_margin = margin``
    -> prediction must equal sigmoid(margin) to numerical tolerance, and its ranking must equal
       the ranking of the teacher probability p.
If that fails, the whole module is invalid and we stop.

Train/serve consistency
-----------------------
A residual learner is only coherent if the teacher's quality is the same on the rows it trains on
and the rows it scores. A teacher evaluated *in-sample* on its own training rows is over-confident,
so the student would be trained against a different input distribution than it will see at serve
time. Two ways to avoid that are supported here:

  * ``cross_fitted`` -- the teacher margins for the fit rows come from an inner K-fold
    cross-fit that never trains on the row it predicts, while the scored rows get a full-fit
    teacher. This is the correct default for any teacher trained on competition labels.
  * ``label_free``  -- the teacher depends on no competition label at all (the original-survey
    teacher), so one global prediction is valid everywhere and no nesting is needed. This is the
    cleanest possible test of the mechanism.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def to_margin(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def to_prob(m: np.ndarray) -> np.ndarray:
    m = np.clip(np.asarray(m, dtype="float64"), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-m))


def xgb_residual(Xf, y_fit, Xv, teacher_fit, teacher_val, params=None, seed=1,
                 device="cuda", n_rounds=2000, es_X=None, es_y=None, y_val=None, verbose=True):
    """Fit a residual XGBoost on top of a teacher margin.

    ``teacher_fit`` supplies margins for the fit rows, ``teacher_val`` margins for the scored rows.
    Both are probabilities; they are converted to raw scores internally. ``y_val`` labels the
    scored rows (used only for the AUC metric that drives early stopping).
    """
    import xgboost as xgb

    p = dict(objective="binary:logistic", eval_metric="auc", n_estimators=n_rounds,
             learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.9,
             colsample_bytree=0.85, reg_lambda=5.0, reg_alpha=0.05, max_bin=256,
             tree_method="hist", device=device, n_jobs=8, random_state=seed,
             early_stopping_rounds=200)
    if params:
        p.update(params)

    m_fit = to_margin(teacher_fit)
    m_val = to_margin(teacher_val)
    yv = np.asarray(y_fit if y_val is None else y_val, dtype="float64")
    dtr = xgb.DMatrix(Xf, label=np.asarray(y_fit, dtype="float64"), base_margin=m_fit)
    dva = xgb.DMatrix(Xv, label=yv, base_margin=m_val)
    bst = xgb.train(p, dtr, num_boost_round=p["n_estimators"], evals=[(dva, "val")],
                    verbose_eval=False)
    pred = bst.predict(dva)
    best = int(getattr(bst, "best_iteration", 0) or p["n_estimators"])
    if verbose:
        print(f"    xgb residual: best_iter={best} auc={roc_auc_score(yv, pred):.6f}", flush=True)
    return pred, bst, best


def predict_with_booster(booster, X, teacher_prob) -> np.ndarray:
    """Apply a fitted residual booster, supplying the teacher's raw score as base_margin.

    The same booster is reused for the test block, so the test prediction cannot drift from the
    validated configuration and no refit (or test labels) are needed.
    """
    import xgboost as xgb

    d = xgb.DMatrix(X, base_margin=to_margin(teacher_prob))
    return booster.predict(d)


def identity_test(X, teacher_p, y=None, device="cuda") -> dict:
    """Zero-round identity test. Returns a report; the caller must require all checks to pass."""
    import xgboost as xgb

    m = to_margin(teacher_p)
    expected = to_prob(m)
    d = xgb.DMatrix(X, label=y, base_margin=m)
    params = dict(objective="binary:logistic", eval_metric="auc", max_depth=3, n_estimators=1,
                  tree_method="hist", device=device, n_jobs=8)
    bst = xgb.train(params, d, num_boost_round=0, verbose_eval=False)
    got = bst.predict(d)
    max_abs = float(np.max(np.abs(got - expected)))
    rank_match = float(np.corrcoef(np.argsort(np.argsort(got)), np.argsort(np.argsort(expected)))[0, 1])
    raw_check = float(np.max(np.abs(bst.predict(d, output_margin=True) - m)))
    rep = {
        "n": int(len(got)),
        "max_abs_prob_error": max_abs,
        "max_abs_raw_margin_error": raw_check,
        "rank_corr_with_expected": rank_match,
        "passes": bool(max_abs < 1e-4 and raw_check < 1e-4 and rank_match > 0.999999),
    }
    print(f"  identity test: max|dP|={max_abs:.3e}  max|dMargin|={raw_check:.3e}  "
          f"rank_corr={rank_match:.9f}  -> {'PASS' if rep['passes'] else 'FAIL'}")
    return rep


def cross_fit_teacher_margin(teacher_predict_fn, fit_idx, y_fit, n_inner=5, seed=0):
    """Inner-cross-fitted teacher probabilities for the fit rows.

    ``teacher_predict_fn(rows, fit_rows) -> probs`` must fit on ``fit_rows`` and predict ``rows``.
    Returns probabilities for every row of ``fit_idx``; each is produced by a model that never
    saw that row.
    """
    from sklearn.model_selection import StratifiedKFold

    p = np.zeros(len(fit_idx))
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    for i, (a_rel, b_rel) in enumerate(skf.split(np.zeros(len(fit_idx)), y_fit)):
        rows = fit_idx[b_rel]
        src = fit_idx[a_rel]
        p[b_rel] = teacher_predict_fn(rows, src, i)
    return p