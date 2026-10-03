"""Ensemble laboratory.

Design goals
------------
* ROC-AUC is rank-based, so probability averaging, logit averaging and rank averaging are all
  tested rather than assumed.
* Searching many weight vectors against one OOF vector overfits it. Two controls are used:
    - **nested meta-validation**: weights are fitted on 4 folds and scored on the 5th, repeated
      over all folds, so the reported stack score never sees a weight fitted on that fold;
    - **admission gate**: a member is admitted only if the mean paired fold gain clears a
      threshold, exceeds a multiple of the paired fold standard error, and is positive in a
      majority of folds (the rule the S6E9 runner-up used).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit as _logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TRANSFORMS = {
    "prob": lambda p: np.asarray(p, dtype="float64"),
    "logit": lambda p: _logit(np.clip(np.asarray(p, dtype="float64"), 1e-7, 1 - 1e-7)),
    "rank": lambda p: rankdata(np.asarray(p, dtype="float64")) / len(p),
}


def tform(p, kind: str) -> np.ndarray:
    return TRANSFORMS[kind](p)


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def fold_aucs(y, p, folds) -> list[float]:
    return [auc(y[folds == k], p[folds == k]) for k in sorted(set(np.asarray(folds).tolist()))]


def corr_matrix(P: dict[str, np.ndarray], kind="logit") -> pd.DataFrame:
    names = list(P)
    M = np.column_stack([tform(P[n], kind) for n in names])
    C = np.corrcoef(M, rowvar=False)
    return pd.DataFrame(C, index=names, columns=names)


def spearman_matrix(P: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(P)
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = spearmanr(P[names[i]], P[names[j]]).statistic
    return pd.DataFrame(M, index=names, columns=names)


def equal_blend(P: dict[str, np.ndarray], y, folds, kind="logit", weights=None) -> dict:
    names = list(P)
    w = np.ones(len(names)) / len(names) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    M = np.column_stack([tform(P[n], kind) for n in names])
    p = M @ w
    return {"auc": auc(y, p), "fold_aucs": fold_aucs(y, p, folds), "weights": w.tolist(),
            "names": names, "kind": kind}


def greedy_hill_climb(P: dict[str, np.ndarray], y, folds, kind="logit", rounds=60,
                      seed_idx=None) -> dict:
    """Caruana-style greedy ensemble selection with replacement, scored on pooled OOF AUC."""
    names = list(P)
    M = np.column_stack([tform(P[n], kind) for n in names])
    counts = np.zeros(len(names))
    cur = np.zeros(M.shape[0])
    best_hist = []
    for it in range(rounds):
        best_a, best_j = -1, None
        for j in range(len(names)):
            a = auc(y, (cur * it + M[:, j]) / (it + 1))
            if a > best_a:
                best_a, best_j = a, j
        cur = (cur * it + M[:, best_j]) / (it + 1)
        counts[best_j] += 1
        best_hist.append(best_a)
    w = counts / counts.sum()
    return {"auc": auc(y, M @ w), "weights": w.tolist(), "names": names, "kind": kind,
            "history": best_hist}


def logistic_stack_nested(P: dict[str, np.ndarray], y, folds, kind="logit", C=1.0,
                          n_meta_folds=5, seed=0) -> dict:
    """Nested-CV logistic stack on transformed member scores.

    For each outer fold: inner meta-folds inside the OTHER folds fit the coefficients, which are
    then applied to the held-out outer fold. The reported AUC is therefore out-of-sample with
    respect to the stacker as well as the base models.
    """
    names = list(P)
    X = np.column_stack([tform(P[n], kind) for n in names])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    oof = np.zeros(len(y))
    coefs = []
    for k in sorted(set(np.asarray(folds).tolist())):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        rng = np.random.default_rng(seed + k)
        mf = rng.permutation(len(tr)) % n_meta_folds
        pred_te = np.zeros(len(te))
        for m in range(n_meta_folds):
            a, b = tr[mf != m], tr[mf == m]
            lr = LogisticRegression(C=C, max_iter=400)
            lr.fit(X[a], y[a])
            pred_te += lr.decision_function(X[te]) / n_meta_folds
        oof[te] = pred_te
    return {"auc": auc(y, oof), "fold_aucs": fold_aucs(y, oof, folds), "names": names,
            "kind": kind, "C": C, "coefs": coefs, "oof": oof}


def logistic_stack_oof_predict(P: dict[str, np.ndarray], y, test: dict[str, np.ndarray],
                               kind="logit", C=1.0, n_meta_folds=5, seed=0) -> np.ndarray:
    """Fit meta coefficients per outer fold and average their test predictions."""
    names = list(P)
    X = np.column_stack([tform(P[n], kind) for n in names])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xt = np.column_stack([tform(test[n], kind) for n in names])
    Xt = (Xt - X.mean(0)) / (X.std(0) + 1e-12)
    folds = np.zeros(len(y), dtype=int)
    pred = np.zeros(len(Xt))
    for k in sorted(set(np.asarray(folds).tolist())):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        rng = np.random.default_rng(seed + k)
        mf = rng.permutation(len(tr)) % n_meta_folds
        for m in range(n_meta_folds):
            a, b = tr[mf != m], tr[mf == m]
            lr = LogisticRegression(C=C, max_iter=400)
            lr.fit(X[a], y[a])
            pred += lr.decision_function(Xt) / n_meta_folds
    return pred


def admission_gate(name: str, base_folds: list[float], cand_folds: list[float],
                   min_units: float = 1.5, se_mult: float = 2.5, min_pos: int = 4,
                   unit: float = 1e-5) -> dict:
    """The S6E9 runner-up's gate: mean gain >= `min_units` units, >= `se_mult` x fold SE,
    positive in at least `min_pos` folds."""
    d = np.array(cand_folds) - np.array(base_folds)
    units = d / unit
    mean = float(units.mean())
    se = float(units.std(ddof=1) / np.sqrt(len(units))) if len(units) > 1 else float("inf")
    pos = int((units > 0).sum())
    ok = bool(mean >= min_units and mean >= se_mult * se and pos >= min_pos)
    return {"name": name, "mean_units": round(mean, 3), "fold_se_units": round(se, 3),
            "pos_folds": pos, "admit": ok, "per_fold_units": [round(x, 2) for x in units]}