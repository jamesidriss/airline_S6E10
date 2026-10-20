"""Phase-24 frozen composition rules shared by the lab and the deployer.

Every rule maps two parent probability vectors to one score vector. Rules that use
labels only ever receive FIT-row arrays; callers are responsible for the split and
the lab asserts FIT/eval disjointness.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
WEIGHT_GRID = np.round(np.arange(0.30, 0.7001, 0.025), 4)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


# ---------------------------------------------------------------- zero-parameter
def c0_control(r, a):
    return sigmoid(0.5 * logit(r) + 0.5 * logit(a))


def c2_block_rank(r, a):
    def unit(x):
        return (rankdata(np.asarray(x, dtype=np.float64), method="average") - 0.5) / len(x)

    return unit(r) + unit(a)


def c6_probability_mean(r, a):
    return 0.5 * np.asarray(r, dtype=np.float64) + 0.5 * np.asarray(a, dtype=np.float64)


def c7_geometric_mean(r, a):
    return np.sqrt(np.clip(np.asarray(r, dtype=np.float64), 0, 1) * np.clip(np.asarray(a, dtype=np.float64), 0, 1))


# ---------------------------------------------------------------- FIT-fitted
def c1_fit_weight(z_r, z_a, y):
    """Return the grid weight maximising FIT AUC of (1-w)*z_r + w*z_a."""
    from sklearn.metrics import roc_auc_score

    best_w, best = 0.5, -np.inf
    for w in WEIGHT_GRID:
        s = (1 - w) * z_r + w * z_a
        v = roc_auc_score(y, s)
        if v > best:
            best, best_w = v, float(w)
    return best_w, float(best)


def c1_apply(w, z_r, z_a):
    return sigmoid((1 - w) * z_r + w * z_a)


def c3_fit_cdf(x):
    """Empirical CDF of FIT rows, returned as sorted values for interpolation."""
    return np.sort(np.asarray(x, dtype=np.float64))


def c3_apply(sorted_fit, x):
    x = np.asarray(x, dtype=np.float64)
    # mid-rank style CDF: fraction of FIT values strictly below + half of equals
    lo = np.searchsorted(sorted_fit, x, side="left")
    hi = np.searchsorted(sorted_fit, x, side="right")
    return (lo + 0.5 * (hi - lo)) / len(sorted_fit)


def c4_fit(z_r, z_a, y):
    X = np.column_stack([z_r, z_a, np.abs(z_r - z_a), np.minimum(z_r, z_a), np.maximum(z_r, z_a)])
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    m.fit(X, y)
    return m


def c4_features(z_r, z_a):
    return np.column_stack([z_r, z_a, np.abs(z_r - z_a), np.minimum(z_r, z_a), np.maximum(z_r, z_a)])


def c5_fit_platt(z, y):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    m.fit(np.asarray(z, dtype=np.float64).reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def c5_apply(z, coef, intercept):
    return coef * np.asarray(z, dtype=np.float64) + intercept


RULE_NAMES = (
    "C1_nested_weight",
    "C2_block_rank",
    "C3_fit_cdf",
    "C4_confidence_logistic",
    "C5_platt_average",
    "C6_probability_mean",
    "C7_geometric_mean",
)
