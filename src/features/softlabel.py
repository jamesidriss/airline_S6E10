"""Fold-safe calibration of a *label-free* external teacher.

The teacher is the model trained only on the original 129,880-row survey, so its predictions
depend on no competition label at all. The calibration is the only part that touches labels, and
it is fitted exclusively on the rows the caller declares as the fit set -- never on rows that will
be scored.

Two calibrators:
  ``logit``   one-feature logistic regression in logit space -> a slope and an intercept. Smooth,
              always defined, and with a single parameter it cannot overfit much.
  ``isotonic`` isotonic regression on the raw probability. Flexible, non-parametric, and
              therefore only trustworthy when the fit set is large relative to the teacher's
              distinct output values -- ``support_ok`` reports that, and callers must respect it.

Both return a calibrated probability for arbitrary rows, using parameters derived only from the
fit rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype="float64"), -35.0, 35.0)))


class ExternalTeacherCalibrator:
    """Fit a calibration of a label-free teacher on the fit rows only."""

    def __init__(self, kind: str = "logit"):
        self.kind = kind
        self.params: dict = {}
        self.support_ok = True
        self.n_fit = 0
        self.fit_auc_teacher = None
        self.fit_auc_calibrated = None

    def fit(self, teacher_prob: np.ndarray, y: np.ndarray) -> "ExternalTeacherCalibrator":
        tp = np.clip(np.asarray(teacher_prob, dtype="float64"), EPS, 1.0 - EPS)
        y = np.asarray(y, dtype="float64")
        self.n_fit = len(y)
        self.fit_auc_teacher = float(roc_auc_score(y, tp))
        if self.kind == "logit":
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(to_logit(tp).reshape(-1, 1), y)
            self.params = {"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}
        elif self.kind == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(tp, y)
            # A non-parametric calibrator needs enough distinct teacher outputs to be identified.
            self.support_ok = len(np.unique(tp)) >= 200 and self.n_fit >= 2000
            self.params = {"x": iso.X_thresholds_.astype("float64"),
                           "y": iso.y_thresholds_.astype("float64")}
        else:
            raise ValueError(self.kind)
        self.fit_auc_calibrated = float(roc_auc_score(y, self.apply(tp)))
        return self

    def apply(self, teacher_prob: np.ndarray) -> np.ndarray:
        tp = np.clip(np.asarray(teacher_prob, dtype="float64"), EPS, 1.0 - EPS)
        if self.kind == "logit":
            z = self.params["coef"] * to_logit(tp) + self.params["intercept"]
            return sigmoid(z)
        x, y = self.params["x"], self.params["y"]
        # piecewise-linear interpolation with flat extrapolation, matching out_of_bounds='clip'
        return np.interp(tp, x, y, left=float(y[0]), right=float(y[-1]))


def soft_targets(y: np.ndarray, p_cal: np.ndarray, lam: float) -> np.ndarray:
    """q = (1 - lambda) * y + lambda * p_calibrated.

    This is a *shrinkage of the observed label toward the prior*. With y drawn as a Bernoulli(p)
    sample, y is an unbiased but high-variance estimate of p; shrinking it toward a lower-variance
    (if biased) prior reduces the target's variance, which is the standard denoising argument.
    Never coerces to binary: q stays fractional for any 0 < lambda < 1.
    """
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p_cal, dtype="float64")
    return (1.0 - lam) * y + lam * p


def optimal_lambda_shrinkage(y: np.ndarray, p_cal: np.ndarray, lams) -> tuple[float, list]:
    """Diagnostic: which lambda best predicts the *hard* labels under log loss.

    This is a label-space diagnostic, not a competition metric -- it tells us whether the teacher
    prior is strong enough to be worth shrinking toward, using only the fit rows' labels.
    """
    out = []
    y = np.asarray(y, dtype="float64")
    p = np.clip(np.asarray(p_cal, dtype="float64"), EPS, 1 - EPS)
    for lam in lams:
        q = np.clip(soft_targets(y, p, lam), EPS, 1 - EPS)
        ll = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
        out.append((lam, ll))
    best = max(out, key=lambda t: -t[1])
    return best[0], out