"""Conditional paired ROC-AUC uncertainty; never a private-score forecast.

DeLong et al. (1988), https://pubmed.ncbi.nlm.nih.gov/3203132/.
Placements preserve half credit for ties. OOF model dependence and adaptive
model selection are outside this conditional row-sampling calculation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def placements(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Aligned finite one-dimensional predictions required")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes required")
    pos, neg = scores[y == 1], scores[y == 0]
    if min(len(pos), len(neg)) < 2:
        raise ValueError("At least two observations in each class required")
    sp, sn = np.sort(pos), np.sort(neg)
    vp = (np.searchsorted(sn, pos, side="left") +
          np.searchsorted(sn, pos, side="right")) / (2 * len(neg))
    vn = 1 - (np.searchsorted(sp, neg, side="left") +
              np.searchsorted(sp, neg, side="right")) / (2 * len(pos))
    return vp, vn


def paired_auc(y, candidate, baseline, target_size=None):
    ap, an = placements(y, candidate)
    bp, bn = placements(y, baseline)
    m, n = len(ap), len(an)
    if target_size is not None:
        if not isinstance(target_size, (int, np.integer)) or target_size < 4:
            raise ValueError("Integer target population of at least four required")
        m = int(round(target_size * m / (m + n)))
        n = target_size - m
        if min(m, n) < 2:
            raise ValueError("Target population must contain both classes")
    delta = float(ap.mean() - bp.mean())
    se = float(np.sqrt(np.var(ap - bp, ddof=1) / m + np.var(an - bn, ddof=1) / n))
    standalone = float(np.sqrt(np.var(ap, ddof=1) / m + np.var(an, ddof=1) / n))
    p = float(norm.sf(delta / se)) if se else (0.0 if delta > 0 else 1.0)
    return {"candidate_auc": float(ap.mean()), "baseline_auc": float(bp.mean()),
            "delta": delta, "paired_se": se, "candidate_standalone_se": standalone,
            "ci95": [delta - 1.959963984540054 * se, delta + 1.959963984540054 * se],
            "one_sided_p": p, "positive_rows": m, "negative_rows": n,
            "scope": "Conditional IID row uncertainty; excludes refitting and model-selection uncertainty"}


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Finite probabilities required")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted


def categorical_total_variation(a, b):
    """Align actual category values, never independently assigned category codes."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Nonempty populations required")
    pa = a.astype(object).value_counts(normalize=True, dropna=False)
    pb = b.astype(object).value_counts(normalize=True, dropna=False)
    pa, pb = pa.align(pb, join="outer", fill_value=0)
    return float(0.5 * np.abs(pa - pb).sum())
