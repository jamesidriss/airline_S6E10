"""Exact weighted AUC for reproducible pseudo leaderboard partitions.

Only ranks are cached. Public/private partitions never fit or modify predictions.
"""
from __future__ import annotations

import numpy as np


class RankedAUC:
    def __init__(self, y, scores):
        y, scores = np.asarray(y), np.asarray(scores)
        if y.shape != scores.shape or y.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("misaligned/nonfinite prediction vector")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("binary labels with both classes required")
        self.order = np.argsort(scores, kind="stable")
        self.y = y[self.order]
        s = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]

    def auc(self, weights):
        w = np.asarray(weights, dtype=float)
        if w.shape != self.y.shape or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("invalid sample weights")
        w = w[self.order]
        pos = np.add.reduceat(w * self.y, self.starts)
        neg = np.add.reduceat(w * (1 - self.y), self.starts)
        np_, nn = pos.sum(), neg.sum()
        if np_ == 0 or nn == 0:
            raise ValueError("partition contains only one class")
        before = np.cumsum(neg) - neg
        return float(np.dot(pos, before + neg / 2) / (np_ * nn))


def summarize(public, private, full_delta):
    public, private = np.asarray(public), np.asarray(private)
    return {"simulations": len(private), "public_quantiles": np.quantile(public, [.05, .5, .95]).tolist(),
            "private_quantiles": np.quantile(private, [.05, .5, .95]).tolist(),
            "private_p05": float(np.quantile(private, .05)), "private_median": float(np.median(private)),
            "private_worst_5pct_mean": float(np.mean(np.sort(private)[:max(1, int(np.ceil(.05 * len(private))))])),
            "public_private_sign_disagreement_rate": float(np.mean(public * private < 0)),
            "private_reversal_vs_full_oof_rate": float(np.mean(private * full_delta < 0)),
            "public_reversal_vs_full_oof_rate": float(np.mean(public * full_delta < 0))}
