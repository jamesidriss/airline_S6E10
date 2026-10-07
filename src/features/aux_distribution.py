"""Auxiliary rating distribution signatures: satisfaction-label-free covariates.

Provenance: aux_* is a new fold cross-fitted, covariate-target block. Its
probabilities must be predicted out of sample; satisfaction is never an input.
"""
from __future__ import annotations

import numpy as np

LEVELS = np.arange(6, dtype=float)
EPS = 1e-12


def signatures(probabilities, ratings, stage, surprise_threshold=None):
    p, r = np.asarray(probabilities, dtype=float), np.asarray(ratings)
    if stage not in range(5):
        raise ValueError("stage must be A0..A4")
    if p.shape != (*r.shape, 6) or r.ndim != 2 or r.shape[1] != 13:
        raise ValueError("rating/probability shape mismatch")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=-1), 1, atol=1e-8):
        raise ValueError("invalid probability simplex")
    if not np.isfinite(r).all() or (r != r.astype(int)).any() or (r < 0).any() or (r > 5).any():
        raise ValueError("ratings must be integers 0..5")
    ev = p @ LEVELS
    residual = r - ev
    observed = np.take_along_axis(p, r.astype(int)[..., None], axis=-1)[..., 0]
    surprise = -np.log(np.clip(observed, EPS, 1))
    entropy = -np.sum(p * np.log(np.clip(p, EPS, 1)), axis=-1)
    variance = np.maximum(p @ (LEVELS ** 2) - ev ** 2, 0)
    blocks = [("ev", ev)]
    if stage >= 1:
        blocks += [("residual", residual), ("abs_residual", np.abs(residual))]
    if stage >= 2:
        blocks += [("p_observed", observed), ("surprisal", surprise)]
    if stage >= 3:
        blocks += [("entropy", entropy), ("variance", variance), ("max_prob", p.max(axis=-1)),
                   ("low_rating_p12", p[..., 1:3].sum(axis=-1)),
                   ("high_rating_p45", p[..., 4:6].sum(axis=-1))]
    mats, names = [b for _, b in blocks], [f"aux_{kind}_{j}" for kind, _ in blocks for j in range(13)]
    if stage >= 4:
        if surprise_threshold is None or np.shape(surprise_threshold) != (13,):
            raise ValueError("A4 requires predeclared FIT-only 95th percentile thresholds")
        aggs = []
        for kind, values in (("abs_residual", np.abs(residual)), ("surprisal", surprise),
                             ("entropy", entropy), ("variance", variance)):
            for stat, fn in (("mean", np.mean), ("max", np.max), ("std", np.std)):
                aggs.append(fn(values, axis=1))
                names.append(f"aux_agg_{kind}_{stat}")
        aggs += [(surprise > surprise_threshold).sum(axis=1), (residual > 0).sum(axis=1),
                 (residual < 0).sum(axis=1)]
        names += ["aux_agg_surprising_count", "aux_agg_residual_positive_count", "aux_agg_residual_negative_count"]
        mats.append(np.column_stack(aggs))
    out = np.column_stack(mats)
    assert out.shape[1] == len(names) and np.isfinite(out).all()
    return out, names


def fit_surprise_threshold(probabilities, ratings):
    # Fixed per-rating 95th percentile on FIT covariates only. No satisfaction labels.
    observed = np.take_along_axis(probabilities, np.asarray(ratings, dtype=int)[..., None], axis=-1)[..., 0]
    return np.quantile(-np.log(np.clip(observed, EPS, 1)), .95, axis=0)
