"""Aligned paired comparison utilities for close experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def fold_auc(y, oof, folds):
    return [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in sorted(set(folds.tolist()))]


def paired_fold_deltas(y, oof_a, oof_b, folds):
    fa, fb = fold_auc(y, oof_a, folds), fold_auc(y, oof_b, folds)
    return float(np.mean(fa) - np.mean(fb)), [x - z for x, z in zip(fa, fb)]


def strat_bootstrap_auc_delta(y, oof_a, oof_b, n_boot=200, seed=0, alpha=0.05):
    """Bootstrap the paired AUC difference (rows resampled; AUC recomputed on each sample)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    idx_all = np.arange(n)
    d = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, oof_a[idx])
        b = roc_auc_score(yy, oof_b[idx])
        d.append(a - b)
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))


def corr(a, b):
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)


def rankit(a):
    from scipy.stats import rankdata

    return rankdata(np.asarray(a, dtype="float64")) / len(a)


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def report(name_a, auc_a, oof_a, name_b, auc_b, oof_b, folds, y, boot=False, n_boot=100):
    m, per = paired_fold_deltas(y, oof_a, oof_b, folds)
    line = (f"{name_a}={auc_a:.6f}  {name_b}={auc_b:.6f}  delta={auc_a-auc_b:+.6f}  "
            f"mean_paired_fold_delta={m:+.6f}  per_fold={[round(x,6) for x in per]}  "
            f"corr={corr(oof_a, oof_b):.5f}  spearman={spearman(oof_a, oof_b):.5f}")
    if boot:
        bm, lo, hi = strat_bootstrap_auc_delta(y, oof_a, oof_b, n_boot=n_boot)
        line += f"  boot_delta={bm:+.6f} [{lo:+.6f},{hi:+.6f}]"
    return line