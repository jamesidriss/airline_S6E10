"""Phase 19 H19-A -- fold-0-safe difficulty reconstruction + controlled A/B helpers.

Anti-leakage contract (the whole point of this module):
  The champion route/aux for fold-k rows (k in 1..4) were trained on folds != k,
  which CONTAINS fold 0.  Therefore every Atlas diagnostic for folds 1-4
  (per_row_burden, pred_a, pred_b, stored route/aux) is a (transitive) function
  of fold-0 labels and is FORBIDDEN as a training weight for a fold-0 eval.

  The difficulty weights used here are rebuilt strictly inside FIT (folds 1-4):
  every judge / difficulty-model prediction for a FIT row comes from a model
  trained ONLY on folds subseteq {1,2,3,4} (inner cross-fit), so no fold-0 label
  can reach the weights, targets, features, or early-stopping.

Everything here is CPU-only (H19-A budget: 0 GPU jobs, no heavy GPU).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr

# --------------------------------------------------------------------- weights
def rank01(d: np.ndarray) -> np.ndarray:
    """Midrank of d scaled to [0,1] (lowest -> 0, highest -> 1). Ties -> average."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0:
        return d
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    r = rankdata(d, method="average")  # 1..n
    return (r - 1.0) / (n - 1.0)


def weights_from_score(d: np.ndarray, lam: float = 0.5, lo: float = 0.25,
                       hi: float = 2.0) -> tuple[np.ndarray, dict]:
    """Frozen weight function (prereg h19a_prereg.json :: weight_function).

        r = midrank(d) in [0,1]
        w_raw = 1 + lam*(2r - 1)          # lam=0.5 -> w_raw in [0.5, 1.5]
        w = clip(w_raw, lo, hi) / mean(clip(w_raw, lo, hi))   # mean exactly 1

    Deterministic, finite, strictly positive, mean 1.  No post-hoc search.
    """
    r = rank01(d)
    w_raw = 1.0 + lam * (2.0 * r - 1.0)
    w_clip = np.clip(w_raw, lo, hi)
    mean = float(w_clip.mean())
    assert mean > 0, "non-positive mean weight"
    w = w_clip / mean
    info = {
        "lambda": lam, "clip_before_norm": [lo, hi],
        "raw_min": float(w_raw.min()), "raw_max": float(w_raw.max()),
        "mean_before_norm": mean,
        "min": float(w.min()), "max": float(w.max()), "mean": float(w.mean()),
        "std": float(w.std()),
        "n": int(len(w)),
    }
    return w, info


def assert_weight_valid(w: np.ndarray, lo: float = 0.25, hi: float = 2.0,
                        tol: float = 1e-9) -> None:
    w = np.asarray(w, dtype=np.float64)
    assert np.isfinite(w).all(), "non-finite weight"
    assert (w > 0).all(), "non-positive weight"
    assert abs(float(w.mean()) - 1.0) < 1e-6, f"weight mean {w.mean()} != 1"
    assert w.min() >= lo - tol and w.max() <= hi + tol, \
        f"weight bounds [{w.min()}, {w.max()}] outside [{lo}, {hi}]"


# --------------------------------------------------------------- fold-0 safety
def assert_no_fold0(fit_idx: np.ndarray, folds: np.ndarray, what: str) -> None:
    """Assert a supervised fit's row indices contain no eval-fold (0) rows."""
    hit = np.flatnonzero(folds[fit_idx] == 0)
    assert len(hit) == 0, f"{what}: {len(hit)} fold-0 rows leaked into FIT"


def assert_partition(fit_idx: np.ndarray, eval_idx: np.ndarray,
                     folds: np.ndarray, eval_fold: int = 0) -> None:
    """Assert FIT and eval are disjoint, eval==eval_fold, together cover all."""
    assert len(np.intersect1d(fit_idx, eval_idx)) == 0, "FIT/eval overlap"
    assert np.all(folds[eval_idx] == eval_fold), "eval idx are not the eval fold"
    assert np.all(folds[fit_idx] != eval_fold), "FIT idx contain the eval fold"


# ------------------------------------------------------------- weighted lightgbm
def fit_lgbm_weighted(X, y, Xv, params, seed, es_X=None, es_y=None,
                      sample_weight=None):
    """Mirror scripts.run_views._fit_lgbm_es but honour sample_weight.

    Control passes sample_weight=ones (== LightGBM's default None exactly);
    treatment passes the fold-0-safe difficulty weights.  Identical code path,
    params, seed, and inner-ES early stopping -- only the weights differ.
    Returns (pred_on_Xv, best_iteration).
    """
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", n_estimators=6000, learning_rate=0.02,
             num_leaves=127, min_child_samples=40, colsample_bytree=0.8, subsample=0.8,
             subsample_freq=1, reg_lambda=1.0, max_bin=255, verbose=-1, n_jobs=8,
             random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    p.update(params)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float64)
    ds = lgb.Dataset(X, label=y, weight=np.asarray(sample_weight, dtype=np.float64))
    dv = lgb.Dataset(es_X, label=es_y, reference=ds)  # unweighted ES (AUC is rank-based)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict(Xv, num_iteration=m.best_iteration), int(m.best_iteration)


# ---------------------------------------------------------- pair sampling (frozen)
def sample_pair_stats(y, route, aux, treat, ctrl, n_pairs=50_000_000,
                      seed=20261019) -> dict:
    """Frozen-seed uniform sampling of cross-class fold pairs.

    For each sampled (positive p, negative n):
      shared        = (route_p <= route_n) & (aux_p <= aux_n)   # both halves fail
      credit_m      = 1 strict-right, 0.5 tie, 0 wrong (tie-aware, AUC convention)
      corrected     = credit_treat > credit_ctrl
      degraded      = credit_treat < credit_ctrl
    Reported per stratum (shared / non-shared) with binomial SEs.
    No quadratic matrix is materialised: O(n_pairs) work.
    """
    from sklearn.metrics import roc_auc_score  # noqa: F401  (kept for parity checks)

    y = np.asarray(y)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    assert len(pos) and len(neg), "need both classes"

    rr, ra = np.asarray(route, dtype=np.float64), np.asarray(aux, dtype=np.float64)
    rt, rc = np.asarray(treat, dtype=np.float64), np.asarray(ctrl, dtype=np.float64)

    rng = np.random.default_rng(seed)
    pi = pos[rng.integers(0, len(pos), size=n_pairs)]
    ni = neg[rng.integers(0, len(neg), size=n_pairs)]

    shared = (rr[pi] <= rr[ni]) & (ra[pi] <= ra[ni])

    def credit(a):
        s = a[pi] - a[ni]
        return np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))

    ct, cc = credit(rt), credit(rc)

    def block(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        t, c = ct[mask], cc[mask]
        diff = t - c
        se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        return {
            "n": n,
            "acc_treat": float(t.mean()), "acc_treat_se": se(t),
            "acc_ctrl": float(c.mean()), "acc_ctrl_se": se(c),
            "net_rate": float(diff.mean()), "net_rate_se": se(diff),
            "corrected": int((diff > 0).sum()),
            "degraded": int((diff < 0).sum()),
            "count_net": int((diff > 0).sum() - (diff < 0).sum()),
            "ties_treat": int((t == 0.5).sum()),
            "ties_ctrl": int((c == 0.5).sum()),
        }

    out = {
        "n_pairs": int(n_pairs), "seed": seed, "with_replacement": True,
        "sampled_shared_rate": float(shared.mean()),
        "sampled_shared_se": float(shared.std(ddof=1) / np.sqrt(n_pairs)),
        "shared": block(shared),
        "non_shared": block(~shared),
        "all": block(np.ones(n_pairs, dtype=bool)),
    }
    return out


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt((a * a).sum())
    db = np.sqrt((b * b).sum())
    if da == 0 or db == 0:
        return float("nan")
    return float((a * b).sum() / (da * db))


def spearman(a, b) -> float:
    return float(spearmanr(np.asarray(a), np.asarray(b)).statistic)
