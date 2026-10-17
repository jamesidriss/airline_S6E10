"""Phase 20 -- H19-C: fold-safe smoothed NEAREST-KEY target features.

Mechanism (distinct from the closed H18-C exact-key TE): for a row whose exact
13-rating key is rare or unseen in the encoding table, aggregate the smoothed
target means of the FIT keys at Hamming distance exactly 1 (65 candidate keys
per row under base-6 arithmetic over the 13 integer ratings). Exact-key TE only
helps rows whose key repeats; nearest-key smoothing transports label information
to the 36%-of-rows unique-key weak region (atlas segment AUC 0.95032).

Fold-safety contract (identical to FoldSafeTE):
  * ``apply(fit_keys, y, apply_keys)`` builds the table from fit rows ONLY.
  * ``crossfit(fit_keys, y)`` encodes fit rows themselves with an inner K-fold
    cross-fit so no row is ever encoded by a table containing its own label.
No eval-fold label is ever read. The key counts used to SEGMENT rows are
transductive and label-free (permitted per the provenance table).

The prior and the smoothing constant are FROZEN in the prereg; the estimator is
(count-weighted Bayesian smoothing of neighbor means):
    nk = (sum_k cnt_k * mean_k + lam * prior) / (sum_k cnt_k + lam)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The 13 service ratings, in the atlas order (base-6 encoding positions).
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
_BASE = 6
_NPOS = 13
_POW = _BASE ** np.arange(_NPOS, dtype=np.int64)  # 6^p, p=0..12


def rating_key_ints(df: pd.DataFrame) -> np.ndarray:
    """Base-6 integer encoding of the 13-rating tuple (unique per key)."""
    R = df[RATING_COLS].to_numpy(dtype=np.int64)
    assert R.min() >= 0 and R.max() < _BASE, "rating outside 0..5"
    return R @ _POW


def _neighbor_keys(base: np.ndarray, R: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """(n, 78) int64 array of Hamming-distance-<=1 candidate keys per row.

    For each of the 13 positions and each of the 6 values in 0..5, replace the
    digit: key' = key - r_p * 6^p + v * 6^p. Thirteen of the 78 columns equal
    the row's own key (v == r_p); they are masked at lookup time.
    """
    n = base.shape[0]
    out = np.empty((n, _NPOS * _BASE), dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        b = base[s:e][:, None]                      # (m,1)
        Rp = R[s:e]                                  # (m,13)
        cols = []
        for p in range(_NPOS):
            rp = Rp[:, p][:, None] * _POW[p]         # (m,1)
            for v in range(_BASE):
                cols.append(b - rp + v * _POW[p])    # (m,1) broadcast->(m,)
        # 13 positions x 6 values = 78 columns; for each row exactly one column
        # per position equals the row's own key (v == r_p). Those self-matches
        # are zeroed at lookup time (a row must not contribute its own label,
        # and exact-key TE is the separately-closed H18-C mechanism).
        out[s:e] = np.column_stack(cols)
    return out


def _self_mask(base: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Boolean (n, 78) mask marking columns equal to the row's own key."""
    return nb == base[:, None]


class NearestKeyTE:
    """Fold-safe Hamming-d-1 smoothed target features over the 13-rating key.

    Frozen estimator: count-weighted Bayesian smoothing of the neighbor keys'
    target means, with smoothing constant ``lam`` and prior = table mean.
    Output features per apply row:
        nk1      : smoothed neighbor target mean
        nk1_logc : log1p(total neighbor count)
        nk1_nk   : number of distinct neighbor keys found (log1p)
    """

    def __init__(self, lam: float = 30.0, min_count: int = 2, inner_folds: int = 5,
                 inner_seed: int = 0):
        self.lam = float(lam)
        self.min_count = int(min_count)
        self.inner_folds = int(inner_folds)
        self.inner_seed = int(inner_seed)

    # ---------------------------------------------------------- table + apply
    def _encode(self, fit_key: np.ndarray, y_fit: np.ndarray,
                apply_key: np.ndarray, apply_R: np.ndarray) -> np.ndarray:
        prior = float(np.mean(y_fit))
        # key -> (count, sum) over rows with count >= 1; rare keys (1 row) still
        # contribute as NEIGHBORS (their mean is noisy but they carry signal);
        # the min_count gate is applied to the aggregate, not the neighbor.
        s = pd.Series(y_fit)
        idx = pd.Index(fit_key)
        cnt = s.groupby(idx).size()
        sm = s.groupby(idx).sum()
        k_uniq = cnt.index.to_numpy(dtype=np.int64)
        c_arr = cnt.to_numpy(dtype=np.float64)
        m_arr = sm.to_numpy(dtype=np.float64)

        base = apply_key
        nb = _neighbor_keys(base, apply_R)
        pos = np.searchsorted(k_uniq, nb)            # sorted index assumed below
        # k_uniq from groupby is sorted ascending -> searchsorted valid
        valid = (pos >= 0) & (pos < len(k_uniq))
        safe = np.where(valid, pos, 0)
        hit = valid & (k_uniq[safe] == nb)
        c_hit = np.where(hit, c_arr[safe], 0.0)
        m_hit = np.where(hit, m_arr[safe], 0.0)
        # exclude self-match (a row must not contribute its own key/label)
        c_hit[hit & (nb == base[:, None])] = 0.0
        m_hit[hit & (nb == base[:, None])] = 0.0

        tot_c = c_hit.sum(axis=1)
        tot_m = m_hit.sum(axis=1)
        n_keys = (c_hit > 0).sum(axis=1).astype(np.float64)
        nk = np.where(tot_c >= self.min_count,
                      (tot_m + self.lam * prior) / (tot_c + self.lam),
                      prior)
        return np.column_stack([nk.astype(np.float32),
                                np.log1p(tot_c).astype(np.float32),
                                np.log1p(n_keys).astype(np.float32)])

    def apply(self, fit_key: np.ndarray, y_fit: np.ndarray,
              apply_key: np.ndarray, apply_R: np.ndarray) -> np.ndarray:
        return self._encode(fit_key, y_fit, apply_key, apply_R)

    # ------------------------------------------------------------- crossfit
    def crossfit(self, fit_key: np.ndarray, y_fit: np.ndarray,
                 fit_R: np.ndarray) -> np.ndarray:
        n = len(y_fit)
        out = np.full((n, 3), np.nan, dtype=np.float32)
        rng = np.random.default_rng(self.inner_seed)
        inner = (np.arange(n) % self.inner_folds)[rng.permutation(n)]
        for i in range(self.inner_folds):
            a = np.flatnonzero(inner != i)
            b = np.flatnonzero(inner == i)
            out[b] = self._encode(fit_key[a], y_fit[a], fit_key[b], fit_R[b])
        assert np.isfinite(out).all(), "crossfit produced non-finite values"
        return out


def build_segment_flags(comb: pd.DataFrame) -> np.ndarray:
    """Transductive, LABEL-FREE duplicate/unique flags over train+test rows
    (identical definition to the phase-19 atlas unique_rating_key segment)."""
    key = None
    for c in RATING_COLS:
        s = comb[c].astype(str) + "|"
        key = s if key is None else key + s
    cnt = key.map(key.value_counts()).to_numpy()
    return cnt > 1  # True = duplicate key
