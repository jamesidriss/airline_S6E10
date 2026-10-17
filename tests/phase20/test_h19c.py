"""Tests for Phase-20 H19-C nearest-key TE (fold-safety + arithmetic)."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase20.h19c_nearest_key import (  # noqa: E402
    RATING_COLS, NearestKeyTE, build_segment_flags, rating_key_ints,
)


def _toy(n=400, seed=0):
    """Structured toy: a pool of base keys, rows are exact copies OR one-position
    perturbations. This guarantees both exact duplicates and distance-1
    neighbors exist (uniform random keys in a 6^13 space would have neither)."""
    rng = np.random.default_rng(seed)
    n_base = max(4, n // 8)
    base = rng.integers(0, 6, (n_base, 13))
    R = np.empty((n, 13), dtype=np.int64)
    for i in range(n):
        b = base[i % n_base].copy()
        if i >= n_base and rng.random() < 0.5:
            p = int(rng.integers(0, 13))
            b[p] = (b[p] + int(rng.integers(1, 6))) % 6
        R[i] = b
    df = pd.DataFrame(R, columns=RATING_COLS)
    score = R.sum(axis=1)
    y = (score + rng.standard_normal(n) * 3 > 32).astype(int)
    if y.std() == 0:  # degenerate split guard
        y = (score > np.median(score)).astype(int)
    return df, y


def test_key_encoding_unique_per_key():
    df, _ = _toy()
    k = rating_key_ints(df)
    dup = df.duplicated(subset=RATING_COLS).to_numpy()
    assert (k[dup].shape[0] > 0), "toy needs some duplicate keys"
    # two rows share a key iff their ints are equal
    same = k[:, None] == k[None, :]
    dfk = df[RATING_COLS].astype(str).agg("|".join, axis=1).to_numpy()
    same_df = dfk[:, None] == dfk[None, :]
    assert np.array_equal(same, same_df), "base-6 key must match tuple identity"


def test_neighbor_arithmetic():
    df, _ = _toy(n=50)
    k = rating_key_ints(df)
    R = df[RATING_COLS].to_numpy(dtype=np.int64)
    from src.phase20.h19c_nearest_key import _neighbor_keys, _POW
    nb = _neighbor_keys(k, R)
    assert nb.shape == (50, 78)
    row = 7
    r0 = R[row]
    # column for position p, value v:
    for p in (0, 5, 12):
        for v in (0, 3, 5):
            col = p * 6 + v
            expect = k[row] - r0[p] * _POW[p] + v * _POW[p]
            assert nb[row, col] == expect, f"neighbor arithmetic wrong at p={p} v={v}"
    # exactly 13 columns equal the row's own key (v == r_p per position)
    selfmatches = int((nb[row] == k[row]).sum())
    assert selfmatches == 13, f"expected 13 self-matches, got {selfmatches}"


def test_apply_excludes_self_label():
    """apply() must exclude the encoded row's own key contribution.

    Simultaneously flip two rows' labels so the CLASS COUNT (hence the prior)
    is unchanged: row i (the row under test) and row j whose key is NOT within
    Hamming distance 1 of row i's key. Row i's encoding must be bit-identical:
    its own key is masked, and key(j) is not its neighbor. (Flipping only row i
    would also move the fit-table prior -- that channel is by design and not a
    self-label leak, since in real usage apply-rows are disjoint from the table.)
    """
    df, y = _toy(n=300, seed=1)
    k = rating_key_ints(df)
    R = df[RATING_COLS].to_numpy(dtype=np.int64)
    from src.phase20.h19c_nearest_key import _neighbor_keys
    te = NearestKeyTE(lam=1.0, min_count=1, inner_folds=3, inner_seed=0)
    enc = te.apply(k, y.astype(float), k, R)
    assert np.isfinite(enc).all()

    i = len(y) - 1
    nb_i = set(_neighbor_keys(k[i:i + 1], R[i:i + 1])[0].tolist())
    j = next(t for t in range(len(y) - 1)
             if t != i and k[t] not in nb_i and y[t] != y[i])
    y2 = y.astype(float).copy()
    y2[i] = 1.0 - y2[i]
    y2[j] = 1.0 - y2[j]
    enc2 = te.apply(k, y2, k, R)
    assert np.array_equal(enc[i], enc2[i]), \
        "row encoding changed when only its own label (+a non-neighbor's) flipped"
    # sanity: SOME other row (a neighbor of key i or j) must react, proving the
    # estimator actually uses neighbor labels at all.
    assert not np.array_equal(enc, enc2), "estimator ignores neighbor labels"


def test_crossfit_no_own_label():
    """Cross-fitted encoding of a row must not change when only that row's
    label flips: it is held out of its OWN fold's encoding table, so neither
    the table nor the prior for its fold can contain its label.

    Rows in OTHER folds legitimately see the flipped row inside their tables,
    so only the target row's own encoding is compared (not the whole array).
    """
    df, y = _toy(n=600, seed=2)
    k = rating_key_ints(df)
    R = df[RATING_COLS].to_numpy(dtype=np.int64)
    te = NearestKeyTE(lam=5.0, min_count=1, inner_folds=3, inner_seed=0)
    e1 = te.crossfit(k, y.astype(float), R)
    y2 = y.astype(float).copy()
    target = 137
    y2[target] = 1.0 - y2[target]
    e2 = te.crossfit(k, y2, R)
    assert np.array_equal(e1[target], e2[target]), \
        "crossfit encoding leaked the held-out row's own label"
    # sanity: the estimator must react somewhere (other folds' tables change)
    assert not np.array_equal(e1, e2), "estimator ignores labels entirely"


def test_crossfit_determinism():
    df, y = _toy(n=400, seed=3)
    k = rating_key_ints(df)
    R = df[RATING_COLS].to_numpy(dtype=np.int64)
    a = NearestKeyTE(inner_seed=0).crossfit(k, y.astype(float), R)
    b = NearestKeyTE(inner_seed=0).crossfit(k, y.astype(float), R)
    assert np.array_equal(a, b), "crossfit not deterministic"


def test_min_count_falls_back_to_prior():
    """If no neighbor aggregate clears min_count, encoding == table prior.

    Uses apply() (not crossfit) so the prior is exactly mean(y_fit) over the
    single encoding table.
    """
    df, y = _toy(n=200, seed=4)
    k = rating_key_ints(df)
    R = df[RATING_COLS].to_numpy(dtype=np.int64)
    te = NearestKeyTE(lam=30.0, min_count=10_000, inner_folds=2, inner_seed=0)
    enc = te.apply(k, y.astype(float), k, R)
    prior = float(np.mean(y))
    assert np.allclose(enc[:, 0], prior), "min_count gate must return prior"


def test_segment_flags_match_atlas_definition():
    """build_segment_flags returns True for DUPLICATE rating keys, identical to
    the atlas is_dup = key.map(cnt) > 1 (transductive, label-free): ALL rows
    whose key appears more than once, including the first occurrence.
    (Note df.duplicated() marks only 2nd+ occurrences, so it is NOT the oracle.)
    """
    df, _ = _toy(n=300, seed=5)
    dup = build_segment_flags(df)
    # independent oracle: atlas is_dup = key.map(value_counts(key)) > 1
    key = df[RATING_COLS].astype(str).agg("|".join, axis=1)
    manual = (key.map(key.value_counts()) > 1).to_numpy()
    assert np.array_equal(dup, manual), \
        "segment flag must equal key-count>1 (atlas is_dup definition)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS {f.__name__}")
    print(f"{len(fns)} H19-C tests passed")
