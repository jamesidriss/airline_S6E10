"""Phase 19 H19-B tests: bag-of-five deployment policy, fold-safety, exact pair math.

Registered in tests/run_tests.py as T19B. Complements the 12 H19-A tests (T19).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase19.h19b_bag5 import (INNER_PARTITIONS_SEED, bag_fit_rows, build_inner_partitions,
                                   exact_pair_flips, fit_ids_manifest, mean_prob,
                                   _counts_single)


def _toy_data(n=400, seed=7):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype("int8")
    ids = np.arange(1000, 1000 + n)
    # primary-like 5 folds
    folds = rng.integers(0, 5, n).astype("int8")
    return y, ids, folds


def test_inner_partitions_deterministic_and_disjoint():
    y, ids, folds = _toy_data()
    fit_idx = np.flatnonzero(folds != 4)
    val_idx = np.flatnonzero(folds == 4)
    p1 = build_inner_partitions(fit_idx, y, n_splits=5, seed=INNER_PARTITIONS_SEED)
    p2 = build_inner_partitions(fit_idx, y, n_splits=5, seed=INNER_PARTITIONS_SEED)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2)), "partitions not deterministic"
    # partitions are disjoint and cover FIT
    seen = np.concatenate(p1)
    assert len(seen) == len(fit_idx) and len(np.unique(seen)) == len(fit_idx)
    # no pseudo-test row in any partition
    for p in p1:
        assert np.intersect1d(p, val_idx).size == 0, "pseudo-test leaked into a partition"
    # each bag fit is FIT minus one partition, size ~80% of FIT
    for i, p in enumerate(p1):
        bf = bag_fit_rows(fit_idx, p1, i)
        assert np.intersect1d(bf, p).size == 0, "bag fit overlaps its held-out partition"
        assert np.array_equal(np.sort(np.concatenate([bf, p])), fit_idx)
        assert abs(len(bf) - 0.8 * len(fit_idx)) <= len(p)


def test_fit_ids_manifest_proves_fold_safety():
    y, ids, folds = _toy_data()
    fit_idx = np.flatnonzero(folds != 4)
    val_idx = np.flatnonzero(folds == 4)
    parts = build_inner_partitions(fit_idx, y, n_splits=5, seed=INNER_PARTITIONS_SEED)
    m = fit_ids_manifest(fit_idx, parts)
    assert m["fit_size"] == len(fit_idx)
    assert len(m["bags"]) == 5
    for i, bag in enumerate(m["bags"]):
        bf = bag_fit_rows(fit_idx, parts, i)
        from src.common import arr_sha256
        assert bag["bag_fit_ids_sha256"] == arr_sha256(bf)
        assert bag["bag_fit_size"] == len(bf)


def test_stratification_preserves_class_balance():
    y, ids, folds = _toy_data(n=2000)
    fit_idx = np.flatnonzero(folds != 4)
    parts = build_inner_partitions(fit_idx, y, n_splits=5, seed=INNER_PARTITIONS_SEED)
    rate = y[fit_idx].mean()
    for p in parts:
        assert abs(y[p].mean() - rate) < 0.03, "inner partition class balance off"


def test_exact_pair_flips_identity_and_bruteforce():
    rng = np.random.default_rng(3)

    def brute(y, p):
        right = tie = wrong = 0
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        for i in pos:
            for j in neg:
                if p[i] > p[j]:
                    right += 1
                elif p[i] == p[j]:
                    tie += 1
                else:
                    wrong += 1
        return right, tie, wrong

    for trial in range(15):
        n = 200
        y = (rng.random(n) < 0.3).astype("int8")
        pc = rng.integers(0, 5, n).astype(float)  # many ties
        pt = rng.integers(0, 5, n).astype(float)
        r = exact_pair_flips(y, pc, pt)
        assert r["identity_abs_error"] < 1e-9, r
        for name, p in (("ctrl", pc), ("treat", pt)):
            rb, tb, wb = brute(y, p)
            rc, tc = _counts_single(y, p)
            assert int(round(rc)) == rb and int(round(tc)) == tb, (trial, name)


def test_pair_credit_semantics_correct_tie_half_wrong_zero():
    # explicit tiny case: 1 pos, 2 neg
    y = np.array([1, 0, 0])
    p = np.array([0.5, 0.4, 0.6])
    # pairs: (pos,neg0): 0.5>0.4 -> right(1); (pos,neg1): 0.5<0.6 -> wrong(0)
    r = exact_pair_flips(y, p, p)  # control vs itself -> zero net
    assert r["pairs"] == 2
    assert r["ctrl_right"] == 1 and r["ctrl_wrong"] == 1
    assert abs(r["net_credit"]) < 1e-12 and abs(r["identity_abs_error"]) < 1e-12


def test_mean_prob_equal_average_no_logit_mixing():
    a = np.array([0.2, 0.6])
    b = np.array([0.4, 0.8])
    m = mean_prob([a, b])
    assert np.allclose(m, [0.3, 0.7]), m
    assert m.dtype == np.float64


def test_partition_seed_is_frozen_constant():
    assert INNER_PARTITIONS_SEED == 20261022
