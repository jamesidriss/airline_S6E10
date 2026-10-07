"""Tests of exact ties, weights and split isolation, using independently computed AUC."""
import numpy as np
from sklearn.metrics import roc_auc_score
from src.validation.private_sim import RankedAUC
from scripts.private_lb_simulator import make_partition


def test_ranked_auc_matches_sklearn_with_ties_and_bootstrap_weights():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 1000)
    scores = rng.integers(0, 15, 1000).astype(float)
    scorer = RankedAUC(y, scores)
    for _ in range(10):
        w = rng.integers(0, 5, 1000)
        assert abs(scorer.auc(w) - roc_auc_score(y, scores, sample_weight=w)) < 1e-12


def test_private_split_definitions_are_deterministic_and_disjoint():
    y = np.tile([0, 1], 1000)
    folds = np.arange(2000) % 5
    for mode in ("random_stratified", "fold_aware", "segment_stressed"):
        a = make_partition(y, folds, 7, mode, 1000, stress=np.arange(2000) < 600, fold=2)
        b = make_partition(y, folds, 7, mode, 1000, stress=np.arange(2000) < 600, fold=2)
        assert np.array_equal(a, b)
        assert (a == 1).sum() == 200 and (a == 2).sum() == 800
        assert y[a == 1].sum() == 100 and y[a == 2].sum() == 400
        if mode == "fold_aware":
            assert (folds[a == 1] == 2).all() and (folds[a == 2] != 2).all()
