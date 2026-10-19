"""Independent pair-enumeration checks for uncertainty and category alignment."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.validation.paired_auc import placements, paired_auc, holm_adjust, categorical_total_variation


def test_placements_equal_brute_pair_matrix_with_ties():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    a = np.array([.4, .4, .1, .2, .9, .7, .7, .2])
    matrix = (a[y == 1, None] > a[None, y == 0]).astype(float)
    matrix += .5 * (a[y == 1, None] == a[None, y == 0])
    p, n = placements(y, a)
    np.testing.assert_allclose(p, matrix.mean(axis=1))
    np.testing.assert_allclose(n, matrix.mean(axis=0))
    assert abs(p.mean() - roc_auc_score(y, a)) < 1e-15


def test_paired_variance_matches_direct_covariance():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    a = np.array([.4, .4, .1, .2, .9, .7, .7, .2])
    b = np.array([.3, .4, .6, .8, .9, .2, .5, .2])
    ap, an = placements(y, a); bp, bn = placements(y, b)
    cov = np.cov([ap, bp], ddof=1) / 4 + np.cov([an, bn], ddof=1) / 4
    expected = np.array([1., -1.]) @ cov @ np.array([1., -1.])
    assert abs(paired_auc(y, a, b)['paired_se'] ** 2 - expected) < 1e-15


def test_identical_predictions_have_zero_paired_uncertainty():
    y = np.array([0, 1, 0, 1]); p = np.array([.1, .1, .8, .7])
    r = paired_auc(y, p, p)
    assert r['paired_se'] == 0 and r['delta'] == 0 and r['one_sided_p'] == 1
    assert r['candidate_standalone_se'] > 0


def test_target_private_four_times_public_halves_standard_error():
    y = np.array([0, 1, 0, 1]); a = np.array([.1, .1, .8, .7]); b = a[::-1]
    pu, pr = paired_auc(y, a, b, 100), paired_auc(y, a, b, 400)
    assert abs(pu['paired_se'] / pr['paired_se'] - 2) < 1e-14


def test_holm_matches_hand_computed_stepdown():
    np.testing.assert_allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])


def test_category_drift_aligns_values_not_codes():
    assert categorical_total_variation(pd.Series(['a', 'a']), pd.Series(['b', 'b'])) == 1


def test_category_drift_handles_order_and_missing_values():
    a = pd.Series(pd.Categorical(['a', 'b', None], categories=['a', 'b']))
    b = pd.Series(pd.Categorical([None, 'b', 'a'], categories=['b', 'a']))
    assert categorical_total_variation(a, b) == 0


def test_invalid_auc_inputs_rejected():
    for y, a, b in [([0, 0, 0], [.1, .2, .3], [.1, .2, .3]),
                    ([0, 1, 0, 1], [0, 1, np.nan, .3], [0, 1, .2, .3])]:
        try:
            paired_auc(y, a, b)
        except ValueError:
            continue
        raise AssertionError('Invalid inputs accepted')
