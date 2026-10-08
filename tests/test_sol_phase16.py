"""Analytical noise assumptions, pair accounting and applied-fold rank isolation."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scripts.phase16_common import noise_ceiling, error_pairs, rank_geometry
from scripts.phase16_tabpfn import probabilities, sequential_predict


def test_symmetric_noise_ceiling_matches_explicit_pair_confusion():
    # Independently enumerate the four latent-class pairs among observed +/−.
    pi, e, clean_auc = .31, .038, .93
    q = pi*(1-e)+(1-pi)*e
    observed_pos = np.array([pi*(1-e), (1-pi)*e])/q
    observed_neg = np.array([pi*e, (1-pi)*(1-e)])/(1-q)
    pair_success = np.array([[.5, clean_auc], [1-clean_auc, .5]])
    expected = np.sum(observed_pos[:, None]*observed_neg[None, :]*pair_success)
    assert abs(noise_ceiling(q, e, clean_auc)-expected) < 1e-14
    assert abs(noise_ceiling(.5, e)-(1-e)) < 1e-14
    for q0, e0 in ((0., .03), (.2, .25), (.5, .5), (.4, -.01)):
        try:
            noise_ceiling(q0, e0)
        except ValueError:
            continue
        raise AssertionError('Invalid latent distribution accepted')


def test_exact_error_pairs_counts_ties_and_region_exclusion():
    y = np.array([1, 0, 1, 0, 0, 1])
    p = np.array([.3, .3, .7, .9, .1, .7])
    pos, neg = p[y == 1], p[y == 0]
    expected = sum(float(a < b)+.5*float(a == b) for a in pos for b in neg)
    errors, count = error_pairs(y, p)
    assert errors == expected and count == 9
    mask = np.arange(len(y)) < 4
    assert error_pairs(y, p, mask) == (2.5, 4)


def test_rank_geometry_cannot_use_other_applied_folds():
    a = np.array([.1, .4, .4, .9, .2, .3, .8, .85])
    b = np.array([.3, .1, .1, .8, .1, .4, .6, .95])
    groups = np.repeat([0, 1], 4)
    original = rank_geometry(a, b, groups)
    changed_a, changed_b = a.copy(), b.copy()
    changed_a[4:] = 1000; changed_b[4:] = -1000
    changed = rank_geometry(changed_a, changed_b, groups)
    assert np.array_equal(changed[:4], original[:4])
    assert not np.array_equal(rank_geometry(a, b)[:4], rank_geometry(changed_a, changed_b)[:4])
    assert original[1] == original[2]
    assert np.array_equal(rank_geometry(np.exp(a), b**3, groups), original)


def test_sequential_cache_release_and_official_probability_order():
    # Fake engines keep a live count; the real orchestration must never retain two.
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(n_estimators=2, average_before_softmax=False)
    clf.softmax_temperature_ = .7; clf.n_classes_ = 2
    clf.downsample_correction_weights_ = None; clf.executor_ = None
    clf.inference_config_ = SimpleNamespace(USE_SKLEARN_16_DECIMAL_PRECISION=False)
    clf._maybe_reweight_probas = lambda *, probas: probas
    alive = []
    members = [SimpleNamespace(X_train=np.ones((3, 1)), y_train=np.array([0, 1, 0])) for _ in range(2)]
    values = [np.array([[[3., 0.], [0., 2.], [1., 1.]]], dtype='float32'),
              np.array([[[-1., 0.], [3., 0.], [2., 1.]]], dtype='float32')]

    def factory(**kwargs):
        assert not alive
        member = kwargs['ensemble_preprocessor'].member
        i = next(i for i, m in enumerate(members) if m is member)
        alive.append(i)
        return SimpleNamespace(index=i)

    def release_fake(model):
        assert len(alive) == 1
        alive.clear(); model.executor_ = None

    clf.predict_raw_logits = lambda applied: values[clf.executor_.index]
    prep = (factory, {'ensemble_preprocessor': None}, members, {'members': [{}, {}]})
    with patch('scripts.phase16_tabpfn.release', release_fake), \
         patch('torch.cuda.reset_peak_memory_stats'), \
         patch('torch.cuda.max_memory_allocated', return_value=42), \
         patch('torch.cuda.memory_allocated', return_value=0):
        p, _, _ = sequential_predict(clf, prep, np.zeros((3, 1)))
    raw = np.concatenate(values)
    expected = torch.softmax(torch.tensor(raw)/.7, dim=-1).mean(dim=0).numpy()
    assert np.max(np.abs(p-expected)) < 1e-7 and not alive and clf.executor_ is None
    wrong_order = torch.softmax(torch.tensor(raw).mean(dim=0)/.7, dim=-1).numpy()
    assert np.max(np.abs(p-wrong_order)) > .1
    clf.average_before_softmax = True
    assert np.max(np.abs(probabilities(clf, raw)-wrong_order)) < 1e-7
