"""Exact rescue/damage credits for all positive-negative pairs, including ties."""
from __future__ import annotations
import numpy as np
from scripts.phase16_common import error_pairs


def pair_rescue_damage(y, candidate, baseline):
    y = np.asarray(y); candidate = np.asarray(candidate); baseline = np.asarray(baseline)
    assert y.shape == candidate.shape == baseline.shape and np.isfinite(candidate).all() and np.isfinite(baseline).all()
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    assert len(pos) and len(neg)
    order = neg[np.argsort(baseline[neg], kind='stable')]
    sorted_negative_baseline = baseline[order]
    coordinates = np.unique(candidate[neg]); tree = np.zeros(len(coordinates)+1, dtype='int64')
    neg_rank = np.searchsorted(coordinates, candidate[order])+1
    pos_order = pos[np.argsort(-baseline[pos], kind='stable')]
    less = np.searchsorted(coordinates, candidate[pos_order], side='left')
    equal_or_less = np.searchsorted(coordinates, candidate[pos_order], side='right')
    cursor = len(order)-1; rescue = 0.; previous = None; equal_group = np.empty(0)

    def prefix(index):
        count = 0
        while index:
            count += int(tree[index]); index -= index & -index
        return count

    for j, p in enumerate(pos_order):
        value = baseline[p]
        while cursor >= 0 and baseline[order[cursor]] > value:
            index = int(neg_rank[cursor])
            while index < len(tree):
                tree[index] += 1; index += index & -index
            cursor -= 1
        # Baseline-wrong pairs: new-correct receives1, new-tied receives1/2.
        rescue += .5*(prefix(int(less[j]))+prefix(int(equal_or_less[j])))
        if previous is None or previous != value:
            left = np.searchsorted(sorted_negative_baseline, value, side='left')
            right = np.searchsorted(sorted_negative_baseline, value, side='right')
            equal_group = np.sort(candidate[order[left:right]])
            previous = value
        # Baseline-tied pairs improved to new-correct receive1/2 credit.
        rescue += .5*int(np.searchsorted(equal_group, candidate[p], side='left'))
    old_errors, total = error_pairs(y, baseline); new_errors, new_total = error_pairs(y, candidate)
    assert total == new_total
    damage = rescue+new_errors-old_errors
    assert rescue >= 0 and damage >= 0
    return {'all_positive_negative_pairs': total, 'rescued_pair_credit': rescue,
        'damaged_pair_credit': damage, 'rescued_pair_fraction': rescue/total,
        'damaged_pair_fraction': damage/total, 'net_auc_gain': (rescue-damage)/total,
        'baseline_error_pair_credit': old_errors, 'candidate_error_pair_credit': new_errors,
        'policy': 'Exact pair credits; correct1/tied1/2/wrong0; no sampling'}
