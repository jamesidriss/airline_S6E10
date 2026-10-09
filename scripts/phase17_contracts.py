"""Reject row/class/config drift before numerical model admission."""
from __future__ import annotations
import numpy as np


def equivalence(native, sequential, native_ids, sequential_ids, native_configs, sequential_configs,
                tolerance=2e-6):
    a,b=np.asarray(native),np.asarray(sequential)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('Expected two class columns with identical shape')
    if len(native_ids) != len(a) or not np.array_equal(native_ids,sequential_ids):
        raise ValueError('Applied row order mismatch')
    if native_configs != sequential_configs or len(native_configs) != 2:
        raise ValueError('Official estimator configuration/order mismatch')
    for p in (a,b):
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
            raise ValueError('Invalid probability')
        if not np.allclose(p.sum(axis=1),1.,atol=2e-7,rtol=0):
            raise ValueError('Invalid probability normalization')
    gap=float(np.max(np.abs(a.astype('float64')-b.astype('float64'))))
    return {'maximum_absolute_probability_gap':gap,'tolerance':tolerance,'passes':gap<=tolerance}
