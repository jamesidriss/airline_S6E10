"""Exact four-config native reference admission; no ranking-only equivalence."""
from __future__ import annotations
import numpy as np


def equivalence_four(native,sequential,native_ids,sequential_ids,native_configs,sequential_configs,tolerance=2e-6):
    a,b=np.asarray(native),np.asarray(sequential)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2:
        raise ValueError('Expected same two class columns')
    if len(native_ids)!=len(a) or not np.array_equal(native_ids,sequential_ids):
        raise ValueError('Applied row order mismatch')
    if len(native_configs)!=4 or native_configs!=sequential_configs:
        raise ValueError('All four official configurations must match in order')
    for p in (a,b):
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all() or not np.allclose(p.sum(1),1.,atol=2e-7,rtol=0):
            raise ValueError('Invalid normalized probabilities')
    gap=float(np.max(np.abs(a.astype('float64')-b.astype('float64'))))
    return {'maximum_absolute_probability_gap':gap,'tolerance':tolerance,'passes':gap<=tolerance}
