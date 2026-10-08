"""Official estimator configurations with one live KV cache; no model policy change.

The official classifier still fits its encoders, configs and CPU preprocessing.
Only its inference-engine construction is intercepted, then the official engine
is constructed for each already-prepared member. Raw, class-unpermuted logits
are combined with the classifier's public postprocessor and final normalization.
"""
from __future__ import annotations
import dataclasses
import gc
import hashlib
import json
import time
from enum import Enum
from unittest.mock import patch
import numpy as np
import torch
from src.common import arr_sha256, file_sha256


def serializable(value):
    if dataclasses.is_dataclass(value):
        return serializable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, 'features'):
        return serializable(value.features)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unserializable official configuration: {type(value)}')


def config_hash(config):
    return hashlib.sha256(json.dumps(serializable(config), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def library_sources():
    import tabpfn
    from pathlib import Path
    root = Path(tabpfn.__file__).parent
    return {name: file_sha256(root/name) for name in (
        'classifier.py', 'inference.py', 'base.py', 'inference_config.py',
        'preprocessing/ensemble.py', 'preprocessing/datamodel.py')}


def prepare(clf, x, y):
    """Run official fit up to engine construction, recording untouched members."""
    from tabpfn.classifier import create_inference_engine
    captured = {}

    def intercept(**kwargs):
        prep = kwargs['ensemble_preprocessor']
        assert prep.subsample_row_indices is None, 'Require every frozen FIT row'
        members = prep.fit_transform_ensemble_members(X_train=kwargs['X_train'],
                                                       y_train=kwargs['y_train'])
        captured.update(kwargs=kwargs, members=members)
        return None

    with patch('tabpfn.classifier.create_inference_engine', intercept):
        clf.fit(x, y)
    members, kwargs = captured['members'], captured['kwargs']
    assert len(members) == clf.n_estimators_ == clf.n_estimators
    assert clf.tuning_config is None and clf.downsample_correction_weights_ is None
    prep = kwargs['ensemble_preprocessor']
    manifests = []
    for i, member in enumerate(members):
        a, b = np.asarray(member.X_train), np.asarray(member.y_train)
        assert len(a) == len(b) == len(x)
        manifests.append({'index': i, 'config': serializable(member.config),
            'config_sha256': config_hash(member.config), 'pipeline_seed': int(prep.pipeline_seeds[i]),
            'prepared_x_sha256': arr_sha256(a), 'prepared_y_sha256': arr_sha256(b),
            'prepared_shape': list(a.shape), 'feature_schema': serializable(member.feature_schema),
            'feature_indices': serializable(member.feature_indices),
            'gpu_preprocessor_type': type(member.gpu_preprocessor).__name__})
    metadata = {'members': manifests, 'softmax_temperature': float(clf.softmax_temperature_),
        'average_before_softmax': clf.average_before_softmax,
        'balance_probabilities': clf.balance_probabilities,
        'row_subsampling': None, 'n_estimators_resolved': clf.n_estimators_,
        'checkpoint_n_estimators': serializable(clf.inference_config_.N_ESTIMATORS),
        'official_inference_config': serializable(clf.inference_config_)}
    return create_inference_engine, kwargs, members, metadata


class OneMemberPreprocessor:
    def __init__(self, member, parent):
        self.member, self.parent = member, parent

    def fit_transform_ensemble_members(self, X_train, y_train):
        assert self.member.X_train is not None, 'Member cache has already been consumed'
        return [self.member]

    def any_estimator_uses_gpu_svd(self):
        return self.parent.any_estimator_uses_gpu_svd()


def probabilities(clf, logits):
    """Same public logit conversion and final operations as _predict_proba."""
    from tabpfn.constants import PROBABILITY_EPSILON_ROUND_ZERO, SKLEARN_16_DECIMAL_PRECISION
    p = clf.logits_to_probabilities(logits).float().detach().cpu().numpy()
    p = clf._maybe_reweight_probas(probas=p)
    if clf.inference_config_.USE_SKLEARN_16_DECIMAL_PRECISION:
        p = np.around(p, decimals=SKLEARN_16_DECIMAL_PRECISION)
        p = np.where(p < PROBABILITY_EPSILON_ROUND_ZERO, 0., p)
    return p / p.sum(axis=1, keepdims=True)


def release(clf):
    clf.executor_ = None
    gc.collect()
    torch.cuda.empty_cache()


def sequential_predict(clf, prepared, applied, *, batch_size=1024, context=None,
                       progress=None, estimator_complete=None):
    from contextlib import nullcontext
    factory, kwargs, members, metadata = prepared
    raw = []
    for i, member in enumerate(members):
        assert clf.executor_ is None, 'Previous GPU engine must be released'
        start = time.monotonic()
        guard = context(i, 'fit') if context else nullcontext()
        torch.cuda.reset_peak_memory_stats()
        try:
            with guard:
                clf.executor_ = factory(**{**kwargs, 'ensemble_preprocessor':
                    OneMemberPreprocessor(member, kwargs['ensemble_preprocessor'])})
            fit_seconds = time.monotonic()-start
            chunks = []
            guard = context(i, 'predict') if context else nullcontext()
            with guard:
                for begin in range(0, len(applied), batch_size):
                    stop = min(begin+batch_size, len(applied))
                    logits = clf.predict_raw_logits(applied[begin:stop])
                    if isinstance(logits, torch.Tensor):
                        logits = logits.detach().float().cpu().numpy()
                    assert logits.shape == (1, stop-begin, clf.n_classes_)
                    chunks.append(logits)
                    if progress:
                        progress(i, stop, fit_seconds, time.monotonic()-start)
            values = np.concatenate(chunks, axis=1).astype('float32')
            assert np.isfinite(values).all()
            raw.append(values)
            member_stats = {'index': i, 'fit_seconds': fit_seconds,
                'total_seconds': time.monotonic()-start,
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                'raw_logits_sha256': arr_sha256(values)}
        finally:
            release(clf)
        member_stats['gpu_allocated_after_release'] = torch.cuda.memory_allocated()
        metadata['members'][i].update(member_stats)
        if estimator_complete:
            estimator_complete(i, values, member_stats)
    stacked = np.concatenate(raw, axis=0)
    return probabilities(clf, stacked).astype('float32'), stacked, metadata
