"""Resume only a verified contiguous prefix of complete official member outputs."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.common import arr_sha256
from scripts.phase16_tabpfn import OneMemberPreprocessor, probabilities, release


def completed_prefix(folder, count, rows):
    out = []; missing = False
    for i in range(count):
        rp = Path(folder)/f'member_{i}_raw_logits.npy'; sp = Path(folder)/f'member_{i}_stats.json'
        if not rp.exists() and not sp.exists():
            missing = True; continue
        if missing or not rp.exists() or not sp.exists():
            raise ValueError('Incomplete or noncontiguous member prefix')
        raw = np.load(rp); stats = json.loads(sp.read_text())
        if raw.shape != (1, rows, 2) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError('Invalid saved member logits')
        if stats['index'] != i or stats['raw_logits_sha256'] != arr_sha256(raw):
            raise ValueError('Saved member index/hash mismatch')
        if 'gpu_allocated_after_release' not in stats:
            raise ValueError('No completed cache-release evidence')
        out.append((raw, stats))
    return out


def resume_predict(clf, prepared, applied, prefix, context, progress, complete):
    factory, kwargs, members, metadata = prepared; raw = []
    for i, member in enumerate(members):
        if i < len(prefix):
            values, stats = prefix[i]; raw.append(values); metadata['members'][i].update(stats)
            complete(i, values, stats); continue
        assert clf.executor_ is None
        start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        try:
            with context(i, 'fit'):
                clf.executor_ = factory(**{**kwargs, 'ensemble_preprocessor':
                    OneMemberPreprocessor(member, kwargs['ensemble_preprocessor'])})
            fit_seconds = time.monotonic()-start; chunks = []
            with context(i, 'predict'):
                for begin in range(0, len(applied), 1024):
                    stop = min(begin+1024, len(applied)); values = clf.predict_raw_logits(applied[begin:stop])
                    if isinstance(values, torch.Tensor):
                        values = values.detach().float().cpu().numpy()
                    assert values.shape == (1, stop-begin, clf.n_classes_)
                    chunks.append(values); progress(i, stop, fit_seconds, time.monotonic()-start)
            values = np.concatenate(chunks, axis=1).astype('float32'); assert np.isfinite(values).all()
            stats = {'index': i, 'fit_seconds': fit_seconds, 'total_seconds': time.monotonic()-start,
                     'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'raw_logits_sha256': arr_sha256(values)}
        finally:
            release(clf)
        stats['gpu_allocated_after_release'] = torch.cuda.memory_allocated()
        raw.append(values); metadata['members'][i].update(stats); complete(i, values, stats)
    return probabilities(clf, np.concatenate(raw, axis=0)).astype('float32'), metadata
