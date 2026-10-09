"""Verify complete OOF with explicit source paths for any verified resume tags."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.common import ARTIFACTS, arr_sha256, file_sha256


def load_primary(tag, ids):
    path = Path('reports')/(tag+'_primary.json'); r = json.loads(path.read_text())
    assert r['status'] == 'PRIMARY_PASS_REQUIRES_INDEPENDENT_CONFIRMATION_AND_TEST'
    assert r['train_ids_sha256'] == arr_sha256(ids)
    assert all(file_sha256(s) == h for s, h in r['source_sha256'].items())
    assert {f['fold'] for f in r['folds']} == set(range(5)) and len(r['folds']) == 5
    for f in r['folds']:
        k = f['fold']; source_tag = f.get('source_tag', tag)
        rp = Path('reports')/(source_tag+f'_f{k}.json'); ep = Path('reports')/(source_tag+f'_f{k}_evaluation.json')
        assert Path(f.get('run_report_path', rp)).resolve() == rp.resolve()
        assert Path(f.get('evaluation_path', ep)).resolve() == ep.resolve()
        assert file_sha256(rp) == f['run_report_sha256'] and file_sha256(ep) == f['evaluation_sha256']
    root = ARTIFACTS/tag/'primary'; assert np.array_equal(np.load(root/'train_ids.npy'), ids)
    out = {}
    for name, field in (('route', 'route_sha256'), ('candidate', 'portfolio_sha256')):
        p = np.load(root/('route_oof.npy' if name == 'route' else 'portfolio_oof.npy'))
        assert p.shape == (len(ids),) and p.dtype == np.float32 and arr_sha256(p) == r[field]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(); out[name] = p
    return r, out
