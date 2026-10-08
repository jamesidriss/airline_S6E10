"""Covariate-only expectations from previously certified outer-fold caches."""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
from src.common import ARTIFACTS, arr_sha256, file_sha256
from scripts.run_phase13 import RATINGS


def expected_values(aux, xfit, xapply, names, fit_ids, apply_ids, fold_hash):
    assert not np.intersect1d(fit_ids, apply_ids).size, 'FIT/apply overlap'
    assert len(np.unique(fit_ids)) == len(fit_ids) and len(np.unique(apply_ids)) == len(apply_ids)
    assert len(xfit) == len(fit_ids) and len(xapply) == len(apply_ids)
    raw_names = list(RATINGS) + ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Age',
                                 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    assert len(raw_names) == 21 and 'id' not in raw_names and 'satisfaction' not in raw_names
    positions = [names.index(c) for c in raw_names]
    contract = aux['upstream_contract']
    assert contract == {'fit_ids_sha256': arr_sha256(fit_ids), 'apply_ids_sha256': arr_sha256(apply_ids),
                        'fit_raw_sha256': arr_sha256(xfit[:, positions]), 'apply_raw_sha256': arr_sha256(xapply[:, positions]),
                        'fold_sha256': fold_hash, 'seed': 1, 'rounds': 250, 'inner_folds': 3,
                        'builder_sha256': file_sha256('scripts/run_phase13.py'), 'raw_names': raw_names}
    fingerprint = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    assert fingerprint == aux['upstream_fingerprint']
    root = ARTIFACTS / 'aux_distribution' / fingerprint
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['contract'] == contract and manifest['fingerprint'] == fingerprint
    outputs = []
    for part, matrix, row_ids, proof_key in (
            ('fit', xfit, fit_ids, 'fit_expected_values_sha256'),
            ('apply', xapply, apply_ids, 'validation_expected_values_sha256')):
        assert np.array_equal(np.load(root / (part + '_ids.npy')), row_ids)
        p = np.load(root / (part + '.npy'), mmap_mode='r')
        assert p.shape == (len(matrix), 13, 6)
        assert arr_sha256(p) == manifest[part + '_probability_sha256']
        # Bound temporary memory while checking every simplex.
        for start in range(0, len(p), 32768):
            chunk = p[start:start+32768]
            assert np.isfinite(chunk).all() and (chunk >= 0).all()
            assert np.allclose(chunk.sum(axis=-1), 1, atol=1e-8)
        ev = p @ np.arange(6, dtype=float)
        assert arr_sha256(ev) == aux[proof_key], 'Expected-rating arithmetic differs from certified control'
        outputs.append(ev.astype('float32'))
    proof = {'upstream_contract': contract, 'upstream_fingerprint': fingerprint,
             'cache_manifest_sha256': file_sha256(path),
             'fit_probability_sha256': manifest['fit_probability_sha256'],
             'apply_probability_sha256': manifest['apply_probability_sha256'],
             'fit_ev_sha256': arr_sha256(outputs[0]), 'apply_ev_sha256': arr_sha256(outputs[1]),
             'provenance': 'aux_* covariate targets only; FIT innerOOS, apply outer-FIT-only; no satisfaction or id input'}
    return outputs[0], outputs[1], proof
