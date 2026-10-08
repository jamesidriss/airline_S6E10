"""Immutable Phase16 controls and label-free/diagnostic numerical helpers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from src.common import ARTIFACTS, arr_sha256, file_sha256, load_cached_parquet
from scripts.assemble_sol_foundation_test import verify_primary, checked_probability

SCOPE = Path('research/phase16_scope_20261009.json')


def bank():
    tr, te = load_cached_parquet()
    ids, tids = tr.id.to_numpy(), te.id.to_numpy()
    y = tr.satisfaction.to_numpy(dtype='int8')
    dh = {s: file_sha256(f'data/raw/{s}.csv') for s in ('train', 'test')}
    old = json.loads(Path('reports/sol_phase15_final_proof_index_20261008.json').read_text())
    for p, h in old['evidence_file_sha256'].items():
        if p == 'experiments/private_finalists.json':
            p = 'reports/phase16_private_strategy_before.json'
        assert file_sha256(p) == h, p
    assert file_sha256('submissions/v6_sol_tabpfn_route_aux10.csv') == old['v6_csv_sha256']
    path = 'reports/sol_route_aux10_primary_final.json'
    primary, vectors, folds = verify_primary(path, ids, y, dh)
    assert abs(roc_auc_score(y, vectors['candidate']) - primary['pooled_oof_auc']) < 1e-14
    for f in primary['folds']:
        k = f['fold']; ref = Path('reports') / f['foundation']['tag'] / f'route_f{k}.json'
        assert file_sha256(ref) == f['foundation']['report_sha256']
        r = json.loads(ref.read_text())
        sources = r['source_sha256']
        if isinstance(sources, str):
            sources = {'scripts/run_sol_tabpfn.py': sources}
        sources = {**sources, 'src/models/windows_attention.py': r['backend_source_sha256'],
                   'src/models/pointwise_inference.py': r['decoder_chunk_source_sha256'],
                   'src/models/resource_guard.py': r['resource_guard_source_sha256']}
        assert all(file_sha256(p) == h for p, h in sources.items())
        assert r['params']['n_estimators'] == 1 and r['seed'] == 1201
        assert file_sha256(r['params']['model_path']) == r['checkpoint']['checkpoint_sha256']
        assert abs(roc_auc_score(y[folds == k], vectors['route'][folds == k]) - r['auc']) < 1e-14
    cert_path = 'reports/sol_route_aux10_primary_final_test_certificate.json'
    cert = json.loads(Path(cert_path).read_text()); root = Path(cert['test_artifact_root'])
    assert cert['data_sha256'] == dh and cert['primary_report_sha256'] == file_sha256(path)
    assert cert['ordered_test_ids_sha256'] == arr_sha256(tids)
    assert cert['oof_sha256'] == {n: arr_sha256(p) for n, p in vectors.items()}
    tests = {n: checked_probability(root / (n + '.npy'), root / 'test_ids.npy', h, tids)
             for n, h in cert['test_sha256'].items()}
    proof = {'primary_report_sha256': file_sha256(path), 'test_certificate_sha256': file_sha256(cert_path),
             'data_sha256': dh, 'fold_sha256': arr_sha256(folds), 'train_ids_sha256': arr_sha256(ids),
             'test_ids_sha256': arr_sha256(tids), 'oof_sha256': {n: arr_sha256(p) for n, p in vectors.items()},
             'test_sha256': {n: arr_sha256(p) for n, p in tests.items()}, 'v6_csv_sha256': old['v6_csv_sha256']}
    return tr, te, y, folds, vectors, tests, proof


def midpercentile(p):
    p = np.asarray(p)
    assert p.ndim == 1 and len(p) and np.isfinite(p).all()
    return (rankdata(p, method='average') - .5) / len(p)


def rank_geometry(a, b, groups=None):
    assert len(a) == len(b)
    out = np.empty(len(a), dtype='float32')
    if groups is None:
        out[:] = .5 * (midpercentile(a) + midpercentile(b))
    else:
        groups = np.asarray(groups); assert groups.shape == np.asarray(a).shape
        for k in np.unique(groups):
            m = groups == k
            out[m] = .5 * (midpercentile(np.asarray(a)[m]) + midpercentile(np.asarray(b)[m]))
    return out


def noise_ceiling(observed_positive_rate, flip_rate, clean_auc=1.0):
    q, e = float(observed_positive_rate), float(flip_rate)
    if not 0 < q < 1 or not 0 <= e < min(q, 1-q) or not .5 <= clean_auc <= 1:
        raise ValueError('Invalid symmetric independent flip assumptions')
    pi = (q-e)/(1-2*e)
    a, b = pi*(1-e)/q, pi*e/(1-q)
    return .5+(a-b)*(clean_auc-.5)


def effective_flip_rate(q, auc):
    lo, hi = 0., min(q, 1-q) - 1e-10
    for _ in range(80):
        mid = (lo+hi)/2
        if noise_ceiling(q, mid) > auc: lo = mid
        else: hi = mid
    return (lo+hi)/2


def auc_se(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    assert len(pos) > 1 and len(neg) > 1
    sn, sp = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sn,pos,'left')+np.searchsorted(sn,pos,'right'))/(2*len(neg))
    v01 = 1-(np.searchsorted(sp,neg,'left')+np.searchsorted(sp,neg,'right'))/(2*len(pos))
    return float(np.sqrt(v10.var(ddof=1)/len(pos)+v01.var(ddof=1)/len(neg)))


def error_pairs(y, p, mask=None):
    y, p = np.asarray(y), np.asarray(p)
    if mask is not None: y, p = y[mask], p[mask]
    pos, neg = p[y == 1], np.sort(p[y == 0])
    errors = (len(neg)-(np.searchsorted(neg,pos,'left')+np.searchsorted(neg,pos,'right'))/2).sum()
    return float(errors), int(len(pos))*int(len(neg))
