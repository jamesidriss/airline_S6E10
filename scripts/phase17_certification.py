"""Verify saved official contributions and reconstruct the unchanged auxiliary vote."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.common import ARTIFACTS, arr_sha256, file_sha256
from src.validation.compare import logit
from src.submission import store
from scripts.phase16_tabpfn import probabilities
from scripts.assemble_sol_clean import certify
from scripts.replay_sol_classical import frozen_roles, resolved_params, PROTOCOL


def replay_members(folder, metadata, params, rows):
    from tabpfn import TabPFNClassifier
    count = params['n_estimators']
    assert count in (2, 4) and metadata['n_estimators_resolved'] == count
    assert metadata['row_subsampling'] is None and len(metadata['members']) == count
    assert not metadata['balance_probabilities'] and not metadata['average_before_softmax']
    raw = []
    for i, member in enumerate(metadata['members']):
        assert member['index'] == i
        stats = json.loads((Path(folder)/f'member_{i}_stats.json').read_text())
        values = np.load(Path(folder)/f'member_{i}_raw_logits.npy')
        assert values.shape == (1, rows, 2) and values.dtype == np.float32 and np.isfinite(values).all()
        assert stats['index'] == i and stats['raw_logits_sha256'] == member['raw_logits_sha256'] == arr_sha256(values)
        raw.append(values)
    clf = TabPFNClassifier(**params)
    clf.softmax_temperature_ = metadata['softmax_temperature']
    clf.inference_config_ = SimpleNamespace(USE_SKLEARN_16_DECIMAL_PRECISION=
        metadata['official_inference_config']['USE_SKLEARN_16_DECIMAL_PRECISION'])
    assert clf.average_before_softmax == metadata['average_before_softmax']
    assert clf.balance_probabilities == metadata['balance_probabilities']
    return probabilities(clf, np.concatenate(raw, axis=0)).astype('float32')[:, 1]


def load_primary(tag, ids):
    path = Path('reports')/(tag+'_primary.json'); r = json.loads(path.read_text())
    assert r['status'] == 'PRIMARY_PASS_REQUIRES_INDEPENDENT_CONFIRMATION_AND_TEST'
    assert r['train_ids_sha256'] == arr_sha256(ids)
    assert all(file_sha256(s) == h for s, h in r['source_sha256'].items())
    assert {f['fold'] for f in r['folds']} == set(range(5)) and len(r['folds']) == 5
    for f in r['folds']:
        k = f['fold']
        assert file_sha256(Path('reports')/(tag+f'_f{k}.json')) == f['run_report_sha256']
        assert file_sha256(Path('reports')/(tag+f'_f{k}_evaluation.json')) == f['evaluation_sha256']
    root = ARTIFACTS/tag/'primary'
    assert np.array_equal(np.load(root/'train_ids.npy'), ids)
    out = {}
    for name, field in (('route', 'route_sha256'), ('candidate', 'portfolio_sha256')):
        p = np.load(root/('route_oof.npy' if name == 'route' else 'portfolio_oof.npy'))
        assert p.shape == (len(ids),) and p.dtype == np.float32 and arr_sha256(p) == r[field]
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        out[name] = p
    return r, out


def unchanged_auxiliary_test(ids, test_ids, y, bank_test, bank_proof):
    original = json.loads(Path('reports/sol_route_aux10_primary_final_test_certificate.json').read_text())
    assert file_sha256('reports/sol_route_aux10_primary_final_test_certificate.json') == bank_proof['test_certificate_sha256']
    roles = [r for r in frozen_roles() if r['aux_arm']]; assert len(roles) == 10
    index = store._load_index(); total = np.zeros(len(test_ids)); proofs = []
    for role in roles:
        entry = {'source_role': role['member'], 'store_id': f'sol_clean_aux10_primary_{role["member"]}',
                 'tag': 'sol_clean_aux10_primary', 'scheme': 'primary', 'protocol': PROTOCOL,
                 'family': role['family'], 'view': role['view']}
        _, p, proof = certify(entry, index, ids, test_ids, y)
        old = next(t for t in original['classical_proofs'] if t['store_id'] == entry['store_id'])
        assert all(proof[k] == old[k] for k in proof)
        path = Path('reports/sol_clean_aux10_primary')/(role['member']+'_test.json')
        assert file_sha256(path) == old['test_report_sha256']
        c = json.loads(path.read_text())['contract']
        assert c['role'] == role and c['params'] == resolved_params(role)
        assert c['data_sha256'] == bank_proof['data_sha256']
        counts = [json.loads((path.parent/(role['member']+f'_f{k}.json')).read_text())['n_trees'] for k in range(5)]
        assert c['capacity']['selected_fold_counts'] == counts and c['capacity']['n_trees'] == int(np.median(counts))
        assert c['upstream']['contract']['inner_folds'] == 3
        total += logit(p)/10
        proofs.append({**proof, 'test_report_sha256': file_sha256(path), 'median_trees': int(np.median(counts))})
    auxiliary = (1/(1+np.exp(-total))).astype('float32')
    assert np.array_equal(auxiliary, bank_test['aux10'])
    compose = lambda route: (1/(1+np.exp(-(logit(route)+total)/2))).astype('float32')
    assert np.array_equal(compose(bank_test['route']), bank_test['candidate']), 'Exact test portfolio control drift'
    return total, proofs
