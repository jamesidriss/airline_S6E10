"""Certify the frozen repaired portfolio; refuse partial or legacy members."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, load_cached_parquet, save_json
from src.submission import store
from src.validation.folds import get_scheme
from src.validation.compare import logit
from src.models.realmlp import TRAINING_PROTOCOL
from scripts.replay_sol_classical import PROTOCOL, frozen_roles


def frozen_plan(classical_tag, neural_tag):
    original = json.loads((REPORTS/'finalist_v3_final.json').read_text(encoding='utf-8'))['members']
    classical = {r['member']: r for r in frozen_roles()}
    entries = []
    for m in original:
        name = m['exp_id']
        is_classical = name in classical
        tag = classical_tag if is_classical else neural_tag
        entries.append({'source_role': name, 'store_id': f'{tag}_{name}',
            'family': m['family'], 'view': m['featureset'], 'tag': tag,
            'scheme': classical[name]['scheme'] if is_classical else 'primary',
            'protocol': PROTOCOL if is_classical else TRAINING_PROTOCOL,
            'is_auxiliary': is_classical and classical[name]['aux_arm'] is not None})
    assert len(entries) == 59 and len({e['source_role'] for e in entries}) == 59
    assert sum(e['family'] in ('realmlp', 'tabm') for e in entries) == 12
    assert sum(e['is_auxiliary'] for e in entries) == 10
    return {'members': entries, 'geometry': 'equal arithmetic mean of member logits',
        'diagnostic_groups': ['all59', 'classical47', 'auxiliary10'],
        'selection': 'frozen original role list; no AUC filter or fitted weights',
        'test_policy': 'classical full-label fixed-count fits; neural exact CV-model probability average'}


def certify(entry, index, train_ids, test_ids, y):
    sid, role = entry['store_id'], entry['source_role']
    record = index[sid]
    meta = record['meta']
    assert record['fold_scheme'] == entry['scheme']
    assert meta['training_protocol'] == entry['protocol'] and meta['source_role'] == role
    assert meta['family'] == entry['family'] and meta['featureset'] == entry['view']
    assert meta['ordered_train_ids_sha256'] == arr_sha256(train_ids)
    assert meta['ordered_test_ids_sha256'] == arr_sha256(test_ids)
    fold_array = get_scheme(entry['scheme'], y, train_ids).folds
    reports = REPORTS/entry['tag']
    complete = reports/f'{role}_complete.json'
    assert complete.exists(), 'Complete member certification is required'
    fold_paths = [reports/f'{role}_f{k}.json' for k in sorted(set(fold_array.tolist()))]
    assert [file_sha256(p) for p in fold_paths] == meta['fold_reports_sha256']
    neural = entry['family'] in ('realmlp', 'tabm')
    reconstructed = np.full(len(train_ids), np.nan, dtype='float32')
    neural_test = np.zeros(len(test_ids), dtype='float64') if neural else None
    for k, p in enumerate(fold_paths):
        r = json.loads(p.read_text(encoding='utf-8'))
        c = r['contract']
        assert c['fold'] == k and c['fold_sha256'] == arr_sha256(fold_array)
        assert c['validation_ids_sha256'] == arr_sha256(train_ids[fold_array == k])
        assert all(file_sha256(s) == h for s, h in c['source_sha256'].items())
        assert all(file_sha256(f'data/raw/{s}.csv') == h for s, h in c['data_sha256'].items())
        artifact = ARTIFACTS/entry['tag']
        prediction = np.load(artifact/(p.stem+'.npy'))
        va = np.flatnonzero(fold_array == k)
        assert prediction.shape == (len(va),) and arr_sha256(prediction) == r['prediction_sha256']
        sidecar = artifact/(f'ids_f{k}.npy' if neural else p.stem+'_ids.npy')
        assert np.array_equal(np.load(sidecar), train_ids[va])
        reconstructed[va] = prediction
        if neural:
            t = np.load(artifact/f'{role}_test_f{k}.npy')
            assert t.shape == (len(test_ids),) and arr_sha256(t) == r['test_prediction_sha256']
            assert np.isfinite(t).all() and ((t >= 0) & (t <= 1)).all()
            assert r['test_ids_sha256'] == arr_sha256(test_ids)
            neural_test += t/len(fold_paths)
    oof, test = store.load_oof(sid), store.load_test(sid)
    assert oof.shape == (len(train_ids),) and test.shape == (len(test_ids),)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    assert ((oof >= 0) & (oof <= 1)).all() and ((test >= 0) & (test <= 1)).all()
    assert abs(float(roc_auc_score(y, oof)) - meta['auc']) < 1e-14
    assert np.array_equal(reconstructed, oof)
    if neural:
        assert np.array_equal(neural_test.astype('float32'), test)
    summary = json.loads(complete.read_text(encoding='utf-8'))
    assert summary['oof_sha256' if neural else 'prediction_sha256'] == arr_sha256(oof)
    assert summary['test_sha256'] == arr_sha256(test)
    if not neural:
        test_report = json.loads((reports/f'{role}_test.json').read_text(encoding='utf-8'))
        assert test_report['prediction_sha256'] == arr_sha256(test)
        c = test_report['contract']
        assert c['test_ids_sha256'] == arr_sha256(test_ids) and c['fit_ids_sha256'] == arr_sha256(train_ids)
        assert all(file_sha256(s) == h for s, h in c['source_sha256'].items())
    return oof, test, {'store_id': sid, 'oof_sha256': arr_sha256(oof),
        'test_sha256': arr_sha256(test), 'complete_report_sha256': file_sha256(complete)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classical-tag', default='sol_clean_classical')
    ap.add_argument('--neural-tag', default='sol_neural_clean_compact')
    ap.add_argument('--name', default='sol_clean_v5_replay')
    args = ap.parse_args()
    plan = frozen_plan(args.classical_tag, args.neural_tag)
    path = REPORTS/f'{args.name}_plan.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == plan
    else:
        save_json(plan, path)
    index = store._load_index()
    missing = [e['store_id'] for e in plan['members'] if e['store_id'] not in index or 'test' not in index[e['store_id']]]
    if missing:
        raise RuntimeError(f'No portfolio assembled: {len(missing)} complete OOF/test members are missing')
    tr, te = load_cached_parquet()
    y, ids, test_ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy(), te['id'].to_numpy()
    primary = get_scheme('primary', y, ids).folds
    sums = {g: [np.zeros(len(tr)), np.zeros(len(te)), 0, []] for g in plan['diagnostic_groups']}
    for entry in plan['members']:
        oof, test, proof = certify(entry, index, ids, test_ids, y)
        groups = ['all59']
        if entry['family'] in ('lgbm', 'xgb', 'cat'):
            groups.append('classical47')
        if entry['is_auxiliary']:
            groups.append('auxiliary10')
        for group in groups:
            sums[group][0] += logit(oof)
            sums[group][1] += logit(test)
            sums[group][2] += 1
            sums[group][3].append(proof)
    for group, (oo, tt, count, proof) in sums.items():
        assert count == {'all59': 59, 'classical47': 47, 'auxiliary10': 10}[group]
        oo /= count; tt /= count
        oof = (1/(1+np.exp(-oo))).astype('float32')
        test = (1/(1+np.exp(-tt))).astype('float32')
        auc = float(roc_auc_score(y, oof))
        meta = {'family': 'blend', 'training_protocol': 'sol_clean_frozen_portfolio_v1',
            'auc': auc, 'group': group, 'member_count': count, 'geometry': plan['geometry'],
            'member_certificates': proof, 'plan_sha256': file_sha256(path),
            'ordered_train_ids_sha256': arr_sha256(ids), 'ordered_test_ids_sha256': arr_sha256(test_ids),
            'validation': 'Every row excluded from its own member fits; original primary/shadow/block10 member schemes retained',
            'test_policy': plan['test_policy'], 'verdict': 'CLEAN_REPLAY_REQUIRES_INDEPENDENT_CONFIRMATION_AND_SCORECARD'}
        key = f'{args.name}_{group}'
        record = store.save(key, oof, test, fold_scheme='mixed_immutable_schemes', meta=meta)
        save_json({'store_id': key, 'meta': meta, 'oof_sha256': record['oof_sha'], 'test_sha256': record['test_sha'],
            'primary_segment_aucs': [float(roc_auc_score(y[primary == k], oof[primary == k])) for k in range(5)],
            'limitation': 'Primary segments summarize the mixed-scheme OOF; they are not an independent shadow confirmation'},
            REPORTS/f'{key}.json')
        print(f'{group}: {count} certified members, AUC {auc:.9f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
