"""Certify all five context predictions, preserve auxiliary10, and build a local CSV."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json
from src.submission.make import build
from src.validation.compare import logit
from scripts.phase16_common import bank
from scripts.phase16_tabpfn import library_sources
from scripts.phase17_certification import replay_members, unchanged_auxiliary_test
from scripts.phase17_primary_io import load_primary
from scripts.score_sol_foundation import correlations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary-tag', required=True)
    ap.add_argument('--test-tag', required=True)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    assert Path(args.name).name == args.name and args.name not in ('.', '..', 'v6_sol_tabpfn_route_aux10')
    output = Path('reports')/(args.name+'_test_certificate.json'); root = ARTIFACTS/(args.name+'_test')
    if output.exists() or root.exists():
        raise FileExistsError('Preserve the existing certificate and predictions')
    tr, te, y, folds, _, tests, proof = bank(); ids, test_ids = tr.id.to_numpy(), te.id.to_numpy()
    assert len(test_ids) == 299844 and len(np.unique(test_ids)) == len(test_ids)
    primary_path = Path('reports')/(args.primary_tag+'_primary.json')
    primary, vectors = load_primary(args.primary_tag, ids)
    assert primary['bank'] == proof
    shadow_path = Path('reports')/(args.primary_tag+'_shadow_confirmation.json'); shadow = json.loads(shadow_path.read_text())
    assert shadow['status'] == 'MATCHED_SHADOW_CONFIRMATION_PASS' and shadow['primary_report_sha256'] == file_sha256(primary_path)
    assert all(file_sha256(s) == h for s, h in shadow['source_sha256'].items())
    scope_path = Path('research/phase17_scope_20261009.json')
    assert primary['scope_sha256'] == shadow['scope_sha256'] == file_sha256(scope_path)
    total = np.zeros(len(test_ids)); contexts = []
    for k in range(5):
        path = Path('reports')/(args.test_tag+f'_f{k}.json'); r = json.loads(path.read_text()); c = r['contract']
        assert r['status'] == 'COMPLETE_FROZEN_PRIMARY_CONTEXT_TEST'
        assert c['mode'] == 'test' and c['scheme'] == 'primary' and c['fold'] == k and c['seed'] == 1201
        assert c['bank'] == proof and c['scope_sha256'] == file_sha256(scope_path)
        assert c['primary_report_sha256'] == file_sha256(primary_path) and c['confirmation_report_sha256'] == file_sha256(shadow_path)
        assert not c['entire_training_context'] and not c['outer_labels_used_for_fit_or_configuration'] and c['no_test_labels']
        assert c['train_rows'] == int((folds != k).sum()) and c['apply_rows'] == len(test_ids)
        assert c['fit_ids_sha256'] == arr_sha256(ids[folds != k]) and c['apply_ids_sha256'] == arr_sha256(test_ids)
        assert c['fold_sha256'] == arr_sha256(folds) and c['library_source_sha256'] == library_sources()
        assert all(file_sha256(s) == h for s, h in c['source_sha256'].items())
        fold_evidence = next(f for f in primary['folds'] if f['fold'] == k)
        original_path = Path(fold_evidence.get('run_report_path', Path('reports')/(args.primary_tag+f'_f{k}.json')))
        assert file_sha256(original_path) == fold_evidence['run_report_sha256']
        original = json.loads(original_path.read_text())
        assert c['primary_fold_report_sha256'] == file_sha256(original_path)
        for field in ('params', 'train_rows', 'feature_fit_sha256', 'fit_ids_sha256', 'feature_names', 'label_free_category_maps', 'checkpoint'):
            assert c[field] == original['contract'][field]
        folder = ARTIFACTS/args.test_tag/f'f{k}'
        assert np.array_equal(np.load(folder/'apply_ids.npy'), test_ids)
        assert np.array_equal(np.load(folder/'fit_ids.npy'), ids[folds != k])
        p = np.load(folder/'prediction.npy')
        assert p.shape == (len(test_ids),) and p.dtype == np.float32 and arr_sha256(p) == r['prediction_sha256']
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        config = json.loads((folder/'configuration.json').read_text())
        assert config['contract'] == c
        for field in ('config_sha256', 'pipeline_seed', 'prepared_x_sha256', 'prepared_y_sha256'):
            assert [m[field] for m in config['metadata']['members']] == [m[field] for m in r['metadata']['members']]
            assert [m[field] for m in r['metadata']['members']] == [m[field] for m in original['metadata']['members']]
        replayed = replay_members(folder, r['metadata'], c['params'], len(test_ids))
        assert np.array_equal(replayed, p), 'Saved member aggregation drift'
        total += p/5
        contexts.append({'fold': k, 'report_sha256': file_sha256(path), 'prediction_sha256': arr_sha256(p),
            'fit_ids_sha256': c['fit_ids_sha256'], 'train_rows': c['train_rows'], 'seconds': r['seconds'],
            'members': r['metadata']['members'], 'saved_member_aggregation_bitexact': True})
    route = total.astype('float32')
    aux, aux_proofs = unchanged_auxiliary_test(ids, test_ids, y, tests, proof)
    p = (1/(1+np.exp(-(logit(route)+aux)/2))).astype('float32')
    root.mkdir(); np.save(root/'route.npy', route); np.save(root/'candidate.npy', p)
    np.save(root/'aux10.npy', tests['aux10']); np.save(root/'test_ids.npy', test_ids)
    csv_path = build(p, args.name, oof_auc=primary['full_pooled_oof_auc'],
        members=[f'official_internal{primary["internal_estimators"]}_five_contexts', 'unchanged_auxiliary10'],
        notes='Frozen Phase17 recipe; complete primary and matched shadow; local certified CSV still requires robustness decision before upload')
    csv = pd.read_csv(csv_path, float_precision='round_trip')
    assert list(csv) == ['id', 'satisfaction'] and np.array_equal(csv.id.to_numpy(), test_ids)
    assert not csv.id.duplicated().any() and np.array_equal(csv.satisfaction.to_numpy(), p.astype('float64'))
    certificate = {'utc': datetime.now(timezone.utc).isoformat(), 'git': git_commit(), 'name': args.name,
        'status': 'CERTIFIED_PHASE17_TEST_REQUIRES_ROBUSTNESS_DECISION', 'bank': proof,
        'primary_tag': args.primary_tag, 'primary_report_sha256': file_sha256(primary_path),
        'shadow_report_sha256': file_sha256(shadow_path), 'scope_sha256': file_sha256(scope_path),
        'internal_estimators': primary['internal_estimators'], 'contexts': contexts,
        'test_rows': len(test_ids), 'test_ids_sha256': arr_sha256(test_ids), 'train_ids_sha256': arr_sha256(ids),
        'weights': {'route': .5, 'equal_logit_auxiliary10': .5},
        'test_policy': 'Official internal probability aggregation followed by equal probability average of five exact primary FIT contexts; unchanged auxiliary10 mean logits; fixed50/50 logit portfolio',
        'test_artifact_root': str(root), 'test_sha256': {n: arr_sha256(v) for n, v in
            (('route', route), ('candidate', p), ('aux10', tests['aux10']))},
        'oof_sha256': {n: arr_sha256(v) for n, v in vectors.items()}, 'auxiliary_proofs': aux_proofs,
        'exact_auxiliary_logits_sha256': arr_sha256(aux), 'original_test_portfolio_reproduced_bitexact': True,
        'test_correlation_vs_v6': correlations(p, tests['candidate']),
        'submission_path': str(csv_path), 'submission_sha256': file_sha256(csv_path),
        'source_sha256': {s: file_sha256(s) for s in ('scripts/assemble_phase17_test.py', 'scripts/phase17_certification.py', 'scripts/phase17_primary_io.py',
            'scripts/assemble_sol_clean.py', 'src/submission/make.py', 'scripts/phase16_tabpfn.py')},
        'uploaded': False}
    save_json(certificate, output)
    print(json.dumps({k: certificate[k] for k in ('status', 'name', 'test_rows', 'submission_sha256', 'test_correlation_vs_v6')}))


if __name__ == '__main__':
    main()
