"""Matched frozen-recipe confirmation on the two predeclared alternative folds."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, save_json
from src.validation.compare import logit
from src.validation.folds import get_scheme
from scripts.phase16_common import bank
from scripts.evaluate_sol_foundation import foundation, classical
from scripts.replay_sol_classical import frozen_roles
from scripts.phase16_pair_metrics import pair_rescue_damage
from scripts.score_sol_foundation import correlations, paired_bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary-tag', required=True)
    ap.add_argument('--candidate-tag', required=True)
    ap.add_argument('--control-tag', default='phase17_shadow_B0')
    ap.add_argument('--auxiliary-tag', default='phase17_shadow_aux10')
    args = ap.parse_args()
    output = Path('reports')/(args.primary_tag+'_shadow_confirmation.json')
    if output.exists():
        raise FileExistsError('Preserve the confirmation measurement')
    pp = Path('reports')/(args.primary_tag+'_primary.json'); primary = json.loads(pp.read_text())
    assert primary['status'] == 'PRIMARY_PASS_REQUIRES_INDEPENDENT_CONFIRMATION_AND_TEST'
    assert all(file_sha256(s) == h for s, h in primary['source_sha256'].items())
    tr, _, y, _, _, _, proof = bank(); ids = tr.id.to_numpy()
    assert primary['bank'] == proof
    scope = Path('research/phase17_scope_20261009.json')
    assert primary['scope_sha256'] == file_sha256(scope)
    folds = get_scheme('shadow', y, ids).folds
    assert arr_sha256(folds) == '5d181e32357e67b2ad8381c33549ef905bb45b969de7b8e09a3e4e7cb24af758'
    roles = [r for r in frozen_roles() if r['aux_arm']]; assert len(roles) == 10
    for role in roles:
        role.update(source_scheme=role['scheme'], scheme='shadow')
    records = []; combined = []; control_combined = []; labels = []
    for k in (2, 3):
        rp = Path('reports')/(args.candidate_tag+f'_f{k}.json'); r = json.loads(rp.read_text()); c = r['contract']
        va, fi = np.flatnonzero(folds == k), np.flatnonzero(folds != k)
        assert r['status'] == 'COMPLETE_FROZEN_SHADOW_FOLD' and c['mode'] == 'shadow' and c['scheme'] == 'shadow' and c['fold'] == k
        assert c['bank'] == proof and c['scope_sha256'] == file_sha256(scope)
        assert c['primary_report_sha256'] == file_sha256(pp) and c['seed'] == 1201
        assert c['train_rows'] == len(fi) and c['apply_rows'] == len(va)
        assert c['fit_ids_sha256'] == arr_sha256(ids[fi]) and c['apply_ids_sha256'] == arr_sha256(ids[va])
        assert c['fold_sha256'] == arr_sha256(folds) and not c['outer_labels_used_for_fit_or_configuration']
        assert not c['entire_training_context'] and all(file_sha256(s) == h for s, h in c['source_sha256'].items())
        folder = ARTIFACTS/args.candidate_tag/f'f{k}'
        route = np.load(folder/'prediction.npy')
        assert np.array_equal(np.load(folder/'apply_ids.npy'), ids[va]) and arr_sha256(route) == r['prediction_sha256']
        assert route.shape == (len(va),) and np.isfinite(route).all() and ((route >= 0) & (route <= 1)).all()
        assert abs(float(roc_auc_score(y[va], route))-r['auc']) < 1e-14
        old, control, cp = foundation(k, [args.control_tag], ids, folds, y, proof['data_sha256'], 'shadow')
        assert control['status'] == 'FROZEN_INDEPENDENT_CONFIRMATION_PARTIAL' and not control['head_views'] and not control['query_activation_reuse']
        assert c['params'] == {**control['params'], 'n_estimators': primary['internal_estimators']}
        for field in ('feature_names', 'feature_fit_sha256', 'label_free_category_maps', 'checkpoint'):
            assert c[field] == control[field]
        assert c['feature_apply_sha256'] == control['feature_val_sha256']
        aux, apf = classical(k, args.auxiliary_tag, roles, ids, folds, y, proof['data_sha256'])
        compose = lambda p: (1/(1+np.exp(-(logit(p)+aux)/2))).astype('float32')
        p, base = compose(route), compose(old)
        np.save(folder/'portfolio.npy', p); np.save(folder/'matched_control_portfolio.npy', base)
        delta = float(roc_auc_score(y[va], p)-roc_auc_score(y[va], base))
        pairs = pair_rescue_damage(y[va], p, base)
        assert abs(pairs['net_auc_gain']-delta) < 1e-14
        segments = []
        for col in ('Class', 'Type of Travel', 'Customer Type', 'Gender'):
            for value in sorted(tr[col].astype(str).unique()):
                m = tr.iloc[va][col].astype(str).to_numpy() == value
                segments.append({'column': col, 'value': value, 'rows': int(m.sum()),
                    'delta': float(roc_auc_score(y[va][m], p[m])-roc_auc_score(y[va][m], base[m]))})
        records.append({'fold': k, 'route_auc': r['auc'], 'control_route_auc': control['auc'],
            'portfolio_auc': float(roc_auc_score(y[va], p)), 'control_portfolio_auc': float(roc_auc_score(y[va], base)),
            'portfolio_delta': delta, 'correlation': correlations(p, base, y[va]), 'pairs': pairs, 'segments': segments,
            'candidate_report_sha256': file_sha256(rp), 'candidate_prediction_sha256': arr_sha256(route),
            'portfolio_prediction_sha256': arr_sha256(p), 'control_portfolio_prediction_sha256': arr_sha256(base),
            'fit_ids_sha256': c['fit_ids_sha256'], 'apply_ids_sha256': c['apply_ids_sha256'],
            'auxiliary_mean_logits_sha256': arr_sha256(aux), 'control_proof': cp, 'auxiliary_proofs': apf,
            'candidate_seconds': r['seconds'], 'control_seconds': control['seconds']})
        combined.append(p); control_combined.append(base); labels.append(y[va])
    deltas = [r['portfolio_delta'] for r in records]
    passed = bool(min(deltas) >= 0 and np.mean(deltas) > 0)
    yl, p, b = np.concatenate(labels), np.concatenate(combined), np.concatenate(control_combined)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'bank': proof, 'scheme': 'shadow',
        'scope': 'Two predeclared matched alternative folds; not full OOF or unseen labels',
        'folds': records, 'paired_fold_deltas': deltas, 'mean_paired_delta': float(np.mean(deltas)),
        'pooled_selected_fold_delta': float(roc_auc_score(yl, p)-roc_auc_score(yl, b)),
        'bootstrap': paired_bootstrap(yl, {'candidate': p, 'control': b}, reference='control'),
        'primary_report_sha256': file_sha256(pp), 'scope_sha256': file_sha256(scope),
        'source_sha256': {s: file_sha256(s) for s in ('scripts/evaluate_phase17_shadow.py',
            'scripts/evaluate_sol_foundation.py', 'scripts/phase16_pair_metrics.py', 'scripts/score_sol_foundation.py')},
        'status': 'MATCHED_SHADOW_CONFIRMATION_PASS' if passed else 'MATCHED_SHADOW_CONFIRMATION_FAIL_NO_PROMOTION'}
    save_json(result, output)
    row = {'exp_id': args.primary_tag+'_shadow_confirmation', 'kind': 'FROZEN_MATCHED_SHADOW_CONFIRMATION',
        'ts': result['utc'], 'report': str(output), 'report_sha256': file_sha256(output),
        'paired_fold_deltas': deltas, 'correlation_with_champion': [r['correlation'] for r in records],
        'full_oof': False, 'verdict': result['status']}
    with Path('experiments/ledger.jsonl').open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'paired_fold_deltas', 'mean_paired_delta')}))


if __name__ == '__main__':
    main()
