"""Score frozen applied-fold predictions; no fitting or weight selection."""
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
from scripts.phase16_common import bank, SCOPE
from scripts.phase16_pair_metrics import pair_rescue_damage
from scripts.score_sol_foundation import correlations, paired_bootstrap
from scripts.evaluate_sol_foundation import classical
from scripts.replay_sol_classical import frozen_roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--fold', type=int, choices=range(5), default=0)
    args = ap.parse_args()
    path = Path('reports')/(args.tag+f'_f{args.fold}_evaluation.json')
    if path.exists():
        raise FileExistsError('Preserve measured result')
    source = Path('reports')/(args.tag+f'_f{args.fold}.json'); r = json.loads(source.read_text()); c = r['contract']
    assert r['status'] == 'COMPLETE_FROZEN_PRIMARY_FOLD'
    tr, _, y, folds, vectors, _, proof = bank(); ids = tr.id.to_numpy(); va = np.flatnonzero(folds == args.fold)
    fi = np.flatnonzero(folds != args.fold)
    assert c['fold'] == args.fold and c['scheme'] == 'primary' and c['train_rows'] == len(fi)
    assert c['fit_ids_sha256'] == arr_sha256(ids[fi]) and c['apply_ids_sha256'] == arr_sha256(ids[va])
    assert c['bank'] == proof and c['scope_sha256'] == file_sha256(SCOPE)
    assert c['seed'] == 1201 and c['params']['random_state'] == 1201 and not c['entire_training_context']
    assert not c['outer_labels_used_for_fit_or_configuration']
    assert all(file_sha256(p) == h for p, h in c['source_sha256'].items())
    root = ARTIFACTS/args.tag
    route = np.load(root/'prediction.npy'); applied_ids = np.load(root/'apply_ids.npy')
    assert np.array_equal(applied_ids, ids[va]) and arr_sha256(route) == r['prediction_sha256']
    assert route.shape == (len(va),) and np.isfinite(route).all() and ((route >= 0) & (route <= 1)).all()
    roles = [r for r in frozen_roles() if r['aux_arm']]
    assert len(roles) == 10
    aux_logits, aux_proofs = classical(args.fold, 'sol_clean_aux10_primary', roles, ids, folds, y, proof['data_sha256'])
    auxiliary = (1/(1+np.exp(-aux_logits))).astype('float32')
    assert np.array_equal(auxiliary, vectors['aux10'][va]), 'Auxiliary10 component drift'
    compose = lambda a: (1/(1+np.exp(-(logit(a)+aux_logits)/2))).astype('float32')
    assert np.array_equal(compose(vectors['route'][va]), vectors['candidate'][va]), 'Exact portfolio control drift'
    p = compose(route); base = vectors['candidate'][va]; old_route = vectors['route'][va]; labels = y[va]
    route_auc, portfolio_auc = float(roc_auc_score(labels, route)), float(roc_auc_score(labels, p))
    assert abs(route_auc-r['auc']) < 1e-14
    standalone_gain = route_auc-float(roc_auc_score(labels, old_route))
    portfolio_gain = portfolio_auc-float(roc_auc_score(labels, base))
    corr = correlations(p, base, labels); pairs = pair_rescue_damage(labels, p, base)
    assert abs(pairs['net_auc_gain']-portfolio_gain) < 1e-14
    scope = json.loads(SCOPE.read_text()); g = scope['discovery']; hedge = g['hedge_path']
    hedge_pass = portfolio_gain >= hedge['portfolio_gain_min'] and corr['spearman'] <= hedge['portfolio_oof_spearman_max'] and pairs['rescued_pair_fraction'] >= hedge['exact_rescued_pair_fraction_min']
    discovery = portfolio_gain >= g['portfolio_gain'] or (standalone_gain >= g['standalone_route_gain'] and portfolio_gain > 0) or hedge_pass
    segments = {}
    for col in scope['geometry']['segment_columns']:
        segments[col] = {}
        for value in sorted(tr[col].astype(str).unique()):
            mask = tr.iloc[va][col].astype(str).to_numpy() == value
            if len(np.unique(labels[mask])) != 2:
                continue
            segments[col][value] = {'rows': int(mask.sum()), 'auc': float(roc_auc_score(labels[mask], p[mask])),
                'delta': float(roc_auc_score(labels[mask], p[mask])-roc_auc_score(labels[mask], base[mask]))}
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'source_report_sha256': file_sha256(source),
        'scope_sha256': file_sha256(SCOPE), 'bank': proof, 'fold': args.fold, 'complete_folds': [args.fold],
        'full_pooled_oof_auc': None, 'standalone_auc': route_auc, 'standalone_delta': standalone_gain,
        'portfolio_auc': portfolio_auc, 'portfolio_delta': portfolio_gain,
        'oof_correlation_vs_v6': corr, 'route_correlation_vs_B0': correlations(route, old_route, labels),
        'exact_pair_rescue_damage_vs_v6': pairs, 'exact_route_pair_rescue_damage': pair_rescue_damage(labels, route, old_route),
        'bootstrap': paired_bootstrap(labels, {'candidate': p, 'v6': base}, reference='v6'),
        'segments': segments, 'discovery_pass': bool(discovery), 'hedge_discovery_pass': bool(hedge_pass),
        'replication_pass': bool(portfolio_gain >= 15e-6 or (standalone_gain >= 50e-6 and portfolio_gain > 0)),
        'portfolio_prediction_sha256': arr_sha256(p), 'exact_auxiliary_logits_sha256': arr_sha256(aux_logits),
        'unchanged_auxiliary_proofs': aux_proofs, 'runtime_seconds': r['seconds'],
        'source_sha256': {p: file_sha256(p) for p in ('scripts/evaluate_phase16_fold.py', 'scripts/phase16_pair_metrics.py')},
        'verdict': 'DISCOVERY_PASS_REQUIRES_FROZEN_REPLICATION' if discovery else 'DISCOVERY_FAIL_CLOSE_NO_PROMOTION'}
    np.save(root/'portfolio.npy', p); save_json(result, path)
    with Path('experiments/ledger.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'exp_id': args.tag+f'_f{args.fold}', 'kind': 'FROZEN_PRIMARY_DISCOVERY',
            'ts': result['utc'], 'report': str(path), 'report_sha256': file_sha256(path),
            'paired_fold_deltas': [portfolio_gain], 'correlation_with_champion': corr,
            'auc': portfolio_auc, 'complete_folds': [args.fold], 'full_oof': False, 'verdict': result['verdict']})+'\n')
    print(json.dumps({k: result[k] for k in ('standalone_auc', 'standalone_delta', 'portfolio_auc',
        'portfolio_delta', 'discovery_pass', 'replication_pass', 'verdict')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
