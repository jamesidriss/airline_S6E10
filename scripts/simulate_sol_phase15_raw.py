"""Frozen-OOF private split diagnostics for the completed raw/route candidate."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.private_sim import RankedAUC, summarize
from scripts.assemble_sol_foundation_test import verify_primary, checked_probability
from scripts.private_lb_simulator import make_partition, segments, SEED
from scripts.audit_sol_state import reconstruct_v5, pair_diagnostic
from scripts.score_sol_foundation import correlations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', default='reports/sol_phase15_raw_primary.json')
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    output = REPORTS / (args.name + '.json')
    if output.exists():
        raise FileExistsError('Preserve previous simulation evidence')
    tr, _ = load_cached_parquet()
    y, ids = tr.satisfaction.to_numpy(dtype='int8'), tr.id.to_numpy()
    dh = {s: file_sha256(f'data/raw/{s}.csv') for s in ('train', 'test')}
    _, base, folds = verify_primary('reports/sol_route_aux10_primary_final.json', ids, y, dh)
    path = Path(args.primary)
    r = json.loads(path.read_text())
    assert r['scope'] == 'full primary OOF' and len(r['folds']) == 5
    assert [f['fold'] for f in r['folds']] == list(range(5))
    assert r['weights'] == {'route': .375, 'raw': .125, 'aux10': .5}
    assert r['fold_sha256'] == arr_sha256(folds) and r['data_sha256'] == dh
    assert r['selected_ids_sha256'] == arr_sha256(ids)
    assert all(file_sha256(p) == h for p, h in r['source_sha256'].items())
    root = ARTIFACTS / path.stem
    candidate = checked_probability(root / 'candidate_oof.npy', root / 'train_ids.npy', r['selected_oof_sha256'], ids)
    assert abs(roc_auc_score(y, candidate) - r['selected_pooled_auc']) < 1e-14
    vectors = {'v6': base['candidate'], 'raw_route': candidate}
    scorers = {k: RankedAUC(y, p) for k, p in vectors.items()}
    full = {k: float(roc_auc_score(y, p)) for k, p in vectors.items()}
    _, legacy, _ = reconstruct_v5(y, folds)
    stress = segments(tr, legacy)
    plans = [('random_stratified', None, j) for j in range(200)]
    plans += [('fold_aware', None, j) for j in range(200)]
    plans += [('segment_stressed', key, j) for key in sorted(stress) for j in range(30)]
    draws, definitions = [], []
    for run, (mode, key, j) in enumerate(plans):
        seed = SEED + run * 104729
        mask = make_partition(y, folds, seed, mode, 299844, stress.get(key), fold=j % 5)
        public, private = mask == 1, mask == 2
        pb, pr = scorers['v6'].auc(public), scorers['v6'].auc(private)
        draws.append({'mode': mode, 'segment': key,
                      'public_delta': scorers['raw_route'].auc(public) - pb,
                      'private_delta': scorers['raw_route'].auc(private) - pr})
        definitions.append({'run': run, 'mode': mode, 'segment': key, 'seed': seed,
                            'public_fold': j % 5 if mode == 'fold_aware' else None,
                            'mask_sha256': arr_sha256(mask), 'public_rows': int(public.sum()),
                            'private_rows': int(private.sum())})
        if (run + 1) % 100 == 0:
            print(f'private split diagnostics {run + 1}/{len(plans)}', flush=True)
    pooled_delta = full['raw_route'] - full['v6']
    def diagnostic(rows):
        return summarize([d['public_delta'] for d in rows], [d['private_delta'] for d in rows], pooled_delta)
    segment_metrics = []
    for column in ('Class', 'Type of Travel', 'Customer Type', 'Gender'):
        for level in sorted(tr[column].unique()):
            keep = (tr[column] == level).to_numpy()
            a, b = roc_auc_score(y[keep], candidate[keep]), roc_auc_score(y[keep], vectors['v6'][keep])
            segment_metrics.append({'column': column, 'level': str(level), 'rows': int(keep.sum()),
                                    'candidate_auc': float(a), 'v6_auc': float(b), 'delta': float(a-b)})
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'git': git_commit(),
              'primary_report_sha256': file_sha256(path), 'baseline_report_sha256': file_sha256('reports/sol_route_aux10_primary_final.json'),
              'source_sha256': {p: file_sha256(p) for p in ('scripts/simulate_sol_phase15_raw.py', 'scripts/private_lb_simulator.py', 'src/validation/private_sim.py')},
              'data_sha256': dh, 'fold_sha256': arr_sha256(folds), 'ordered_ids_sha256': arr_sha256(ids),
              'prediction_sha256': {k: arr_sha256(p) for k, p in vectors.items()},
              'seed': SEED, 'population': 299844, 'public_fraction': .2, 'stress_odds_multiplier': 4,
              'confidence_partition_reference': 'fixed legacy v5, unchanged from previous campaign simulation',
              'total_draws': len(draws), 'full_auc': full, 'pooled_delta_vs_v6': pooled_delta,
              'modes': {m: diagnostic([d for d in draws if d['mode'] == m]) for m in ('random_stratified', 'fold_aware', 'segment_stressed')},
              'stress_segments': {k: diagnostic([d for d in draws if d['segment'] == k]) for k in stress},
              'segment_metrics': segment_metrics, 'correlation_vs_v6': correlations(candidate, vectors['v6']),
              'pair_rescue_damage': pair_diagnostic(y, candidate, vectors['v6']),
              'split_definitions_sha256': arr_sha256(np.frombuffer(json.dumps(definitions, sort_keys=True).encode(), dtype='uint8')),
              'disclaimer': 'Fixed OOF predictions only. Resampling does not simulate refitting, unknown private labels, distribution shift or future competition rank. It cannot override the predeclared admission gate.',
              'test_correlation_vs_v6': None, 'test_correlation_status': 'Requires separately certified full test inference; no OOF proxy supplied'}
    save_json(result, output)
    print(json.dumps({'pooled_delta_vs_v6': pooled_delta, 'total_draws': len(draws), 'modes': result['modes']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
