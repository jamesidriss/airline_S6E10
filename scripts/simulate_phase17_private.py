"""Replay the exact banked 820 partition definitions against certified candidates."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json
from src.validation.compare import logit
from src.validation.private_sim import RankedAUC, summarize
from scripts.phase16_common import bank
from scripts.phase17_primary_io import load_primary
from scripts.private_lb_simulator import make_partition, segments
from scripts.audit_sol_state import reconstruct_v5
from scripts.score_sol_foundation import correlations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--certificate', action='append', required=True)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    output = Path('reports')/(args.tag+'.json'); root = ARTIFACTS/args.tag
    if output.exists() or root.exists():
        raise FileExistsError('Preserve the previous simulation')
    tr, te, y, folds, vectors, tests, proof = bank(); ids, test_ids = tr.id.to_numpy(), te.id.to_numpy()
    candidates = {'v6': vectors['candidate'], 'unchanged_aux10': vectors['aux10'], 'original_route': vectors['route']}
    certificates = {}; primaries = {}; test_correlation = {}; eligibility = {}
    for item in args.certificate:
        path = Path(item); c = json.loads(path.read_text())
        assert c['status'] == 'CERTIFIED_PHASE17_TEST_REQUIRES_ROBUSTNESS_DECISION' and c['bank'] == proof
        assert all(file_sha256(s) == h for s, h in c['source_sha256'].items())
        primary, p = load_primary(c['primary_tag'], ids)
        assert file_sha256(Path('reports')/(c['primary_tag']+'_primary.json')) == c['primary_report_sha256']
        assert c['name'] not in candidates and c['oof_sha256']['candidate'] == arr_sha256(p['candidate'])
        test = np.load(Path(c['test_artifact_root'])/'candidate.npy')
        assert test.shape == (len(test_ids),) and arr_sha256(test) == c['test_sha256']['candidate']
        assert np.array_equal(np.load(Path(c['test_artifact_root'])/'test_ids.npy'), test_ids)
        assert file_sha256(c['submission_path']) == c['submission_sha256']
        candidates[c['name']] = p['candidate']; candidates[c['name']+'_route_component'] = p['route']
        certificates[c['name']] = {'path': str(path), 'sha256': file_sha256(path)}
        primaries[c['name']] = primary
        test_correlation[c['name']] = correlations(test, tests['candidate'])
        eligibility[c['name']] = 'Certified candidate; simulation cannot replace admission or independent confirmation'
    old_path = Path('reports/sol_private_sim_recovery.json'); old = json.loads(old_path.read_text())
    assert old['portfolio_contract_sha256'] == proof['primary_report_sha256']
    assert old['fold_sha256'] == arr_sha256(folds) and old['ids_sha256'] == arr_sha256(ids)
    assert all(file_sha256(s) == h for s, h in old['source_sha256'].items())
    payload = json.loads(Path(old['split_and_draw_artifact']).read_text()); definitions = payload['definitions']
    assert len(definitions) == 820
    assert hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest() == old['split_definitions_sha256']
    _, legacy, _ = reconstruct_v5(y, folds); stress = segments(tr, legacy)
    scorers = {n: RankedAUC(y, logit(p)) for n, p in candidates.items()}
    full = {n: float(roc_auc_score(y, p)) for n, p in candidates.items()}
    assert all(abs(scorers[n].auc(np.ones(len(y)))-full[n]) < 1e-14 for n in candidates)
    draws = {n: [] for n in candidates}
    for d in definitions:
        assert d['population'] == 299844
        mask = make_partition(y, folds, d['seed'], d['mode'], d['population'],
                              stress.get(d['segment']), fold=d['public_fold'])
        assert arr_sha256(mask) == d['mask_sha256']
        public, private = mask == 1, mask == 2
        assert int(public.sum()) == d['public_rows'] and int(private.sum()) == d['private_rows']
        pb, pr = scorers['v6'].auc(public), scorers['v6'].auc(private)
        for name, scorer in scorers.items():
            draws[name].append({'mode': d['mode'], 'segment': d['segment'],
                'public_delta': scorer.auc(public)-pb, 'private_delta': scorer.auc(private)-pr})
        if (d['run']+1) % 100 == 0:
            print(f'Frozen private diagnostics {d["run"]+1}/820', flush=True)
    summary = {}
    for name, rows in draws.items():
        delta = full[name]-full['v6']
        def describe(chosen):
            return summarize([r['public_delta'] for r in chosen], [r['private_delta'] for r in chosen], delta)
        summary[name] = {'oof_auc': full[name], 'pooled_delta_vs_v6': delta, 'all_820': describe(rows),
            'modes': {m: describe([r for r in rows if r['mode'] == m]) for m in ('random_stratified', 'fold_aware', 'segment_stressed')},
            'stress_segments': {k: describe([r for r in rows if r['segment'] == k]) for k in stress},
            'test_correlation_vs_v6': test_correlation.get(name),
            'eligibility': eligibility.get(name, 'Certified component diagnostic; not a separately eligible finalist')}
    root.mkdir(); save_json({'definitions': definitions, 'draws': draws}, root/'draws.json')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'git': git_commit(), 'bank': proof,
        'status': 'COMPLETE_FROZEN_820_PRIVATE_DIAGNOSTICS', 'certificates': certificates,
        'reference': 'v6', 'total_draws': 820, 'old_report_sha256': file_sha256(old_path),
        'split_definitions_sha256': old['split_definitions_sha256'], 'every_original_mask_hash_matched': True,
        'confidence_partition_reference': 'Fixed legacy v5 defines confidence segments only; unchanged banked partitions',
        'candidates': summary, 'prediction_sha256': {n: arr_sha256(p) for n, p in candidates.items()},
        'draws_artifact': str(root/'draws.json'), 'draws_file_sha256': file_sha256(root/'draws.json'),
        'source_sha256': {s: file_sha256(s) for s in ('scripts/simulate_phase17_private.py', 'scripts/private_lb_simulator.py',
            'src/validation/private_sim.py', 'scripts/phase17_primary_io.py', 'scripts/audit_sol_state.py')},
        'limitation': 'Fixed OOF resampling; unknown private labels, refitting uncertainty and unseen shifts are not simulated. Historical v3/v5 model scores are ineligible diagnostics.'}
    save_json(result, output)
    print(json.dumps({n: summary[n]['all_820'] for n in certificates}))


if __name__ == '__main__':
    main()
