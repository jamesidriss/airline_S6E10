"""Create a concrete submission decision only from complete verified evidence."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import arr_sha256, file_sha256, git_commit, save_json
from scripts.phase16_common import bank
from scripts.phase17_primary_io import load_primary
from scripts.phase16_pair_metrics import pair_rescue_damage
from scripts.score_sol_foundation import correlations


def checked_certificate(path, proof, ids, test_ids):
    c = json.loads(Path(path).read_text())
    assert c['status'] == 'CERTIFIED_PHASE17_TEST_REQUIRES_ROBUSTNESS_DECISION' and c['bank'] == proof
    assert all(file_sha256(s) == h for s, h in c['source_sha256'].items())
    primary, vectors = load_primary(c['primary_tag'], ids)
    pp = Path('reports')/(c['primary_tag']+'_primary.json'); sp = Path('reports')/(c['primary_tag']+'_shadow_confirmation.json')
    assert c['primary_report_sha256'] == file_sha256(pp) and c['shadow_report_sha256'] == file_sha256(sp)
    shadow = json.loads(sp.read_text())
    assert shadow['status'] == 'MATCHED_SHADOW_CONFIRMATION_PASS' and shadow['primary_report_sha256'] == file_sha256(pp)
    assert all(file_sha256(s) == h for s, h in shadow['source_sha256'].items())
    assert shadow['mean_paired_delta'] > 0 and min(shadow['paired_fold_deltas']) >= 0
    root = Path(c['test_artifact_root']); p = np.load(root/'candidate.npy')
    assert np.array_equal(np.load(root/'test_ids.npy'), test_ids) and p.shape == (len(test_ids),)
    assert arr_sha256(p) == c['test_sha256']['candidate'] and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert c['oof_sha256']['candidate'] == arr_sha256(vectors['candidate'])
    assert file_sha256(c['submission_path']) == c['submission_sha256']
    return c, primary, vectors['candidate'], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--certificate', required=True); ap.add_argument('--simulation', required=True)
    ap.add_argument('--role', choices=('A', 'B'), required=True)
    ap.add_argument('--against-certificate', help='Chosen A certificate; mandatory for a proposed B')
    args = ap.parse_args()
    tr, te, y, folds, _, tests, proof = bank(); ids, test_ids = tr.id.to_numpy(), te.id.to_numpy()
    c, primary, p, test = checked_certificate(args.certificate, proof, ids, test_ids)
    output = Path('reports')/(c['name']+'_decision.json')
    if output.exists(): raise FileExistsError('Preserve the pre-result decision')
    sim = json.loads(Path(args.simulation).read_text())
    assert sim['status'] == 'COMPLETE_FROZEN_820_PRIVATE_DIAGNOSTICS' and sim['bank'] == proof and sim['total_draws'] == 820
    assert sim['every_original_mask_hash_matched'] and all(file_sha256(s) == h for s, h in sim['source_sha256'].items())
    assert sim['certificates'][c['name']]['sha256'] == file_sha256(args.certificate)
    assert sim['prediction_sha256'][c['name']] == arr_sha256(p)
    contract_path = Path('research/phase17_robustness_contract_20261009.json'); contract = json.loads(contract_path.read_text())
    test_corr = correlations(test, tests['candidate'])
    original_hedge = primary['hedge_primary_admission'] and test_corr['spearman'] <= .999
    eligible = bool(primary['performance_admission'] or original_hedge)
    metrics = sim['candidates'][c['name']]['all_820']
    a = contract['A']; worst = min(s['delta'] for s in primary['segments'])
    robust = metrics['private_p05'] >= a['private_820_p05_delta_min'] and worst >= a['worst_primary_segment_delta_min']
    joint = None
    if args.role == 'A':
        passed = bool(primary['performance_admission'] and robust and metrics['private_median'] > 0)
    else:
        assert args.against_certificate, 'A genuine B requires a chosen-A comparison'
        ac, _, pa, ta = checked_certificate(args.against_certificate, proof, ids, test_ids)
        assert sim['certificates'][ac['name']]['sha256'] == file_sha256(args.against_certificate)
        h = json.loads(Path('research/phase17_joint_finalist_contract_20261009.json').read_text())['B_comparison_to_chosen_A']
        ds = [float(roc_auc_score(y[folds == k], p[folds == k])-roc_auc_score(y[folds == k], pa[folds == k])) for k in range(5)]
        pairs = pair_rescue_damage(y, p, pa); oof_corr, tc = correlations(p, pa, y), correlations(test, ta)
        segment_deltas = []
        for col in ('Class', 'Type of Travel', 'Customer Type', 'Gender'):
            for value in sorted(tr[col].astype(str).unique()):
                m = tr[col].astype(str).to_numpy() == value
                segment_deltas.append(float(roc_auc_score(y[m], p[m])-roc_auc_score(y[m], pa[m])))
        assert file_sha256(sim['draws_artifact']) == sim['draws_file_sha256']
        draws = json.loads(Path(sim['draws_artifact']).read_text())['draws']
        pr = np.array([r['private_delta'] for r in draws[c['name']]])-np.array([r['private_delta'] for r in draws[ac['name']]])
        joint = {'chosen_A': ac['name'], 'pooled_delta': float(roc_auc_score(y, p)-roc_auc_score(y, pa)),
            'mean_paired_delta': float(np.mean(ds)), 'paired_fold_deltas': ds, 'oof_correlation': oof_corr, 'test_correlation': tc,
            'pair_credit': pairs, 'worst_segment_delta': min(segment_deltas), 'private_820_p05': float(np.quantile(pr, .05))}
        passed = bool(eligible and robust and joint['pooled_delta'] >= -h['full_pooled_auc_loss_max']
            and joint['mean_paired_delta'] >= -h['mean_paired_fold_auc_loss_max'] and oof_corr['spearman'] <= h['oof_spearman_max']
            and tc['spearman'] <= h['test_spearman_max'] and pairs['rescued_pair_fraction'] >= h['rescued_pair_fraction_min']
            and pairs['rescued_pair_fraction']/max(pairs['damaged_pair_fraction'], 1e-30) >= h['rescue_damage_ratio_min']
            and joint['worst_segment_delta'] >= h['worst_segment_delta_min'] and joint['private_820_p05'] >= h['private_simulation_5th_delta_min'])
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'git': git_commit(), 'candidate': c['name'], 'intended_role': args.role,
        'verdict': 'SUBMIT_FROZEN_CERTIFIED_PHASE17_CANDIDATE' if passed else 'REJECT_NO_UPLOAD',
        'reason': 'Complete fixed primary admission, matched independent confirmation, exact test certification and frozen private robustness; no public-based choice' if passed else 'Predeclared complete eligibility or robustness/complementarity gate failed',
        'primary_auc': primary['full_pooled_oof_auc'], 'pooled_delta_vs_v6': primary['pooled_delta'],
        'certificate_sha256': file_sha256(args.certificate), 'simulation_sha256': file_sha256(args.simulation),
        'robustness_contract_sha256': file_sha256(contract_path), 'joint_contract_sha256': file_sha256('research/phase17_joint_finalist_contract_20261009.json'),
        'test_correlation_vs_v6': test_corr, 'private_820': metrics, 'worst_primary_segment_delta': worst, 'joint_B': joint,
        'submission_path': c['submission_path'], 'submission_sha256': c['submission_sha256'],
        'test_prediction_sha256': c['test_sha256']['candidate'], 'uploaded': False,
        'source_sha256': {s: file_sha256(s) for s in ('scripts/decide_phase17_submission.py', 'scripts/phase17_primary_io.py', 'scripts/phase16_pair_metrics.py')}}
    save_json(result, output)
    print(json.dumps({k: result[k] for k in ('candidate', 'intended_role', 'verdict', 'primary_auc', 'pooled_delta_vs_v6')}))


if __name__ == '__main__': main()
