"""Bounded, preregistered saved-array discovery; no fitting or submission."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common import arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.submission.store import load_oof
from src.validation.compare import logit
from src.validation.folds import get_scheme
from src.validation.paired_auc import paired_auc, holm_adjust
from src.validation.private_sim import RankedAUC, summarize
from scripts.private_lb_simulator import make_partition
from scripts.phase17_primary_io import load_primary
from scripts.endgame.finalist_compare import drift

PREREG = ROOT / 'configs/phase21/private_reassessment_20261019.json'
PREREG_SHA = 'e819d73b7e37ff6d2f2c115ac8b39234b9f68fb715fa062dd1eed160932aa3ae'


def main():
    started = time.monotonic()
    assert file_sha256(PREREG) == PREREG_SHA
    frozen = subprocess.check_output(['git', 'show', '4704803:configs/phase21/private_reassessment_20261019.json'])
    assert frozen == PREREG.read_bytes(), 'Preregistration must precede predictive results'
    spec = json.loads(frozen)
    out = ROOT / spec['output']; assert not out.exists()
    for p, h in spec['protected_files'].items():
        assert file_sha256(p) == h, p
    def check_time():
        if time.monotonic() - started > 60 * spec['CPU_wall_limit_minutes']:
            raise TimeoutError('Preregistered CPU wall limit reached')
    train, test = load_cached_parquet()
    y = train.satisfaction.to_numpy(dtype='int8'); ids = train.id.to_numpy()
    folds = get_scheme('primary', y, ids).folds
    cert = json.loads(Path('reports/v7_sol_tabpfn_internal2_aux10_test_certificate.json').read_text())
    assert arr_sha256(ids) == cert['train_ids_sha256']
    assert arr_sha256(test.id.to_numpy()) == cert['test_ids_sha256']
    inputs = {}
    for name, record in spec['store_inputs'].items():
        a = load_oof(name)
        assert arr_sha256(a) == record['oof_sha'] and a.shape == y.shape
        assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
        inputs[name] = a
    base = inputs[spec['baseline']]; base_z = logit(base)
    assert arr_sha256(base) == spec['baseline_oof_sha256']
    primary, vectors = load_primary('phase17_B1', ids)
    assert np.array_equal(vectors['candidate'], base)
    pool = spec['candidate_pool']
    aux_z = np.mean([logit(inputs[n]) for n in pool['certified_aux_members']], axis=0)
    route_z = logit(vectors['route'])
    sigmoid = lambda z: (1 / (1 + np.exp(-z))).astype('float32')
    reference_replay = sigmoid((route_z + aux_z) / 2)
    replay_gap = float(np.max(np.abs(reference_replay - base)))
    assert replay_gap <= 6e-8, 'Source member geometry does not reproduce champion'
    candidates = []
    for member in pool['certified_aux_members']:
        for weight in pool['aux_added_weights']:
            candidates.append((f'aux_{member.removeprefix("sol_clean_aux10_primary_")}_w{weight:.2f}',
                               sigmoid((1 - weight) * base_z + weight * logit(inputs[member])), True))
    for weight in pool['route_weights']:
        candidates.append((f'route_weight_{weight:.2f}', sigmoid(weight * route_z + (1 - weight) * aux_z), True))
    neural_z = np.mean([logit(inputs[n]) for n in pool['legacy_neural_members']], axis=0)
    for weight in pool['legacy_added_weights']:
        candidates.append((f'legacy_tabm_w{weight:.2f}', sigmoid((1 - weight) * base_z + weight * neural_z), False))
    assert len(candidates) == 24
    masks = []
    for i in range(spec['simulation_repeats']):
        seed = spec['simulation_seed'] + i * 104729
        mask = make_partition(y, folds, seed, 'random_stratified', spec['population'])
        assert int((mask == 1).sum()) == spec['public_rows']
        assert int((mask == 2).sum()) == spec['private_rows']
        masks.append(mask)
    base_scorer = RankedAUC(y, base)
    base_public = np.array([base_scorer.auc(m == 1) for m in masks])
    base_private = np.array([base_scorer.auc(m == 2) for m in masks])
    base_fold = [roc_auc_score(y[folds == k], base[folds == k]) for k in range(5)]
    rows = []
    for name, candidate, certified_inputs in candidates:
        check_time()
        paired = paired_auc(y, candidate, base)
        assert abs(paired['candidate_auc'] - roc_auc_score(y, candidate)) < 1e-14
        deltas = [float(roc_auc_score(y[folds == k], candidate[folds == k]) - base_fold[k]) for k in range(5)]
        mean, se = float(np.mean(deltas)), float(np.std(deltas, ddof=1) / np.sqrt(5))
        scorer = RankedAUC(y, candidate)
        pu, pr = [], []
        for i, mask in enumerate(masks):
            check_time()
            pu.append(scorer.auc(mask == 1) - base_public[i])
            pr.append(scorer.auc(mask == 2) - base_private[i])
        segments = []
        for col in ('Class', 'Type of Travel', 'Customer Type', 'Gender'):
            for val in sorted(train[col].astype(str).unique()):
                m = train[col].astype(str).to_numpy() == val
                segments.append({'column': col, 'value': val, 'rows': int(m.sum()),
                                 'delta': float(roc_auc_score(y[m], candidate[m]) - roc_auc_score(y[m], base[m]))})
        row = {'name': name, 'prediction_sha256': arr_sha256(candidate), 'certified_source_members': certified_inputs,
               'pooled': paired, 'paired_fold_deltas': deltas, 'mean_fold_delta': mean, 'fold_se': se,
               'positive_folds': int((np.array(deltas) > 0).sum()), 'fold_t': mean / se if se else None,
               'correlation': {'spearman': float(spearmanr(candidate, base).statistic),
                               'logit': float(np.corrcoef(logit(candidate), base_z)[0, 1])},
               'private80': summarize(pu, pr, paired['delta']), 'segments': segments,
               'public_size_IID_projection': paired_auc(y, candidate, base, spec['public_rows']),
               'private_size_IID_projection': paired_auc(y, candidate, base, spec['private_rows'])}
        rows.append(row)
        print(json.dumps({'candidate': name, 'delta': paired['delta'], 'positive_folds': row['positive_folds'],
                          'private_p05': row['private80']['private_p05']}), flush=True)
    adjusted = holm_adjust([r['pooled']['one_sided_p'] for r in rows])
    nominations = []
    for row, p in zip(rows, adjusted):
        row['within_study_Holm_p'] = float(p)
        passes = (row['mean_fold_delta'] >= 1.5e-5 and row['mean_fold_delta'] >= 2.5 * row['fold_se']
                  and row['positive_folds'] == 5 and p <= .05 and row['private80']['private_p05'] >= 0)
        row['numerical_discovery_gate_pass'] = bool(passes)
        row['verdict'] = ('NOMINATE_FOR_INDEPENDENT_CONFIRMATION' if row['certified_source_members'] else
                          'NOMINATE_FOR_FRESH_FOLD_SAFE_RETRAINING') if passes else 'NO_ADMISSION'
        if passes:
            nominations.append(row['name'])
    proof = {p: file_sha256(p) for p in spec['protected_files']}
    assert proof == spec['protected_files']
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'COMPLETE_DISCOVERY_ONLY',
              'git': git_commit(), 'prereg_sha256': PREREG_SHA,
              'source_sha256': {p: file_sha256(p) for p in [__file__, 'src/validation/paired_auc.py',
                              'scripts/endgame/finalist_compare.py', 'scripts/private_lb_simulator.py']},
              'baseline_auc': float(roc_auc_score(y, base)), 'baseline_sha256': arr_sha256(base),
              'baseline_member_replay_maxabs': replay_gap, 'primary_folds_sha256': arr_sha256(folds),
              'v7_vs_v6_paired': paired_auc(y, base, inputs[spec['v6_store_name']]),
              'corrected_label_aligned_covariate_drift': drift(train, test),
              'simulation': {'population': spec['population'], 'public_rows': spec['public_rows'],
                             'private_rows': spec['private_rows'], 'seed': spec['simulation_seed'],
                             'repeats': len(masks), 'mask_sha256': [arr_sha256(m) for m in masks],
                             'limitation': 'Conditional finite OOF resampling; not private labels, refit uncertainty or protection against adaptive reuse'},
              'candidates': rows, 'nominations': nominations, 'protected_files_after': proof,
              'GPU_hours': 0, 'CPU_wall_seconds': time.monotonic() - started, 'new_submissions': 0,
              'selection_note': 'Holm covers these24 comparisons only. Independent matched confirmation remains mandatory.'}
    out.parent.mkdir(exist_ok=True)
    save_json(report, out)
    print(json.dumps({'status': report['status'], 'candidates': len(rows), 'nominations': nominations,
                      'seconds': report['CPU_wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
