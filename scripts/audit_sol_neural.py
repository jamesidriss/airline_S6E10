"""Quantify historical neural contribution and withdraw invalid selection CV."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.submission import store
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.audit_sol_state import reconstruct_v5, pair_diagnostic


def main():
    tr, _ = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    manifest = json.loads((REPORTS / 'finalist_v3_final.json').read_text(encoding='utf-8'))
    neural = [m for m in manifest['members'] if m['family'] in ('realmlp', 'tabm')]
    assert len(neural) == 12 and sum(m['family'] == 'realmlp' for m in neural) == 10
    _, v5, _ = reconstruct_v5(y, folds)
    z = np.zeros(len(y))
    for m in neural:
        p = store.load_oof(m['exp_id'])
        m['prediction_sha256'] = arr_sha256(p)
        z += logit(p)
    classical = (59 * v5 - z) / 47
    comparison = []
    for k in range(5):
        mask = folds == k
        a, b = roc_auc_score(y[mask], v5[mask]), roc_auc_score(y[mask], classical[mask])
        comparison.append({'fold': k, 'legacy_v5_auc': a, 'legacy_classical47_auc': b,
                           'classical_minus_v5': b - a})
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'git': git_commit(),
        'status': 'REJECT_LEGACY_NEURAL_OUTER_CHECKPOINT_SELECTION', 'affected_members': neural,
        'affected_count': 12, 'legacy_v5_auc': float(roc_auc_score(y, v5)),
        'legacy_neural12_auc': float(roc_auc_score(y, z)),
        'legacy_classical47_auc': float(roc_auc_score(y, classical)), 'paired_fold_metrics': comparison,
        'classical_logit_corr_v5': float(np.corrcoef(classical, v5)[0,1]),
        'pair_diagnostic': pair_diagnostic(y, classical, v5),
        'evidence': {'realmlp_use_early_stopping': False, 'realmlp_use_best_epoch_default': True,
                    'realmlp_checkpoint_callback_restores_best': True, 'tabm_checkpoint_restores_best': True,
                    'legacy_validation_inputs': 'outer evaluation rows and labels',
                    'fixed_validation_inputs': 'stratified inner10 percent of outer FIT, seed1',
                    'twins_old': 'separately factorized category codes; vocabularies could alias values',
                    'twins_new': 'literal rounded numeric values; estimator FIT-only ordinal mapping'},
        'source_sha256': {p: file_sha256(p) for p in ('src/models/realmlp.py',
                 '.venv/Lib/site-packages/pytabkit/models/training/nn_creator.py',
                 '.venv/Lib/site-packages/pytabkit/models/training/lightning_modules.py',
                 '.venv/Lib/site-packages/pytabkit/models/alg_interfaces/tabm_interface.py')},
        'fold_sha256': arr_sha256(folds), 'ordered_ids_sha256': arr_sha256(ids),
        'limitation': 'Fixed legacy-vector diagnostics only. Classical47 still includes historical own-label TE priors. Neither group is a clean reconstructed finalist; no new ensemble admission.'}
    path = REPORTS / 'sol_neural_audit.json'
    save_json(report, path)
    ledger = Path('experiments/ledger.jsonl')
    existing = {r.get('exp_id', r.get('id')) for r in map(json.loads, ledger.read_text(encoding='utf-8').splitlines())}
    eid = 'sol_neural_outer_validation_audit'
    if eid not in existing:
        with ledger.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'exp_id': eid, 'ts': report['updated_utc'], 'git': git_commit(),
                'verdict': report['status'], 'paired_fold_deltas': [r['classical_minus_v5'] for r in comparison],
                'corr_with_champion': report['classical_logit_corr_v5'], 'report': str(path),
                'report_sha256': file_sha256(path), 'affected_count': 12,
                'limitation': report['limitation']}) + '\n')
    finalist_path = Path('experiments/private_finalists.json')
    finalists = json.loads(finalist_path.read_text(encoding='utf-8'))
    finalists['updated_utc'] = report['updated_utc']
    finalists['BEST_A']['status'] = 'BANKED_LEGACY_ONLY; OOF_AND_TEST_CONTRACT_REPAIR_REQUIRED'
    finalists['BEST_A']['limitations'] = ['12 neural members select checkpoints on outer labels',
        'historical TE prior sees each fit row label', 'six auxiliary XT test configurations collapsed; auxiliary/test iteration mismatch']
    finalists['verified_clean_BEST_A'] = None
    finalists['final_selection_locked'] = False
    save_json(finalists, finalist_path)
    print(json.dumps({k: report[k] for k in ('affected_count', 'legacy_v5_auc', 'legacy_neural12_auc', 'legacy_classical47_auc', 'paired_fold_metrics')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
