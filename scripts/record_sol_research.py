"""Recompute completed SOL fold metrics; append evidence without rewriting history."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.folds import get_scheme
from src.validation.compare import logit
from src.submission import store
from scripts.audit_sol_state import reconstruct_v5
from scripts.run_phase14c import counter_path
from scripts.run_phase14 import XGB_BASE, CAT_BASE
from scripts.run_phase12 import CHAMPION_PARAMS, CAT_PARAMS

FOUNDATION_TAGS = ('sol_tabpfn35_verified', 'sol_tabpfn35_memory', 'sol_tabpfn35_solo',
                  'sol_tabpfn35_query', 'sol_tabpfn35_query_full', 'sol_tabpfn35_query_serial',
                  'sol_tabpfn35_reuse_control', 'sol_tabpfn35_reuse', 'sol_tabpfn35_reuse_full',
                  'sol_tabpfn35_decoder', 'sol_tabpfn35_decoder_full',
                  'sol_tabpfn35_gelu', 'sol_tabpfn35_gelu_full', 'sol_tabpfn35_predict_guard',
                  'sol_tabpfn35_route', 'sol_tabpfn35_route_reserve')


def main():
    tr, _ = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    v3, v5, _ = reconstruct_v5(y, folds)
    ledger = Path('experiments/ledger.jsonl')
    existing = {r.get('id', r.get('exp_id')) for r in map(json.loads, ledger.read_text(encoding='utf-8').splitlines())}
    data_hash = {s: file_sha256(f'data/raw/{s}.csv') for s in ('train','test')}
    verified = []
    repair_sum = {k: v5[folds == k].copy() for k in range(5)}
    repair_counts = {k: 0 for k in range(5)}
    now = datetime.now(timezone.utc).isoformat()
    with ledger.open('a', encoding='utf-8') as out:
        for tag in ('sol_repair', 'sol_b', 'sol_b_xt', 'sol_a_ladder', 'sol_neural_clean', 'sol_neural_clean_compact') + FOUNDATION_TAGS:
            for path in sorted((REPORTS / tag).glob('*.json')):
                rec = json.loads(path.read_text(encoding='utf-8'))
                if 'prediction_sha256' not in rec:
                    continue
                c = rec.get('contract', rec)
                k = c['fold']
                va = np.flatnonzero(folds == k)
                pred = np.load(ARTIFACTS / tag / (path.stem + '.npy'))
                assert arr_sha256(pred) == rec['prediction_sha256']
                expected_ids = c.get('validation_ids_sha256')
                assert expected_ids is None or expected_ids == arr_sha256(ids[va])
                assert pred.shape == (len(va),) and np.isfinite(pred).all()
                auc = float(roc_auc_score(y[va], pred))
                assert abs(auc - rec['auc']) < 1e-14
                member = c.get('spec', c).get('member')
                if tag.startswith('sol_tabpfn'):
                    blended = (59 * v5[va] + logit(pred)) / 60
                    params = c['params']
                else:
                    old = store.load_oof(member)[va] if tag.startswith('sol_neural_clean') else np.load(counter_path(member, k))
                    blended = v5[va] + (logit(pred) - logit(old)) / 59
                    params = c.get('params', c.get('spec', {}).get('params', {}))
                    if tag == 'sol_repair':
                        spec = c['spec']
                        if spec['family'] == 'lgbm_xt':
                            params = dict(CHAMPION_PARAMS, **CAT_PARAMS)
                            params.update(spec['params'])
                            params.update(random_state=spec['seed'], bagging_seed=spec['seed']+1, feature_fraction_seed=spec['seed']+2)
                        elif spec['family'] == 'xgb':
                            params = dict(XGB_BASE, **spec['params'], random_state=spec['seed'])
                        else:
                            params = dict(CAT_BASE, **spec['params'], random_seed=spec['seed'], boosting_type='Plain', early_stopping_rounds=300)
                        repair_sum[k] += (logit(pred) - logit(old)) / 59
                        repair_counts[k] += 1
                marginal = float(roc_auc_score(y[va], blended) - roc_auc_score(y[va], v5[va]))
                corr = float(np.corrcoef(logit(pred), v5[va])[0,1])
                row = {'exp_id': f'{tag}_{path.stem}', 'ts': now, 'git': rec.get('git', git_commit()),
                       'scope': 'single primary fold, not full OOF', 'fold': k, 'validation_auc': auc,
                       'paired_fold_deltas': [marginal], 'comparison': 'actual historical v5 slot or append one fixed equal-logit member',
                       'corr_with_champion': corr, 'params': params,
                       'seed': c.get('seed', c.get('spec', {}).get('seed', params.get('random_state', params.get('random_seed')))),
                       'train_rows': c.get('train_rows'), 'validation_rows': len(va), 'test_policy': c.get('test_policy'),
                       'duration_s': rec['seconds'], 'data_hash': data_hash, 'fold_hash': arr_sha256(folds),
                       'prediction_hash': arr_sha256(pred), 'report_hash': file_sha256(path), 'report': str(path.relative_to(Path.cwd())) if path.is_absolute() else str(path),
                       'contract': c, 'verdict': rec['status']}
                if tag == 'sol_repair':
                    row['verdict'] = 'REPAIRED_BASELINE_REQUIRES_ALL_FOLDS'
                    row['report_correction'] = 'Original operational_v5_slot_delta field compared A0 to itself. This entry recomputes replacement of the actual historical auxiliary slot.'
                if tag.startswith('sol_neural_clean'):
                    row['comparison'] = 'Corrected neural contract versus a legacy outer-selected slot, diagnostic only; not an honest matched comparison or admission evidence'
                verified.append({key: row[key] for key in ('exp_id','fold','validation_auc','paired_fold_deltas','corr_with_champion','verdict')})
                if row['exp_id'] not in existing:
                    out.write(json.dumps(row, default=str) + '\n')
        # Timing failures remain visible, distinct from scientific negatives.
        for tag in ('sol_tabpfn35', 'sol_tabpfn35_b64', 'sol_tabpfn35_efficient', 'sol_tabpfn35_iclbf16', 'sol_tabpfn35_compact', 'sol_tabpfn35_bounded', 'sol_neural_clean', 'sol_neural_clean_compact') + FOUNDATION_TAGS:
            for path in sorted((REPORTS / tag).glob('*.json')):
                r = json.loads(path.read_text(encoding='utf-8'))
                if 'prediction_sha256' in r:
                    continue
                eid = f'{tag}_{path.stem}'
                if eid not in existing:
                    out.write(json.dumps({'exp_id': eid, 'ts': now, 'git': r['git'], 'kind': 'RESOURCE_PROBE',
                                          'verdict': r['status'], 'report_hash': file_sha256(path), 'report': str(path), **r}, default=str) + '\n')
                if r['status'] == 'INVALID_IMPLEMENTATION_NOT_NEGATIVE' and any(
                        phrase in r.get('error', '') for phrase in ('available host RAM', 'free disk reserve')):
                    correction_id = eid + '_resource_classification'
                    if correction_id not in existing:
                        out.write(json.dumps({'exp_id': correction_id, 'ts': now, 'git': git_commit(),
                            'verdict': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE', 'corrects': eid,
                            'reason': 'Historical generic exception handler classified a resource preflight refusal as implementation failure; no CV predictions exist',
                            'report_hash': file_sha256(path), 'report': str(path)}) + '\n')
            abort = ARTIFACTS / tag / 'resource_abort.json'
            eid = f'{tag}_guard_abort'
            if abort.exists() and eid not in existing:
                r = json.loads(abort.read_text(encoding='utf-8'))
                out.write(json.dumps({'exp_id': eid, 'ts': now, 'kind': 'RESOURCE_PROBE',
                    'verdict': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE', 'report_hash': file_sha256(abort),
                    'report': str(abort), 'contract_and_telemetry': r,
                    'reason': 'Own worker stopped before completed predictions; no model-performance verdict'}, default=str) + '\n')
    combined = {str(k): {'completed_slots': repair_counts[k],
                         'auc': float(roc_auc_score(y[folds == k], repair_sum[k])),
                         'delta_vs_legacy_v5': float(roc_auc_score(y[folds == k], repair_sum[k])-roc_auc_score(y[folds == k], v5[folds == k])),
                         'delta_vs_v3': float(roc_auc_score(y[folds == k], repair_sum[k])-roc_auc_score(y[folds == k], v3[folds == k]))}
                for k in range(5) if repair_counts[k]}
    save_json({'updated_utc': now, 'verified_records': verified, 'repaired_blend_partial': combined,
               'limitation': 'Unreplaced legacy members retain historical inner crossfit TE priors; all 12 legacy neural members also selected checkpoints on outer evaluation labels and factorized twins separately. Full strict finalist reproduction is not complete.'}, REPORTS / 'sol_metrics_verified.json')
    print(json.dumps({'verified_records': len(verified), 'repaired_blend_partial': combined}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
