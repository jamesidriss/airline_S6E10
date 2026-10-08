"""Predeclared fixed-vector sensitivity and complementarity; no model fitting."""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.validation.compare import logit
from src.submission import store
from scripts.assemble_sol_foundation_test import verify_primary, checked_probability
from scripts.audit_sol_state import reconstruct_v5, pair_diagnostic
from scripts.score_sol_foundation import paired_bootstrap, correlations
from scripts.run_sol_tabpfn import frames

SCOPE = Path('research/sol_phase15_scope_20261008.json')
OUTPUT = REPORTS / 'sol_phase15_zero_training_20261008.json'


def raw_control(tr, te, ids, y, folds, data_hash):
    path = REPORTS / 'sol_tabpfn35_predict_guard/raw_f0.json'
    r = json.loads(path.read_text(encoding='utf-8'))
    route = json.loads((REPORTS / 'sol_tabpfn35_route/route_f0.json').read_text(encoding='utf-8'))
    fi, va = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    assert r['arm'] == 'raw' and r['fold'] == 0 and not r['timing_only'] and r['full_intended_population']
    assert r['train_rows'] == len(fi) and r['validation_rows'] == len(va)
    assert r['data_sha256'] == data_hash and r['fold_sha256'] == arr_sha256(folds)
    assert r['fit_ids_sha256'] == arr_sha256(ids[fi]) and r['validation_ids_sha256'] == arr_sha256(ids[va])
    assert r['params'] == dict(route['params'], categorical_features_indices=[13, 14, 15, 16])
    for key in ('checkpoint', 'library_version', 'seed', 'precision', 'icl_bf16', 'decoder_inplace_gelu',
                'inference_chunk_cells', 'inference_col_chunk_size', 'batch_size', 'gpu_memory_fraction'):
        assert r[key] == route[key], key
    for source, key in [('scripts/run_sol_tabpfn.py', 'source_sha256'),
                        ('src/models/windows_attention.py', 'backend_source_sha256'),
                        ('src/models/pointwise_inference.py', 'decoder_chunk_source_sha256'),
                        ('src/models/resource_guard.py', 'resource_guard_source_sha256')]:
        assert file_sha256(source) == r[key]
    assert file_sha256(r['params']['model_path']) == r['checkpoint']['checkpoint_sha256']
    x, _, names, cats, maps = frames(tr, te, False)
    assert names == r['feature_names'] and cats == r['params']['categorical_features_indices']
    assert maps == r['label_free_category_maps']
    assert arr_sha256(x[fi]) == r['feature_fit_sha256'] and arr_sha256(x[va]) == r['feature_val_sha256']
    p = checked_probability(ARTIFACTS / path.parent.name / 'raw_f0.npy',
                            ARTIFACTS / path.parent.name / 'ids_f0.npy', r['prediction_sha256'], ids[va])
    assert abs(roc_auc_score(y[va], p) - r['auc']) < 1e-14
    return p, {'report': str(path), 'report_sha256': file_sha256(path), 'prediction_sha256': arr_sha256(p),
               'validation_ids_sha256': arr_sha256(ids[va]), 'auc': r['auc'], 'matched_protocol': True}


def evidence(y, folds, candidate, control, frame, scope):
    ds = [float(roc_auc_score(y[folds == k], candidate[folds == k]) -
                roc_auc_score(y[folds == k], control[folds == k])) for k in sorted(set(folds))]
    segments = []
    for col in scope['audit']['segment_columns']:
        for value in sorted(frame[col].dropna().unique()):
            mask = (frame[col] == value).to_numpy()
            if len(np.unique(y[mask])) < 2:
                continue
            segments.append({'column': col, 'value': str(value), 'rows': int(mask.sum()),
                             'auc': float(roc_auc_score(y[mask], candidate[mask])),
                             'delta': float(roc_auc_score(y[mask], candidate[mask]) - roc_auc_score(y[mask], control[mask]))})
    return {'pooled_auc': float(roc_auc_score(y, candidate)), 'pooled_delta_vs_v6': float(roc_auc_score(y, candidate) - roc_auc_score(y, control)),
            'paired_fold_deltas': ds, 'mean_paired_gain': float(np.mean(ds)),
            'paired_se': float(np.std(ds, ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else None,
            'positive_folds': int(np.sum(np.array(ds) > 0)), 'segments': segments,
            'oof_correlation_vs_v6': correlations(candidate, control, y),
            'pair_rescue_damage': pair_diagnostic(y, candidate, control, n=scope['audit']['pair_samples']),
            'prediction_sha256': arr_sha256(candidate)}


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the prior audit')
    start = time.monotonic()
    scope = json.loads(SCOPE.read_text(encoding='utf-8'))
    index = json.loads((REPORTS / 'sol_final_recovery_proof_index_20261008.json').read_text())
    for path, expected in index['evidence_file_sha256'].items():
        assert file_sha256(path) == expected, path
    tr, te = load_cached_parquet()
    ids, test_ids, y = tr.id.to_numpy(), te.id.to_numpy(), tr.satisfaction.to_numpy(dtype='int8')
    data_hash = {s: file_sha256(f'data/raw/{s}.csv') for s in ('train', 'test')}
    primary, vectors, folds = verify_primary('reports/sol_route_aux10_primary_final.json', ids, y, data_hash)
    cert_path = REPORTS / 'sol_route_aux10_primary_final_test_certificate.json'
    cert = json.loads(cert_path.read_text()); root = Path(cert['test_artifact_root'])
    assert cert['data_sha256'] == data_hash and cert['ordered_test_ids_sha256'] == arr_sha256(test_ids)
    assert cert['primary_report_sha256'] == file_sha256('reports/sol_route_aux10_primary_final.json')
    tests = {n: checked_probability(root / (n + '.npy'), root / 'test_ids.npy', h, test_ids)
             for n, h in cert['test_sha256'].items()}
    control = vectors['candidate']; cv, rows = {'v6': control}, {}
    for w in scope['audit']['route_weights']:
        name = f'route_{int(w*100)}'
        p = expit(w * logit(vectors['route']) + (1-w) * logit(vectors['aux10'])).astype('float32')
        if w == .5:
            assert np.max(np.abs(p - control)) <= 2e-6
            # Use the exact banked vector as the comparison baseline.
            p = control
        cv[name] = p
        rows[name] = evidence(y, folds, p, control, tr, scope)
        test = expit(w * logit(tests['route']) + (1-w) * logit(tests['aux10'])).astype('float32')
        rows[name].update(eligibility='CERTIFIED_COMPONENTS_SENSITIVITY_ONLY', test_correlation_vs_v6=correlations(test, tests['candidate']),
                          test_prediction_sha256=arr_sha256(test))
    _, old_v5, _ = reconstruct_v5(y, folds)
    manifest = json.loads((REPORTS / 'finalist_v3_final.json').read_text())
    neural = [m for m in manifest['members'] if m['family'] in ('realmlp', 'tabm')]
    assert len(neural) == 12
    neural_sum = sum((logit(store.load_oof(m['exp_id'])) for m in neural), np.zeros(len(y)))
    classical = (59 * old_v5 - neural_sum) / 47
    for w in scope['audit']['historical_classical_weights']:
        name = f'legacy_classical_{int(w*100)}'
        p = expit((1-w) * logit(control) + w * classical).astype('float32'); cv[name] = p
        rows[name] = evidence(y, folds, p, control, tr, scope)
        rows[name].update(eligibility='DIAGNOSTIC_ONLY_LEGACY_TE_AND_INFERENCE_DEFECTS', test_correlation_vs_v6=None)
    boot = paired_bootstrap(y, cv, reference='v6', repeats=scope['audit']['bootstrap_repeats'])
    for name in rows:
        rows[name]['paired_bootstrap'] = boot['comparisons'][name]
    va = np.flatnonzero(folds == 0); raw, raw_proof = raw_control(tr, te, ids, y, folds, data_hash)
    route, aux, base = vectors['route'][va], vectors['aux10'][va], control[va]
    p = expit(.375*logit(route) + .125*logit(raw) + .5*logit(aux)).astype('float32')
    raw_mix = evidence(y[va], np.zeros(len(va), dtype=int), p, base, tr.iloc[va], scope)
    mixed_pfn = expit(.75*logit(route) + .25*logit(raw)).astype('float32')
    standalone_gain = float(roc_auc_score(y[va], mixed_pfn) - roc_auc_score(y[va], route))
    raw_mix.update(scope='primary fold0 only', eligibility='MATCHED_CERTIFIED_DISCOVERY_ONLY', raw_proof=raw_proof,
                   raw_route_correlation=correlations(raw, route), raw_route_pair_rescue_damage=pair_diagnostic(y[va], raw, route),
                   raw_route_opposite_order_pairs=pair_diagnostic(y[va], raw, route)['rescue_count']+pair_diagnostic(y[va], raw, route)['damage_count'],
                   standalone_mixture_auc=float(roc_auc_score(y[va], mixed_pfn)), standalone_mixture_gain=standalone_gain,
                   discovery_pass=bool(standalone_gain >= .00005 or raw_mix['pooled_delta_vs_v6'] >= .00001))
    raw_mix['paired_bootstrap'] = paired_bootstrap(y[va], {'v6':base,'raw_route':p}, reference='v6', repeats=200)['comparisons']['raw_route']
    artifact = ARTIFACTS / 'sol_phase15_zero_training_20261008'; artifact.mkdir(exist_ok=False)
    np.save(artifact / 'raw_route_f0.npy', p); np.save(artifact / 'ids_f0.npy', ids[va])
    for name, vector in cv.items():
        if name != 'v6': np.save(artifact / (name+'_oof.npy'), vector)
    np.save(artifact / 'train_ids.npy', ids)
    out = {'utc':datetime.now(timezone.utc).isoformat(), 'git':git_commit(), 'scope_sha256':file_sha256(SCOPE),
           'proof_index_verification':'ALL_V6_FILES_UNCHANGED', 'data_sha256':data_hash, 'fold_sha256':arr_sha256(folds),
           'ordered_train_ids_sha256':arr_sha256(ids), 'v6_auc':float(roc_auc_score(y,control)), 'comparisons':rows,
           'raw_route':raw_mix, 'bootstrap_limitation':boot['limitation'], 'historical_classical_auc':float(roc_auc_score(y,classical)),
           'historical_classical_logit_sha256':arr_sha256(classical), 'seconds':time.monotonic()-start,
           'verdict':'SENSITIVITY_AND_DISCOVERY_ONLY_NO_FINALIST_SELECTION',
           'meta_validation_warning':'Meta-only CV of fixed OOF is not fully nested at base level; no unbiased weight-selection claim.'}
    save_json(out, OUTPUT)
    with Path('experiments/ledger.jsonl').open('a',encoding='utf-8') as f:
        for name, row in {**rows, 'raw_route_f0':raw_mix}.items():
            f.write(json.dumps({'exp_id':'sol_phase15_'+name,'kind':'PREDECLARED_ZERO_TRAINING_AUDIT', 'ts':out['utc'],
                'report':str(OUTPUT), 'report_sha256':file_sha256(OUTPUT), 'scope_sha256':out['scope_sha256'],
                'paired_fold_deltas':row['paired_fold_deltas'], 'pooled_auc':row['pooled_auc'],
                'corr_with_champion':row['oof_correlation_vs_v6'], 'verdict':row['eligibility']})+'\n')
    print(json.dumps({'comparisons':{n:{k:v[k] for k in ('pooled_auc','pooled_delta_vs_v6','paired_fold_deltas')} for n,v in rows.items()},
                      'raw_route':{k:raw_mix[k] for k in ('pooled_auc','pooled_delta_vs_v6','standalone_mixture_gain','discovery_pass')},'seconds':out['seconds']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
