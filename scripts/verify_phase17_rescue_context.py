"""Audit a completed frozen rescue context without fitting or changing predictions."""
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
from src.validation.folds import get_scheme
from scripts.phase16_common import bank
from scripts.phase16_tabpfn import library_sources
from scripts.phase17_certification import replay_members
from scripts.run_sol_tabpfn import frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('shadow', 'test'), required=True)
    ap.add_argument('--fold', type=int, choices=range(5), required=True)
    ap.add_argument('--telemetry-tag', required=True)
    ap.add_argument('--output-tag', required=True)
    args = ap.parse_args()
    assert args.mode != 'shadow' or args.fold == 3
    output = Path('reports')/(args.output_tag+'.json')
    assert not output.exists(), 'Preserve existing verification'
    tag = 'phase17_shadow_B1' if args.mode == 'shadow' else 'phase17_B1_test'
    report_path = Path('reports')/(tag+f'_f{args.fold}.json')
    result = json.loads(report_path.read_text()); c = result['contract']
    expected = 'COMPLETE_FROZEN_SHADOW_FOLD' if args.mode == 'shadow' else 'COMPLETE_FROZEN_PRIMARY_CONTEXT_TEST'
    assert result['status'] == expected and c['mode'] == args.mode and c['fold'] == args.fold
    assert c['seed'] == 1201 and c['params']['n_estimators'] == 2
    scope_path = Path('research/phase17_rescue_scope_20261011.json')
    scope = json.loads(scope_path.read_text())
    assert all(file_sha256(p) == h for p, h in scope['protected_source_sha256'].items())
    assert all(file_sha256(p) == h for p, h in c['source_sha256'].items())
    assert c['library_source_sha256'] == library_sources()
    assert not c['outer_labels_used_for_fit_or_configuration'] and not c['entire_training_context'] and c['no_test_labels']
    assert c['scope_sha256'] == file_sha256('research/phase17_scope_20261009.json')
    assert c['primary_report_sha256'] == file_sha256('reports/phase17_B1_primary.json')
    assert file_sha256(c['params']['model_path']) == c['checkpoint']['checkpoint_sha256']
    tr, te, y, primary_folds, _, _, proof = bank()
    assert c['bank'] == proof
    ids, test_ids = tr.id.to_numpy(), te.id.to_numpy()
    scheme = 'shadow' if args.mode == 'shadow' else 'primary'
    folds = get_scheme(scheme, y, ids).folds
    assert c['scheme'] == scheme and arr_sha256(folds) == c['fold_sha256']
    if scheme == 'primary':
        assert np.array_equal(folds, primary_folds)
        assert c['confirmation_report_sha256'] == file_sha256('reports/phase17_B1_shadow_confirmation.json')
        assert json.loads(Path('reports/phase17_B1_shadow_confirmation.json').read_text())['status'] == 'MATCHED_SHADOW_CONFIRMATION_PASS'
    fi, va = np.flatnonzero(folds != args.fold), np.flatnonzero(folds == args.fold)
    apply_ids = ids[va] if args.mode == 'shadow' else test_ids
    assert c['train_rows'] == len(fi) == 559708 and c['apply_rows'] == len(apply_ids)
    assert c['fit_ids_sha256'] == arr_sha256(ids[fi]) and c['apply_ids_sha256'] == arr_sha256(apply_ids)
    assert not np.intersect1d(ids[fi], apply_ids).size
    folder = ARTIFACTS/tag/f'f{args.fold}'
    assert np.array_equal(np.load(folder/'fit_ids.npy'), ids[fi])
    assert np.array_equal(np.load(folder/'apply_ids.npy'), apply_ids)
    x, xt, names, cats, maps = frames(tr, te, True)
    assert names == c['feature_names'] and cats == c['params']['categorical_features_indices'] and maps == c['label_free_category_maps']
    assert arr_sha256(x[fi]) == c['feature_fit_sha256']
    assert arr_sha256(x[va] if args.mode == 'shadow' else xt) == c['feature_apply_sha256']
    config = json.loads((folder/'configuration.json').read_text())
    assert config['contract'] == c
    probe = json.loads(Path('reports/phase17_internal2_serial_equivalence_20261009.json').read_text())
    assert probe['status'] == 'PASS_REQUIRES_EXACT_FULL_B0' and probe['maximum_absolute_probability_gap'] <= 2e-6
    assert c['probe_report_sha256'] == file_sha256('reports/phase17_internal2_serial_equivalence_20261009.json')
    for field in ('config_sha256', 'pipeline_seed'):
        assert [m[field] for m in result['metadata']['members']] == [m[field] for m in probe['official_metadata']['members']]
    for field in ('config_sha256', 'pipeline_seed', 'prepared_x_sha256', 'prepared_y_sha256'):
        assert [m[field] for m in config['metadata']['members']] == [m[field] for m in result['metadata']['members']]
    p = np.load(folder/'prediction.npy')
    assert p.shape == (len(apply_ids),) and p.dtype == np.float32 and arr_sha256(p) == result['prediction_sha256']
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.array_equal(replay_members(folder, result['metadata'], c['params'], len(apply_ids)), p)
    if args.mode == 'shadow':
        assert abs(float(roc_auc_score(y[va], p))-result['auc']) < 1e-14
    telemetry = Path('reports')/args.telemetry_tag
    summary_path = telemetry/'summary.json'; summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'FROZEN_DRIVER_COMPLETE'
    repair = json.loads(Path('research/phase17_rescue_observer_repair_20261011.json').read_text())
    assert summary['wrapper_sha256'] == repair['new_wrapper_sha256'] == file_sha256('scripts/run_phase17_monitored.py')
    assert summary['driver_sha256'] == file_sha256('scripts/run_phase17_confirmation.py')
    assert summary['driver_args'] == ['--mode',args.mode,'--primary-tag','phase17_B1','--fold',str(args.fold),'--tag',tag]
    assert not (telemetry/'observer_abort.json').exists() and not (telemetry/'resource_abort.json').exists()
    observations_path = telemetry/'observations.jsonl'
    rows = [json.loads(line) for line in observations_path.read_text().splitlines()]
    assert len(rows) >= 2 and max(b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(rows,rows[1:])) < 60
    assert rows[0]['c_free_bytes'] >= 40*2**30 and min(r['c_free_bytes'] for r in rows) >= 20*2**30
    resources = {'sample_interval_seconds':summary['interval_seconds'],'samples':len(rows),
        'sampled_disk_min_gib':min(r['c_free_bytes'] for r in rows)/2**30,
        'sampled_ram_min_gib':min(r['host_available_bytes'] for r in rows)/2**30,
        'sampled_process_commit_max_gib':max(r['process_commit_charge_bytes'] for r in rows)/2**30,
        'sampled_gpu_allocated_max_gib':max((r.get('gpu_allocated_bytes') or 0) for r in rows)/2**30,
        'sampled_gpu_reserved_max_gib':max((r.get('gpu_reserved_bytes') or 0) for r in rows)/2**30,
        'pagefile_allocated_unique_gib':sorted({p['allocated_bytes']/2**30 for r in rows for p in r.get('pagefiles',[])}),
        'observations_sha256':file_sha256(observations_path),'summary_sha256':file_sha256(summary_path)}
    report = {'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPLETE_FROZEN_CONTEXT_REPLAY_VERIFIED',
        'mode':args.mode,'fold':args.fold,'bank':proof,'internal_estimators':2,'fit_rows':len(fi),'apply_rows':len(apply_ids),
        'report_sha256':file_sha256(report_path),'prediction_sha256':arr_sha256(p),'saved_raw_members_bitexact_replay':True,
        'actual_wrapper_worker_seconds':summary['seconds'],'driver_seconds':result['seconds'],'resources':resources,
        'source_sha256':file_sha256(__file__),'new_submissions':0}
    save_json(report, output)
    ledger = Path('experiments/ledger.jsonl'); before = ledger.read_bytes()
    assert not any(json.loads(line).get('exp_id') == args.output_tag for line in before.splitlines())
    entry = {'exp_id':args.output_tag,'ts':report['utc'],'kind':'FROZEN_CONTEXT_ARTIFACT_VERIFICATION',
        'report':str(output),'report_sha256':file_sha256(output),'seconds':summary['seconds'],
        'paired_fold_deltas':[],'correlation_with_champion':None,'verdict':report['status'],
        'scope':'Saved vector verification only; shadow comparison and final test admission remain protected gates'}
    with ledger.open('ab') as stream:
        stream.write((json.dumps(entry)+'\n').encode('utf-8'))
    assert ledger.read_bytes().startswith(before)
    prior = json.loads(Path('reports/phase17_rescue_compute_after_observer_failure_20261011.json').read_text())
    workers = list(prior['unique_completed_model_workers'])
    known = {r['tag'] for r in workers}
    for path in sorted(Path('reports').glob('phase17_rescue_telemetry_*/summary.json')):
        if path.parent.name in known:
            continue
        item = json.loads(path.read_text())
        assert item['status'] == 'FROZEN_DRIVER_COMPLETE', 'Investigate invalid worker and account its duration separately'
        workers.append({'tag':path.parent.name,'seconds':item['seconds'],'status':item['status'],'sha256':file_sha256(path)})
        known.add(path.parent.name)
    total = prior['prior_model_hours']+(sum(r['seconds'] for r in workers)+sum(r['seconds'] for r in prior['invalid_worker_attempts']))/3600
    assert total < prior['total_ceiling_hours']
    save_json({**prior,'utc':report['utc'],'unique_completed_model_workers':workers,'cumulative_model_hours':total,
        'remaining_model_hours':prior['total_ceiling_hours']-total,'next':'Continue only the frozen gated rescue queue'},
        Path('reports')/(args.output_tag+'_compute.json'))
    print(json.dumps({k:report[k] for k in ('status','mode','fold','prediction_sha256','actual_wrapper_worker_seconds','resources')}),flush=True)


if __name__ == '__main__':
    main()
