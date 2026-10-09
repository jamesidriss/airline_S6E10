"""Copy complete verified folds into a fresh continuation namespace."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json
from scripts.phase16_common import bank


def complete_record(run, evaluation, run_path, fold):
    c = run['contract']
    assert run['status'] == 'COMPLETE_FROZEN_PRIMARY_FOLD'
    assert c['scheme'] == 'primary' and c['fold'] == evaluation['fold'] == fold
    assert c['seed'] == c['params']['random_state'] == 1201
    assert c['params']['n_estimators'] in (2, 4) and c['params']['memory_saving_mode'] is True
    assert not c['outer_labels_used_for_fit_or_configuration'] and not c['entire_training_context']
    assert evaluation['source_report_sha256'] == file_sha256(run_path)
    assert c['bank'] == evaluation['bank'] and c['scope_sha256'] == evaluation['scope_sha256']
    for record in (c, evaluation):
        assert all(file_sha256(p) == h for p, h in record['source_sha256'].items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--sources', required=True, help='JSON fold->complete source tag mapping')
    args = ap.parse_args()
    assert Path(args.tag).name == args.tag and args.tag not in ('.', '..')
    sources = json.loads(Path(args.sources).read_text(encoding='utf-8'))
    assert sources and set(sources) == {str(k) for k in range(len(sources))} and len(sources) <= 5
    assert all(Path(tag).name == tag and tag not in ('.', '..', args.tag) for tag in sources.values())
    root = ARTIFACTS/args.tag; output = Path('reports')/(args.tag+'_staging.json')
    if root.exists() or output.exists():
        raise FileExistsError('Preserve the existing continuation namespace')
    for k in sources:
        assert not (Path('reports')/(args.tag+f'_f{k}.json')).exists()
        assert not (Path('reports')/(args.tag+f'_f{k}_evaluation.json')).exists()
    tr, _, y, folds, old, _, proof = bank(); ids = tr.id.to_numpy()
    from scripts.phase17_certification import replay_members
    from scripts.phase16_tabpfn import library_sources
    verified = []; recipes = []
    for sk, tag in sorted(sources.items()):
        k = int(sk); rp = Path('reports')/(tag+f'_f{k}.json'); ep = Path('reports')/(tag+f'_f{k}_evaluation.json')
        run, evaluation = json.loads(rp.read_text()), json.loads(ep.read_text()); c = run['contract']
        complete_record(run, evaluation, rp, k)
        assert c['bank'] == proof and c['scope_sha256'] == file_sha256('research/phase17_scope_20261009.json')
        assert c['library_source_sha256'] == library_sources()
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        folder = ARTIFACTS/tag/f'f{k}'
        assert c['train_rows'] == len(fi) and c['apply_rows'] == len(va) and c['fold_sha256'] == arr_sha256(folds)
        configuration = json.loads((folder/'configuration.json').read_text())
        assert configuration['contract'] == c
        for field in ('config_sha256','pipeline_seed','prepared_x_sha256','prepared_y_sha256'):
            assert [m[field] for m in configuration['metadata']['members']] == [m[field] for m in run['metadata']['members']]
        assert np.array_equal(np.load(folder/'fit_ids.npy'), ids[fi]) and np.array_equal(np.load(folder/'apply_ids.npy'), ids[va])
        assert c['fit_ids_sha256'] == arr_sha256(ids[fi]) and c['apply_ids_sha256'] == arr_sha256(ids[va])
        route, p = np.load(folder/'prediction.npy'), np.load(folder/'portfolio.npy')
        assert route.shape == p.shape == (len(va),) and route.dtype == p.dtype == np.float32
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert arr_sha256(route) == run['prediction_sha256'] and arr_sha256(p) == evaluation['portfolio_prediction_sha256']
        assert np.array_equal(replay_members(folder, run['metadata'], c['params'], len(va)), route)
        assert abs(float(roc_auc_score(y[va], p)-roc_auc_score(y[va], old['candidate'][va]))-evaluation['portfolio_delta']) < 1e-14
        recipes.append({f:c[f] for f in ('params','seed','feature_names','label_free_category_maps','checkpoint','library_versions','library_source_sha256','test_policy')})
        files = {p.name:file_sha256(p) for p in folder.iterdir() if p.is_file() and
                 (p.suffix == '.npy' or p.name == 'configuration.json' or p.name.endswith('_stats.json'))}
        verified.append({'fold':k,'source_tag':tag,'run_report':str(rp),'run_report_sha256':file_sha256(rp),
            'evaluation':str(ep),'evaluation_sha256':file_sha256(ep),'files':files})
    assert all(r == recipes[0] for r in recipes), 'Cannot mix completed recipes'
    root.mkdir()
    for v in verified:
        k, tag = v['fold'], v['source_tag']; dest = root/f'f{k}'; dest.mkdir()
        for name, digest in v['files'].items():
            shutil.copyfile(ARTIFACTS/tag/f'f{k}'/name, dest/name)
            assert file_sha256(dest/name) == digest
        for source, target in ((v['run_report'], Path('reports')/(args.tag+f'_f{k}.json')),
                               (v['evaluation'], Path('reports')/(args.tag+f'_f{k}_evaluation.json'))):
            shutil.copyfile(source, target); assert file_sha256(source) == file_sha256(target)
        save_json(v, dest/'staged_from.json')
    save_json({'utc':datetime.now(timezone.utc).isoformat(),'git':git_commit(),'status':'VERIFIED_COMPLETED_FOLD_REUSE_NO_NEW_FIT',
        'tag':args.tag,'bank':proof,'source_mapping_sha256':file_sha256(args.sources),'folds':verified,
        'source_sha256':{p:file_sha256(p) for p in ('scripts/stage_phase17_completed_folds.py','scripts/phase17_certification.py')},
        'scope':'Exact byte copies of complete evidence; failed original and recovery attempts preserved; no new fit, prediction, AUC experiment or changed recipe'}, output)
    print(json.dumps({'status':'VERIFIED_COMPLETED_FOLD_REUSE_NO_NEW_FIT','tag':args.tag,'folds':[v['fold'] for v in verified]}))


if __name__ == '__main__':
    main()
