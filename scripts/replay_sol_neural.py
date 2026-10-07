"""Replay fixed legacy neural recipes with corrected inner selection and twins.

This repairs a validation contract; a legacy model-selected OOF is not an honest
matched control. No source experiment, weight, seed or architecture is tuned.
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from importlib.metadata import version
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.features.view import ViewBuilder
from src.models import realmlp as RM
from src.submission import store
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.run_views import _inner_es_split
from scripts.audit_sol_state import reconstruct_v5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default='z3_rm_bs256_e6')
    ap.add_argument('--folds', default='0')
    ap.add_argument('--tag', default='sol_neural_clean')
    ap.add_argument('--save-test', action='store_true', help='Save inference from the same fitted CV model')
    args = ap.parse_args()
    from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.85)
    manifest = json.loads((REPORTS / 'finalist_v3_final.json').read_text(encoding='utf-8'))
    specs = {m['exp_id']: m for m in manifest['members'] if m['family'] in ('realmlp', 'tabm')}
    tr, te = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    _, v5, _ = reconstruct_v5(y, folds)
    root, reports = ARTIFACTS / args.tag, REPORTS / args.tag
    root.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    chosen = list(specs) if args.members == 'all' else args.members.split(',')
    for member in chosen:
        spec = specs[member]
        assert store._load_index()[member]['fold_scheme'] == 'primary', 'Select a primary neural recipe'
        params = dict(RM.REALMLP_BASE if spec['family'] == 'realmlp' else RM.TABM_BASE)
        params.update(spec.get('params', {}))
        if member.startswith('prod5_realmlp_'):
            params.update(n_epochs=6, n_ens=8)
        seed = params.pop('random_seed', spec.get('seed', 1))
        params.update(device='cuda', random_state=seed, verbosity=1, n_threads=4)
        cls = RealMLP_TD_Classifier if spec['family'] == 'realmlp' else TabM_D_Classifier
        for k in map(int, args.folds.split(',')):
            vb = ViewBuilder(tr, te, spec['featureset'])
            vb.build_static()
            fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            itr, es = _inner_es_split(fi, y, 1)
            test_idx = np.arange(len(tr), len(tr)+len(te)) if args.save_test else None
            Xf, Xa, names = vb.assemble(fi, y, va, test_idx, inner_seed=k)
            contract = {'member': member, 'fold': k, 'family': spec['family'], 'params': params,
                'training_protocol': RM.TRAINING_PROTOCOL, 'feature_names': names,
                'fit_ids_sha256': arr_sha256(ids[itr]), 'es_ids_sha256': arr_sha256(ids[es]),
                'validation_ids_sha256': arr_sha256(ids[va]), 'fold_sha256': arr_sha256(folds),
                'feature_fit_sha256': arr_sha256(Xf), 'feature_val_sha256': arr_sha256(Xa['val']),
                'source_sha256': {p: file_sha256(p) for p in ('scripts/replay_sol_neural.py',
                    'src/models/realmlp.py', 'src/features/view.py', 'src/features/s6e10.py', 'scripts/run_views.py')},
                'data_sha256': {s: file_sha256(f'data/raw/{s}.csv') for s in ('train','test')},
                'library_versions': {p: version(p) for p in ('pytabkit','torch','numpy','scikit-learn')},
                'train_rows': len(itr), 'es_rows': len(es), 'validation_rows': len(va),
                'save_test': args.save_test,
                'test_policy': 'average the same fitted CV models with corrected twins/TE and inner selection; no separate refit' if args.save_test else 'no test prediction in this diagnostic',
                'limitation': 'Legacy checkpoint-selected OOF is a diagnostic, not an honest baseline; no ensemble admission from one fold'}
            fingerprint = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            rp, pp = reports / f'{member}_f{k}.json', root / f'{member}_f{k}.npy'
            if rp.exists():
                previous = json.loads(rp.read_text(encoding='utf-8'))
                assert previous['fingerprint'] == fingerprint
                assert arr_sha256(np.load(pp)) == previous['prediction_sha256']
                if args.save_test:
                    assert arr_sha256(np.load(root / f'{member}_test_f{k}.npy')) == previous['test_prediction_sha256']
                print(f'reuse {member} f{k}', flush=True)
                del Xf, Xa
                del vb
                continue
            fit_frame, val_frame = RM._twin_frame(Xf, names), RM._twin_frame(Xa['val'], names)
            # The assembled frames are self-contained. Release the static cache
            # and combined covariate frame before PyTabKit allocates its dataset.
            del vb
            gc.collect()
            model = cls(**params)
            start = time.monotonic()
            from src.models.resource_guard import inference_guard
            try:
                with inference_guard(root, contract, max_seconds=2700, min_available_gib=4):
                    RM.fit_inner_validation(model, fit_frame, y[fi], seed=1)
                    pred = model.predict_proba(val_frame)[:, 1].astype('float32')
                    test_pred = model.predict_proba(RM._twin_frame(Xa['test'], names))[:, 1].astype('float32') if args.save_test else None
            except Exception as error:
                from src.models.resource_guard import ResourcePreflightError
                resource = isinstance(error, (torch.cuda.OutOfMemoryError, ResourcePreflightError))
                save_json({'contract': contract, 'fingerprint': fingerprint, 'git': git_commit(),
                    'status': 'INVALID_RESOURCE_LIMIT_NOT_NEGATIVE' if resource else 'INVALID_IMPLEMENTATION_NOT_NEGATIVE',
                    'error': repr(error), 'seconds': time.monotonic()-start}, rp)
                raise
            assert pred.shape == (len(va),) and np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
            np.save(pp, pred)
            np.save(root / f'ids_f{k}.npy', ids[va])
            if test_pred is not None:
                assert test_pred.shape == (len(te),) and np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
                np.save(root / f'{member}_test_f{k}.npy', test_pred)
                np.save(root / 'test_ids.npy', te['id'].to_numpy())
            old = store.load_oof(member)[va]
            spliced = v5[va] + (logit(pred) - logit(old)) / 59
            rec = {'contract': contract, 'fingerprint': fingerprint, 'git': git_commit(),
                'auc': float(roc_auc_score(y[va], pred)), 'prediction_sha256': arr_sha256(pred),
                'legacy_slot_delta_diagnostic': float(roc_auc_score(y[va], spliced) - roc_auc_score(y[va], v5[va])),
                'corr_with_legacy_v5': float(np.corrcoef(logit(pred), v5[va])[0,1]),
                'corr_with_legacy_member': float(np.corrcoef(logit(pred), logit(old))[0,1]),
                'selected_epochs': getattr(model.alg_interface_, 'fit_params', None),
                'seconds': time.monotonic() - start, 'status': 'CORRECTED_BASELINE_UNCONFIRMED'}
            if test_pred is not None:
                rec['test_prediction_sha256'] = arr_sha256(test_pred)
                rec['test_ids_sha256'] = arr_sha256(te['id'].to_numpy())
            rec['resolved_library_params'] = model.get_config()
            save_json(rec, rp)
            print(f'{member} f{k}: corrected AUC {rec["auc"]:.9f}; legacy slot diagnostic {rec["legacy_slot_delta_diagnostic"]:+.9f}', flush=True)
            del model, fit_frame, val_frame, Xf, Xa
            gc.collect()
            torch.cuda.empty_cache()
        gc.collect()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
