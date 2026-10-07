"""Matched distribution ablations and the predeclared SSL/aux interaction.

The operational delta replaces the HISTORICAL v5 member, rather than incorrectly
treating a newly fitted A0 as that member. Existing repair records remain intact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.features.view import ViewBuilder
from src.features.aux_distribution import signatures, fit_surprise_threshold
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.run_views import _inner_es_split, _fit_xgb_es
from scripts.run_phase13 import RATINGS
from scripts.run_phase14 import XGB_ARMS, XGB_BASE, fit_xgb
from scripts.run_phase14c import counter_path
from scripts.sol_aux_cache import probability_cache
from scripts.audit_sol_state import reconstruct_v5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0')
    ap.add_argument('--variants', default='A1,A2')
    ap.add_argument('--tag', default='sol_a_ladder')
    args = ap.parse_args()
    tr, te = load_cached_parquet()
    y = tr['satisfaction'].to_numpy(dtype='int8')
    ids = tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    _, champ, _ = reconstruct_v5(y, folds)
    vb = ViewBuilder(tr, te, 'full')
    vb.build_static()
    member, seed, overrides = XGB_ARMS['X2']
    root, records = ARTIFACTS / args.tag, REPORTS / args.tag
    root.mkdir(exist_ok=True)
    records.mkdir(exist_ok=True)
    variants = args.variants.split(',')
    embedding = None
    if 'SSL_A0' in variants:
        rep = ARTIFACTS / 'sol_ssl12'
        rep_manifest = json.loads((rep / 'manifest.json').read_text(encoding='utf-8'))
        assert rep_manifest['status'] == 'COMPLETE'
        assert np.array_equal(np.load(rep / 'ids.npy'), np.r_[ids, te['id'].to_numpy()])
        embedding = np.load(rep / 'embedding.npy')
        assert arr_sha256(embedding) == rep_manifest['embedding_sha256']
    for k in map(int, args.folds.split(',')):
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert not np.intersect1d(fi, va).size
        Xf, Xa, names = vb.assemble(fi, y, va, None, inner_seed=k)
        Xv = Xa['val']
        pf, pv, cache = probability_cache(Xf, Xv, names, ids[fi], ids[va], arr_sha256(folds), label=f'fold{k}')
        pos = [names.index(r) for r in RATINGS]
        rf, rv = Xf[:, pos], Xv[:, pos]
        threshold = fit_surprise_threshold(pf, rf)
        af0, _ = signatures(pf, rf, 0)
        av0, _ = signatures(pv, rv, 0)
        assert np.max(np.abs(af0 - np.load(REPORTS / f'p13b_auxfit_f{k}.npy'))) <= 1e-10
        assert np.max(np.abs(av0 - np.load(REPORTS / f'p13b_auxval_f{k}.npy'))) <= 1e-10
        itr, es = _inner_es_split(fi, y, 1)
        tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
        assert np.array_equal(fi[tl], itr) and np.array_equal(fi[el], es)
        F0, V0 = np.column_stack([Xf, af0]).astype(float), np.column_stack([Xv, av0]).astype(float)
        control, control_it = _fit_xgb_es(F0[tl], y[itr], V0, overrides, seed, F0[el], y[es])
        a0 = control.astype('float32')
        saved_control = ARTIFACTS / 'sol_repair' / f'X2_A0_f{k}.npy'
        if saved_control.exists():
            assert np.array_equal(a0, np.load(saved_control)), 'STOP: independent A0 control failed'
        else:
            second_control, second_it = fit_xgb(F0[tl], y[itr], V0, seed, overrides, F0[el], y[es])
            assert np.array_equal(a0, second_control.astype('float32')) and control_it == second_it, 'STOP: independent A0 runner control failed'
        auc0 = float(roc_auc_score(y[va], a0))
        old = np.load(counter_path(member, k))
        np.save(root / f'A0_f{k}.npy', a0)
        np.save(root / f'ids_f{k}.npy', ids[va])
        del F0, V0
        for variant in variants:
            stage = 0 if variant == 'SSL_A0' else int(variant[1:])
            af, extra_names = signatures(pf, rf, stage, threshold)
            av, _ = signatures(pv, rv, stage, threshold)
            F, V = np.column_stack([Xf, af]), np.column_stack([Xv, av])
            cols = names + extra_names
            if variant == 'SSL_A0':
                F, V = np.column_stack([F, embedding[fi]]), np.column_stack([V, embedding[va]])
                cols += [f'ssl_latent_{j}' for j in range(embedding.shape[1])]
            F, V = F.astype(float), V.astype(float)
            path = records / f'{variant}_f{k}.json'
            if path.exists():
                raise RuntimeError(f'Preserve existing experiment: {path}')
            start = time.monotonic()
            pred, it = fit_xgb(F[tl], y[itr], V, seed, overrides, F[el], y[es])
            pred = pred.astype('float32')
            assert pred.shape == (len(va),) and np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
            np.save(root / f'{variant}_f{k}.npy', pred)
            auc = float(roc_auc_score(y[va], pred))
            actual = champ[va] + (logit(pred) - logit(old)) / 59
            incremental = champ[va] + (logit(pred) - logit(a0)) / 59
            rec = {'exp_id': f'{args.tag}_{variant}_f{k}', 'git': git_commit(), 'fold': k, 'variant': variant,
                   'model_family': 'xgb', 'member': member, 'params': dict(XGB_BASE, **overrides, random_state=seed),
                   'seed': seed, 'train_rows': len(itr), 'es_rows': len(es), 'validation_rows': len(va),
                   'fit_ids_sha256': arr_sha256(ids[itr]), 'es_ids_sha256': arr_sha256(ids[es]),
                   'validation_ids_sha256': arr_sha256(ids[va]), 'outer_fit_ids_sha256': arr_sha256(ids[fi]),
                   'fold_sha256': arr_sha256(folds), 'feature_names': cols, 'feature_fit_sha256': arr_sha256(F),
                   'feature_val_sha256': arr_sha256(V), 'aux_fingerprint': cache['fingerprint'],
                   'data_sha256': {s: file_sha256(f'data/raw/{s}.csv') for s in ('train','test')},
                   'source_sha256': {s: file_sha256(s) for s in ('scripts/evaluate_sol_aux.py','scripts/run_phase14.py','scripts/run_views.py','src/features/aux_distribution.py')},
                   'representation_fingerprint': None if variant != 'SSL_A0' else rep_manifest['fingerprint'],
                   'auc': auc, 'a0_auc': auc0, 'delta_vs_A0': auc - auc0,
                   'actual_legacy_v5_slot_delta': float(roc_auc_score(y[va], actual) - roc_auc_score(y[va], champ[va])),
                   'incremental_vs_repaired_A0_slot_delta': float(roc_auc_score(y[va], incremental) - roc_auc_score(y[va], champ[va])),
                   'logit_corr_vs_v5': float(np.corrcoef(logit(pred), champ[va])[0,1]),
                   'logit_corr_vs_A0': float(np.corrcoef(logit(pred), logit(a0))[0,1]),
                   'prediction_sha256': arr_sha256(pred), 'a0_sha256': arr_sha256(a0), 'n_trees': it + 1,
                   'canonical_control_bit_identical': True, 'control_n_trees': control_it + 1,
                   'seconds': time.monotonic() - start, 'hardware': 'RTX 5070 Ti 16GB; eight CPU threads',
                   'test_policy': 'full-label median-tree refit; same three-inner-fold auxiliary recipe; fixed transductive SSL encoder if present',
                   'status': 'POSITIVE_UNCONFIRMED' if auc > auc0 else 'NEGATIVE'}
            save_json(rec, path)
            print(f'{variant} f{k}: {auc:.9f}; A0 {auc-auc0:+.9f}; actual v5 {rec["actual_legacy_v5_slot_delta"]:+.9f}; incremental {rec["incremental_vs_repaired_A0_slot_delta"]:+.9f}', flush=True)
            del F, V, af, av
        del Xf, Xv, Xa, pf, pv
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
