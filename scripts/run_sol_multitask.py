"""Conditional matched SOL-C experiment; checkpoint selection is inner-FIT only."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.features.view import RAW21, ViewBuilder
from src.models.masked import prepare_covariates
from src.models.multitask import SatisfactionMultitask, normalization_fit, normalized_gpu
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.run_views import _inner_es_split
from scripts.audit_sol_state import reconstruct_v5

WEIGHTS = {0.: 'L0', .05: 'L005', .2: 'L02'}


@torch.no_grad()
def predict(model, features, cat, numeric, missing, rows, batch=8192):
    model.eval()
    return torch.cat([model(features[ii], cat[ii], numeric[ii], missing[ii]).sigmoid()
                      for ii in rows.split(batch)]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--folds', default='0')
    ap.add_argument('--weights', default='0,0.05,0.2')
    ap.add_argument('--tag', default='sol_c')
    args = ap.parse_args()
    weights = list(map(float, args.weights.split(',')))
    if any(w not in WEIGHTS for w in weights):
        raise ValueError('Only the three predeclared weights are permitted')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tr, te = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    _, reference, _ = reconstruct_v5(y, folds)
    cat, numeric, missing, prep = prepare_covariates(pd.concat([tr[RAW21], te[RAW21]], ignore_index=True))
    ssl = ARTIFACTS / 'sol_ssl12'
    manifest = json.loads((ssl / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'COMPLETE' and prep == manifest['contract']['preprocessing']
    vb = ViewBuilder(tr, te, 'full')
    vb.build_static()
    root, records = ARTIFACTS / args.tag, REPORTS / args.tag
    root.mkdir(exist_ok=True)
    records.mkdir(exist_ok=True)
    source = {p: file_sha256(p) for p in ('scripts/run_sol_multitask.py', 'src/models/multitask.py',
            'src/models/masked.py', 'src/features/s6e10.py', 'src/features/view.py', 'research/sol_multitask_protocol.md')}
    for k in map(int, args.folds.split(',')):
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        itr, es = _inner_es_split(fi, y, 1)
        tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
        Xf, Xa, names = vb.assemble(fi, y, va, None, inner_seed=k)
        mean, scale = normalization_fit(Xf, tl)
        feature_hash = {'fit': arr_sha256(Xf), 'validation': arr_sha256(Xa['val'])}
        # GPU data are label-free covariates/features; only FIT labels are sent
        # to training. Outer labels remain exclusively in the final AUC call.
        F, V = normalized_gpu(Xf, mean, scale), normalized_gpu(Xa['val'], mean, scale)
        C, N, M = [torch.tensor(a[fi], device='cuda') for a in (cat, numeric, missing)]
        CV, NV, MV = [torch.tensor(a[va], device='cuda') for a in (cat, numeric, missing)]
        Y = torch.tensor(y[fi], dtype=torch.float32, device='cuda')
        ti, ei = torch.tensor(tl, device='cuda'), torch.tensor(el, device='cuda')
        del Xf, Xa
        base = None
        for weight in weights:
            name = WEIGHTS[weight]
            report = records / (f'timing_{name}_f{k}.json' if args.timing else f'{name}_f{k}.json')
            pp = root / f'{name}_f{k}.npy'
            contract = {'fold': k, 'weight': weight, 'seed': 1201, 'view': 'full', 'feature_names': names,
                'feature_sha256': feature_hash, 'source_sha256': source, 'normalization_mean': mean.tolist(),
                'normalization_scale': scale.tolist(), 'normalization_fit_ids_sha256': arr_sha256(ids[itr]),
                'fit_ids_sha256': arr_sha256(ids[itr]), 'es_ids_sha256': arr_sha256(ids[es]),
                'validation_ids_sha256': arr_sha256(ids[va]), 'fold_sha256': arr_sha256(folds),
                'initialization_sha256': file_sha256(ssl / 'best.pt'), 'ssl_fingerprint': manifest['fingerprint'],
                'train_rows': len(itr), 'es_rows': len(es), 'validation_rows': len(va),
                'epochs': 24, 'patience': 6, 'batch_size': 2048, 'lr': .002, 'weight_decay': .0001,
                'checkpoint_selection': 'inner ES AUC only, outer evaluation exactly once',
                'test_policy': 'same transductive initializer; primary median selected epoch full-FIT supervised refit',
                'comparison_policy': 'append one fixed equal-logit member to legacy-v5 diagnostic; never fitted weights'}
            fingerprint = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            if report.exists():
                rec = json.loads(report.read_text(encoding='utf-8'))
                assert rec['fingerprint'] == fingerprint
                if not args.timing:
                    pred = np.load(pp)
                    assert arr_sha256(pred) == rec['prediction_sha256']
                    if weight == 0:
                        base = pred
                print(f'reuse {name} f{k}', flush=True)
                continue
            torch.manual_seed(1201)
            np.random.seed(1201)
            model = SatisfactionMultitask(prep['cardinalities'], F.shape[1]).cuda()
            model.covariate.load_state_dict(torch.load(ssl / 'best.pt', weights_only=True))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24)
            mask_rng = torch.Generator(device='cuda').manual_seed(20261010)
            start, best_auc, bad, history = time.monotonic(), -1., 0, []
            best_path = root / f'{name}_f{k}_best.pt'
            for epoch in range(1 if args.timing else 24):
                model.train()
                order = ti[torch.randperm(len(ti), device='cuda')]
                losses, epoch_start = [], time.monotonic()
                batches = order.split(2048)
                if args.timing:
                    batches = batches[:60]
                for ii in batches:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(F[ii], C[ii], N[ii], M[ii])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y[ii])
                    mask = torch.rand((len(ii), 21), generator=mask_rng, device='cuda') < .25
                    if weight:
                        loss = loss + weight * model.auxiliary_loss(C[ii], N[ii], M[ii], mask)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                if args.timing:
                    torch.cuda.synchronize()
                    seconds = time.monotonic() - epoch_start
                    save_json({'contract': contract, 'fingerprint': fingerprint, 'git': git_commit(),
                               'status': 'TIMING_ONLY_NO_OUTER_AUC', 'seconds': seconds,
                               'epoch_seconds_estimate': seconds / len(batches) * np.ceil(len(ti)/2048),
                               'peak_gpu_bytes': torch.cuda.max_memory_allocated()}, report)
                    print(f'{name}: timing60 steps {seconds:.1f}s', flush=True)
                    break
                es_pred = predict(model, F, C, N, M, ei)
                score = float(roc_auc_score(y[es], es_pred))
                improved = score > best_auc
                bad = 0 if improved else bad + 1
                if improved:
                    best_auc, best_epoch = score, epoch + 1
                    torch.save(model.state_dict(), best_path)
                scheduler.step()
                history.append({'epoch': epoch + 1, 'inner_es_auc': score, 'loss': float(np.mean(losses)),
                                'seconds': time.monotonic() - epoch_start})
                save_json({'contract': contract, 'fingerprint': fingerprint, 'history': history,
                           'status': 'TRAINING'}, root / f'{name}_f{k}_progress.json')
                print(f'{name} f{k} epoch{epoch+1}: inner ES {score:.8f}', flush=True)
                if bad >= 6 or time.monotonic() - start > 1200:
                    break
            if not args.timing:
                model.load_state_dict(torch.load(best_path, weights_only=True))
                pred = predict(model, V, CV, NV, MV, torch.arange(len(va), device='cuda')).astype('float32')
                np.save(pp, pred)
                np.save(root / f'ids_f{k}.npy', ids[va])
                auc = float(roc_auc_score(y[va], pred))
                if weight == 0:
                    base = pred
                if base is None:
                    base = np.load(root / f'L0_f{k}.npy')
                blend = (59 * reference[va] + logit(pred)) / 60
                rec = {'contract': contract, 'fingerprint': fingerprint, 'git': git_commit(), 'auc': auc,
                       'best_epoch': best_epoch, 'inner_es_auc': best_auc, 'history': history,
                       'delta_vs_L0': auc - float(roc_auc_score(y[va], base)),
                       'v5_equal_member_delta': float(roc_auc_score(y[va], blend) - roc_auc_score(y[va], reference[va])),
                       'corr_with_v5': float(np.corrcoef(logit(pred), reference[va])[0,1]),
                       'prediction_sha256': arr_sha256(pred), 'seconds': time.monotonic() - start,
                       'status': 'BASELINE' if weight == 0 else 'UNCONFIRMED'}
                save_json(rec, report)
                print(f'{name} f{k}: AUC {auc:.9f}; delta L0 {rec["delta_vs_L0"]:+.9f}; v5 {rec["v5_equal_member_delta"]:+.9f}', flush=True)
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        del F, V, C, N, M, CV, NV, MV, Y
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
