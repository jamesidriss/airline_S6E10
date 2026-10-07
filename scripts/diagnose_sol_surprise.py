"""FIT-thresholded auxiliary surprise and pair errors; diagnostics, no correction fit."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.run_phase13 import RATINGS


def summaries(p, r):
    observed = np.take_along_axis(p, r.astype(int)[..., None], axis=-1)[..., 0]
    return -np.log(np.clip(observed, 1e-12, 1)).mean(axis=1), np.abs(r - p @ np.arange(6)).mean(axis=1)


def main():
    tr, _ = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    fi, va = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    control = json.loads((REPORTS / 'sol_repair/X2_A0_f0.json').read_text(encoding='utf-8'))
    cache = ARTIFACTS / 'aux_distribution' / control['contract']['aux_fingerprint']
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    assert np.array_equal(np.load(cache / 'fit_ids.npy'), ids[fi])
    assert np.array_equal(np.load(cache / 'apply_ids.npy'), ids[va])
    pf, pv = np.load(cache / 'fit.npy', mmap_mode='r'), np.load(cache / 'apply.npy', mmap_mode='r')
    assert arr_sha256(pf) == manifest['fit_probability_sha256']
    assert arr_sha256(pv) == manifest['apply_probability_sha256']
    sf, rf = summaries(pf, tr[RATINGS].to_numpy()[fi])
    sv, rv = summaries(pv, tr[RATINGS].to_numpy()[va])
    thresholds = {'surprise': float(np.quantile(sf, .95)), 'residual': float(np.quantile(rf, .95))}
    high = {'surprise': sv > thresholds['surprise'], 'residual': rv > thresholds['residual']}
    base = np.load(ARTIFACTS / 'sol_s0/v5_logit_oof.npy')[va]
    new = base + (logit(np.load(ARTIFACTS / 'sol_a_ladder/A1_f0.npy')) - logit(np.load(REPORTS / 'p14_X2_aux_f0.npy'))) / 59
    yy = y[va]
    rng = np.random.default_rng(20261012)
    positive, negative = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    bins = np.array([0, .5, 1, 2, 4, 8, np.inf])
    counts = {name: np.zeros((2, len(bins)-1, 5), dtype='int64') for name in high}
    # 20M IID positive/negative pairs, chunked to keep host memory bounded.
    for _ in range(20):
        p = rng.choice(positive, 1000000, replace=True)
        n = rng.choice(negative, 1000000, replace=True)
        margin = base[p] - base[n]
        altered = new[p] - new[n]
        band = np.searchsorted(bins, np.abs(margin), side='right') - 1
        error, tie = margin < 0, margin == 0
        rescue, damage = error & (altered > 0), (margin > 0) & (altered < 0)
        for name, values in high.items():
            group = values[p] | values[n]
            for g in (0, 1):
                for b in range(len(bins)-1):
                    m = (group == g) & (band == b)
                    counts[name][g,b] += [m.sum(), (m & error).sum(), (m & tie).sum(), (m & rescue).sum(), (m & damage).sum()]
    result = {}
    for name, table in counts.items():
        totals = table.sum(axis=1)
        groups = []
        for g in (0,1):
            N, errors, ties, rescues, damages = totals[g].tolist()
            bands = []
            for b, row in enumerate(table[g]):
                nn, ee, tt, rr, dd = row.tolist()
                bands.append({'abs_v5_margin_range': bins[b:b+2].tolist(), 'pairs': nn, 'errors': ee, 'ties': tt,
                              'error_rate': None if nn == 0 else (ee+tt/2)/nn, 'rescue_count': rr, 'damage_count': dd})
            groups.append({'high_pair': bool(g), 'pairs': N, 'errors': errors, 'ties': ties,
                           'error_rate': (errors+ties/2)/N, 'rescue_count': rescues, 'damage_count': damages, 'margin_bands': bands})
        result[name] = {'fit_95th_percentile': thresholds[name], 'validation_high_row_rate': float(high[name].mean()),
                        'groups': groups, 'high_pair_share_of_errors': float(totals[1,1]/totals[:,1].sum())}
    rec = {'exp_id': 'sol_d_surprise_diagnostic_f0', 'git': git_commit(), 'fold': 0, 'seed': 20261012,
           'pairs': 20000000, 'threshold_protocol': 'fixed FIT-covariate 95th percentile, no satisfaction tuning',
           'results': result, 'fold_hash': arr_sha256(folds), 'validation_ids_sha256': arr_sha256(ids[va]),
           'aux_fingerprint': manifest['fingerprint'], 'verdict': 'DIAGNOSTIC_NOT_SPECIALIST_ADMISSION',
           'limitations': 'Discovery fold only. Repeated pairs are Monte Carlo diagnostics, not independent data for inferential confidence. Concentration can reflect irreducible noise; no label-dependent correction is fitted.'}
    save_json(rec, REPORTS / 'sol_surprise_diagnostic.json')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
