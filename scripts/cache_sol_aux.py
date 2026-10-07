"""CPU-only construction of pending fold caches; no satisfaction model fit."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import arr_sha256, load_cached_parquet
from src.features.view import RAW21, _prep_raw
from src.validation.folds import get_scheme
from scripts.sol_aux_cache import probability_cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='1,2')
    args = ap.parse_args()
    tr, te = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    # Auxiliary targets and covariates use only the raw 21-column block. Match
    # ViewBuilder's label-free joint vocabulary without loading its 237 static
    # columns or computing any satisfaction target encoding.
    raw = _prep_raw(pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)).to_numpy(dtype='float32')
    raw = np.nan_to_num(raw, nan=-999.0, posinf=1e9, neginf=-1e9)
    names = list(RAW21)
    for k in map(int, args.folds.split(',')):
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        Xf, Xv = raw[fi], raw[va]
        pf, pv, manifest = probability_cache(Xf, Xv, names, ids[fi], ids[va], arr_sha256(folds), label=f'fold{k}')
        print(f'fold{k} COMPLETE {manifest["fingerprint"]}', flush=True)
        del Xf, Xv, pf, pv
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
