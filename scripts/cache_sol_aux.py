"""CPU-only construction of pending fold caches; no satisfaction model fit."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import arr_sha256, load_cached_parquet
from src.features.view import ViewBuilder
from src.validation.folds import get_scheme
from scripts.sol_aux_cache import probability_cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='1,2')
    args = ap.parse_args()
    tr, te = load_cached_parquet()
    y, ids = tr['satisfaction'].to_numpy(dtype='int8'), tr['id'].to_numpy()
    folds = get_scheme('primary', y, ids).folds
    vb = ViewBuilder(tr, te, 'full')
    vb.build_static()
    for k in map(int, args.folds.split(',')):
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        Xf, Xa, names = vb.assemble(fi, y, va, None, inner_seed=k)
        pf, pv, manifest = probability_cache(Xf, Xa['val'], names, ids[fi], ids[va], arr_sha256(folds), label=f'fold{k}')
        print(f'fold{k} COMPLETE {manifest["fingerprint"]}', flush=True)
        del Xf, Xa, pf, pv
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
