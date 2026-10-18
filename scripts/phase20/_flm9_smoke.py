"""Engineering smoke test for FLM9 Xiaomi-TabLDM screen (NOT a scientific result).

Tiny context + eval to verify: import, checkpoint auto-download, categorical_indices
handling, kv_cache fit/predict, proba shape, timing extrapolation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import load_cached_parquet  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import CONTEXT_SEED, stratified_context  # noqa: E402


def main() -> int:
    t0 = time.time()
    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)

    from scripts.run_sol_tabpfn import frames
    Xall, _Xte, names, cats, maps = frames(tr, te, route=True)
    print(f"route features: {Xall.shape[1]} cats={list(cats)}")

    ctx = stratified_context(FIT, y, 2000, seed=CONTEXT_SEED)
    ev = EVAL[:5000]
    X_fit, y_fit = Xall[ctx], y[ctx]
    X_eval = Xall[ev]
    print(f"ctx={len(ctx)} eval={len(ev)} dtypes={X_fit.dtype}")

    from tabldm import TabLDMClassifier

    clf = TabLDMClassifier(
        n_estimators=2,
        kv_cache=False,  # conflicts with enhance_candidates; single predict call
        enhance_candidates=False,  # library broadcast bug in gaussian-rank perturbation

        device="cuda:0",  # library calls torch.cuda.set_device(device) which needs an index
        categorical_indices=[int(c) for c in cats],
        random_state=42,
        verbose=True,
    )
    t1 = time.time()
    clf.fit(X_fit, y_fit)
    print(f"fit: {time.time()-t1:.1f}s")
    t2 = time.time()
    pr = np.asarray(clf.predict_proba(X_eval))
    print(f"predict: {time.time()-t2:.1f}s shape={pr.shape}")
    assert pr.shape == (len(ev), 2), f"bad shape {pr.shape}"
    assert np.isfinite(pr).all() and (pr >= 0).all() and (pr <= 1).all()

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y[ev], pr[:, 1]))
    print(f"smoke auc (diagnostic only): {auc:.4f}")
    print(f"total {time.time()-t0:.1f}s")
    import torch

    print(f"peak vram: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
