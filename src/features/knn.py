"""Fold-safe k-NN target encoding -- a genuinely local, non-parametric estimator of p(x).

Why this, and why now
---------------------
Every model in the pool is a global parametric function (GBDT axis partitions, MLP affine+ReLU) and
they sit at logit-correlation 0.995-0.999 with the finalist. TabR was the attempt at a retrieval
architecture; it earned its keep as evidence (its retrieval layer is exact and GPU-native) but its
predictions are 1.2e-3 weaker standalone and no predeclared fixed blend weight improved the finalist
at either context size tried.

A k-NN target encoding is the cheapest genuinely different estimator available: instead of learning a
global function of x, it estimates p(x) as the (distance-weighted) target mean of the k nearest
labelled rows. That is local and non-parametric, so its errors are structurally unlike a GBDT's, and
AGENTS.md already records that our largest single-model lever is `extra_trees` on a *richer* view --
which is a statement that extra decorrelated columns, not more trees, are what pays.

Leakage discipline
------------------
This is a target-dependent transform, so it is the single most leak-prone block in the repository.
Three separate protections, all required:

1. The retrieval database is built ONLY from the outer-FIT rows. Outer-validation and test rows are
   queries, never database entries.
2. A fit row never retrieves itself: the inner cross-fit that produces a fit row's encoding excludes
   that row from its own database.
3. Cross-fitting, not in-sample. If fit rows were encoded against a database containing themselves,
   their neighbour labels would be partly their own -- the feature would look far more informative in
   training than it can ever be at serve time, and the model would over-trust it. Every fit row's
   encoding therefore comes from an inner fold that excluded it, exactly as for the other fold-safe
   TE blocks.

Retrieval
---------
Uses src/models/tabr_retrieval.TorchExactL2Index: exact squared-L2 on the GPU, fp32 with TF32 disabled
inside the search, query-chunked so memory does not scale with n_queries x n_database. Already
validated against a float64 reference (relative error ~2.4e-07, top-k set overlap exact).

Metric space
------------
Default is the 21 raw columns standardised, because Euclidean distance in ~285 engineered dimensions
is a poor neighbourhood metric (we measured that cost cliff behaviour in TabR) whereas the raw
survey variables are on a meaningful shared scale. This is target-free, so it is fitted on the
allowed population without any label involvement; a PCA variant is available and, being target-free
too, is likewise safe -- but any *supervised* projection must be fitted strictly inside the fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
NEIGHBOUR_KS = (16, 64, 256)


def to_logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def knn_target_encode(X_fit, y_fit, X_query, ks=NEIGHBOUR_KS, device="cuda", query_chunk=256,
                      batch_report=None):
    """Distance-weighted k-NN target encoding of ``X_query`` against a database of ``X_fit``.

    Returns a dict of column arrays. For each k:
      ``te_k``         distance-weighted mean target of the k nearest neighbours
      ``te_k_unw``     unweighted mean target of the k nearest neighbours
      ``dist_k``       mean distance to the k nearest neighbours (a confidence signal)
      ``disp_k``       weighted dispersion of neighbour targets, i.e. how mixed the neighbourhood is

    The dispersion column is what makes this block informative rather than just another smoothed
    label: two rows with the same neighbour mean but very different neighbour dispersion are very
    different points, and a GBDT can use that.

    ``y_fit`` must not contain any row that also appears in ``X_query``; the caller enforces that by
    cross-fitting (see build_knn_block).
    """
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index, full_fp32_matmul

    Xf = torch.as_tensor(np.ascontiguousarray(X_fit), dtype=torch.float32, device=device)
    Xq = torch.as_tensor(np.ascontiguousarray(X_query), dtype=torch.float32, device=device)
    y = np.asarray(y_fit, dtype="float64")
    assert len(y) == len(Xf), "y_fit must align with the retrieval database"
    kmax = max(ks)

    out = {}
    for k in ks:
        for nm in ("te", "te_unw", "dist", "disp"):
            out[f"{nm}_{k}"] = np.zeros(len(Xq), dtype="float64")

    idx = TorchExactL2Index(d_main=Xf.shape[1], device=torch.device(device),
                            query_chunk=query_chunk)
    idx.add(Xf)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    start = 0
    for s in range(0, len(Xq), 4096):
        e = min(s + 4096, len(Xq))
        d, nn = idx.search(Xq[s:e], kmax)            # ascending distances
        dy = yt[nn]                                    # (chunk, kmax) neighbour targets
        for k in ks:
            dk = d[:, :k]
            yk = dy[:, :k]
            # weight by inverse distance, shifted so the nearest neighbour dominates but no weight
            # becomes infinite when two rows coincide exactly
            w = 1.0 / (dk + 1e-3)
            wsum = w.sum(1)
            wm = (w * yk).sum(1) / wsum
            mu = yk.mean(1)
            # weighted variance of the neighbour targets
            var = (w * (yk - wm[:, None]) ** 2).sum(1) / wsum
            out[f"te_{k}"][s:e] = wm.cpu().numpy()
            out[f"te_unw_{k}"][s:e] = mu.cpu().numpy()
            out[f"dist_{k}"][s:e] = dk.mean(1).cpu().numpy()
            out[f"disp_{k}"][s:e] = np.sqrt(var.cpu().numpy())
        start = e
        if batch_report is not None and (s // 4096) % 25 == 0:
            batch_report(e, len(Xq))

    st = idx.stats()
    return out, st


def _assemble_cols(block: dict) -> tuple[np.ndarray, list[str]]:
    names = sorted(block.keys())
    M = np.column_stack([block[n] for n in names]).astype("float32")
    return M, names


def build_knn_block(X_all, y, fit_idx, val_idx, test_idx, raw_cols, ks=NEIGHBOUR_KS,
                    n_inner=5, seed=0, device="cuda", query_chunk=256, es_frac=0.10):
    """Fold-safe k-NN target encoding for one outer fold.

    ``X_all``   standardised metric-space matrix for ALL rows (target-free, so passing every row is
                safe -- it is the *labels* and the *database membership* that must be restricted)
    ``fit_idx`` outer-FIT rows: the only rows allowed in any retrieval database
    ``val_idx`` outer-validation rows: scored, never a database entry
    ``test_idx`` test rows: scored, never a database entry

    Inner cross-fitting
    -------------------
    Every fit row's encoding is produced from a database that excludes it. Validation and test rows
    are encoded against the full outer-FIT database, which is the honest serve-time situation: at
    prediction time we have all the training labels.

    The inner early-stopping rows (a 10% slice of the outer-FIT rows) are treated like validation
    rows -- i.e. excluded from their own encoding's database -- so the inner-ES block is honest too.

    Returns (columns, names, diagnostics).
    """
    y = np.asarray(y, dtype="float64")
    fit_idx = np.asarray(fit_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx) if test_idx is not None and len(test_idx) else np.array([], int)

    rng = np.random.default_rng(seed + 991)
    pos, neg = fit_idx[y[fit_idx] == 1], fit_idx[y[fit_idx] == 0]
    n_es = int(len(fit_idx) * es_frac)
    es_idx = np.unique(np.concatenate([
        rng.choice(pos, max(1, int(n_es * len(pos) / len(fit_idx))), replace=False),
        rng.choice(neg, max(1, int(n_es * len(neg) / len(fit_idx))), replace=False)]))
    pool_idx = np.setdiff1d(fit_idx, es_idx)          # rows allowed to act as database entries

    n_all = len(X_all)
    blocks = {n: np.zeros(n_all, dtype="float64") for n in (
        [f"{p}_{k}" for k in ks for p in ("te", "te_unw", "dist", "disp")])}
    diag = {"inner_folds": 0, "retrieval_stats": None, "n_pool": int(len(pool_idx))}

    # ---- inner cross-fit: every pool row encoded from a database excluding it ----
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 500)
    for a_rel, b_rel in skf.split(np.zeros(len(pool_idx)), y[pool_idx]):
        db = pool_idx[a_rel]
        qy = pool_idx[b_rel]
        blk, st = knn_target_encode(X_all[db], y[db], X_all[qy], ks=ks, device=device,
                                    query_chunk=query_chunk)
        for n, v in blk.items():
            blocks[n][qy] = v
        diag["inner_folds"] += 1
        diag["retrieval_stats"] = st

    # ---- inner-ES, outer-validation and test: full outer-FIT-minus-ES database ----
    # The ES rows are excluded so their encoding cannot see their own label.
    db = pool_idx
    for name, idxs in (("es", es_idx), ("val", val_idx), ("test", test_idx)):
        if len(idxs) == 0:
            continue
        blk, st = knn_target_encode(X_all[db], y[db], X_all[idxs], ks=ks, device=device,
                                    query_chunk=query_chunk)
        for n, v in blk.items():
            blocks[n][idxs] = v
        diag["retrieval_stats"] = st

    M, names = _assemble_cols(blocks)
    diag["n_es"] = int(len(es_idx))
    diag["n_val"] = int(len(val_idx))
    diag["n_test"] = int(len(test_idx))
    diag["names"] = names
    diag["metric_space"] = list(raw_cols)
    return M, names, diag


def standardised_raw_metric(train_df, all_df, cols):
    """Target-free metric space: standardised numerics + one-hot low-cardinality categoricals.

    Fitted on the raw TRAIN columns only. No labels are involved at any point, and no validation or
    test row influences the fit, so this is leakage-safe by construction -- it is a transform of x
    only, not of y.

    The categorical columns are one-hot rather than dropped or ordinal-coded: ``Gender``,
    ``Customer Type``, ``Type of Travel`` and ``Class`` carry real distance information for a
    neighbourhood, and an arbitrary integer code would place 'Economy' at 0 and 'Business Plus' at 2
    with a fabricated unit. One-hot keeps every pair equidistant, which is the honest encoding for a
    k-NN metric.
    """
    import pandas as pd

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in cols if c not in num_cols]

    parts, names = [], []
    if num_cols:
        tr = train_df[num_cols].to_numpy(dtype="float64")
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd < 1e-9] = 1.0
        parts.append(((all_df[num_cols].to_numpy(dtype="float64") - mu) / sd).astype("float32"))
        names += [f"num:{c}" for c in num_cols]
    for c in cat_cols:
        levels = pd.Index(sorted(train_df[c].astype(str).unique()))
        d = pd.get_dummies(all_df[c].astype(str), prefix=f"cat:{c}").reindex(
            columns=[f"{c}_{v}" for v in levels], fill_value=0)
        parts.append(d.to_numpy(dtype="float32"))
        names += [f"cat:{c}={v}" for v in levels]
    if not parts:
        raise ValueError("metric space is empty")
    M = np.hstack(parts).astype("float32")
    # constant columns carry no distance information and can destabilise the scale
    keep = M[:: max(1, len(M) // 5000)].std(0) > 1e-9
    return M[:, keep], [n for n, k in zip(names, keep) if k]
