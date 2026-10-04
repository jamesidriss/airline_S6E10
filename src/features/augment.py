"""In-distribution training-support augmentation for the champion GBDT.

Motivation (measured, not assumed)
----------------------------------
The learning curve says a genuine doubling of unique in-distribution rows is worth ~+5e-4 to +7e-4
(fitted AUC ~ a + b*n^(-1/5), R^2 = 0.99894). That is roughly twice the entire rank-1-to-rank-45
spread, and about an order of magnitude more than any ensemble-level gain in this campaign. So the
highest-value remaining direction is manufacturing additional effective training support.

The honest caveat, stated up front: the learning curve proves UNIQUE GENUINE signal pays. It does NOT
prove augmentation does. Duplicating rows, or interpolating between neighbours, creates no new
information. This module exists to TEST that, with a matched control, and a null result is a real
result -- it would mean the limit is unique information rather than sample count.

Leakage discipline
------------------
Every augmentation is built from the outer-FIT rows only, using their features AND their labels.
Outer-validation rows never contribute a feature vector or a label to the donor pool. The
neighbourhood search uses TorchExactL2Index: exact fp32 on GPU, validated against a float64
reference.

Feature semantics
-----------------
A convex combination of two raw rows is only meaningful if every column has an interpolable meaning:

  * continuous (Age, Flight Distance)      -> interpolate
  * integer-like (departure/arrival delay) -> interpolate then round, staying a valid integer
  * categorical (Gender, Customer Type, Type of Travel, Class) -> NEVER interpolated numerically;
    a fractional `Class` of 1.5 is meaningless, so the value is inherited from one of the two parents
  * missingness (`Arrival Delay` NaN)     -> preserved: NaN only when both parents are NaN, and a
    single NaN parent is treated as "unknown" rather than as a value to interpolate

Interleaving those rules is what makes a naive row-mixing augmentation quietly poisonous: it would
manufacture off-manifold rows exactly where the model is most sensitive.
"""

from __future__ import annotations

import numpy as np

from src.models.realmlp import TWIN_CAPS

EPS = 1e-6

# Columns whose values are categories: inherit from a parent, never interpolate.
CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
# Columns that are integers in the survey instrument: interpolate then round back to integers.
INTEGER_LIKE = ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
                "Inflight entertainment", "Online boarding", "Seat comfort", "Food and drink",
                "WiFi service", "In-flight entertainment", "Baggage handling"]
# Genuinely continuous measures.
CONTINUOUS = ["Age"]


def _is_categorical(col):
    return (col in CATEGORICAL) or (col.endswith("__cat"))


def classify_columns(cols):
    """Partition feature names into (categorical, integer-like, continuous)."""
    cat, integer, cont = [], [], []
    for c in cols:
        if _is_categorical(c):
            cat.append(c)
        elif c in CONTINUOUS:
            cont.append(c)
        else:
            integer.append(c)
    return cat, integer, cont


def find_same_class_neighbours(X, y, k=1, device="cuda", query_chunk=256):
    """k-th nearest same-class neighbour for every row, via one retrieval index per class.

    One index per class is what GUARANTEES the donor is same-class, rather than retrieving a wide
    neighbourhood and filtering it, which silently fails whenever a class is locally sparse.

    Self-exclusion is exact and needs no special case: under exact squared-L2 a row's distance to
    itself is 0, so it is always the first hit. Asking for k+1 neighbours and dropping the first
    therefore removes exactly the self-match.
    """
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y)
    n = len(X)
    donors = np.zeros((n, k), dtype=np.int64)

    for cls in (0, 1):
        rows = np.where(y == cls)[0]
        if len(rows) <= k + 1:
            continue
        Xc = torch.as_tensor(X[rows], device=device)
        idx = TorchExactL2Index(d_main=X.shape[1], device=torch.device(device),
                                query_chunk=query_chunk)
        idx.add(Xc)
        nk = min(k + 1, len(rows))
        for s in range(0, len(rows), 4096):
            e = min(s + 4096, len(rows))
            q = Xc[torch.arange(s, e, device=device)]
            _d, nn = idx.search(q, nk)
            donors[rows[s:e]] = rows[nn[:, 1:k + 1].cpu().numpy()]
        del idx, Xc
        torch.cuda.empty_cache()

    # any row left without a donor (a class too small to search) falls back to a same-class pick
    missing = np.where((donors <= 0).any(axis=1))[0]
    if len(missing):
        for r in missing:
            same = np.where(y == y[r])[0]
            donors[r] = same[rng_pick(r, len(same), k)]
    return donors


def rng_pick(r, n_same, k):
    return np.arange(k) % max(1, n_same)


def _mix(a, b, lam, how, rng):
    """Combine two rows. ``lam`` is the weight on row ``a``."""
    if how == "cont":
        return lam * a + (1.0 - lam) * b
    if how == "int":
        v = lam * a + (1.0 - lam) * b
        v = np.where(np.isnan(a) | np.isnan(b), a, v)     # a NaN parent means "unknown", keep it
        return np.rint(v)
    # categorical: inherit wholesale from one parent
    pick_a = rng.random(len(a)) < lam
    return np.where(pick_a, a, b)


def build_local_augmentation(X_fit, y_fit, cols, ratio=1.0, alpha_lo=0.35, alpha_hi=0.65,
                             mode="interp", seed=0, device="cuda", query_chunk=256,
                             ks=(1, 2, 4), report=None):
    """Create additional training rows by local, same-class manifold mixing.

    ``X_fit``  feature matrix restricted to the outer-FIT rows
    ``y_fit``  their labels
    ``ratio``  augmented rows as a multiple of the fit rows (0.5 => half as many again)
    ``mode``   ``interp``  numeric convex combination, categories inherited
               ``donor``   numeric values inherited from a single parent, categories inherited
                           (a harder, strictly on-manifold variant)
    ``ks``     neighbour ranks to mix with; each contributes equally, so a row is averaged over
               several local donor pairs rather than relying on one arbitrary neighbour

    Returns (X_aug, y_aug, provenance) where provenance records exactly what was built so the
    experiment can be fingerprinted and reproduced.
    """
    rng = np.random.default_rng(seed)
    X_fit = np.asarray(X_fit, dtype=np.float32)
    y_fit = np.asarray(y_fit)
    n, d = X_fit.shape
    n_target = int(round(n * ratio))
    if n_target <= 0:
        return (np.zeros((0, d), np.float32), np.zeros(0, np.float32),
                {"n_aug": 0, "ratio": ratio, "mode": mode})

    cat, integer, cont = classify_columns(cols)
    cat_ix = np.array([cols.index(c) for c in cat], dtype=int)
    int_ix = np.array([cols.index(c) for c in integer], dtype=int)
    cont_ix = np.array([cols.index(c) for c in cont], dtype=int)

    # sample which source rows get augmented, and which donor rank each uses
    src = rng.choice(n, size=n_target, replace=True)
    donor_rank = rng.choice(np.asarray(ks), size=n_target, p=np.ones(len(ks)) / len(ks))

    aug = np.empty((n_target, d), dtype=np.float32)
    for k in np.unique(donor_rank):
        sel = np.where(donor_rank == k)[0]
        donors = find_same_class_neighbours(X_fit, y_fit, k=int(k), device=device,
                                            query_chunk=query_chunk)
        # donors is (n, k): ONE donor per source row is taken from column 0. Taking donors[src[sel]]
        # whole would make B three-dimensional and silently index the NEIGHBOUR axis instead of the
        # feature axis.
        b = donors[src[sel], 0]
        lam = rng.uniform(alpha_lo, alpha_hi, size=len(sel))
        A = X_fit[src[sel]]
        B = X_fit[b]
        out = np.empty_like(A)
        if cont_ix.size:
            out[:, cont_ix] = _mix(A[:, cont_ix], B[:, cont_ix], lam[:, None], "cont", rng)
        if int_ix.size:
            if mode == "donor":
                pick_a = (rng.random(len(sel)) < lam)[:, None]
                out[:, int_ix] = np.where(pick_a, A[:, int_ix], B[:, int_ix])
            else:
                out[:, int_ix] = _mix(A[:, int_ix], B[:, int_ix], lam[:, None], "int", rng)
        if cat_ix.size:
            pick_a = (rng.random(len(sel)) < lam)[:, None]
            out[:, cat_ix] = np.where(pick_a, A[:, cat_ix], B[:, cat_ix])
        aug[sel] = out

    y_aug = y_fit[src]
    prov = {"n_orig": int(n), "n_aug": int(n_target), "ratio": ratio, "mode": mode,
            "alpha_lo": alpha_lo, "alpha_hi": alpha_hi, "ks": list(map(int, ks)),
            "seed": seed, "n_categorical": len(cat_ix), "n_integer_like": len(int_ix),
            "n_continuous": len(cont_ix),
            "same_class_guaranteed": True,
            "note": "categorical values inherited from a parent, never interpolated numerically"}
    if report is not None:
        report(prov)
    return aug, y_aug, prov


def build_duplicated(X_fit, y_fit, ratio=1.0, seed=0):
    """Negative control: more physical rows with NO new information.

    If plain duplication moved the score materially, then the augmentation experiments would be
    confounded by row count, bagging or iteration count rather than measuring new support.
    """
    rng = np.random.default_rng(seed + 7717)
    X_fit = np.asarray(X_fit, dtype=np.float32)
    y_fit = np.asarray(y_fit)
    n_target = int(round(len(X_fit) * ratio))
    idx = rng.choice(len(X_fit), size=n_target, replace=True)
    return X_fit[idx].copy(), y_fit[idx].copy(), {"n_orig": int(len(X_fit)), "n_aug": int(n_target),
                                                 "ratio": ratio, "mode": "duplicate", "seed": seed}
