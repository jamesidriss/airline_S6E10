"""RealMLP (PyTabKit) and TabM runners.

RealMLP is fed a *twin* representation: every numeric column is duplicated as a categorical
column so the network gets a dedicated embedding per exact value. Twins retain
their literal values: the estimator's FIT-only ordinal encoder maps those values
consistently at validation and inference. Checkpoint selection uses an inner
10 percent partition of outer FIT, never the outer evaluation labels.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from src.features.s6e10 import META4, NUMS, SURVEY13  # noqa: E402

# Numeric columns whose cardinality makes an exact-value categorical twin useful.
TWIN_CAPS: dict[str, int] = {
    "Flight Distance": 6000,
    "Age": 200,
    "Departure Delay in Minutes": 400,
    "Arrival Delay in Minutes": 400,
}

TRAINING_PROTOCOL = "sol_v2_inner10_literal_twins"


def fit_inner_validation(model, frame, labels, seed=1):
    """Fit/select checkpoints using only outer-FIT rows and labels.

    ``use_early_stopping=False`` alone does not disable PyTabKit's default
    best-epoch restoration. Both its validation and restoration must therefore
    be isolated from the outer evaluation population.
    """
    labels = np.asarray(labels)
    if len(frame) != len(labels):
        raise ValueError("FIT frame/label lengths differ")
    from scripts.run_views import _inner_es_split
    train, es = _inner_es_split(np.arange(len(labels)), labels, seed)
    assert not np.intersect1d(train, es).size
    model.fit(frame.iloc[train], labels[train],
              X_val=frame.iloc[es], y_val=labels[es])
    return model


def build_twin_frame(X: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Return a copy of the design matrix with numeric->categorical twin columns.

    Returns (frame, cat_feature_indices). Only *raw* columns get twins; engineered columns are
    left numeric because RealMLP penalises feature bloat.
    """
    df = _twin_frame(X, names)
    cat_idx = [i for i, c in enumerate(df.columns) if str(df[c].dtype) == "category"]
    return df, cat_idx


REALMLP_BASE = dict(
    n_ens=8, n_epochs=4, batch_size=4096, use_early_stopping=False,
    val_metric_name="1-auc_ovr", lr=0.053, wd=0.015, sq_mom=0.988,
    lr_sched="flat_anneal", wd_sched="cos_log_15", first_layer_lr_factor=0.25,
    embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512, 256, 128],
    act="silu", p_drop=0.05, p_drop_sched="invsqrtp1e-3",
    plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151,
    plr_sigma=2.33, ls_eps=0.01, ls_eps_sched="sqrt_cos",
    add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
    n_cv=1, n_refit=0,
)

TABM_BASE = dict(
    n_epochs=60, batch_size=2048, val_metric_name="1-auc_ovr", patience=12,
    tabm_k=8, d_block=256, n_blocks=3, n_cv=1, n_refit=0,
)


def realmlp(X, y, Xte, folds, names, params=None, seed=1, cat_idx=None, device="cuda",
            n_splits=5):
    """Train RealMLP per fold on the *given* folds. Returns (oof, test, fold_aucs, seconds)."""
    from pytabkit import RealMLP_TD_Classifier

    p = {**REALMLP_BASE, "device": device, "random_state": seed, "verbosity": 0}
    if params:
        p.update({k: v for k, v in params.items() if k not in ("random_seed", "verbose")})
    p["random_state"] = (params or {}).get("random_seed", seed)
    p.pop("random_seed", None)

    oof = np.zeros(len(y), dtype="float64")
    test = np.zeros(len(Xte), dtype="float64")
    t0 = time.time()
    ks = sorted(set(np.asarray(folds).tolist()))
    for k in ks:
        a = np.where(np.asarray(folds) != k)[0]
        b = np.where(np.asarray(folds) == k)[0]
        tr_d = _frame(X[a], names, cat_idx)
        va_d = _frame(X[b], names, cat_idx)
        te_d = _frame(Xte, names, cat_idx)
        m = RealMLP_TD_Classifier(**p)
        fit_inner_validation(m, tr_d, y[a])
        oof[b] = m.predict_proba(va_d)[:, 1]
        test += m.predict_proba(te_d)[:, 1] / len(ks)
        print(f"    realmlp fold{k} auc={roc_auc_score(y[b], oof[b]):.6f}", flush=True)
    fa = [float(roc_auc_score(y[np.asarray(folds) == k], oof[np.asarray(folds) == k])) for k in ks]
    return oof, test, fa, time.time() - t0


def _frame(X, names, cat_idx=None):
    """Build a model-ready frame from a raw numpy matrix, marking twin columns as category."""
    df = pd.DataFrame(X, columns=names)
    if cat_idx is not None and len(cat_idx):
        for i in cat_idx:
            df[names[i]] = df[names[i]].astype("category")
    return df


def realmlp_view(vb, folds, y, ntr, nte, params=None, seed=1, device="cuda", twin=True):
    """Fit RealMLP on a ViewBuilder's per-fold assembled matrices (TE included)."""
    from pytabkit import RealMLP_TD_Classifier

    p = {**REALMLP_BASE, "device": device, "random_state": seed, "verbosity": 0}
    if params:
        p.update({k: v for k, v in params.items() if k not in ("random_seed", "verbose")})
    p["random_state"] = (params or {}).get("random_seed", seed)
    p.pop("random_seed", None)

    oof = np.zeros(ntr, dtype="float64")
    test = np.zeros(nte, dtype="float64")
    t0 = time.time()
    ks = sorted(set(folds.tolist()))
    cat_names = None
    for k in ks:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte), inner_seed=k)
        frame_builder = _twin_frame if twin else _frame
        tr_d = frame_builder(Xf, names)
        va_d = frame_builder(Xa["val"], names)
        te_d = frame_builder(Xa["test"], names)
        cat_idx = tr_d.dtypes.index[tr_d.dtypes.astype(str) == "category"].tolist()
        m = RealMLP_TD_Classifier(**p)
        fit_inner_validation(m, tr_d, y[fit])
        oof[val] = m.predict_proba(va_d)[:, 1]
        test += m.predict_proba(te_d)[:, 1] / len(ks)
        print(f"    realmlp fold{k} auc={roc_auc_score(y[val], oof[val]):.6f} "
              f"cols={tr_d.shape[1]} ncat={len(cat_idx)}", flush=True)
    fa = [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in ks]
    return oof, test, fa, time.time() - t0


def _twin_frame(X, names):
    """Raw design matrix -> model frame, with an exact-value categorical twin of the
    high-value-numeric raw columns. The twin vocabulary is derived from the values themselves
    only (never a label), so it is label-free and fold-safe."""
    df = pd.DataFrame(X, columns=names)
    for c in TWIN_CAPS:
        if c in df.columns:
            v = df[c]
            v = v.round(6) if getattr(v.dtype, "kind", "i") == "f" else v
            # Preserve semantic values; independent factorization would alias
            # different distances whenever split vocabularies differ.
            df[c + "__tw"] = pd.Categorical(v)
    return df


def tabm_view(vb, folds, y, ntr, nte, params=None, seed=1, device="cuda", twin=True):
    """TabM (deep ensemble of parameter-efficient linear layers)."""
    from pytabkit import TabM_D_Classifier

    p = {**TABM_BASE, "device": device, "random_state": seed, "verbosity": 0}
    if params:
        p.update({k: v for k, v in params.items() if k not in ("random_seed", "verbose")})
    p["random_state"] = (params or {}).get("random_seed", seed)
    p.pop("random_seed", None)
    oof = np.zeros(ntr)
    test = np.zeros(nte)
    t0 = time.time()
    ks = sorted(set(folds.tolist()))
    for k in ks:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte), inner_seed=k)
        frame_builder = _twin_frame if twin else _frame
        tr_d = frame_builder(Xf, names)
        va_d = frame_builder(Xa["val"], names)
        te_d = frame_builder(Xa["test"], names)
        m = TabM_D_Classifier(**p)
        fit_inner_validation(m, tr_d, y[fit])
        oof[val] = m.predict_proba(va_d)[:, 1]
        test += m.predict_proba(te_d)[:, 1] / len(ks)
        print(f"    tabm fold{k} auc={roc_auc_score(y[val], oof[val]):.6f}", flush=True)
    fa = [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in ks]
    return oof, test, fa, time.time() - t0
