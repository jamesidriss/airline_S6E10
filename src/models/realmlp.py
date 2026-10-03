"""RealMLP (PyTabKit) and TabM runners.

RealMLP is fed a *twin* representation: every numeric column is duplicated as a categorical
column so the network gets a dedicated embedding per exact value. The twin vocabulary is fitted
on train+test only (never on any label) so it is label-free and therefore fold-safe; see
`tests/test_leakage.py::test_twin_vocabulary_is_label_free`.
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


def build_twin_frame(X: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, list[int]]:
    """Return a copy of the design matrix with numeric->categorical twin columns.

    Returns (frame, cat_feature_indices). Only *raw* columns get twins; engineered columns are
    left numeric because RealMLP penalises feature bloat.
    """
    df = pd.DataFrame(X, columns=names)
    add = []
    for c in names:
        if c in TWIN_CAPS:
            v = df[c]
            v = v.round(6) if v.dtype.kind == "f" else v
            codes = pd.factorize(v.astype("object"), sort=True)[0].astype("int32")
            add.append((c + "__tw", codes))
    for nm, codes in add:
        df[nm] = pd.Series(codes).astype("category")
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
    p["random_state"] = p.get("random_seed", seed)
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
        m.fit(tr_d, y[a], X_val=va_d, y_val=y[b])
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
            df.iloc[:, i] = df.iloc[:, i].astype("category")
    return df


def realmlp_view(vb, folds, y, ntr, nte, params=None, seed=1, device="cuda", twin=True):
    """Fit RealMLP on a ViewBuilder's per-fold assembled matrices (TE included)."""
    from pytabkit import RealMLP_TD_Classifier

    p = {**REALMLP_BASE, "device": device, "random_state": seed, "verbosity": 0}
    if params:
        p.update({k: v for k, v in params.items() if k not in ("random_seed", "verbose")})
    p["random_state"] = p.get("random_seed", seed)
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
        tr_d = _twin_frame(Xf, names)
        va_d = _twin_frame(Xa["val"], names)
        te_d = _twin_frame(Xa["test"], names)
        cat_idx = tr_d.dtypes.index[tr_d.dtypes.astype(str) == "category"].tolist()
        m = RealMLP_TD_Classifier(**p)
        m.fit(tr_d, y[fit], X_val=va_d, y_val=y[val])
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
            codes = pd.factorize(np.asarray(v, dtype="object"), sort=True)[0]
            df[c + "__tw"] = pd.Series(codes).astype("category")
    return df


def tabm_view(vb, folds, y, ntr, nte, params=None, seed=1, device="cuda", twin=True):
    """TabM (deep ensemble of parameter-efficient linear layers)."""
    from pytabkit import TabM_D_Classifier

    p = {**TABM_BASE, "device": device, "random_state": seed, "verbosity": 0}
    if params:
        p.update({k: v for k, v in params.items() if k not in ("random_seed", "verbose")})
    p["random_state"] = p.get("random_seed", seed)
    p.pop("random_seed", None)
    oof = np.zeros(ntr)
    test = np.zeros(nte)
    t0 = time.time()
    ks = sorted(set(folds.tolist()))
    for k in ks:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte), inner_seed=k)
        tr_d = _twin_frame(Xf, names)
        va_d = _twin_frame(Xa["val"], names)
        te_d = _twin_frame(Xa["test"], names)
        m = TabM_D_Classifier(**p)
        m.fit(tr_d, y[fit], X_val=va_d, y_val=y[val])
        oof[val] = m.predict_proba(va_d)[:, 1]
        test += m.predict_proba(te_d)[:, 1] / len(ks)
        print(f"    tabm fold{k} auc={roc_auc_score(y[val], oof[val]):.6f}", flush=True)
    fa = [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in ks]
    return oof, test, fa, time.time() - t0