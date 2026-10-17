"""Phase 19 -- foundation-model screen core (fold-0 discovery protocol).

Five installable tabular foundation models (TabICL-v2, TabDPT, Causilo, TabFM,
xRFM) are each given a chance to become an 11th ensemble member. The champion
route arm is TabPFN-v3.5 full-context on the raw-21 view; to keep the screen
apples-to-apples every candidate is fed the SAME raw representation (via the
champion's own `frames()` builder) and a fold-0-safe stratified context drawn
only from FIT (folds != eval).

Anti-leakage: context rows are a subset of FIT (never the eval fold); no target
encoding is used (raw view only); the champion reference is loaded from hash-
asserted phase17 artifacts. This is a SCREEN (fold 0 only) -- it cannot pass
the OOF admission gate, it only decides whether a model merits full 5-fold OOF.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.validation.compare import logit

# Fold-safe stratified context seed (frozen).
CONTEXT_SEED = 20261023
# Clip bound for logit transforms of probabilities at the 0/1 boundary.
_EPS = 1e-7


def stratified_context(fit_idx: np.ndarray, y: np.ndarray, n: int, seed: int = CONTEXT_SEED) -> np.ndarray:
    """Stratified subsample of FIT rows for in-context learning. Fold-safe by
    construction (subset of `fit_idx`, which never contains the eval fold)."""
    rng = np.random.default_rng(seed)
    yf = y[fit_idx]
    pos = fit_idx[yf == 1]
    neg = fit_idx[yf == 0]
    rate = float((yf == 1).mean())
    n_pos = min(int(round(n * rate)), len(pos))
    n_neg = min(n - n_pos, len(neg))
    sel = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    ctx = np.sort(sel)
    assert np.isin(ctx, fit_idx).all(), "context leaked outside FIT"
    assert len(np.unique(ctx)) == len(ctx), "duplicate context rows"
    return ctx


def _cl(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)


def screen_metrics(y_eval: np.ndarray, pred: np.ndarray, port0: np.ndarray, route0: np.ndarray) -> dict:
    """Standalone AUC + decorrelation + equal-logit portfolio delta vs champion."""
    pred = np.asarray(pred, dtype=np.float64)
    auc = float(roc_auc_score(y_eval, pred))
    auc_port = float(roc_auc_score(y_eval, port0))
    auc_route = float(roc_auc_score(y_eval, route0))
    # equal-logit blend of champion portfolio + candidate (an 11th-member proxy)
    blend_logit = (logit(_cl(port0)) + logit(_cl(pred))) / 2.0
    blend = 1.0 / (1.0 + np.exp(-blend_logit))
    auc_blend = float(roc_auc_score(y_eval, blend))
    lp_port = float(np.corrcoef(logit(_cl(pred)), logit(_cl(port0)))[0, 1])
    lp_route = float(np.corrcoef(logit(_cl(pred)), logit(_cl(route0)))[0, 1])
    return {
        "standalone_auc": auc,
        "champion_portfolio_auc": auc_port,
        "champion_route_auc": auc_route,
        "equal_logit_blend_auc": auc_blend,
        "portfolio_delta": auc_blend - auc_port,
        "logit_pearson_vs_portfolio": lp_port,
        "logit_pearson_vs_route": lp_route,
        "pred_std": float(pred.std()),
        "pred_mean": float(pred.mean()),
    }


# ------------------------------------------------------------------ adapters
def build_model(name: str, device: str, seed: int = 1201):
    """Return a fitted-ready estimator with .fit(X,y) / .predict_proba(X).

    Each foundation model is an in-context learner (fit = ingest context rows,
    predict = forward pass). xRFM is the exception (a fitted kernel machine) and
    is wrapped to expose the same fit(X,y)/predict_proba signature by carving an
    inner validation split out of the context (fold-safe).
    """
    if name == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=device, n_estimators=8, allow_auto_download=True)
    if name == "tabdpt":
        from tabdpt import TabDPTClassifier
        # compile=False avoids a long recompile per call; context_reduction handles
        # large context by subsampling inside the model. use_flash=False because the
        # flash-attention kernel is unavailable on the RTX 5070 Ti (sm_120) -- verified
        # to otherwise raise "No available kernel".
        return TabDPTClassifier(device=device, context_reduction="subsample", compile=False,
                                use_flash=False, verbose=False)
    if name == "causilo":
        from causilo import CausiloClassifier
        return CausiloClassifier(device=device, n_estimators=8, random_state=seed)
    if name == "tabfm":
        from tabfm.src.jax.tabfm_v1_0_0 import load
        from tabfm.src.classifier_and_regressor import TabFMClassifier
        model = load("classification")
        return TabFMClassifier(model=model, n_estimators=8, batch_size=8, random_state=seed)
    if name == "xrfm":
        from xrfm import xRFM
        return _XRFMWrapper(xRFM(device=device, random_state=seed, verbose=False))
    raise ValueError(f"unknown foundation model: {name}")


class _XRFMWrapper:
    """Adapt xRFM (fit(X,y,X_val,y_val)) to the fit(X,y)/predict_proba interface
    by carving a stratified inner validation split out of the context."""

    def __init__(self, inner):
        self.inner = inner

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        Xtr, Xv, ytr, yv = train_test_split(
            np.asarray(X), np.asarray(y), test_size=0.15, random_state=1201, stratify=np.asarray(y))
        self.inner.fit(Xtr, ytr, Xv, yv)
        return self

    def predict_proba(self, X):
        p = np.asarray(self.inner.predict_proba(np.asarray(X)))
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        return np.stack([1.0 - p, p], axis=1)
