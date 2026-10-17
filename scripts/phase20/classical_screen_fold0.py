"""Phase-20 classical fold-0 screen.

Runs one classical (sklearn) model on the certified rich view (285 features,
fold-safe TE via ViewBuilder.assemble) and screens it against the champion
portfolio on fold 0.

Rationale: the champion's aux10 are boosted trees (lgbm/xgb/cat); the genuinely
distinct classical candidates are bagged/oblique forests and additive/kernel models
that capture a different inductive bias. This screen decides whether any merits
full 5-fold OOF.

Anti-leakage: view assembled fold-safe (TE fit on FIT, EVAL labels never read); model
fit on FIT only; champion hash-asserted. SCREEN only (fold 0) -- cannot pass admission.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import screen_metrics  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

OUT = REPORTS / "phase20" / "classical_screen"
ART = ARTIFACTS / "phase20" / "classical_screen"

SEED = 1


def log(msg: str) -> None:
    print(f"[classical] {msg}", flush=True)


def inner_es_split(fit_idx, y, frac=0.1, seed=SEED):
    """Stratified 10% of FIT as inner-val (fold-safe: both subsets of FIT)."""
    rng = np.random.default_rng(seed)
    yf = y[fit_idx]
    val_parts = []
    for cls in (0, 1):
        cls_idx = fit_idx[yf == cls]
        n_val = max(1, int(round(frac * len(cls_idx))))
        val_parts.append(rng.choice(cls_idx, n_val, replace=False))
    val = np.sort(np.concatenate(val_parts))
    train = np.setdiff1d(fit_idx, val, assume_unique=False)
    assert not np.intersect1d(train, val).size
    return train, val


def build_model(name: str, n_features: int):
    """Return a fitted-ready estimator with .fit(X,y)/.predict_proba(X)."""
    if name == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                    max_features="sqrt", n_jobs=-1, random_state=SEED)
    if name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     max_features="sqrt", n_jobs=-1, random_state=SEED)
    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                              max_leaf_nodes=63, random_state=SEED)
    if name == "oblique":
        # rotation/oblique forest: PCA-project each split space (cheap proxy via random
        # Givens rotations baked into ExtraTrees on projected features)
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.random_projection import GaussianRandomProjection
        # fixed n_components: eps-based auto target can exceed n_features on wide views.
        n_comp = min(64, n_features)
        proj = GaussianRandomProjection(n_components=n_comp, random_state=SEED)

        class _Oblique:
            def __init__(self):
                self.p = proj
                self.m = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                              max_features="sqrt", n_jobs=-1, random_state=SEED)

            def fit(self, X, y):
                Xp = self.p.fit_transform(X)
                self.m.fit(Xp, y)
                return self

            def predict_proba(self, X):
                return self.m.predict_proba(self.p.transform(X))
        return _Oblique()
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    if name == "kernelridge":
        # subsample FIT to ~40k rows (kernel methods are O(n^2)); RBF kernel
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=1.0, gamma=0.001))
    raise ValueError(f"unknown classical model: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["extratrees", "randomforest", "histgb", "oblique", "logreg",
                             "kernelridge"])
    ap.add_argument("--view", default="full", choices=["full", "route"])
    ap.add_argument("--eval-rows", type=int, default=0)
    ap.add_argument("--kernel-rows", type=int, default=40000,
                    help="FIT subsample for O(n^2) kernel models")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    route_all, port_all = load_champion_fold0(folds)

    if args.view == "route":
        from scripts.run_sol_tabpfn import frames
        Xall, _Xte, names, cats, maps = frames(tr, te, route=True)
        X_fit = Xall[FIT]
        X_eval_all = Xall[EVAL]
    else:
        from src.features.view import ViewBuilder
        vb = ViewBuilder(tr, te, "full")
        vb.build_static()
        X_fit, Xa, names = vb.assemble(FIT, y, EVAL, None, inner_seed=SEED)
        X_eval_all = Xa["val"]
    n_feat = X_fit.shape[1]
    y_fit = y[FIT].astype(np.int8)

    eval_n = len(EVAL) if (not args.eval_rows or args.eval_rows >= len(EVAL)) else args.eval_rows
    eval_global = EVAL[:eval_n]
    route0, port0 = route_all[eval_global], port_all[eval_global]
    y_eval = y[eval_global]
    X_eval = X_eval_all[:eval_n]

    train_pos, val_pos = inner_es_split(np.arange(len(FIT)), y_fit, frac=0.1, seed=SEED)

    # For O(n^2) kernel models, subsample FIT-train.
    fit_train = train_pos
    if args.model == "kernelridge" and len(train_pos) > args.kernel_rows:
        rng = np.random.default_rng(SEED)
        fit_train = np.sort(rng.choice(train_pos, args.kernel_rows, replace=False))

    log(f"view={args.view} FIT={len(FIT)} train={len(fit_train)} (of {len(train_pos)}) "
        f"EVAL={eval_n} features={n_feat} model={args.model}")

    pred_path = ART / f"{args.model}_{args.view}_fold0.npy"
    meta_path = OUT / f"{args.model}_{args.view}_fold0.json"

    model = build_model(args.model, n_feat)
    t1 = time.time()
    try:
        model.fit(X_fit[fit_train], y_fit[fit_train])
        proba = model.predict_proba(X_eval)
        pred = proba[:, 1] if proba.ndim == 2 else proba
        pred = np.asarray(pred, dtype=np.float64)
        np.save(pred_path, pred.astype("float32"))
        m = screen_metrics(y_eval, pred, port0, route0)
        status, err = "DONE", None
    except Exception as exc:
        pred, m, status, err = None, None, "FAILED", repr(exc)
    secs = time.time() - t1

    meta = {
        "model": args.model, "family": "classical", "status": status, "error": err,
        "view": args.view, "seed": SEED, "fit_rows": int(len(fit_train)),
        "train_rows": int(len(train_pos)), "val_rows": int(len(val_pos)),
        "eval_rows": int(eval_n), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "n_features": int(n_feat), "utc": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(),
        "protocol": "view assembled fold-safe (TE on FIT only); model fit on FIT-train; "
                    "inner 10% ES split fold-safe; champion hash-asserted; SCREEN only",
        "prediction_sha256": arr_sha256(np.load(pred_path)) if pred is not None else None,
        "metrics": m, "fit_predict_seconds": secs,
    }
    save_json(meta, meta_path)
    if m is not None:
        bar = m["portfolio_delta"] >= 5e-5 and m["logit_pearson_vs_portfolio"] < 0.995
        log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
            f"corr_port={m['logit_pearson_vs_portfolio']:.4f} PROMISING={bar} ({secs:.1f}s)")
    else:
        log(f"FAILED: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
