"""P20-ENS74-NEAR-TIE-SPECIALIST -- prereg (configs/phase20/ens74_..._prereg.json),
frozen before results.

Portfolio-margin-weighted rich-view LightGBM: near-tie rows (|port_OOF - 0.5| <= tau,
tau = 30% quantile of FIT margins) get sample_weight 3.0, all others 1.0. Portfolio
margins are fold-safe (PORTFOLIO_REF OOF; fold-0 rows gated by fold-0 predictions only).

Candidate = specialist probabilities on fold-0. Frozen rule:
port_delta = AUC(logit-blend(cand, PORTFOLIO_REF)) - AUC(PORTFOLIO_REF) >= +5e-5
AND Spearman(cand, PORTFOLIO_REF) < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/ens74_near_tie_specialist.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_cached_parquet  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "ens74_near_tie_specialist"
ART = ROOT / "artifacts" / "phase20" / "ens74_near_tie_specialist"
PREREG = ROOT / "configs" / "phase20" / "ens74_near_tie_specialist_prereg.json"
PREREG_SHA = "febd665e1823126a245eca51f0c9e6396dfb9dd5a1b7338d33fa81b4a9d95474"  # pinned before the first run (preregister-before-result)
PORTFOLIO_REF = "sol_route_aux10_primary_final"
TIE_QUANTILE = 0.30
TIE_WEIGHT = 3.0
SEED = 1

LGBM_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.02,
    "num_leaves": 127, "min_child_samples": 40, "colsample_bytree": 0.8,
    "subsample": 0.8, "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255,
    "extra_trees": True, "n_estimators": 6000, "verbose": -1, "n_jobs": 8,
    "random_state": SEED, "bagging_seed": SEED + 1, "feature_fraction_seed": SEED + 2,
}


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def inner_es_split(fit: np.ndarray, y: np.ndarray, frac: float = 0.10, seed: int = SEED):
    rng = np.random.default_rng(seed + 991)
    pos, neg = fit[y[fit] == 1], fit[y[fit] == 0]
    n = int(len(fit) * frac)
    es = np.concatenate([rng.choice(pos, max(1, int(n * len(pos) / len(fit))), replace=False),
                         rng.choice(neg, max(1, int(n * len(neg) / len(fit))), replace=False)])
    es = np.unique(es)
    return np.setdiff1d(fit, es), es


def main() -> int:
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"
    print(f"prereg sha256 {sha}")

    t0 = time.time()
    tr, _te = load_cached_parquet()
    y = tr[TARGET].to_numpy(dtype="int8")
    ids = tr[ID_COL].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    port = load_oof(PORTFOLIO_REF).astype(np.float64)
    assert np.isfinite(port).all() and 0.0 <= port.min() and port.max() <= 1.0

    # Fold-safe margins: FIT-side OOF margins set tau; fold-0 gating uses fold-0 preds.
    marg_fit = np.abs(port[FIT] - 0.5)
    tau = float(np.quantile(marg_fit, TIE_QUANTILE))
    tie_fit = marg_fit <= tau

    # Rich view: fold-safe TE (fit on FIT), returns FIT-positioned train + EVAL matrix.
    from src.features.view import ViewBuilder
    vb = ViewBuilder(tr, _te, "full")
    vb.build_static()
    X_fit, Xa, names = vb.assemble(FIT, y, EVAL, None, inner_seed=SEED)
    X_eval = Xa["val"]
    y_fit = y[FIT].astype(np.int8)
    n_feat = X_fit.shape[1]

    trn, es = inner_es_split(np.arange(len(FIT)), y_fit)
    w = np.ones(len(FIT), dtype=np.float64)
    w[tie_fit] = TIE_WEIGHT

    import lightgbm as lgb

    ds = lgb.Dataset(X_fit[trn], label=y_fit[trn], weight=w[trn])
    # NOTE: early stopping must NEVER watch fold-0 AUC (leaks selection) -> carve ES
    # from FIT-train rows only (fold-safe); fold-0 labels are read at final metrics.
    dv = lgb.Dataset(X_fit[es], label=y_fit[es], reference=ds)
    m = lgb.train(dict(LGBM_PARAMS), ds, num_boost_round=LGBM_PARAMS["n_estimators"],
                  valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    best = int(m.best_iteration)
    pred = m.predict(X_eval, num_iteration=best)

    y_e = y[EVAL]
    auc_cand = float(roc_auc_score(y_e, pred))
    auc_port = float(roc_auc_score(y_e, port[EVAL]))
    blend = (logit(pred) + logit(port[EVAL])) / 2.0
    auc_blend = float(roc_auc_score(y_e, blend))
    pd74 = auc_blend - auc_port
    corr = float(spearmanr(pred, port[EVAL]).statistic)
    prom = bool(pd74 >= 5e-5 and corr < 0.995)

    # Descriptive: near-tie subset behavior (fold-0 rows gated by fold-0 port preds)
    tie_eval = np.abs(port[EVAL] - 0.5) <= tau
    n_tie = int(tie_eval.sum())
    tie_port_auc = float(roc_auc_score(y_e[tie_eval], port[EVAL][tie_eval])) if n_tie > 1 else None
    tie_blend_auc = float(roc_auc_score(y_e[tie_eval], blend[tie_eval])) if n_tie > 1 else None

    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "fold0_candidate.npy", pred.astype("float32"))

    rep = {
        "status": "DONE",
        "prereg": "configs/phase20/ens74_near_tie_specialist_prereg.json",
        "prereg_sha256": sha,
        "protocol": "portfolio-margin-weighted rich-view LGBM (tie rows 3x, tau = 30% FIT "
                    "margin quantile, PORTFOLIO_REF OOF; fold-0 gated by predictions only); "
                    "inner 10% ES from FIT-train; candidate = specialist proba on fold-0.",
        "metrics": {
            "specialist_fold0_auc": round(auc_cand, 6),
            "portfolio_fold0_auc": round(auc_port, 6),
            "equal_logit_blend_fold0_auc": round(auc_blend, 6),
            "port_delta": round(pd74, 8),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "promising_frozen_rule": prom,
            "tau": round(tau, 6),
            "fit_tie_fraction": round(float(tie_fit.mean()), 4),
            "eval_tie_rows": n_tie,
            "eval_tie_fraction": round(float(tie_eval.mean()), 4),
            "tie_subset_portfolio_auc": round(tie_port_auc, 6) if tie_port_auc else None,
            "tie_subset_blend_auc": round(tie_blend_auc, 6) if tie_blend_auc else None,
            "best_iteration": best,
            "n_features": int(n_feat),
            "eval_rows": int(len(EVAL)),
            "fit_rows": int(len(FIT)),
        },
    }
    (OUT / "ens74_near_tie_specialist.json").write_text(
        json.dumps(rep, indent=1) + "\n", encoding="utf-8")

    mm = rep["metrics"]
    print(f"specialist fold0={mm['specialist_fold0_auc']} port={mm['portfolio_fold0_auc']} "
          f"blend={mm['equal_logit_blend_fold0_auc']}")
    print(f"port_delta={mm['port_delta']:+.3e} corr={mm['candidate_spearman_vs_portfolio']} "
          f"-> PROMISING={prom} (tau={mm['tau']} tie_fit={mm['fit_tie_fraction']} "
          f"tie_eval={mm['eval_tie_fraction']} es_iter={best} {round(time.time()-t0,1)}s)")
    print(f"tie subset: port={mm['tie_subset_portfolio_auc']} "
          f"blend={mm['tie_subset_blend_auc']} (n={n_tie})")
    print(f"report -> {(OUT / 'ens74_near_tie_specialist.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
