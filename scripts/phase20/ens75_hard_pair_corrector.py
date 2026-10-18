"""P20-ENS75-HARD-PAIR-CORRECTOR -- prereg (configs/phase20/ens75_..._prereg.json),
frozen before results.

Stacking corrector: rich-view LightGBM trained on ALL FIT rows with UNIFORM weights,
with the portfolio's own cross-fitted logit appended as an input feature. The model
learns residual corrections (hard pairs) while able to reproduce the portfolio on
easy rows (protect-easy by construction). Distinct from H19-A (difficulty reweight)
and ENS74 (margin reweight): neither reweighting here.

Candidate = corrector probabilities on all fold-0 rows (full vector, no gating).
Frozen rule: port_delta = AUC(logit-blend(cand, PORTFOLIO_REF)) - AUC(PORTFOLIO_REF)
>= +5e-5 AND Spearman(cand, PORTFOLIO_REF) < 0.995.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/ens75_hard_pair_corrector.py
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

OUT = ROOT / "reports" / "phase20" / "ens75_hard_pair_corrector"
ART = ROOT / "artifacts" / "phase20" / "ens75_hard_pair_corrector"
PREREG = ROOT / "configs" / "phase20" / "ens75_hard_pair_corrector_prereg.json"
PREREG_SHA = "0f8d059df9977b7715d75dd2b2abb2e736eb7a5808149f8f05c11e6be8fc1e5d"  # pinned before the first run (preregister-before-result)
PORTFOLIO_REF = "sol_route_aux10_primary_final"
TIE_QUANTILE = 0.30
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

    # Fold-safe rich view (TE fit on FIT), then append the stacking feature.
    from src.features.view import ViewBuilder
    vb = ViewBuilder(tr, _te, "full")
    vb.build_static()
    X_fit, Xa, names = vb.assemble(FIT, y, EVAL, None, inner_seed=SEED)
    X_eval = Xa["val"]
    y_fit = y[FIT].astype(np.int8)
    X_fit = np.column_stack([X_fit, logit(port[FIT]).astype("float32")])
    X_eval = np.column_stack([X_eval, logit(port[EVAL]).astype("float32")])
    n_feat = X_fit.shape[1]

    trn, es = inner_es_split(np.arange(len(FIT)), y_fit)

    import lightgbm as lgb

    ds = lgb.Dataset(X_fit[trn], label=y_fit[trn])
    # ES watches a FIT-train carve-out only; fold-0 labels read at final metrics.
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
    pd75 = auc_blend - auc_port
    corr = float(spearmanr(pred, port[EVAL]).statistic)
    pearson = float(np.corrcoef(logit(pred), logit(port[EVAL]))[0, 1])
    prom = bool(pd75 >= 5e-5 and corr < 0.995)

    # Descriptive: bottom-30% portfolio-margin (hard-pair) fold-0 subset, gated by
    # fold-0 predictions only.
    marg = np.abs(port[EVAL] - 0.5)
    tau = float(np.quantile(np.abs(port[FIT] - 0.5), TIE_QUANTILE))
    tie = marg <= tau
    n_tie = int(tie.sum())
    tie_port = float(roc_auc_score(y_e[tie], port[EVAL][tie])) if n_tie > 1 else None
    tie_cand = float(roc_auc_score(y_e[tie], pred[tie])) if n_tie > 1 else None

    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "fold0_candidate.npy", pred.astype("float32"))

    rep = {
        "status": "DONE",
        "prereg": "configs/phase20/ens75_hard_pair_corrector_prereg.json",
        "prereg_sha256": sha,
        "protocol": "stacking corrector: rich-view LGBM, uniform weights, portfolio OOF "
                    "logit appended as input feature; inner 10% ES from FIT-train; "
                    "candidate = corrector proba on all fold-0 rows (no gating).",
        "metrics": {
            "corrector_fold0_auc": round(auc_cand, 6),
            "portfolio_fold0_auc": round(auc_port, 6),
            "equal_logit_blend_fold0_auc": round(auc_blend, 6),
            "port_delta": round(pd75, 8),
            "candidate_spearman_vs_portfolio": round(corr, 6),
            "candidate_logit_pearson_vs_portfolio": round(pearson, 6),
            "promising_frozen_rule": prom,
            "tau": round(tau, 6),
            "tie_rows": n_tie,
            "tie_fraction": round(float(tie.mean()), 4),
            "tie_subset_portfolio_auc": round(tie_port, 6) if tie_port else None,
            "tie_subset_corrector_auc": round(tie_cand, 6) if tie_cand else None,
            "best_iteration": best,
            "n_features": int(n_feat),
            "eval_rows": int(len(EVAL)),
            "fit_rows": int(len(FIT)),
        },
    }
    (OUT / "ens75_hard_pair_corrector.json").write_text(
        json.dumps(rep, indent=1) + "\n", encoding="utf-8")

    mm = rep["metrics"]
    print(f"corrector fold0={mm['corrector_fold0_auc']} port={mm['portfolio_fold0_auc']} "
          f"blend={mm['equal_logit_blend_fold0_auc']}")
    print(f"port_delta={mm['port_delta']:+.3e} corr={mm['candidate_spearman_vs_portfolio']} "
          f"pearson={mm['candidate_logit_pearson_vs_portfolio']} -> PROMISING={prom} "
          f"(es_iter={best} {round(time.time()-t0,1)}s)")
    print(f"tie subset (n={n_tie}): port={mm['tie_subset_portfolio_auc']} "
          f"corrector={mm['tie_subset_corrector_auc']}")
    print(f"report -> {(OUT / 'ens75_hard_pair_corrector.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
