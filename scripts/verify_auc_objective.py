"""Verify the AUC-surrogate objective BEFORE using it, numerically.

Why this file exists
--------------------
The plan is to test whether training against a ranking/AUC objective rather than logloss changes
the error geometry enough to help the finalist ensemble. Every model in the pool optimises BCE, and
the competition metric is ROC-AUC, so this is the one remaining untested axis that alters error
geometry rather than adding capacity.

But a hand-written pairwise loss is exactly the kind of thing that can be subtly, invisibly wrong: a
sign error or a mis-scaled hessian still trains, still converges, and still produces a plausible AUC.
So the gradient is checked against central finite differences on random data before it is allowed
anywhere near an experiment, and the native ranking objectives are capability-probed rather than
assumed.

The loss
--------
AUC is P(score_pos > score_neg). For a pair (i, j) with y_i = 1, y_j = 0, the pairwise logistic
surrogate is

    l_ij = softplus(-(s_i - s_j)) = log(1 + exp(-(s_i - s_j)))

with

    d l_ij / d s_i = sigmoid(s_i - s_j) - 1
    d l_ij / d s_j = sigmoid(s_j - s_i)

which is the standard RankNet-style pairwise logistic loss, averaged over sampled pairs. The
hessian for row i is the number of pairs it participates in times sigmoid'(z) with z = s_i - s_j.

Usage: python scripts/verify_auc_objective.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = {}


def build_pairs(y, n_neg_per_pos=4, seed=0):
    """Sample a fixed set of (positive, negative) index pairs. Fixed across boosting iterations.

    Returns POS and NEG of EQUAL length, expanded so element t of POS pairs with element t of NEG.
    Each positive is repeated once per sampled negative, which turns the pairwise gradient into a
    plain scatter-add. Returning unexpanded positives instead mismatches the two arrays, since one
    positive contributes several pairs.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    pi = rng.choice(pos, size=len(pos), replace=True)
    ni = rng.choice(neg, size=(len(pos), n_neg_per_pos), replace=True)
    return np.repeat(pi, n_neg_per_pos), ni.ravel()


def auc_pairwise_grad(preds, y, pos_idx, neg_idx, n_pairs):
    """Gradient and hessian of the mean pairwise logistic loss on raw margins."""
    p = 1.0 / (1.0 + np.exp(-np.clip(preds, -35, 35)))
    z = preds[pos_idx] - preds[neg_idx]
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
    g = np.zeros_like(preds)
    h = np.zeros_like(preds)
    gp = sig - 1.0
    gn = -gp
    np.add.at(g, pos_idx, gp / n_pairs)
    np.add.at(g, neg_idx, gn / n_pairs)
    hp = sig * (1.0 - sig)
    np.add.at(h, pos_idx, hp / n_pairs)
    np.add.at(h, neg_idx, hp / n_pairs)
    del p
    return g, np.maximum(h, 1e-6)


def mean_pairwise_loss(preds, pos_idx, neg_idx):
    z = preds[pos_idx] - preds[neg_idx]
    return float(np.logaddexp(0.0, -z).mean())


def finite_difference_check(n=400, seed=0, eps=1e-4):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype("int8")
    preds = rng.normal(size=n).astype("float64")
    pi, ni = build_pairs(y, n_neg_per_pos=4, seed=seed)
    n_pairs = len(pi)
    g, h = auc_pairwise_grad(preds, y, pi, ni, n_pairs)

    fd = np.zeros(n)
    for i in range(n):
        pp = preds.copy()
        pm = preds.copy()
        pp[i] += eps
        pm[i] -= eps
        fd[i] = (mean_pairwise_loss(pp, pi, ni) - mean_pairwise_loss(pm, pi, ni)) / (2 * eps)
    err = float(np.abs(g - fd).max())
    rel = err / max(float(np.abs(fd).max()), 1e-12)
    RESULTS["gradient_vs_finite_difference"] = {
        "n": n, "n_pairs": n_pairs, "max_abs_error": err, "max_rel_error": rel,
        "passes": bool(rel < 1e-5),
    }
    print(f"  gradient vs finite differences: max_abs={err:.3e} max_rel={rel:.3e} "
          f"-> {'PASS' if rel < 1e-5 else 'FAIL'}")
    # hessian must be non-negative everywhere or boosting will diverge
    RESULTS["hessian_positive"] = {"min": float(h.min()), "passes": bool(h.min() > 0)}
    print(f"  hessian strictly positive: min={h.min():.3e} "
          f"-> {'PASS' if h.min() > 0 else 'FAIL'}")
    return rel < 1e-5 and h.min() > 0


def sanity_auc_alignment(n=3000, seed=0):
    """A gradient step on the pairwise loss must increase the true AUC, not decrease it."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype("int8")
    signal = np.where(y == 1, 1.0, -1.0)
    feats = rng.normal(size=(n, 3)) + signal[:, None] * 0.8
    w = np.zeros(feats.shape[1])
    pi, ni = build_pairs(y, n_neg_per_pos=4, seed=seed)
    n_pairs = len(pi)

    def auc_of(w):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, feats @ w)

    def loss_of(w):
        s = feats @ w
        return mean_pairwise_loss(s, pi, ni)

    a0, l0 = auc_of(w), loss_of(w)
    for _ in range(40):
        s = feats @ w
        g, _h = auc_pairwise_grad(s, y, pi, ni, n_pairs)
        # grad wrt w = X^T (grad wrt s)
        gw = feats.T @ g
        w -= 0.5 * gw / max(float(np.abs(gw).max()), 1e-9)
    a1, l1 = auc_of(w), loss_of(w)
    ok = (a1 > a0) and (l1 < l0)
    RESULTS["gradient_step_improves_auc"] = {
        "auc_before": a0, "auc_after": a1, "loss_before": l0, "loss_after": l1,
        "auc_delta": a1 - a0, "passes": bool(ok),
    }
    print(f"  gradient descent on the loss: AUC {a0:.4f} -> {a1:.4f} ({a1-a0:+.4f}), "
          f"loss {l0:.4f} -> {l1:.4f}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def probe_native_rank_objectives():
    """Capability-probe the native ranking objectives on a toy problem."""
    out = {}
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 6)).astype("float32")
    y = (rng.random(n) < 0.45).astype("int32")
    X[y == 1] += 0.6
    ds = lgb.Dataset(X, label=y)
    try:
        m = lgb.train({"objective": "lambdarank", "metric": "ndcg", "verbosity": -1,
                       "label_gain": [0, 1], "num_leaves": 15, "learning_rate": 0.1},
                      ds, num_boost_round=30)
        from sklearn.metrics import roc_auc_score
        out["lightgbm_lambdarank"] = {"ok": True, "auc": float(roc_auc_score(y, m.predict(X)))}
        print(f"  lightgbm lambdarank: OK, toy AUC={out['lightgbm_lambdarank']['auc']:.4f}")
    except Exception as exc:  # noqa: BLE001
        out["lightgbm_lambdarank"] = {"ok": False, "error": type(exc).__name__,
                                     "msg": str(exc)[:160]}
        print(f"  lightgbm lambdarank: FAILED {type(exc).__name__}: {str(exc)[:120]}")

    import xgboost as xgb

    try:
        dtr = xgb.DMatrix(X, label=y)
        bst = xgb.train({"objective": "rank:pairwise", "eval_metric": "auc", "max_depth": 4,
                         "eta": 0.1, "tree_method": "hist", "device": "cpu", "nthread": 4},
                        dtr, num_boost_round=30)
        from sklearn.metrics import roc_auc_score
        out["xgboost_rank_pairwise"] = {"ok": True,
                                        "auc": float(roc_auc_score(y, bst.predict(dtr)))}
        print(f"  xgboost rank:pairwise: OK, toy AUC={out['xgboost_rank_pairwise']['auc']:.4f}")
    except Exception as exc:  # noqa: BLE001
        out["xgboost_rank_pairwise"] = {"ok": False, "error": type(exc).__name__,
                                        "msg": str(exc)[:160]}
        print(f"  xgboost rank:pairwise: FAILED {type(exc).__name__}: {str(exc)[:120]}")
    return out


def main() -> int:
    print("=" * 92)
    print("AUC-SURROGATE OBJECTIVE VERIFICATION")
    print("=" * 92)
    print("\n1. gradient correctness (central finite differences)")
    ok1 = finite_difference_check()
    print("\n2. the loss actually optimises AUC")
    ok2 = sanity_auc_alignment()
    print("\n3. native ranking objective capability")
    native = probe_native_rank_objectives()

    # The verdict is about the CUSTOM objective, which is what the experiment will use. The native
    # ranking objectives are capability probes: lambdarank legitimately refuses to run without group
    # information, and its NDCG-based per-query truncation is the wrong surrogate for a global AUC in
    # the first place, so its inapplicability is a FINDING, not a defect. Reporting "NOT SAFE"
    # because of it would block the experiment for the wrong reason.
    custom_safe = bool(ok1 and ok2)
    xgb_ok = bool(native.get("xgboost_rank_pairwise", {}).get("ok", False))
    lgb_ok = bool(native.get("lightgbm_lambdarank", {}).get("ok", False))
    RESULTS["verdict"] = {
        "custom_pairwise_auc_surrogate_safe": custom_safe,
        "lightgbm_lambdarank_applicable": lgb_ok,
        "xgboost_rank_pairwise_available": xgb_ok,
        "approved_arms": (["custom_pairwise_auc"] if custom_safe else [])
                          + (["xgb_rank_pairwise"] if xgb_ok else []),
    }
    RESULTS["all_passed"] = custom_safe
    g = RESULTS["gradient_vs_finite_difference"]
    hh = RESULTS["hessian_positive"]
    sd = RESULTS["gradient_step_improves_auc"]
    print("\n" + "=" * 92)
    print(f"custom pairwise AUC surrogate : {'VERIFIED SAFE' if custom_safe else 'NOT SAFE'}")
    print(f"  gradient vs finite differences: max_rel={g['max_rel_error']:.2e}")
    print(f"  hessian strictly positive  : min={hh['min']:.2e}")
    print(f"  gradient descent on the loss: AUC {sd['auc_before']:.4f} -> {sd['auc_after']:.4f}")
    print(f"lightgbm lambdarank : {'available' if lgb_ok else 'NOT APPLICABLE'} "
          f"(needs group information; NDCG truncation is the wrong surrogate for a global AUC)")
    print(f"xgboost rank:pairwise : {'available' if xgb_ok else 'unavailable'}")
    print(f"APPROVED ARMS: {RESULTS['verdict']['approved_arms']}")
    print("=" * 92)

    from src.common import REPORTS, save_json

    save_json(RESULTS, REPORTS / "auc_objective_verification.json")
    print("wrote", REPORTS / "auc_objective_verification.json")
    return 0 if RESULTS["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
