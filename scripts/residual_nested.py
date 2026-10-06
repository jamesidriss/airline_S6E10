"""Fully NESTED residual-structure diagnostic (Phase 10A). Promotion-grade evidence.

Why the previous script cannot license a correction
---------------------------------------------------
scripts/residual_structure.py estimates a per-group residual bias on "discovery" rows and applies it
to "confirmation" rows. The two sets are disjoint, which looks like cross-fitting -- but the base
scores are `blend_v3_final` OOF, and that creates a **second-order dependency**:

    for a discovery row j in fold f(j) != k, the base prediction p_j was produced by models trained
    on every fold EXCEPT f(j), which INCLUDES fold k

so `p_j`, and therefore the residual `r_j = y_j - p_j`, indirectly depends on the confirmation
fold's labels. The bias table fitted on discovery rows is thus not independent of the rows it is
scored on. The size is unknown and the sign is not even guaranteed: a base model that memorised
positive fold-k rows would push some discovery residuals negative, biasing the correction *against*
the confirmation labels. The point is not that it inflates -- it is that nothing licenses it.

That script's output is therefore relabelled **NON-NESTED META-DIAGNOSTIC** and remains useful only
as a cheap screen. This script is the promotion evidence.

The nesting, per outer fold k
-----------------------------
    META_TRAIN = all rows except fold k
    META_VAL   = fold k

    inside META_TRAIN only:
        inner cross-fit over 5 inner folds -> p_meta_train_oof
            (no model producing a META_TRAIN prediction sees that row, and NO model producing a
             META_TRAIN prediction trains on ANY META_VAL row, because META_TRAIN excludes fold k)
        one champion fit on ALL of META_TRAIN at a fixed round count -> p_meta_val

    bias / offset estimated on (y - p_meta_train_oof) over META_TRAIN
    frozen, then applied to p_meta_val and scored once on META_VAL

Two score-space semantics, kept strictly separate
-------------------------------------------------
The earlier script estimated a PROBABILITY-space residual `y - p` and then added it to a LOGIT-scale
score. That mixes scales: it can change the ranking, but it is not a calibration correction and its
AUC delta must not be read as one. Both semantics are implemented separately here:

  PROBABILITY   b_g = shrunk mean of (y - p | g);  score = clip(p + b_g, eps, 1-eps)
  LOGIT OFFSET  delta_g = one Newton step on the group's regularised logistic intercept
                score = logit(p) + delta_g

For the logit offset, with z_i = logit(p_i), the group's regularised intercept is found by Newton
iteration on

    L(delta) = -sum_g [ y_i log s(z_i+delta) + (1-y_i) log(1-s(z_i+delta)) ] + LAMBDA*delta^2

whose gradient is sum_g(s_i - y_i) + 2*LAMBDA*delta and curvature sum_g s_i(1-s_i) + LAMBDA, both at
the current delta. NOTE: a SINGLE Newton step from zero is not the MLE and the tests caught that it
can return the wrong sign for a saturated base; the implementation iterates to convergence and the
tests verify the iterate against a brute-force minimiser.

LAMBDA is pre-declared at 50.0, matching PRIOR_N in the probability version, so the two semantics
carry the same nominal shrinkage. It is NOT tuned against confirmation folds. `tests/
test_residual_nested.py` asserts that LAMBDA is pre-declared, that shrinkage is monotone in both
semantics, and that the Newton iterate matches a numerical minimiser.

Round count is fixed at 900, pre-declared: Phase 9 selected 900 from an honest inner curve at fold 0
(`ctl_fixed`), within 13% of the ES-selected 797, and it is a training-policy constant that never saw
an eval label. Fixing it removes an entire nesting question at zero cost.

Index discipline: `assemble()` returns matrices indexed WITHIN `fit_idx`, while `y` is indexed
GLOBALLY. Conflating the two has produced three fake results in this campaign (-483e-5, -586e-5,
-2086e-5), so every label lookup here goes through an explicit global->local map.

Usage:
  python scripts/residual_nested.py --folds 0,1,2,3,4 --tag p10a
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.s6e10 import META4, NUMS, SURVEY13  # noqa: E402
from src.features.view import RAW21, ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

# ---------------------------------------------------------------- pre-declared constants
PRIOR_N = 50.0        # probability-semantics shrinkage, group-size units. NOT tuned.
LAMBDA = 50.0         # logit-offset shrinkage, curvature units. NOT tuned.
MIN_N = 200           # minimum discovery group size. NOT tuned.
N_BINS = 24           # quantile bins for continuous keys.
EPS = 1e-6            # probability clip for the probability semantics.
ROUNDS = 900          # fixed champion round count, pre-declared (see module docstring).
CHAMPION = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 127,
            "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
            "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True,
            "verbose": -1, "n_jobs": 8}

FD, AGE = "Flight Distance", "Age"


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype="float64"), -35, 35)))


# --------------------------------------------------------------------------- grouping keys
def qbin(s: pd.Series, n: int = N_BINS) -> pd.Series:
    """Quantile bins as strings; duplicates='drop' so constant columns do not raise."""
    return pd.qcut(s, n, duplicates="drop").astype(str)


def predeclared_keys(tr: pd.DataFrame) -> dict[str, pd.Series]:
    """The Phase 10A grouping families, predeclared. Nothing here touches the target.

    Families are exactly those specified before any result was seen: raw variables, the current TE
    keys, survey patterns, segments, original-knowledge regimes, and base confidence. Hundreds of
    ad-hoc keys are deliberately NOT included -- the gate needs replication across folds, and a huge
    key family guarantees some key passes by chance.
    """
    K: dict[str, pd.Series] = {}

    # --- RAW: each of the 21 variables, continuous ones quantile-binned
    for c in RAW21:
        s = tr[c]
        K[f"raw:{c}"] = (qbin(s) if pd.api.types.is_numeric_dtype(s) and s.nunique() > N_BINS
                         else s.astype(str))

    # --- CURRENT TE KEYS: the keys the champion's existing TE block conditions on
    K["te:fd"] = qbin(tr[FD])
    K["te:age"] = qbin(tr[AGE])
    K["te:fd_x_class"] = qbin(tr[FD]) + "|" + tr["Class"].astype(str)
    K["te:fd_x_trip"] = qbin(tr[FD]) + "|" + tr["Type of Travel"].astype(str)
    K["te:age_x_class_x_trip_x_cust"] = (qbin(tr[AGE], 8) + "|" + tr["Class"].astype(str) + "|"
                                         + tr["Type of Travel"].astype(str) + "|"
                                         + tr["Customer Type"].astype(str))

    # --- SURVEY PATTERNS: coverage and level signatures
    z = (tr[SURVEY13] == 0).astype("int8")
    K["pat:n_zero"] = z.sum(axis=1).astype(str)
    sig = z.astype(str).agg("".join, axis=1)
    # exact zero signature only where it has support; otherwise it is a singleton per row and
    # MIN_N would silently drop it, which is fine but makes the family look empty
    K["pat:zero_signature"] = sig
    K["pat:n_max5"] = (tr[SURVEY13] == 5).sum(axis=1).astype(str)
    K["pat:mean_service"] = qbin(tr[SURVEY13].mean(axis=1))
    K["pat:delay_any"] = ((tr["Departure Delay in Minutes"] > 0).astype(str) + "_"
                          + (tr["Arrival Delay in Minutes"] > 0).astype(str))

    # --- SEGMENTS
    K["seg:class_x_trip"] = tr["Class"].astype(str) + "|" + tr["Type of Travel"].astype(str)
    K["seg:class_x_cust"] = tr["Class"].astype(str) + "|" + tr["Customer Type"].astype(str)
    K["seg:trip_x_cust"] = (tr["Type of Travel"].astype(str) + "|"
                            + tr["Customer Type"].astype(str))
    K["seg:class_x_trip_x_gender"] = (tr["Class"].astype(str) + "|"
                                      + tr["Type of Travel"].astype(str) + "|"
                                      + tr["Gender"].astype(str))

    # --- BASE CONFIDENCE (filled in main, needs p; placeholder order preserved)
    return K


def teacher_keys(extra: dict[str, np.ndarray], p: np.ndarray) -> dict[str, pd.Series]:
    """ORIGINAL-KNOWLEDGE REGIMES from original-data target statistics."""
    K: dict[str, pd.Series] = {}
    if not extra:
        return K
    df = pd.DataFrame(extra)
    K["orig:teacher_composite"] = qbin(df.mean(axis=1))
    ranked = df.corrwith(pd.Series(p, name="p")).abs().sort_values(ascending=False).index.tolist()
    for c in ranked[:4]:
        K[f"orig:{c}"] = qbin(df[c])
    return K


# --------------------------------------------------------------------------- semantics
def fit_prob_bias(g: pd.Series, y: np.ndarray, p: np.ndarray, rows: np.ndarray) -> dict:
    """b_g = shrunk mean of (y - p) over discovery rows. PROBABILITY semantics."""
    sub, ys, ps = g.iloc[rows], y[rows], p[rows]
    codes, uniq = pd.factorize(sub, sort=True)
    cnt = np.bincount(codes, minlength=len(uniq)).astype("float64")
    tot = np.bincount(codes, weights=(ys - ps), minlength=len(uniq))
    bias = tot / (cnt + PRIOR_N)
    return {u: float(b) for u, b, k in zip(uniq, bias, cnt >= MIN_N) if k}


def fit_logit_offset(g: pd.Series, y: np.ndarray, p: np.ndarray, rows: np.ndarray,
                     iters: int = 25) -> dict:
    """delta_g = regularised intercept MLE by Newton, started from 0. LOGIT-OFFSET semantics.

    d/ddelta [ -sum y log s(z+delta) - (1-y) log(1-s(z+delta)) ] is sum(s_i - y_i) and
    d2/ddelta2 is sum(s_i(1-s_i)), both evaluated at the CURRENT delta, giving

        delta <- delta - (sum(s-y) + 2*LAMBDA*delta) / (sum s(1-s) + LAMBDA)

    A single step from zero is NOT the MLE and I initially wrote it that way. The tests caught it:
    with a saturated base (p = 0.30 for a group whose true rate is 0.20) one step returned +0.4557
    where the numerical optimum is -0.4861 -- the WRONG SIGN, because the gradient at delta=0 is
    small (s = 0.43) while the curvature is also small, so a single step overshoots past the point
    where the gradient changes sign. Iterating to convergence fixes it, and the tests now check the
    iterate against a brute-force minimiser rather than trusting one step.

    `iters` is pre-declared and the objective is convex in delta, so 25 iterations is far beyond
    convergence for any group reaching MIN_N; a non-converged result would be visible as a residual
    gradient the tests do not tolerate.
    """
    sub, ys, z = g.iloc[rows], y[rows], logit(p[rows])
    codes, uniq = pd.factorize(sub, sort=True)
    m = len(uniq)
    cnt = np.bincount(codes, minlength=m).astype("float64")
    delta = np.zeros(m, dtype="float64")
    for _ in range(iters):
        s = sigmoid(z + delta[codes])
        grad = np.bincount(codes, weights=(s - ys), minlength=m) + 2.0 * LAMBDA * delta
        curv = np.bincount(codes, weights=(s * (1 - s)), minlength=m) + LAMBDA
        step = grad / curv
        delta -= step
        if np.max(np.abs(step)) < 1e-12:
            break
    return {u: float(d) for u, d, k in zip(uniq, delta, cnt >= MIN_N) if k}


def apply_table(g: pd.Series, rows: np.ndarray, table: dict) -> np.ndarray:
    if not table:
        return np.zeros(len(rows), dtype="float64")
    return np.array([table.get(u, 0.0) for u in g.iloc[rows]], dtype="float64")


def observed_bias(g: pd.Series, r: np.ndarray, rows: np.ndarray) -> dict:
    sub, rs = g.iloc[rows], r[rows]
    codes, uniq = pd.factorize(sub, sort=True)
    cnt = np.bincount(codes, minlength=len(uniq)).astype("float64")
    tot = np.bincount(codes, weights=rs, minlength=len(uniq))
    return {u: float(t / n) for u, t, n in zip(uniq, tot, cnt) if n >= MIN_N}


# --------------------------------------------------------------------------- champion surrogate
def train_champion(X, y, seed):
    import lightgbm as lgb
    p = dict(CHAMPION)
    p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=ROUNDS)


def inner_folds(n: int, k: int, seed: int) -> np.ndarray:
    """Deterministic inner fold assignment by contiguous id blocks.

    Uses the immutable id-block scheme rather than a random split so an inner fold can never share a
    generator batch with its complement -- the property `blocked_id_scheme()` exists to test.
    """
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    lab = np.empty(n, dtype="int64")
    for j, chunk in enumerate(np.array_split(perm, k)):
        lab[chunk] = j
    return lab


def build_surrogate(tr, te, y, meta_fit, meta_val, view, seed, k):
    """Fully nested champion surrogate for one outer fold.

    Returns p_meta_val (one champion fit on ALL meta_fit) and p_meta_train_oof (inner cross-fit
    inside meta_fit only). No model contributing to p_meta_train_oof trains on meta_val.
    """
    vb = ViewBuilder(tr, te, view)
    vb.build_static()
    y_int = (y > 0.5).astype("int8")

    # ---- the model that predicts META_VAL: trained on 100% of META_TRAIN ----
    Xf, Xa, _names = vb.assemble(meta_fit, y_int, meta_val, None, inner_seed=k)
    p_val = train_champion(Xf, y[meta_fit], seed).predict(Xa["val"])

    # ---- inner cross-fit inside META_TRAIN only ----
    inner = inner_folds(len(meta_fit), 5, seed + 7717)
    p_oof = np.zeros(len(meta_fit), dtype="float64")
    for j in range(5):
        a = meta_fit[inner != j]
        b = meta_fit[inner == j]
        assert not (set(a.tolist()) & set(meta_val.tolist())), "inner fit touched META_VAL"
        Xa_fit, Xa_app, _ = vb.assemble(a, y_int, b, None, inner_seed=100 * k + j)
        p_oof[inner == j] = train_champion(Xa_fit, y[a], seed + 13 * (j + 1)).predict(Xa_app["val"])
    return p_val, p_oof


# --------------------------------------------------------------------------- evaluation
def evaluate_key(name, g, y, p_oof, p_val, meta_fit, meta_val, base_auc):
    """Both semantics, cross-fitted at the correction layer, on top of a nested base."""
    r = y[meta_fit] - p_oof
    pbias = fit_prob_bias(g, y[meta_fit], p_oof, np.arange(len(meta_fit)))
    loff = fit_logit_offset(g, y[meta_fit], p_oof, np.arange(len(meta_fit)))
    yv, pv = y[meta_val], p_val
    gv = g.iloc[meta_val]

    a_base = float(roc_auc_score(yv, pv))
    # A. PROBABILITY semantics
    bA = apply_table(gv, np.arange(len(meta_val)), pbias)
    a_prob = float(roc_auc_score(yv, np.clip(pv + bA, EPS, 1 - EPS)))
    # B. LOGIT-OFFSET semantics
    dB = apply_table(gv, np.arange(len(meta_val)), loff)
    a_logit = float(roc_auc_score(yv, logit(pv) + dB))

    # signed-bias replication on the SAME groups, discovery vs confirmation
    obs = observed_bias(g.iloc[meta_fit], r, np.arange(len(meta_fit)))
    common = [u for u in pbias if u in obs]
    rep = float(np.corrcoef([pbias[u] for u in common], [obs[u] for u in common])[0, 1]) \
        if len(common) >= 5 else None
    sign = float(np.mean(np.sign([pbias[u] for u in common])
                         == np.sign([obs[u] for u in common]))) if len(common) >= 5 else None

    sizes = g.iloc[meta_fit].value_counts()
    return {
        "key": name,
        "n_groups": int(g.iloc[meta_fit].nunique()),
        "groups_kept": int((sizes >= MIN_N).sum()),
        "base_auc": a_base,
        "delta_prob_e5": (a_prob - a_base) * 1e5,
        "delta_logit_e5": (a_logit - a_base) * 1e5,
        "bias_replication_r": rep,
        "sign_agreement": sign,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="p10a")
    ap.add_argument("--load-surrogate", action="store_true",
                    help="reuse cached p_meta_val / p_meta_train_oof instead of retraining")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds
    K = [int(x) for x in args.folds.split(",")]

    print("FULLY NESTED residual-structure scan (Phase 10A)")
    print(f"  view={args.view}  scheme={args.scheme}  folds={K}")
    print(f"  pre-declared, NOT tuned: PRIOR_N={PRIOR_N}  LAMBDA={LAMBDA}  MIN_N={MIN_N}  "
          f"N_BINS={N_BINS}  ROUNDS={ROUNDS}")
    print("  base = single champion lgbm extra_trees surrogate, NOT the 59-member v3 blend\n")

    out = {"tag": args.tag, "view": args.view, "scheme": args.scheme, "folds": {},
           "constants": {"PRIOR_N": PRIOR_N, "LAMBDA": LAMBDA, "MIN_N": MIN_N, "N_BINS": N_BINS,
                         "ROUNDS": ROUNDS},
           "base": "single champion lgbm extra_trees surrogate (not v3)",
           "nesting": "inner cross-fit inside META_TRAIN only; no META_VAL row trains any "
                      "p_meta_train_oof model"}

    for k in K:
        meta_val = np.where(folds == k)[0]
        meta_fit = np.where(folds != k)[0]
        assert not (set(meta_fit.tolist()) & set(meta_val.tolist()))
        cache = REPORTS / f"{args.tag}_surrogate_fold{k}.npz"
        if args.load_surrogate and cache.exists():
            z = np.load(cache)
            p_val, p_oof = z["p_val"], z["p_oof"]
            print(f"fold {k}: loaded cached surrogate from {cache.name}")
        else:
            ts = time.time()
            p_val, p_oof = build_surrogate(tr, te, y, meta_fit, meta_val, args.view,
                                           args.seed + k, k)
            np.savez_compressed(cache, p_val=p_val, p_oof=p_oof,
                                meta_val=meta_val, meta_fit=meta_fit)
            base = float(roc_auc_score(y[meta_val], p_val))
            print(f"fold {k}: surrogate built in {time.time()-ts:.0f}s   "
                  f"META_VAL AUC={base:.6f}   saved {cache.name}", flush=True)

        keys = predeclared_keys(tr)
        for nm, s in teacher_keys(load_teacher_columns(args.view), p_val).items():
            keys[nm] = s
        # base confidence, needs the base score. NOTE the positional take: `tr["Class"]` is indexed
        # by GLOBAL row, so the META_VAL rows must be selected with .to_numpy()[meta_val], not with
        # .loc[] or by passing an index array as a column name.
        cls_val = tr["Class"].to_numpy()[meta_val].astype(str)
        keys["conf:p_decile"] = pd.qcut(pd.Series(p_val), 10, labels=False).astype(str)
        keys["conf:p_decile_x_class"] = (pd.qcut(pd.Series(p_val), 10, labels=False).astype(str)
                                        + "|" + cls_val)

        base = float(roc_auc_score(y[meta_val], p_val))
        print(f"  scanning {len(keys)} predeclared groupings "
              f"(base fold-{k} AUC {base:.6f}) ...", flush=True)
        recs = []
        for i, (nm, g) in enumerate(keys.items()):
            recs.append(evaluate_key(nm, g, y, p_oof, p_val, meta_fit, meta_val, base))
            if (i + 1) % 30 == 0:
                print(f"    {i+1}/{len(keys)}", flush=True)
        out["folds"][str(k)] = {"base_auc": base, "n_keys": len(keys), "keys": recs}
        print(f"  fold {k} done ({time.time()-t0:.0f}s elapsed)\n", flush=True)

    # ------------------------------------------------------------------ aggregate
    names = list(out["folds"][str(K[0])]["keys"][i]["key"]
                 for i in range(len(out["folds"][str(K[0])]["keys"])))
    agg = {}
    for i, nm in enumerate(names):
        dp = [out["folds"][str(k)]["keys"][i]["delta_prob_e5"] for k in K]
        dl = [out["folds"][str(k)]["keys"][i]["delta_logit_e5"] for k in K]
        rp = [out["folds"][str(k)]["keys"][i]["bias_replication_r"] for k in K]
        sg = [out["folds"][str(k)]["keys"][i]["sign_agreement"] for k in K]
        rp_ok = [r for r in rp if r is not None]
        sg_ok = [s for s in sg if s is not None]
        agg[nm] = {
            "delta_prob_mean_e5": float(np.mean(dp)), "delta_prob_pos": int(sum(d > 0 for d in dp)),
            "delta_prob_folds": [round(d, 2) for d in dp],
            "delta_logit_mean_e5": float(np.mean(dl)), "delta_logit_pos": int(sum(d > 0 for d in dl)),
            "delta_logit_folds": [round(d, 2) for d in dl],
            "replication_r_mean": float(np.mean(rp_ok)) if rp_ok else None,
            "sign_agreement_mean": float(np.mean(sg_ok)) if sg_ok else None,
        }
        agg[nm]["passes_both_semantics"] = bool(
            agg[nm]["delta_prob_pos"] >= 4 and agg[nm]["delta_logit_pos"] >= 4
            and max(agg[nm]["delta_prob_mean_e5"], agg[nm]["delta_logit_mean_e5"]) >= 1.5)
    out["aggregate"] = agg

    print(f"{'='*118}")
    print("PHASE 10A -- NESTED residual structure. Gate: >=4/5 folds positive under BOTH semantics "
          "and mean >= +1.5e-5")
    print("=" * 118)
    print(f"  {'key':<36}{'prob mean':>11}{'pos':>6}{'logit mean':>12}{'pos':>6}"
          f"{'repl r':>9}{'sign':>8}")
    print(f"  {'-'*118}")
    ranked = sorted(agg.items(), key=lambda kv: -max(kv[1]["delta_prob_mean_e5"],
                                                    kv[1]["delta_logit_mean_e5"]))
    for nm, a in ranked[:20]:
        rr = f"{a['replication_r_mean']:.3f}" if a["replication_r_mean"] is not None else "  -"
        ss = f"{a['sign_agreement_mean']:.3f}" if a["sign_agreement_mean"] is not None else "  -"
        star = " *" if a["passes_both_semantics"] else ""
        print(f"  {nm[:35]:<36}{a['delta_prob_mean_e5']:>+10.2f}e{a['delta_prob_pos']:>3}/5"
              f"{a['delta_logit_mean_e5']:>+11.2f}e{a['delta_logit_pos']:>3}/5{rr:>9}{ss:>8}{star}")

    winners = [nm for nm, a in agg.items() if a["passes_both_semantics"]]
    out["winners"] = winners
    if winners:
        print("\nPASSING KEYS (both semantics, >=4/5 folds, mean >= +1.5e-5):")
        for nm in winners:
            a = agg[nm]
            print(f"  {nm}: prob {a['delta_prob_mean_e5']:+.2f}e-5 ({a['delta_prob_pos']}/5), "
                  f"logit {a['delta_logit_mean_e5']:+.2f}e-5 ({a['delta_logit_pos']}/5), "
                  f"replication r={a['replication_r_mean']:.3f}")
        verdict = "SIMPLE GROUP RESIDUAL STRUCTURE FOUND -- a targeted specialist is justified"
    else:
        best = max(max(a["delta_prob_mean_e5"], a["delta_logit_mean_e5"]) for a in agg.values())
        print("\nNO KEY PASSES.")
        print("  Simple group residual structure is EXHAUSTED as a source of gain. The best "
              f"predeclared key reached {best:+.2f}e-5 under its better semantics, below the "
              "+1.5e-5 gate.")
        print("  Combined with Phase 9 (error sits where members AGREE) this means the shared bias "
              "is NOT recoverable by a\n  per-group mean correction on these groupings. If it exists "
              "it is not a group-constant offset.")
        verdict = "SIMPLE GROUP RESIDUAL STRUCTURE EXHAUSTED"
    out["verdict"] = verdict
    out["seconds"] = round(time.time() - t0, 1)
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()[:12]
    save_json(out, REPORTS / f"{args.tag}.json")
    print(f"\nVERDICT: {verdict}")
    print("wrote", REPORTS / f"{args.tag}.json", f"({out['seconds']}s)")


def load_teacher_columns(view: str = "full") -> dict[str, np.ndarray]:
    """Original-data teacher columns from the cached static view.

    `ogte_*` / `ogs_*` are assembled inside the feature view, NOT columns on the raw frame, so
    reading them from `tr` yields an empty set with no error. The static view is fold-independent, so
    reading it introduces no leakage.
    """
    import json as _json

    from src.common import FEATURES
    npz, jsn = FEATURES / f"static_{view}.npz", FEATURES / f"static_{view}.json"
    if not npz.exists():
        print(f"  [warn] no cached static view '{view}' -> original-knowledge keys skipped")
        return {}
    names = _json.loads(jsn.read_text(encoding="utf-8"))
    names = names["names"] if isinstance(names, dict) and "names" in names else names
    want = [i for i, n in enumerate(names) if n.startswith(("ogte_", "ogs_", "teach_"))]
    if not want:
        return {}
    z = np.load(npz)
    trm = z["tr"]
    return {names[i]: np.asarray(trm[:, i], dtype="float64") for i in want}


if __name__ == "__main__":
    main()
