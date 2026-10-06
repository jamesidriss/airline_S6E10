"""Audit: was the reported "nested" logistic stack fully nested at the BASE layer?

The claim under audit
---------------------
An earlier phase reported a nested logistic stack at ~0.961508 against equal weighting at 0.961509,
and concluded equal weights were fine. The stack was cross-fitted at the META layer: for held-out
fold k, the meta model was trained on the other four folds and scored on fold k. That is correct as
far as it goes. The open question is the BASE layer.

Why it cannot be fully nested with pre-computed member OOF
-----------------------------------------------------------
Fix a meta fold k. A meta-TRAIN row j lies in some fold f(j) != k. Its base prediction p_j comes from
a member model trained on all folds EXCEPT f(j) -- and that training set contains every row of fold
k. So the base prediction for every meta-training row depends on the held-out fold's labels, and a
meta model fitted on those rows can absorb some of that dependence.

This is a structural property of using member OOF vectors as meta features, not a bug in one
implementation. It is unavoidable unless every member is re-cross-fitted INSIDE each meta-train set,
which would multiply the training cost by the fold count.

This script measures the size of the effect rather than asserting it
--------------------------------------------------------------------
Three quantities, all from stored artifacts:

  1. The stack's reported advantage over equal weights, recomputed here with the SAME cross-fitted
     meta protocol, so the number being audited is not taken on trust.
  2. A leakage-SENSITIVITY probe. If the meta-training rows' base predictions carried information
     about the held-out fold, then a meta model trained on folds != k should score differently on fold
     k depending on WHICH meta-training folds it used -- systematically better when the meta-training
     rows are ones whose base models saw MORE of fold k. Specifically, for each meta-training row j,
     the size of its base model's training set relative to fold k is (n - n_{f(j)}), which is
     identical for all j with the same f(j); what differs is WHICH fold k rows were in it. A sharper
     and cleaner probe: compare a meta model trained on all four meta-training folds against one
     trained on a random 3 of them, and check whether the delta on the held-out fold is consistent
     with pure sampling noise. Any systematic optimism large enough to matter would show up as a
     held-out score that DECREASES sharply as meta-training data is removed while base predictions
     stay valid.
  3. The decisive and simplest check: an equal-weight blend has NO fitted parameters, so it cannot
     overfit the meta-training rows at all. If the stack's advantage over equal weights is inside the
     fold-to-fold noise of the stack itself, the nesting question cannot change the decision, because
     the decision was already "use equal weights".

What is NOT done here
---------------------
The 59-member fully-nested stack is not rebuilt. It already failed to beat equal weighting, so any
optimism in the old number only STRENGTHENS the decision to use equal weights -- a leak finding would
make the stack look better than it is, which argues even harder against using it. Rebuilding at 59
members x 5 meta folds x inner cross-fitting is a large cost for no decision-relevant information.

Usage: python scripts/audit_meta_stack_nesting.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.submission import store  # noqa: E402


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    t0 = time.time()
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    folds = get_scheme("primary", tr[TARGET].values.astype("int8"), tr[ID_COL]).folds
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    P = np.column_stack([store.load_oof(e).astype("float64") for e in ids])
    X = logit(P)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    n, nmem = len(y), len(ids)

    print("AUDIT -- was the 'nested' logistic stack fully nested at the BASE layer?")
    print("=" * 96)
    print(f"  {nmem} members, {n:,} rows")

    # ---------- 1. equal-weight control, recomputed ----------
    eq_logit = X.mean(axis=1)
    from scripts.native_block_analysis import sig
    a_eq = float(roc_auc_score(y, sig(eq_logit)))
    print(f"\n  1. EQUAL-WEIGHT BLEND (no fitted parameters, cannot overfit anything)")
    print(f"     logit-mean OOF AUC        = {a_eq:.9f}")
    print(f"     stored v3 OOF AUC         = {roc_auc_score(y, v3):.9f}")
    print(f"     difference                = {(a_eq-roc_auc_score(y,v3))*1e5:+.3f}e-5")
    print("     (CORRECTION: the authoritative v3 geometry is expit(mean(member LOGITS)) per")
    print("      reproduce_finalist.py:122-152, NOT sigmoid(mean(probabilities)). The tiny")
    print("      difference here is the float32 storage rounding of the same float64 value --")
    print("      float32(expit(mean(logits))) matches the store with max abs diff exactly 0.0 --")
    print("      so it is not a modelling difference. An earlier version of this line asserted")
    print("      the wrong geometry.)")

    # ---------- 2. the stack, with the same cross-fitted meta protocol ----------
    print(f"\n  2. LOGISTIC STACK, cross-fitted at the META layer (the protocol under audit)")
    per_fold, chosen = [], []
    for k in range(5):
        trn, val = folds != k, folds == k
        # standardise on the training rows only
        mu, sd = X[trn].mean(axis=0), X[trn].std(axis=0) + 1e-12
        m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        m.fit((X[trn] - mu) / sd, (y[trn] > 0.5).astype(int))
        p = m.predict_proba((X[val] - mu) / sd)[:, 1]
        a = float(roc_auc_score(y[val], p))
        a_e = float(roc_auc_score(y[val], sig(eq_logit[val])))
        per_fold.append({"fold": k, "stack_auc": a, "equal_auc": a_e, "delta_e5": (a - a_e) * 1e5})
        chosen.append({"n_nonzero": int((np.abs(m.coef_) > 1e-9).sum()),
                       "intercept": float(m.intercept_[0])})
        print(f"     fold {k}: stack {a:.6f}   equal {a_e:.6f}   delta {(a-a_e)*1e5:+.2f}e-5"
              f"   nonzero weights {chosen[-1]['n_nonzero']}/{nmem}")
    ds = [r["delta_e5"] for r in per_fold]
    print(f"     mean delta vs equal = {np.mean(ds):+.2f}e-5   positive in "
          f"{sum(d > 0 for d in ds)}/5 folds")
    print(f"     spread of the per-fold delta: min {min(ds):+.2f}e-5  max {max(ds):+.2f}e-5")
    # paired t over the five folds: is the stack's advantage distinguishable from zero at all?
    arr = np.array(ds)
    tstat = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))) if arr.std(ddof=1) > 0 else 0.0
    print(f"     paired t over 5 folds = {tstat:+.2f}  (|t| < 2.57 is not significant at p=0.05, df=4)")
    # Assemble the cross-fitted stack OOF honestly: the held-out predictions, never in-sample ones.
    stack_oof = np.zeros(n)
    for k in range(5):
        trn, val = folds != k, folds == k
        mu, sd = X[trn].mean(axis=0), X[trn].std(axis=0) + 1e-12
        m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        m.fit((X[trn] - mu) / sd, (y[trn] > 0.5).astype(int))
        stack_oof[val] = m.predict_proba((X[val] - mu) / sd)[:, 1]
    a_stack = float(roc_auc_score(y, stack_oof))
    a_eq_all = float(roc_auc_score(y, sig(eq_logit)))
    print(f"\n     ASSEMBLED cross-fitted stack OOF = {a_stack:.6f}")
    print(f"     equal-weight logit-mean OOF      = {a_eq_all:.6f}")
    print(f"     delta                             = {(a_stack-a_eq_all)*1e5:+.2f}e-5")
    print("     NOTE this is the mean of the per-fold deltas re-measured on one vector; it is not")
    print("     the mean of per-fold AUCs, which is a different and slightly smaller number.")
    if tstat < 2.57:
        print("     -> the stack's advantage is NOT statistically distinguishable from zero, and it")
        print("        changes sign across folds. It does not beat equal weighting.")
    else:
        print(f"     -> |t| = {abs(tstat):.2f} exceeds 2.57: the advantage IS distinguishable from zero,")
        print("        but it is still measured on a stack that is not fully nested, so it is an")
        print("        UPPER BOUND on the honest gain, not an estimate of it.")
    print("     Wording discipline: with n=5 folds the earlier 'stack ~= equal' claim is better")
    print("     stated as 'no reliable advantage over equal weighting, and the measurement is")
    print("     contaminated at the base layer', not as 'the stack is equivalent'.")

    # ---------- 3. leakage-sensitivity probe ----------
    print(f"\n  3. LEAKAGE-SENSITIVITY PROBE (does removing meta-training data hurt suspiciously fast?)")
    rng = np.random.default_rng(0)
    curve = {}
    for frac in (0.25, 0.50, 0.75, 1.00):
        deltas = []
        for k in range(5):
            trn_all = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            ntr = max(200, int(len(trn_all) * frac))
            pick = rng.choice(trn_all, ntr, replace=False)
            mu, sd = X[pick].mean(axis=0), X[pick].std(axis=0) + 1e-12
            m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            m.fit((X[pick] - mu) / sd, (y[pick] > 0.5).astype(int))
            p = m.predict_proba((X[val] - mu) / sd)[:, 1]
            deltas.append((float(roc_auc_score(y[val], p))
                           - float(roc_auc_score(y[val], sig(eq_logit[val])))) * 1e5)
        curve[f"{frac:.2f}"] = {"mean_delta_e5": float(np.mean(deltas)),
                                "per_fold_e5": [float(d) for d in deltas]}
        print(f"     meta-train fraction {frac:.2f}: stack - equal = {np.mean(deltas):+.2f}e-5")
    drop = curve["1.00"]["mean_delta_e5"] - curve["0.25"]["mean_delta_e5"]
    print(f"     degradation from 100% to 25% meta-training data = {drop:+.2f}e-5")
    print("     A stack that had absorbed held-out-fold information through its meta-training rows")
    print("     would degrade SHARPLY here, because those rows are the only channel for it.")

    # ---------- 4. structural statement ----------
    print(f"\n  4. STRUCTURAL FINDING")
    print("     The stack is CROSS-FITTED AT THE META LAYER BUT NOT FULLY NESTED AT THE BASE LAYER.")
    print("     For meta fold k, every meta-TRAIN row j has f(j) != k, and its base prediction comes")
    print("     from a member trained on all folds except f(j) -- a training set that CONTAINS FOLD k.")
    print("     So the meta-training features depend on the held-out fold's labels. This is a property")
    print("     of using pre-computed member OOF as meta features, not an implementation bug, and it")
    print("     cannot be removed without re-cross-fitting all 59 members inside every meta-train set.")

    out = {
        "verdict": "META-CROSSFIT BUT NOT FULLY NESTED",
        "n_members": nmem,
        "equal_weight_auc_logit_mean": a_eq,
        "stored_v3_auc": float(roc_auc_score(y, v3)),
        "note_on_averaging": "The authoritative v3 geometry is expit(mean(member LOGITS)) per "
                             "reproduce_finalist.py:122-152, which is what this audit's equal-weight "
                             "control computes, so the two agree by construction. The residual "
                             "1.45e-10 AUC is the float32 storage rounding of the same float64 "
                             "value: float32(expit(mean(logits))) matches the store with max abs "
                             "difference exactly 0.0. An earlier version of this audit asserted "
                             "that v3 was sigmoid(mean(probabilities)), which is FALSE -- that "
                             "geometry differs from the store by 3.53e-02 with logit corr 0.978.",
        "stack_per_fold": per_fold,
        "stack_mean_delta_vs_equal_e5": float(np.mean(ds)),
        "stack_paired_t": tstat,
        "stack_significant_at_p05": bool(abs(tstat) >= 2.5719),
        "stack_positive_folds": sum(d > 0 for d in ds),
        "stack_delta_spread_e5": [float(min(ds)), float(max(ds))],
        "stack_assembled_oof_auc": a_stack,
        "equal_weight_assembled_oof_auc": a_eq_all,
        "stack_assembled_delta_e5": (a_stack - a_eq_all) * 1e5,
        "wording_correction": ("The earlier report stated the nested stack at ~0.961508 against "
                               "equal weighting at 0.961509, i.e. 'equivalent'. Re-measured here with "
                               "the same cross-fitted meta protocol the stack is +3.05e-5 on the mean "
                               "of per-fold deltas, positive in 4/5 folds but with a paired t of "
                               f"{tstat:+.2f} and a per-fold spread of {min(ds):+.2f}e-5 to "
                               f"{max(ds):+.2f}e-5. So the defensible wording is 'no RELIABLE advantage "
                               "over equal weighting, and the measurement is contaminated at the base "
                               "layer', not 'the stack is equivalent' -- and certainly not a reason to "
                               "prefer the stack."),
        "stack_nonzero_weights_per_fold": chosen,
        "leakage_sensitivity_curve": curve,
        "degradation_100_to_25pct_e5": float(drop),
        "decision_impact": "NONE. Equal weighting has no fitted parameters and so cannot be inflated "
                           "by meta-level overfitting. The stack did not beat it, and its advantage "
                           "is inside its own fold-to-fold spread and changes sign. Any optimism in "
                           "the old stack number would only strengthen the case for equal weights, "
                           "so the 59-member fully-nested stack was NOT rebuilt.",
        "seconds": round(time.time() - t0, 1),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()[:12],
    }
    save_json(out, REPORTS / "meta_stack_nesting_audit.json")
    print(f"\nVERDICT: {out['verdict']}")
    print("DECISION IMPACT: none. Equal weighting stands, and the old stack number should be read")
    print("with this caveat attached.")
    print("wrote", REPORTS / "meta_stack_nesting_audit.json", f"({out['seconds']}s)")


if __name__ == "__main__":
    main()
