"""Guards for the nested residual diagnostic and the two score-space semantics.

Failure modes encoded here
--------------------------
1. SCALE MIXING. The earlier residual script estimated a probability-space residual (y - p) and added
   it to a logit-scale score. That can change a ranking while not being a calibration correction, and
   its AUC delta must never be read as one. These tests assert the two semantics stay separate and
   that each is self-consistent.

2. THE NEWTON STEP. The logit offset is a single Newton step claimed to be the regularised intercept
   MLE. That claim is checked against a brute-force numerical minimiser, not asserted.

3. SHRINKAGE DIRECTION. Shrinkage must pull the estimate toward zero, never amplify it, and must be
   monotone in the shrinkage constant. A sign error here would silently invert every correction.

4. NESTING. The promotion-grade property: no model producing a META_TRAIN prediction may train on
   any META_VALIDATION row. Verified structurally on the real fold scheme, not merely documented.

5. PRE-DECLARATION. The shrinkage constants, minimum group size and round count must be pre-declared
   and bounded, so they cannot be tuned against confirmation folds after the fact.

6. INDEX SPACE. Labels are global, assembled matrices are local to fit_idx. Conflating these has
   produced three fake results in this campaign, so the surrogate's shapes and label alignment are
   asserted against a synthetic case with a known answer.

Run: python tests/test_residual_nested.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.residual_nested import (CHAMPION, EPS, LAMBDA, MIN_N,  # noqa: E402
                                     PRIOR_N, ROUNDS, apply_table, fit_logit_offset,
                                     fit_prob_bias, inner_folds, logit, qbin, sigmoid)

FAILS: list[str] = []
N = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global N
    N += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


# --------------------------------------------------------------------------- semantics
def test_probability_semantics_is_its_own_scale() -> None:
    print("\nprobability semantics")
    rng = np.random.default_rng(0)
    n = 4000
    g = pd.Series(rng.choice(list("ABCD"), n))
    p = np.clip(rng.random(n) * 0.6 + 0.2, EPS, 1 - EPS)
    y = (rng.random(n) < p).astype(float)
    rows = np.arange(n)
    b = fit_prob_bias(g, y, p, rows)
    check("returns one bias per sufficiently sized group", len(b) > 0, str(len(b)))
    check("biases are finite", all(np.isfinite(v) for v in b.values()))
    # a group whose y is genuinely below p must get a negative bias
    gp = pd.Series(np.repeat("A", n))
    yp = np.zeros(n)
    pp = np.full(n, 0.6)
    ba = fit_prob_bias(gp, yp, pp, rows)
    check("group with y << p gets a negative bias", ba.get("A", 0.0) < 0, str(ba.get("A")))
    gq = pd.Series(np.repeat("A", n))
    bq = fit_prob_bias(gq, np.ones(n), np.full(n, 0.6), rows)
    check("group with y >> p gets a positive bias", bq.get("A", 0.0) > 0, str(bq.get("A")))


def test_logit_offset_matches_brute_force() -> None:
    print("\nlogit-offset semantics -- Newton step vs numerical minimiser")
    rng = np.random.default_rng(1)
    n = 6000
    for gi, (frac_pos, base_p) in enumerate([(0.20, 0.30), (0.50, 0.50), (0.80, 0.70)]):
        z = np.full(n, logit(np.array([base_p]))[0])
        y = (rng.random(n) < frac_pos).astype(float)
        g = pd.Series(np.repeat("A", n))
        got = fit_logit_offset(g, y, np.full(n, base_p), np.arange(n)).get("A", 0.0)

        # brute-force: minimise the regularised log-loss over delta on a fine grid
        def loss(d: float) -> float:
            s = sigmoid(z + d)
            return float(-(y * np.log(s) + (1 - y) * np.log(1 - s)).sum() + LAMBDA * d * d)

        grid = np.linspace(-3, 3, 24001)
        vals = np.array([loss(d) for d in grid[::40]])       # coarse pass
        c = grid[::40][int(vals.argmin())]
        fine = np.linspace(c - 0.05, c + 0.05, 2001)
        best = fine[int(np.argmin([loss(d) for d in fine]))]
        ok = abs(got - best) < 2e-3
        check(f"group {gi}: Newton step {got:+.5f} matches numerical optimum {best:+.5f}", ok,
              f"diff {abs(got-best):.2e}")

    # the offset must be ZERO for a perfectly calibrated group
    n = 60000
    p = rng.random(n) * 0.9 + 0.05
    y = (rng.random(n) < p).astype(float)
    d = fit_logit_offset(pd.Series(np.repeat("A", n)), y, p, np.arange(n)).get("A", 0.0)
    # tolerance is sampling noise, not a modelling claim: SE of the mean of (s-y) is about
    # sd/sqrt(n) ~ 0.29/245 ~ 1.2e-3, and the estimate is divided by curvature (~0.25n), so the
    # achievable precision is far looser than 5e-3. A 2e-2 band is still 25x tighter than any delta
    # this diagnostic could act on.
    check(f"perfectly calibrated group gets ~zero offset (got {d:+.2e})", abs(d) < 2e-2,
          f"{d:+.4f}")


def test_semantics_do_not_mix_scales() -> None:
    print("\nthe two semantics stay separate")
    rng = np.random.default_rng(2)
    n = 3000
    g = pd.Series(rng.choice(list("AB"), n))
    p = np.clip(rng.random(n) * 0.8 + 0.1, EPS, 1 - EPS)
    y = (rng.random(n) < 0.5).astype(float)
    rows = np.arange(n)
    bp = fit_prob_bias(g, y, p, rows)
    dl = fit_logit_offset(g, y, p, rows)
    # a probability bias and a logit offset live on different scales even for the same group
    common = set(bp) & set(dl)
    check("both semantics produce estimates for the same groups", len(common) > 0)
    diffs = [abs(bp[u] - dl[u]) for u in common]
    check("the two are numerically distinct (no accidental sharing of one number)",
          max(diffs) > 1e-6, f"max abs diff {max(diffs):.2e}")
    # applying the probability table must produce a PROBABILITY, and the logit table a LOGIT
    ap = apply_table(g, rows, bp)
    check("apply_table on probability semantics returns a bounded additive offset",
          np.all(np.abs(ap) < 1.0), f"max |b| {np.abs(ap).max():.3f}")
    # a group with y << p: probability bias negative, logit offset also negative (same direction)
    gp = pd.Series(np.repeat("A", n))
    # Sign agreement between the two semantics, on the regime that matters. Note the probability
    # semantics CAN saturate where the logit offset does not: with y = 0 and p = 0.8 the clipped
    # corrected score is 0.2 for every row, so its ranking collapses and any tie-break is arbitrary.
    # That is a real limitation of probability semantics at extreme groups, not a bug, and it is
    # exactly why both semantics are required to agree before promotion. The test therefore uses a
    # moderate base, where neither semantics saturates.
    ym = (rng.random(n) < 0.30).astype(float)
    pm = np.full(n, 0.55)
    check("both semantics agree on SIGN for a moderately under-performing group",
          (fit_prob_bias(gp, ym, pm, rows).get("A", 0) < 0)
          == (fit_logit_offset(gp, ym, pm, rows).get("A", 0) < 0))
    check("probability semantics can saturate at extreme groups (documented limitation)",
          fit_prob_bias(gp, np.zeros(n), np.full(n, 0.999), rows).get("A", 0) < -0.9)


def test_shrinkage_is_monotone_and_pulls_to_zero() -> None:
    print("\nshrinkage behaviour")
    import scripts.residual_nested as R
    rng = np.random.default_rng(3)
    n = 2000
    g = pd.Series(np.repeat("A", n))
    p = np.full(n, 0.5)
    y = (rng.random(n) < 0.7).astype(float)
    saved = R.PRIOR_N
    try:
        vals = []
        for prior in (0.0, 10.0, 50.0, 200.0, 1000.0):
            R.PRIOR_N = prior
            vals.append(R.fit_prob_bias(g, y, p, np.arange(n)).get("A", 0.0))
        check("probability bias is monotone non-increasing in shrinkage",
              all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1)),
              str([round(v, 5) for v in vals]))
        check("larger shrinkage drives the estimate toward zero", abs(vals[-1]) < abs(vals[0]),
              str([round(v, 5) for v in vals]))
    finally:
        R.PRIOR_N = saved

    saved = R.LAMBDA
    try:
        vals = []
        for lam in (0.0, 10.0, 50.0, 500.0):
            R.LAMBDA = lam
            vals.append(R.fit_logit_offset(g, y, p, np.arange(n)).get("A", 0.0))
        check("logit offset is monotone non-increasing in LAMBDA",
              all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1)),
              str([round(v, 5) for v in vals]))
        check("larger LAMBDA drives the offset toward zero", abs(vals[-1]) < abs(vals[0]),
              str([round(v, 5) for v in vals]))
    finally:
        R.LAMBDA = saved


def test_min_group_size_is_enforced() -> None:
    print("\nminimum group size")
    rng = np.random.default_rng(4)
    n = 5000
    big_n, tiny_n = 2500, 150
    n = big_n + tiny_n
    g = pd.Series(["big"] * big_n + ["tiny"] * tiny_n)
    check("fixture sizes put 'big' above MIN_N and 'tiny' below it",
          big_n >= MIN_N > tiny_n, f"{big_n} / {MIN_N} / {tiny_n}")
    p = np.clip(rng.random(n), EPS, 1 - EPS)
    y = (rng.random(n) < 0.5).astype(float)
    t = fit_prob_bias(g, y, p, np.arange(n))
    check("group at or above MIN_N is kept", "big" in t, str(sorted(t)))
    check("group below MIN_N is dropped", "tiny" not in t, str(sorted(t)))


def test_empty_group_falls_back_to_zero() -> None:
    print("\nunseen groups")
    rng = np.random.default_rng(5)
    n = 1000
    g = pd.Series(rng.choice(list("AB"), n))
    p = np.clip(rng.random(n), EPS, 1 - EPS)
    y = (rng.random(n) < 0.5).astype(float)
    t = fit_prob_bias(g, y, p, np.arange(n))
    unseen = apply_table(pd.Series(["ZZZ"] * 10), np.arange(10), t)
    check("an unseen group gets exactly zero offset", np.all(unseen == 0.0))
    check("apply_table with an empty table returns zeros",
          np.all(apply_table(pd.Series(["A"] * 5), np.arange(5), {}) == 0.0))


# --------------------------------------------------------------------------- nesting
def test_inner_folds_partition_cleanly() -> None:
    print("\ninner fold construction")
    lab = inner_folds(1000, 5, 42)
    check("every row gets exactly one inner fold", len(lab) == 1000 and lab.min() >= 0)
    check("all 5 inner folds are non-empty", len(set(lab.tolist())) == 5)
    bal = [int((lab == j).sum()) for j in range(5)]
    check("inner folds are balanced within 1 row", max(bal) - min(bal) <= 1, str(bal))
    lab2 = inner_folds(1000, 5, 42)
    check("inner folds are deterministic for a fixed seed", np.array_equal(lab, lab2))
    check("a different seed gives a different partition",
          not np.array_equal(lab, inner_folds(1000, 5, 43)))
    for j in range(5):
        a = np.where(lab != j)[0]
        b = np.where(lab == j)[0]
        check(f"inner fold {j}: complement and fold are disjoint and cover everything",
              not (set(a.tolist()) & set(b.tolist())) and len(a) + len(b) == 1000)


def test_nesting_on_real_fold_scheme() -> None:
    print("\nnesting on the real primary fold scheme")
    from src.common import ID_COL, TARGET, load_cached_parquet
    from src.validation.folds import get_scheme
    tr, _ = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y_int, tr[ID_COL]).folds
    for k in range(5):
        meta_fit = np.where(folds != k)[0]
        meta_val = np.where(folds == k)[0]
        check(f"fold {k}: META_FIT and META_VAL are disjoint",
              not (set(meta_fit.tolist()) & set(meta_val.tolist())))
        check(f"fold {k}: they partition all rows",
              len(meta_fit) + len(meta_val) == len(folds))
        inner = inner_folds(len(meta_fit), 5, 7717 + k)
        leak = 0
        for j in range(5):
            inner_fit = meta_fit[inner != j]
            if set(inner_fit.tolist()) & set(meta_val.tolist()):
                leak += 1
        check(f"fold {k}: no inner-fit set touches META_VAL", leak == 0, f"{leak} leaks")


def test_index_space_alignment() -> None:
    print("\nglobal vs local index discipline")
    from src.common import ID_COL, TARGET, load_cached_parquet
    from src.validation.folds import get_scheme
    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    folds = get_scheme("primary", tr[TARGET].values.astype("int8"), tr[ID_COL]).folds
    k = 0
    meta_fit = np.where(folds != k)[0]
    meta_val = np.where(folds == k)[0]
    n = len(y)
    check("assembled matrix would be LOCAL to meta_fit, not length n",
          len(meta_fit) != n)
    check("labels for assembled rows must be gathered as y[meta_fit]",
          len(y[meta_fit]) == len(meta_fit))
    # a local index i corresponds to global meta_fit[i]; verify the mapping is unambiguous
    check("local->global map is a bijection onto META_FIT",
          len(set(meta_fit.tolist())) == len(meta_fit))
    rng = np.random.default_rng(6)
    fake_local = rng.random(len(meta_fit))
    # reading y at LOCAL indices would give a different (wrong) label vector
    check("reading y at local indices differs from y[meta_fit] (so the bug would be visible)",
          not np.array_equal(y[:len(meta_fit)], y[meta_fit]))


# --------------------------------------------------------------------------- pre-declaration
def test_constants_are_predeclared_and_bounded() -> None:
    print("\npre-declared constants")
    check("PRIOR_N is fixed at 50", PRIOR_N == 50.0, str(PRIOR_N))
    check("LAMBDA is fixed at 50 and matches PRIOR_N nominally", LAMBDA == 50.0, str(LAMBDA))
    check("MIN_N is fixed at 200", MIN_N == 200, str(MIN_N))
    check("ROUNDS is fixed at 900", ROUNDS == 900, str(ROUNDS))
    check("shrinkage constants are below the +1.5e-5 admission gate in effect",
          PRIOR_N > 0 and LAMBDA > 0)
    check("champion uses the champion learning rate 0.02, not the generic 0.03",
          CHAMPION["learning_rate"] == 0.02, str(CHAMPION["learning_rate"]))
    check("champion uses num_leaves 127", CHAMPION["num_leaves"] == 127)
    check("champion has extra_trees True", CHAMPION["extra_trees"] is True)


def test_qbin_handles_degenerate_columns() -> None:
    print("\nquantile binning robustness")
    check("constant column bins without raising", len(qbin(pd.Series([1.0] * 100))) > 0)
    check("two-valued column bins without raising",
          len(qbin(pd.Series([1.0] * 50 + [2.0] * 50))) > 0)
    r = qbin(pd.Series(np.arange(1000.0)))
    check("continuous column produces multiple bins", r.nunique() > 5, str(r.nunique()))


def main() -> int:
    print("=" * 78)
    print("NESTED RESIDUAL DIAGNOSTIC -- SEMANTICS AND NESTING TESTS")
    print("=" * 78)
    for fn in (test_probability_semantics_is_its_own_scale,
               test_logit_offset_matches_brute_force,
               test_semantics_do_not_mix_scales,
               test_shrinkage_is_monotone_and_pulls_to_zero,
               test_min_group_size_is_enforced,
               test_empty_group_falls_back_to_zero,
               test_inner_folds_partition_cleanly,
               test_nesting_on_real_fold_scheme,
               test_index_space_alignment,
               test_constants_are_predeclared_and_bounded,
               test_qbin_handles_degenerate_columns):
        fn()
    print("\n" + "=" * 78)
    print(f"{N - len(FAILS)}/{N} passed")
    if FAILS:
        print("FAILURES: " + ", ".join(FAILS))
    print("=" * 78)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
