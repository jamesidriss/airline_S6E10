# STATUS_p11r.md — Phase 11R closed

## Verdict

**Phase 11 is CLOSED. No fold 1, no seven-slot training, no submission.** Both predeclared
questions are answered, both negatively.

The Phase 11 "structural cap" claim is withdrawn as invalid (it was not a mathematical upper
bound). The correct replacement conclusion is empirical and now measured: **native CatBoost
counterparts do not move the v3 ensemble**, and that was measured from actual prediction vectors,
not inferred from family weights.

## Two predeclared questions, both answered

**Question A — can the native CTR mechanism be made materially stronger? No.**
All three arms on primary fold 0, against a budget-matched C2 reference:

| arm | mechanism | AUC | Δ vs C2REF | iters | verdict |
|---|---|---|---|---|---|
| C2REF | the 17-native-category reference | 0.961067 | — | 1043 | reference |
| C4 | + exact categorical **Age** (75 levels) | 0.960938 | **−12.9e-5** | 697 | negative |
| C5 | + exact categorical **Flight Distance** (3474) | 0.961075 | **+0.9e-5** | 1010 | below +2e-5 |
| ctr2 | `max_ctr_complexity` 1 → 2 | 0.961039 | **−2.7e-5** | 1462 | negative |

- Age converges *earlier and worse* (1043 → 697 iters), consistent with a shortcut that saturates
  the model without improving generalisation.
- Flight Distance is the one positive, but +0.9e-5 is an order of magnitude below the +5e-5 fold-1
  trigger. The high-cardinality **cost** concern was real in projection and false in fact: 0.3710
  s/round, converged at 1010 iters.
- `max_ctr_complexity=2` was the right test to run — the precondition was *verified* (dump shows
  `simple_ctr` 0, `combinations_ctr` 0, so pairwise CTRs were genuinely absent) — and it lost,
  while costing 58% more compute.

**Question B — do a few distinct native counterparts preserve enough diversity to move the blend?
Diversity: yes. Effect: no.**

Three **exact** counterparts (seed, view, depth, l2, scheme all matching the slot replaced):

| arm | replaces | standalone Δ | O–N corr | **single-slot v3 Δ** |
|---|---|---|---|---|
| ND6 | `z3_cat_d6` (depth 6, seed 1) | +23.1e-5 | 0.99893 | **+0.24e-5** |
| ND10 | `z3_cat_d10` (depth 10, seed 2, lr 0.03) | +5.1e-5 | 0.99850 | **+0.02e-5** |
| NCORE3 | `z3_cat_core3` (core3 view, seed 4) | −3.4e-5 | 0.99880 | **−0.19e-5** |

Mini block, exact prediction vectors, α=0 control reconstructs at **max|dp| = 0.000e+00**:

| block | α | v3 control | block AUC | Δ |
|---|---|---|---|---|
| MINI_B100 | 1.00 | 0.961501 | 0.961501 | **+0.07e-5** |
| MINI_B50 | 0.50 | 0.961501 | 0.961501 | **+0.04e-5** |

**Diversity is retained, and that is the informative part:**

| subset | min | median | max |
|---|---|---|---|
| ORIGINAL 3 slots | 0.99697 | 0.99748 | 0.99780 |
| NATIVE 3 counterparts | 0.99771 | **0.99779** | 0.99856 |

The native set did **not** collapse toward 1.0. So the hypothesis fails for a reason *other* than
the predicted one: the counterparts are **diverse and interchangeable in effect**. That is a
different and more informative negative than diversity collapse would have been.

## The gate, and why it does not license fold 1

| gate | criterion | result |
|---|---|---|
| A | mini block ≥ +1.0e-5 | **False** — best +0.07e-5, ~14× short |
| B | ≥2 of 3 swaps positive + diversity retained | **True** — 2/3, diversity retained |
| C | mechanism ≥ +5e-5 over C2 | **False** — 0 arms |

Gate B's boolean is **True**, and it carries **no evidential weight**. Under pure noise each swap
delta is positive with probability 0.5, so **P(≥2 of 3) = 0.500** — at n=3 that is close to the
*single most likely* outcome of no signal at all. One of the two positives is +0.02e-5, a
tie-scale effect; the mean swap delta is +0.024e-5.

The predeclared boolean is preserved **exactly as written** in `gates`, with a separate
`gates_weighted` field recording the corrected reading and `substantive_signal: false`. Amending a
criterion after seeing the result is exactly what a predeclared gate exists to prevent.

## Two bugs found, both mine, both with evidence

**1. The ES set was a subset of its own training data.** The harness fitted the early-stopping
model on the *full* outer-fit frame with `eval_set = f.iloc[es_l]`. CatBoost never early-stops in
that configuration. Exposed by an arithmetic impossibility: a *shorter* budget with *shorter*
patience (2500/200 vs 6000/300) selected **2499** where Phase 10's identical config selected
**1043**.

Fix verified, not assumed: the fixed harness reproduces Phase 10 **exactly** — 0.961067 at 1043
iters, identical. Two identical control runs both returned 1043, so CatBoost is deterministic here
and 2499 was a bug, not noise. The harness now aborts if the carve sets overlap or do not partition
the outer fit. Contaminated arms quarantined in `reports/contaminated_es_leak/`.

**2. A cost upper bound used as a point estimate silently skipped two experiments.** The
projection assumed ES never fires, so C5 and ctr2 "projected" 121 and 132 min against a 90-min
budget and were **skipped, not run** — quietly, since a cost skip prints as a skip. Their real
cost is ~31 and ~32 min. Both ran post-fix.

Three further defects I introduced while fixing these are also recorded: an `UnboundLocalError`
from `n` being both a tuple-assignment target and used on the RHS; a test mixing the *position*
index space with the *row* index space; an over-strict changed-row count (`n_changed == n_fold`)
that false-alarmed because a fold row may legitimately be unchanged. A 4th: a literal-string test
assertion broke when its message was deliberately rewritten — asserting on wording, not behaviour,
so it now checks the behaviour.

## Answers to the 12 questions

1. **"+0.79e-5 structural cap" removed as invalid?** **Yes, withdrawn.** AUC is a nonlinear
   functional of the prediction vector; blend gain is neither equal to nor bounded by
   `family_weight × standalone_gain`. The valid Phase 11 conclusion was only that the two zero-cost
   swaps gave ≈0 gain.
2. **Which previous C2 slot swap was exact?** **None.** C2 is seed 4; no slot matches on seed.
   Those swaps are relabelled **APPROXIMATE ZERO-COMPUTE DIAGNOSTICS**. All three swaps in this
   report are exact (seed + view + depth + l2 + scheme).
3. **Age categorical?** No — **−12.9e-5**, and it converges 1043 → 697 iters.
4. **Flight Distance categorical?** Not usably — **+0.9e-5**, below the +2e-5 threshold.
5. **CTR complexity = 2?** No — **−2.7e-5** at +58% compute.
6. **Do 3 distinct counterparts retain diversity?** **Yes** — median logit corr 0.99779 vs the
   originals' 0.99748. No collapse. But they are interchangeable in *effect*.
7. **Do MINI_B50/B100 move v3?** **No** — +0.07e-5 and +0.04e-5.
8. **Inside the 0.96134 notebook?** `sachith7/s6e10-what-each-step-was-worth`. **No CatBoost at
   all** — no `cat_features`, no `CatBoostClassifier`, no `max_ctr_complexity`. It reports a
   negative for a *different* substitution: numerics-as-categories **instead of** target encoding
   (0.960387 vs 0.961014), and flags its own confound. We measured categoricals **on top of** our
   existing TE block. Neither refutes the other. Its reproducible finding is 10-fold members
   (+0.00009 to +0.00018); its gains are otherwise dominated by public-LB observation and imported
   public predictions. **No category-F genuinely new mechanism.**
9. **Any genuinely new public mechanism?** No. Its own ±0.0008 single-submission noise estimate
   agrees with our ±2e-4 paired floor.
10. **HEAD:** see §Git. Tests: **413 assertions, 0 failures** across 6 suites.
11. **Submission justified?** **No.** No gate carrying evidential weight was met.

## What this closes, and what it does not

Closed: Phase 11 (diverse native block), and with it Age-as-category, Flight-Distance-as-category,
and CTR-complexity expansion. Seven-slot training is **not** started; ~12 h of compute avoided.

**This does not explain the 7.8e-4 leaderboard gap.** It rules out one mechanism family and
records a genuinely informative negative: native CatBoost CTRs are a real +6.64e-5 *model-level*
effect (Phase 10B, 4/5 folds) that is **diverse but redundant at the ensemble level**. The two
families already in the blend (lgbm 32/59, cat 7/59) remain weight-capped, and this phase adds
independent evidence that the CatBoost family is not merely capped by weight.

## Git
See `git log`. Every entry above is also in `experiments/ledger.jsonl` (161 lines, append-only),
including the two withdrawn claims. Failed history was not rewritten.