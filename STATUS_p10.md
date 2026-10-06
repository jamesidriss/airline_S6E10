Phase 10 complete -- shared bias tested on two independent axes

MOTIVATION
Phase 9 closed DART and RF. Its most informative result was not either model: our remaining ranking
error concentrates where the 98 ensemble members AGREE, not where they disagree (error-vs-disagreement
is U-shaped; the full-agreement band carries a 1.7-1.8x higher pair error rate than the median band).
That argues the ensemble shares a systematic bias rather than carrying excess variance, and it
contradicts every variance-reduction lever still on the table.

  A. nested residual structure -- is E[y - p | group] reproducibly non-zero?
  B. a model family whose representation of categorical interactions is genuinely different

================================================================================
HEADLINE
================================================================================
A. Simple group residual structure: EXHAUSTED. No specialist built.
B. Native CatBoost categorical CTRs: a REAL, REPLICATED, POSITIVE mechanism at the model level.
     C2 - C0 = +6.64e-5 on the full 5-fold OOF; per-fold mean +6.51e-5, paired SE 3.00e-5,
     t = 2.17, positive in 4/5 folds (+5.87, +11.58, -2.37, +14.44, +3.02).
   Ordered boosting, the other half of the CatBoost hypothesis, is clearly HARMFUL (-71.7e-5).
   BUT the ensemble cannot harvest it: marginal add of C2 = +0.00e-5, single-model swap = -0.39e-5.

================================================================================
CORRECTIONS TO PHASE 9 MADE BEFORE ANY NEW RESEARCH
================================================================================

1. extra_trees ATTENUATES under DART, it does NOT reverse sign. Measured:
     under GBDT  ctl_fixed 0.9613518 - ctl_det   0.9609042 = +44.76e-5
     under DART  dart005_xt 0.9610003 - dart005  0.9609236 =  +7.67e-5
   Both positive. Attenuation factor 5.83x. My earlier "the sign flips" was wrong: I compared the
   two DART arms' deltas against the control (-35.2e-5 vs -42.8e-5) and read their ordering as a
   sign change, when both are negative and their difference is positive. The quantity that identifies
   the extra_trees effect is the WITHIN-family contrast, not that contrast's ordering vs the control.
   The 90k probe claimed -4.82e-3 under DART; the fold measured +7.67e-5. That does NOT replicate --
   opposite in sign, magnitude ratio 62.8x (I wrote ~600x, wrong by an order of magnitude). Only the
   probe's broader conclusion (DART is uncompetitive) survived.
   The mechanistic story that DART renormalisation is actively HARMFUL to random-threshold trees is
   WITHDRAWN. What survives: DART dilutes how much extra_trees helps. No mechanism is claimed.

2. A stale paired z-score claim survived in STATUS.md section 8: "That is z = 3.1-4.4 ... and it
   clears 2 sigma even between uncorrelated predictions." Already retracted in section 6e, but the
   executive summary was never updated. We do not hold the leader's prediction vector, so rho is
   UNMEASURED; across rho in [0,1] the paired z spans 0.44 to 7.88 and clears 2 sigma only above
   rho = 0.9411, and at rho = 0 it is 0.49. Section 8 now states the factual scores only
   (0.960980 vs 0.961760, gap 7.8e-4) plus the rho-INDEPENDENT observation that the top twenty sit in
   a 1.0e-4 band 7.8e-4 above us.
   Process lesson: a correction applied to one section did not propagate to the summary, which is the
   part that gets read.

================================================================================
10A. NESTED RESIDUAL STRUCTURE -- EXHAUSTED
================================================================================

VERDICT: **SIMPLE GROUP RESIDUAL STRUCTURE IS EXHAUSTED.** No specialist built.

AUDIT of the pre-existing scripts/residual_structure.py: NOT promotion-grade. For a discovery row j in
fold f(j) != k, its base score comes from blend_v3_final OOF, i.e. models trained on every fold EXCEPT
f(j) -- which INCLUDES fold k, the confirmation fold. So the residual y_j - p_j, and the bias table
fitted from it, indirectly depend on the labels of the rows it is scored on. Magnitude unknown and sign
not guaranteed. Now labelled NON-NESTED META-DIAGNOSTIC in its docstring, JSON and stdout. It also
MIXED SCALES: a probability-space residual (y - p) added to a logit-scale score.

scripts/residual_nested.py is the promotion evidence. Per outer fold k:
  META_TRAIN = all rows except fold k; META_VAL = fold k
  inside META_TRAIN only: 5-fold inner cross-fit -> p_meta_train_oof
                          (no model producing a META_TRAIN prediction sees that row, and NO model
                           producing one trains on ANY META_VAL row)
  one champion fit on 100% of META_TRAIN at a fixed 900 rounds -> p_meta_val
  bias estimated on (y - p_meta_train_oof) over META_TRAIN, frozen, applied to p_meta_val
Base is a single champion lgbm extra_trees surrogate, NOT v3. Consistency check: the fold-0 surrogate
scores 0.961352, identical to Phase 9's ctl_fixed (0.9613518).

Two score-space semantics kept strictly separate, both cross-fitted:
  PROBABILITY   b_g = shrunk mean(y - p | g);            score = clip(p + b_g, eps, 1-eps)
  LOGIT OFFSET  delta_g = regularised intercept MLE;    score = logit(p) + delta_g
Pre-declared, not tuned: PRIOR_N=50, LAMBDA=50, MIN_N=200, N_BINS=24, ROUNDS=900.

42 predeclared keys: raw 21, current TE keys, survey patterns, segments, original-knowledge regimes,
base confidence. Hundreds of ad-hoc keys deliberately excluded -- the gate demands replication across
folds, and a huge family guarantees some key passes by chance.

RESULT. Exactly one key passed on the surrogate:
  raw:On-board service   prob +2.00e-5 (4/5)   logit +1.92e-5 (4/5)
                          sign agreement 0.720   replication r=0.540   mean bias gap -20.45e-5
Two numbers already argued against believing it: r=0.540 is modest, and the bias gap is TEN TIMES the
size of the gain it produces -- a fitted group offset that does not carry its magnitude out of sample.

TRANSFER TEST (scripts/residual_transfer_test.py) -- the operationally decisive question. The nested
scan used a single champion surrogate, not v3. Same correction (fitted only on META_TRAIN from nested
residuals, so it saw no META_VAL label) applied to v3's OOF on META_VAL rows:

  fold   surrogate    v3 prob    v3 logit
    0      +2.04e      -0.71e       -0.27e
    1      -4.08e      -7.18e       -4.40e
    2      +3.89e      +2.10e       +2.34e
    3      +6.29e      +4.95e       +3.62e
    4      +1.85e      -0.57e       +0.21e
  mean    +2.00e      -0.28e       +0.30e    (2/5 and 3/5 folds positive)

**The +2.00e-5 does NOT transfer.** On v3 the same correction is far below the +1.5e-5 gate and not
positive in 4/5 folds. The structure was a property of the SURROGATE'S residuals, not a shared bias in
the ensemble. Phase 9's models-fail-together result stands, but the shared bias is NOT a
group-constant mean offset on any of the 42 predeclared groupings.

--------------------------------------------------------------------------------
BUGS FOUND IN MY OWN DIAGNOSTIC, kept because they nearly produced a false positive
--------------------------------------------------------------------------------
- **The replication statistic was TAUTOLOGICAL.** I computed the "confirmation-observed" group bias from
  (y - p_oof) over the DISCOVERY rows -- the same rows and same quantity the table was fitted on,
  differing only by shrinkage. It returned r = 1.000 for all 42 keys, which is precisely why every key
  in the first report looked perfectly replicated. It measured shrinkage, not replication, so gate
  criterion 3 was vacuous. Fixed to estimate E[y - p | g] independently on META_VAL using p_val; honest
  correlations are 0.02-0.98 with a median near 0.3, and several keys go NEGATIVE. This is the single
  most important correction in Phase 10: a diagnostic that cannot fail is worse than no diagnostic.
- The logit offset was first a SINGLE Newton step from 0, called the MLE. It is not: with a saturated
  base (p=0.30, true rate 0.20) one step returned +0.4557 where the optimum is -0.4861 -- the WRONG
  SIGN. Now iterated to convergence and verified against a brute-force minimiser.
- `pd.factorize` returns -1 for NaN and `np.bincount` rejects negatives, so any key with a missing
  value crashed the scan. NaN is now its own group, normalised ONCE before the sides are sliced.
- A single malformed key aborted the whole 5-fold scan. Failures are caught per key and RECORDED.
- Aggregation was POSITIONAL, so one fold's failed key shifted every later record. Now by NAME, with
  all-5-folds completeness required.
- BASE-CONFIDENCE keys must be built per side (p-deciles from p_oof on discovery, p_val on
  confirmation); one shared series crashed AND would have used the scored rows to set boundaries.
- residual_nested.py's aggregation wrote the summary into the same dict it was accumulating, raising
  KeyError on the next key. Accumulation and summarisation are now separate passes.

================================================================================
10B. NATIVE CATBOOST CATEGORICAL LEARNING -- POSITIVE AT MODEL LEVEL, UNUSABLE BY THE ENSEMBLE
================================================================================

AUDIT: zero `cat_features` and zero CatBoost `boosting_type` anywhere in src/ or scripts/. Every
CatBoost model ever fitted went through ViewBuilder -> float32 numpy -> CatBoostClassifier.fit(X, y),
so Gender, Customer Type, Type of Travel, Class and all 13 service ratings were ORDINAL FLOATS. Its
defining mechanism -- ordered target statistics for categorical features -- was never exercised. The
best CatBoost ever recorded was 0.9610555 (z3_cat_d10), below the deterministic LightGBM, so
"CatBoost is worse" was never tested; "CatBoost without categoricals" was.

MEASURED cardinalities, three of my guesses wrong: Customer Type 2 (no 'Neutral Customer'), Class 3
(no 'First'), Baggage handling 5 (one rating level unobserved). Sentinel does not collide; 0 missing.

CORRECTION -- CatBoost DOES resolve cat_features by name. My first capability probe reported every
native-categorical arm as unsupported and I concluded "CatBoost resolves cat_features positionally, not
by name". MEASURED, false. My bug: the probe passed SOURCE names (Gender, ...) to a frame whose columns
are TWIN names (ncat__Gender, ...). Verified: positional indices OK; twin NAMES that exist OK; a name
NOT in the frame raises KeyError. It was MY bug, not a CatBoost limitation, so the fix belongs in my
code and not in a claim about the library.

CTR ENGAGEMENT VERIFIED POSITIVELY (scripts/verify_ctr_engaged.py). The probe's original check called
`get_leaf_ctr_description()`, which does not exist on CatBoostClassifier -- it raised, recorded None,
and printed "categoricals may have been silently ignored", a conclusion it never tested. Replaced by:
  T1 model dump contains ctr_type structures           YES (4 occurrences)
  T2 predictions DIVERGE from an ordinal-code control   max |diff| 3.65e-01, corr 0.99914
  T3 declaring a numeric column as categorical RAISES   TypeError
VERDICT: CTR MACHINERY VERIFIED ENGAGED. Necessary because the silent-ignore case is real.

2x2 on fold 0 (primary, full view, champion CatBoost params, fixed-round protocol). Nothing varies
except the mechanism -- asserted in tests/test_native_cat.py:

  arm  boosting  cats  iters   AUC        delta vs C0  logit corr v3  spearman  blend@2%
  C0   Plain     no     851   0.961008   (control)    --             --        -0.23
  C1   Ordered   no     497   0.960291     -71.7e     0.99761       0.97968   -0.60
  C2   Plain     yes   1043   0.961067      +5.9e     0.99895       0.99197   -0.14
  C3   Ordered   yes    629   0.960412     -59.7e     0.99771       0.98058   -0.44

ORDERED BOOSTING HURTS, and is closed. -71.7e-5, the worst arm, selecting far fewer iterations
(497 vs 851). Mechanism stated as a hypothesis consistent with the data, NOT a measured finding:
Ordered exists to prevent target leakage and buys that by computing each tree's target statistics from
an ordered SUBSET of rows. With 559,708 outer-fit rows that safety is unnecessary and the reduced
effective sample is a pure cost; the early iteration count is the same signal. The 2x2 is clean and
additive in the harmful direction -- Ordered hurts regardless of categoricals (C3 - C1 = +12.1e-5) and
its cost swamps the categoricals' benefit (C3 - C2 = -65.5e-5). The two halves of the CatBoost
mechanism have OPPOSITE verdicts, so testing them as one hypothesis would have been wrong.

NATIVE CATEGORICAL CTRs HELP, and this replicated over the full primary 5-fold:

  arm    fold0     fold1     fold2     fold3     fold4     full OOF
  C0   0.961008 0.961221  0.961174  0.960267  0.961665  0.961063
  C2   0.961067 0.961337  0.961151  0.960411  0.961695  0.961121
  d     +5.87e   +11.58e    -2.37e   +14.44e    +3.02e    +6.64e

  mean +6.51e-5   paired SE 3.00e-5   t = 2.17   positive in 4/5 folds

CONTROL VALIDATED: C0 = 0.961008 vs the established view_cat_full_primary fold-0 of 0.960880, i.e.
+12.8e-5. Not a discrepancy: C0 uses the Phase 9 fixed-round protocol (inner ES picks the iteration
count, refit on 100% of outer-fit) where the established path discarded the 10% carve. Phase 9
measured +5.3e-5 for that change on LightGBM; a weaker model gaining more from 11% more data is
plausible. Recorded so the +12.8e-5 is not later mistaken for a bug.

--------------------------------------------------------------------------------
WHY A REAL +6.6e-5 MECHANISM YET NO ENSEMBLE GAIN
--------------------------------------------------------------------------------
scripts/cat_swap_analysis.py, zero training cost:

* C2 (OOF 0.961121) is individually STRONGER than ALL SEVEN of v3's existing CatBoost members, by
  +4.9e-5 (z3_cat_f10) to +25.7e-5 (z3_cat_ogs). So the mechanism is not cosmetic -- it genuinely
  improves the CatBoost family, and every current CatBoost member is strictly dominated.
* Marginal admission of C2 as a NEW member: +0.00e-5 at w=1%, ~0 at 2%, -0.07e-5 at 5%, -0.34e-5 at 10%.
* Swap counterfactual, giving the CatBoost block's 7/59 = 0.119 weight to C2 instead:
  0.961509 -> 0.961505, delta **-0.39e-5**, positive in only 2/5 folds.

The explanation is DIVERSITY, and it is the substantive lesson of Phase 10B. v3's rebuild from its 59
stored members reproduces the stored blend to logit corr 1.000000 and AUC 0.961509 exactly, so the
arithmetic is trustworthy. C2's logit correlation with each existing CatBoost member is 0.997-0.998 --
nearly identical. Replacing SEVEN correlated-but-distinct members with ONE individually stronger model
destroys the averaging that made the block worth 11.9% of the ensemble in the first place. Individual
member quality and block contribution are different quantities, and here they point opposite ways.

THE ONE LIVE UNTESTED HYPOTHESIS. A DIVERSE native-categorical CatBoost block -- several members
mirroring the existing configs (depth 6/8/10, feature_fraction, core3 and ogs views, seed variation),
each individually better than its numeric counterpart AND retaining the diversity the block depends on
-- could be strictly better than the current numeric block. That is untested; the single-model swap
above is pessimistic about diversity and is explicitly an approximation, not a submission. Estimated
cost ~1.5 h for 7 members plus a re-blend. It is the only Phase 10 thread with a plausible route to a
real ensemble gain.

Real-scale timing measured BEFORE committing (s/round on 559,708 rows x 285 features): C0 0.122,
C1 0.689, C2 0.284, C3 0.883, so the slowest arm projected to 0.22 h for the refit rather than the
hours a naive extrapolation from the 15% probe suggested.

SUBMISSION: none. No arm improved v3 by >= +1.5e-5. v3_final (public 0.960980) and v4_fulldata
(public 0.961000) remain the banked finalists, both immutable.

tests/run_tests.py: 54 passed + test_stochastic_protocol 69/69.
tests/test_residual_nested.py: 62/62.  tests/test_native_cat.py: 101/101.
