Phase 10 -- shared bias: nested residual structure, and native CatBoost categoricals

MOTIVATION
Phase 9 closed DART and RF. Its most informative result was not either model: our remaining ranking
error concentrates where the 98 ensemble members AGREE, not where they disagree (error-vs-disagreement
is U-shaped; the full-agreement band carries a 1.7-1.8x higher pair error rate than the median band).
That argues the ensemble shares a systematic bias rather than carrying excess variance, and it
contradicts every variance-reduction lever still on the table.

Two independent ways to test shared bias:
  A. nested residual structure -- is E[y - p | group] reproducibly non-zero?
  B. a model family whose representation of categorical interactions is genuinely different

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
10A. NESTED RESIDUAL STRUCTURE
================================================================================

AUDIT of the pre-existing scripts/residual_structure.py: it is NOT promotion-grade. For a discovery
row j in fold f(j) != k, its base score comes from blend_v3_final OOF, i.e. models trained on every
fold EXCEPT f(j) -- which INCLUDES fold k, the confirmation fold. So the residual y_j - p_j, and the
bias table fitted from it, indirectly depend on the labels of the rows it is scored on. Magnitude
unknown and sign not guaranteed. That script is now labelled NON-NESTED META-DIAGNOSTIC in its
docstring, in its JSON and on stdout, and points at the replacement. It also MIXED SCALES: it
estimated a probability-space residual (y - p) and added it to a logit-scale score.

scripts/residual_nested.py is the promotion evidence. Per outer fold k:
  META_TRAIN = all rows except fold k; META_VAL = fold k
  inside META_TRAIN only: 5-fold inner cross-fit -> p_meta_train_oof
                          (no model producing a META_TRAIN prediction sees that row, and NO model
                           producing one trains on ANY META_VAL row)
  one champion fit on 100% of META_TRAIN at a fixed 900 rounds -> p_meta_val
  bias estimated on (y - p_meta_train_oof) over META_TRAIN, frozen, applied to p_meta_val
Base is a single champion lgbm extra_trees surrogate, NOT the 59-member v3 blend. Round count fixed
at 900 (Phase 9's honest inner selection), so iteration choice raises no nesting question.

Two score-space semantics kept strictly separate, both cross-fitted:
  PROBABILITY   b_g = shrunk mean(y - p | g);            score = clip(p + b_g, eps, 1-eps)
  LOGIT OFFSET  delta_g = regularised intercept MLE;    score = logit(p) + delta_g
Pre-declared and not tuned: PRIOR_N = 50, LAMBDA = 50, MIN_N = 200, N_BINS = 24, ROUNDS = 900.

BUG the tests caught: I first implemented the logit offset as a SINGLE Newton step from 0 and called
it the MLE. It is not. With a saturated base (p = 0.30, true rate 0.20) one step returned +0.4557
where the numerical optimum is -0.4861 -- the WRONG SIGN, because the gradient at delta=0 is small
and so is the curvature, so one step overshoots past the point where the gradient changes sign. Now
iterated to convergence, and the tests verify the iterate against a brute-force minimiser. This would
have silently inverted corrections for exactly the saturated groups where a large bias lives.

Three further bugs found by running the scan, all mine:
  - `pd.factorize` returns -1 for NaN and `np.bincount` rejects negatives, so any key with a missing
    value crashed it. NaN is now its own group, normalised ONCE before the two sides are sliced so
    discovery and confirmation cannot disagree about the sentinel.
  - A single malformed key aborted the whole 5-fold scan. Failures are now caught per key and
    RECORDED, never silently dropped.
  - Aggregation was positional, so one fold's failed key would shift every later record and pair fold
    A's key with fold B's neighbour. Now aggregated by key NAME, and a key must be scored on all 5
    folds to be eligible -- otherwise a key that survived only where it did not crash could pass.
  - BASE-CONFIDENCE keys are built per side (p-deciles from p_oof on discovery, p_val on
    confirmation). Using one series for both made them crash AND would have chosen decile boundaries
    using the rows being scored.

Groupings are PREDECLARED (raw 21, current TE keys, survey patterns, segments, original-knowledge
regimes, base confidence) -- 44 keys. Hundreds of ad-hoc keys are deliberately excluded: the gate
demands replication across folds, and a huge family guarantees some key passes by chance.
Gate: >=4/5 folds positive under BOTH semantics AND mean >= +1.5e-5.

Consistency check: the fold-0 surrogate scores 0.961352, identical to Phase 9's ctl_fixed (0.9613518).

================================================================================
10B. NATIVE CATBOOST CATEGORICAL LEARNING
================================================================================

AUDIT (scripts/native_cat.py): zero occurrences of `cat_features` or CatBoost `boosting_type`
anywhere in src/ or scripts/. Every CatBoost model this campaign ever fitted went through
    ViewBuilder -> float32 numpy -> CatBoostClassifier.fit(X, y)
so Gender, Customer Type, Type of Travel, Class and all 13 service ratings were handed to CatBoost
as ORDINAL FLOATS. Its defining mechanism -- ordered target statistics for categorical features --
was never exercised. The best CatBoost ever recorded is 0.9610555 (z3_cat_d10), below the
deterministic LightGBM, so "CatBoost is worse" was never tested; "CatBoost without categoricals" was.

MEASURED cardinalities, and three of my guesses were wrong:
  Customer Type 2 (not 3 -- no 'Neutral Customer' in this data)
  Class 3 (not 4 -- Business/Eco/Plus, no 'First')
  Baggage handling 5 (not 6 -- one rating level unobserved)
Sentinel '__MISSING__' does not collide with any real level; there are 0 missing ratings.

CORRECTION -- CatBoost DOES resolve cat_features by name. My first capability probe reported every
native-categorical arm as unsupported and I concluded "CatBoost resolves cat_features positionally,
not by name". MEASURED, that is false. The real cause was my own bug: the probe passed the SOURCE
column names (Gender, Type of Travel, ...) to a frame whose columns are the TWIN names
(ncat__Gender, ...). Verified behaviour in 1.2.10:
  positional indices                        OK
  twin NAMES that exist in the frame        OK  (same resolved indices)
  a name NOT in the frame                   raises KeyError "'Gender' is not in list"
So names are honoured; an unknown name is reinterpreted as a position, which read numeric column 0
and produced `Invalid type for cat_feature[...] = 1.0`. Two arms failed for a reason unrelated to
native categorical support, and I nearly recorded a false capability finding. It was MY bug, not a
CatBoost limitation, so the fix belongs in my code and not in a claim about the library.

CTR ENGAGEMENT VERIFIED POSITIVELY (scripts/verify_ctr_engaged.py). The probe's original check
called `get_leaf_ctr_description()`, which does not exist on CatBoostClassifier -- the call raised,
the probe recorded None, and the summary printed a scary "categoricals may have been silently
ignored" that it had never actually tested. Replaced by three decisive checks:
  T1 model dump contains ctr_type structures           -> YES (4 occurrences)
  T2 predictions DIVERGE from an ordinal-code control   -> max |diff| 3.6e-01, corr 0.99914
  T3 declaring a numeric column as categorical RAISES   -> TypeError
Verdict: CTR MACHINERY VERIFIED ENGAGED. The experiment is genuinely testable. This check matters
because the silent-ignore case is real: a categorical column present but undeclared trains happily
as a float and every number looks plausible while nothing is tested.

Frame rules: category identity never passes through float32 (twins are strings, declared by name);
twins come from the raw frame by column name and never touch the target; the existing `*__cat` float
twins inside the champion view are left untouched so arm C0 reproduces the numeric control
bit-for-bit. Flight Distance (~3,474 levels) and Age (~75) are EXCLUDED so the 2x2 stays
interpretable; high-cardinality CTRs are a separate later arm (C4/C5).

2x2 (fold 0 first): C0 numeric Plain (control) / C1 numeric Ordered / C2 numeric + native cats Plain /
C3 numeric + native cats Ordered. Round selection uses the Phase 9 fixed-round protocol: inner ES
picks the iteration count, refit on 100% of outer-fit, score the outer fold once.

Capability probe (15% subsample, 250 rounds -- capability and RATE only, no score is evidence):
all eight arms train. Ordered CPU SUPPORTED. Native cats CPU SUPPORTED. GPU + cats works.
GPU + Ordered works. Ordered is ~4.7x slower than Plain; native cats ~11x slower than numeric
Plain. GPU is worth measuring at real scale before committing.

tests/test_native_cat.py: 101/101, covering all nine required properties plus the name-resolution
rule. tests/test_residual_nested.py: 62/62.
