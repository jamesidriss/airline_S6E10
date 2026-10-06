PHASE 11 -- DIVERSE NATIVE-CATEGORICAL CATBOOST BLOCK

WHY THIS PHASE EXISTS
Phase 10 established that native categorical CTRs genuinely improve a CatBoost model: C2 - C0 =
+6.64e-5 full 5-fold OOF, per-fold mean +6.51e-5, paired SE 3.00e-5, t = 2.17, positive in 4/5 folds.
That is a replicated positive signal and a credible mechanism, not proof: with n_folds = 5 the
inferential power is limited, so the wording is deliberately "replicated positive signal".

But it produced NO ensemble gain. Adding one C2 as a 60th member: +0.00e-5. Replacing all seven
CatBoost slots with that one C2: 0.961509 -> 0.961505, i.e. -0.39e-5, positive in only 2/5 folds.

The reason is diversity, and it is the whole point of this phase. v3's CatBoost block is seven
correlated-but-distinct members (pairwise logit corr 0.99637 / 0.99748 / 0.99803 min/median/max)
differing in depth, seed, feature view and fold scheme. Their 11.86% of the ensemble weight earns its
keep partly through averaging. One individually stronger model cannot replace seven, because it
destroys that averaging. So the correct test is SEVEN DIVERSE native counterparts -- one per slot,
each keeping its own configuration -- not one model repeated seven times.

================================================================================
SLOT INVENTORY (reports/native_cat_slot_inventory.json, hashed)
================================================================================
Recovered from the prediction store metadata, the ledger, reports/all_models_final.md and the
run_views.py defaults, in that priority order. Nothing was inferred from a member's NAME.

  slot  exp_id                    view          scheme   lr     depth  l2     seed
    0   z3_cat_f10                full          block10  0.04   8      3.0    1
    1   z3_cat_d10                full          primary  0.03   10     6.0    2
    2   z3_cat_d8_s2              full          primary  0.04   8      3.0    3
    3   prod5_cat_full_primary    full          primary  0.04   8      3.0    1   [runner defaults]
    4   z3_cat_d6                 full          primary  0.04   6      5.0    1
    5   z3_cat_core3              core3         primary  0.04   8      3.0    4
    6   z3_cat_ogs                full_ogsurf   primary  0.04   8      3.0    5

TWO CORRECTIONS TO THE TASK BRIEF, made from evidence rather than deference:
  * "f10" is NOT feature_fraction=0.1. Its recorded params are lr=0.04, depth=8 -- identical to
    z3_cat_d8_s2 apart from the seed. The name refers to the FOLD SCHEME: slot 0 is the only block10
    member of the seven, and its counterpart must therefore be trained on block10, not primary.
    Getting this wrong would have built the counterpart on a different partition and made the
    comparison meaningless.
  * prod5_cat_full_primary has no explicit params dict anywhere. It is a byte-identical duplicate of
    view_cat_full_primary (oof_sha 935d6778ee54 on both), so its configuration is the run_views.py
    defaults and its seed is 1, the runner's --seed default.

Also recorded, not fixed: the originals' OOF came from `_fit_cat_es`, which early-stops on a 10%
carve and then writes the OOF from the model trained on the OTHER 90% -- the carve is permanently
discarded. And `_fit_full_predict_cat` refits on 100% at the median iteration but with
learning_rate=0.05 while the OOF path used 0.04, a pre-existing inconsistency in the finalist's test
predictions. Any native test inference must decide deliberately whether to reproduce it.

================================================================================
THE CONFOUND, DECLARED NOT HIDDEN
================================================================================
A native counterpart trained under the Phase 10 fixed-round protocol beats its original on TWO axes at
once: the categorical representation AND a 10% larger training fraction (worth +12.8e-5 on CatBoost
per Phase 10's C0 measurement). So:

    native_slot - numeric_counterpart  = the MECHANISM, protocol-matched  (causal)
    numeric_counterpart - original     = the PROTOCOL effect alone
    native_slot - original             = the OPERATIONAL REPLACEMENT delta (the two, added)

The runner trains both variants per slot so the first two are separable. The causal estimate of the
mechanism remains Phase 10's C2 - C0 = +6.64e-5, where both arms shared one protocol.

================================================================================
FIXED BLOCK WEIGHT -- WHY A 66-MEMBER BLEND WOULD BE THE WRONG TEST
================================================================================
v3 is an equal-logit blend of 59 members, so each slot owns exactly 1/59 = 0.016949. Candidates replace
what FILLS each CatBoost slot while leaving the other 52 slots byte-identical and every weight
untouched. Appending seven native members as a 66-member blend would raise CatBoost's family weight
from 11.86% to 23.4% and confound member quality with family weight -- flattering the native block
for the wrong reason. Predeclared candidates, mixed in LOGIT space (the space v3 averages in):

  B0    alpha_native = 0.00   the original seven, == v3
  B25   0.25 per slot      B50  0.50      B75  0.75      B100  1.00

================================================================================
HARNESS GUARD, AND A CORRECTION TO IT
================================================================================
The brief required that alpha=0 reconstruct v3's OOF 0.961508578 to numerical precision, and to STOP
otherwise. It did stop -- at a -1.45e-10 AUC difference with logit correlation 1.000000000. That was
my guard being wrong, not the harness. v3 is STORED as sigmoid(mean(member probabilities)) while the
block harness averages member LOGITS; those are different functions, so bit-equality is impossible
and the achievable agreement is float64 round-off. The guard now asserts round-off plus logit
correlation of 1 to machine precision, and the reported deltas are taken against the harness's own
reconstruction so every candidate shares one geometry. Asserting strict equality would have masked a
real breakage later.

================================================================================
STAGE 1 -- FOLD-0 SCREEN (in progress)
================================================================================
Seven slots x two variants (native, numeric) on fold 0. Runtime measured first, as required:
slot 3 native, 302 features (17 categorical), 0.2905 s/round at real scale, so ~0.10 h for 1200
rounds. Depth 10 and the two non-full views are the expensive ones.

Stage 1 continuation rule, predeclared: continue to fold 1 if EITHER at least one of
B25/B50/B75/B100 improves v3 fold-0 by >= +1.0e-5, OR at least three individual one-slot swaps are
positive AND native block diversity has not collapsed. If every block candidate is negative and fewer
than three slot swaps are positive, the diverse-block hypothesis stops immediately without
rationalisation.

tests/test_native_block.py: 52/52, covering all ten required properties.
