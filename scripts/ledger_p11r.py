"""Append the Phase 11R ledger entries.

Kept as a script rather than an inline command because the findings are long, contain quotes, and
an inline here-string through PowerShell mangled one of them into a syntax error -- which is itself
worth noting, because a half-written ledger entry is worse than none.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()

ROWS = [
    {
        "phase": "p11r",
        "id": "p11r_ES_CARVE_WAS_A_SUBSET_OF_ITS_OWN_TRAINING_ROWS",
        "kind": "CORRECTION",
        "claim": "Phase 11R arms C2REF and C4 selected 2499 of 2500 rounds, while Phase 10's "
                 "identically-configured C2 selected 1043 under a 6000-round budget.",
        "finding": "The cause was a leak in my own harness, not a budget effect. The ES model was "
                   "fitted on the FULL outer-fit frame while eval_set was f.iloc[es_l], making the "
                   "early-stopping set a SUBSET OF ITS OWN TRAINING DATA. CatBoost never "
                   "early-stops in that configuration, because the eval metric keeps improving "
                   "while the model memorises the rows it is being scored on. A shorter budget with "
                   "shorter patience (2500/200 versus 6000/300) cannot plausibly stop LATER, and "
                   "that arithmetic impossibility is what exposed the defect.",
        "evidence": "The fixed harness reproduces Phase 10 EXACTLY: C2REF AUC 0.961067 at 1043 "
                    "iterations, identical to Phase 10 C2 at 0.961067 and 1043 iterations. Two "
                    "identical control runs both returned 1043, so CatBoost is deterministic in "
                    "this configuration and the 2499 was a bug rather than noise. The fix raises "
                    "SystemExit if the carve sets overlap or if inner-train plus ES does not "
                    "partition the outer fit.",
        "verdict": "C2REF and C4 fold-0 arms trained before the fix are CONTAMINATED and are "
                   "quarantined under reports/contaminated_es_leak/ with their logs. Every Phase "
                   "11R number below comes from the post-fix re-run. The earlier reading of C4 as "
                   "-6.9e-5 against a budget-matched reference is WITHDRAWN, not merely "
                   "superseded.",
    },
    {
        "phase": "p11r",
        "id": "p11r_COST_UPPER_BOUND_USED_AS_POINT_ESTIMATE_SKIPPED_TWO_EXPERIMENTS",
        "kind": "CORRECTION",
        "claim": "C5 and ctr2 were skipped as projecting 121 and 132 minutes against a 90-minute "
                 "budget.",
        "finding": "The projection evaluated cost(R) = a*R + b*R^2 at ES_ROUNDS = 2500, which is an "
                   "UPPER BOUND that assumes early stopping never fires. This workload stops at "
                   "about 1043 rounds. At the measured stopping point C5 costs roughly 31 minutes "
                   "and ctr2 roughly 32 minutes, so both fit the budget comfortably. An upper "
                   "bound used as a point estimate is not a conservative choice, it is a decision "
                   "made by default, and it silently removed both arms from Question A.",
        "evidence": "Post-fix projections print the measured stopping point and the cap worst case "
                    "on separate lines. C5 ran in 1160 s and ctr2 in 1160 s actual, against "
                    "cap-worst-case projections of 117 and 132 minutes.",
        "verdict": "FIXED. --project-iters (default 1100, from the measured 1043) drives the budget "
                   "gate; the cap worst case stays visible rather than becoming the estimate.",
    },
    {
        "phase": "p11r",
        "id": "p11r_AGE_TWIN_NEGATIVE",
        "kind": "EXPERIMENT",
        "claim": "C4 = C2 plus an exact categorical Age twin (75 levels), numeric Age retained, "
                 "primary fold 0.",
        "finding": "C4 scored 0.960938 against the budget-matched C2REF at 0.961067, a delta of "
                   "-12.9e-5. Neither arm hit the ES cap, so the comparison is fair. Adding the Age "
                   "categorical also cut the selected iteration count sharply, 1043 to 697, so the "
                   "model converges earlier AND worse.",
        "evidence": "iters C2REF 1043 versus C4 697; logit corr C4/C2REF 0.99911; corr versus v3 "
                    "0.99896; 551 s versus 736 s.",
        "verdict": "NEGATIVE. Age as a native category does not help. The early convergence is "
                   "consistent with a shortcut that saturates the model without improving "
                   "generalisation. Branch closed; no fold 1, since the predeclared promotion rule "
                   "required at least +2e-5.",
    },
    {
        "phase": "p11r",
        "id": "p11r_FLIGHT_DISTANCE_TWIN_EFFECTIVELY_ZERO",
        "kind": "EXPERIMENT",
        "claim": "C5 = C2 plus an exact categorical Flight Distance twin (3474 levels), numeric "
                 "Flight Distance retained, primary fold 0.",
        "finding": "C5 scored 0.961075 against C2REF at 0.961067, a delta of +0.9e-5. Positive but "
                   "below the +2e-5 promotion threshold and an order of magnitude below the +5e-5 "
                   "fold-1 trigger. Notably, 3474 levels did NOT make this expensive in practice: "
                   "0.3710 s/round at the 400-round anchor and convergence at 1010 iterations, so "
                   "the high-cardinality cost concern was real in projection and false in fact.",
        "evidence": "iters 1010; logit corr versus C2REF 0.99910; corr versus v3 0.99899; 836 s.",
        "verdict": "NOT PROMOTED. The mechanistically appealing hypothesis, that route identity "
                   "deserves a native category alongside our manual target statistics, is not "
                   "supported at a usable magnitude. Branch closed without fold 1.",
    },
    {
        "phase": "p11r",
        "id": "p11r_CTR_COMPLEXITY_2_NEGATIVE",
        "kind": "EXPERIMENT",
        "claim": "ctr2 = C2 with max_ctr_complexity = 2 and nothing else changed, primary fold 0.",
        "finding": "ctr2 scored 0.961039 against C2REF at 0.961067, a delta of -2.7e-5. The "
                   "precondition was verified rather than assumed: the CTR audit of C4's dump shows "
                   "simple_ctr 0 and combinations_ctr 0, meaning the default build was simple-CTR "
                   "only, so pairwise CTRs were genuinely absent. The extension still lost.",
        "evidence": "iters 1462, the longest of any C2-family arm and 40 percent more than C2REF, "
                    "for a worse score; logit corr versus C2REF 0.99902; 1160 s versus 736 s.",
        "verdict": "NEGATIVE, and the cost is substantial. Branch closed. Do not sweep complexity "
                   "1/2/3/4.",
    },
    {
        "phase": "p11r",
        "id": "p11r_EXACT_COUNTERPARTS_RETAIN_DIVERSITY_BUT_MOVE_V3_BY_NOTHING",
        "kind": "EXPERIMENT",
        "claim": "Three genuinely distinct native counterparts, each EXACT on view, seed, depth, l2 "
                 "and fold scheme: ND6 for z3_cat_d6 (depth 6, seed 1), ND10 for z3_cat_d10 (depth "
                 "10, seed 2, lr 0.03), NCORE3 for z3_cat_core3 (core3 view, seed 4). Primary fold "
                 "0, Plain boosting, the validated 17 native categories.",
        "finding": "Question B is answered in the negative. Exact single-slot swaps on fold 0: "
                   "ND6 +0.24e-5, ND10 +0.02e-5, NCORE3 -0.19e-5. MINI_B100 at alpha = 1 gives "
                   "+0.07e-5 and MINI_B50 at alpha = 0.5 gives +0.04e-5. Alpha = 0 reconstructs "
                   "the control with max|dp| = 0.000e+00, so the block geometry is exact and these "
                   "deltas are measurements rather than artefacts of a mis-built blend.",
        "evidence": "DIVERSITY IS RETAINED, and that is the informative part. The native subset's "
                    "median pairwise logit correlation is 0.99779 against the original subset's "
                    "0.99748 (min 0.99771 versus 0.99697, max 0.99856 versus 0.99780). The native "
                    "counterparts did NOT collapse toward 1.0.",
        "verdict": "The hypothesis fails for a reason other than diversity collapse: the native "
                   "counterparts are diverse AND interchangeable in effect. Replacing three genuine "
                   "distinct slots with three exact native counterparts moves v3 by +0.07e-5 at "
                   "full weight, which is nothing. No fold 1, no seven-slot training.",
    },
    {
        "phase": "p11r",
        "id": "p11r_GATE_B_IS_UNINFORMATIVE_AT_N3",
        "kind": "METHODOLOGY",
        "claim": "The predeclared Stage-1 gate B reads: at least 2 of 3 single-slot swaps positive "
                 "AND native diversity retained.",
        "finding": "Gate B was MET (2 of 3 positive, diversity retained) but carries no evidential "
                   "weight. Under pure noise each swap delta is positive with probability 0.5, so "
                   "P(at least 2 of 3) = 0.500. At n = 3 that condition is close to the single most "
                   "likely outcome of NO signal at all, so passing it is not evidence of "
                   "complementarity. The two positives are +0.24e-5 and +0.02e-5, and one of those "
                   "is a tie-scale effect. The mean swap delta is +0.024e-5.",
        "evidence": "Computed by binomial enumeration rather than asserted. Gate A, which required "
                    "a mini-block gain of at least +1.0e-5, missed by 0.93e-5, roughly 14x short. "
                    "Gate C, which required a mechanism gain of at least +5e-5 over C2, was met by "
                    "zero arms.",
        "verdict": "The predeclared boolean is preserved EXACTLY as written in the report's gates "
                   "field. A separate gates_weighted field records that no gate carrying "
                   "evidential weight was met, and the substantive signal flag is False. Amending a "
                   "criterion after seeing the result is precisely what a predeclared gate exists "
                   "to prevent, so the criterion is annotated rather than rewritten. Fold 1 is NOT "
                   "run and roughly 2.5 hours of compute is not spent resolving a signal this size.",
    },
]


def main() -> int:
    p = Path("experiments/ledger.jsonl")
    existing = p.read_text(encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        for r in ROWS:
            if f'"{r["id"]}"' in existing:
                print("already present, skipping:", r["id"])
                continue
            f.write(json.dumps({"ts": NOW, **r}) + "\n")
    n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
    print(f"ledger lines now: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())