"""Append the Phase 14 ledger entries, including two findings that reverse earlier conclusions."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()

ROWS = [
    {
        "phase": "p14",
        "id": "p14_AUX_HELP_XGBOOST_AND_CATBOOST_TOO",
        "kind": "EXPERIMENT",
        "claim": "Do the VALIDATED Phase 13 auxiliary rating features help XGBoost and CatBoost as "
                 "well as extra_trees LightGBM? The features are reused EXACTLY from the cached "
                 "per-fold arrays, so the only changed variable per arm is 13 appended columns.",
        "finding": "Yes, and the per-slot ENSEMBLE marginal is as large as extra_trees'. Mean "
                   "standalone gain over five primary folds: X0 +9.8e-5, X1 +8.7e-5, X2 +10.2e-5, "
                   "C0 +10.3e-5, C1 +8.2e-5 -- the same order as extra_trees' +12.25e-5. Per-slot "
                   "marginal on v3: X0 +0.286e-5 (5/5 folds), X2 +0.332e-5 (4/5), X1 +0.228e-5 "
                   "(4/5), C1 +0.203e-5 (5/5), C0 +0.106e-5 (3/5).",
        "evidence": "Every arm was run as a matched ctrl/aux pair in ONE harness, and the reported "
                    "delta is aux minus that control, not minus the stored member. That distinction "
                    "mattered: X1's and X2's stored members came from different runners with "
                    "reproduction gaps of +1.1e-5 and -5.0e-5, which would have been charged to the "
                    "treatment. X0's control reproduces its stored member to +0.0e-5, so the harness "
                    "is the champion for this family.",
        "verdict": "The mechanism is family-general. Its per-slot VALUE is also higher outside "
                   "extra_trees: 0.27e-5 per XGB slot against 0.12e-5 per extra_trees slot, which "
                   "is the cross-family prediction and is what Phase 14C then confirmed.",
    },
    {
        "phase": "p14",
        "id": "p14_AUX_AND_NATIVE_CAT_DO_NOT_COMPOUND",
        "kind": "EXPERIMENT",
        "claim": "Does the auxiliary-task mechanism compound with CatBoost's native categorical "
                 "CTRs? Phase 10B established native CTRs at +6.6e-5 at model level, so CatBoost "
                 "gives a clean 2x2.",
        "finding": "No. Fold 0, matched ctrl/aux pairs: numeric base 0.960841, native-cat base "
                   "0.960945 (so native-cat is +10.4e-5, consistent with Phase 10B's direction), "
                   "numeric+aux 0.961055 (aux on numeric = +21.4e-5), native-cat+aux 0.961003 "
                   "(aux on native-cat = +5.8e-5). native+aux does NOT exceed numeric+aux.",
        "evidence": "The auxiliary effect on native-cat CatBoost is 3.7x SMALLER than on numeric "
                    "CatBoost. The two mechanisms are substitutes, not complements.",
        "verdict": "NO SYNERGY CLAIMED. The instruction was explicit that invented synergy is worse "
                   "than none, and this campaign already carries two withdrawn claims built on "
                   "inferring a mechanism's blend value rather than measuring it. Reported as a "
                   "clean negative.",
    },
    {
        "phase": "p14c",
        "id": "p14c_AUX_CROSS_IS_THE_FIRST_REPLICATED_MULTI_FAMILY_ENSEMBLE_GAIN",
        "kind": "EXPERIMENT",
        "claim": "Replace 10 of v3's 59 slots with aux-equipped counterparts -- 6 extra_trees "
                 "LightGBM, 3 XGBoost, 1 CatBoost -- keeping the member count at 59 and every slot "
                 "weight at exactly 1/59. Slots selected by a STRUCTURAL rule (every slot with a "
                 "trained counterpart), never by measured gain.",
        "finding": "OOF 0.961523, +1.491e-5 versus v3, 5/5 folds positive, paired SE 0.376e-5, "
                   "t +3.97. Per fold: +0.92, +2.80, +1.54, +0.64, +1.78e-5. The decomposition is "
                   "monotone and the cross-family prediction holds: cat1 +0.15, xt6 +0.72, "
                   "xgb3 +0.80, xgb3+cat1 +0.93, all 10 slots +1.49e-5, every one 5/5 positive. "
                   "Per slot that is 0.12e-5 for extra_trees and 0.27e-5 for XGBoost. Rescue rate "
                   "rises with the block (0.0025 -> 0.0103) while damage stays at 0.0001-0.0003.",
        "verdict": "MISSES the predeclared +1.5e-5 admission gate by 0.6 percent. The gate is NOT "
                   "moved. This is nonetheless the cleanest ensemble result of the campaign -- 5/5 "
                   "folds, t +3.97, monotone in block size -- and it is submitted as S1 purely as a "
                   "transfer sanity check, with the public movement recorded as UNKNOWN.",
    },
    {
        "phase": "p14c",
        "id": "p14c_PER_FOLD_TABLE_WAS_FABRICATED_BY_MIXING_FULL_OOF_AND_PER_FOLD_AUC",
        "kind": "CORRECTION",
        "claim": "The first AUX_CROSS report printed a per-fold profile of +2.28, -12.66, +1.14, "
                 "+72.97, -59.79e-5, with paired SE 21.3e-5 and t +0.07, alongside the correct "
                 "aggregate +1.491e-5.",
        "finding": "The per-fold numbers were meaningless. The code computed each fold's delta as "
                   "`a - auc(y[fold], base[fold])`, where `a` is the FULL-OOF AUC. Subtracting a "
                   "full-OOF quantity from a per-fold one produced, for the CONTROL whose per-fold "
                   "delta must be identically zero, the values +0.79, -14.15, -0.35, +71.48, "
                   "-61.28e-5 -- which is simply the full-OOF AUC minus each fold's own control AUC. "
                   "The tell was that the control was non-zero.",
        "evidence": "The aggregate delta_e5 was NOT affected, because both of its terms are "
                    "full-OOF quantities, so +1.491e-5 stands and is confirmed by the corrected run. "
                    "Every per-fold claim, INCLUDING the fold-consistency count that gates "
                    "admission, was fabricated: the corrected profile is +0.92, +2.80, +1.54, "
                    "+0.64, +1.78e-5, all positive, t +3.97 rather than +0.07.",
        "verdict": "FIXED, and the corrected control now carries an assertion that its per-fold "
                   "delta is exactly zero. A reporting bug that makes the control look non-zero is "
                   "precisely what should stop a run, and this is the fifth unfalsifiable-check "
                   "instance in this campaign.",
    },
    {
        "phase": "hedge",
        "id": "hedge_CORRELATIONS_WERE_COMPUTED_ON_A_SATURATED_SIGN_FUNCTION",
        "kind": "CORRECTION",
        "claim": "The hedge report stated that H1_family_diverse correlates 0.96582 with v3 and "
                 "H0_nonXT 0.96585, and that no reweighting beat v3 by much.",
        "finding": "Those correlations were a sign-agreement statistic, not a correlation. `base` is "
                   "the MEAN LOGIT of the 59 members, not a probability, and the module's lg() clips "
                   "to [1e-6, 1-1e-6] before log-ratios -- correct for a probability and "
                   "catastrophic for a mean logit. 97.87 percent of base values fell outside that "
                   "window and saturated to +13.8 or -13.8.",
        "evidence": "True correlations against the stored v3 probabilities: H1 0.99993, H0 0.99986, "
                    "and every individual family mean between 0.99725 (tabm) and 0.99976 "
                    "(lgbm_xt). The runner now asserts that base still looks like a mean logit, so "
                    "the saturation cannot silently return.",
        "verdict": "THE HEDGE CONCLUSION IS REVERSED, and it reverses in the pessimistic direction. "
                   "NO REWEIGHTING OF v3'S MEMBERS IS A PRIVATE HEDGE. Every candidate is at least "
                   "0.997 correlated with v3, because the 59-member equal-logit average has already "
                   "averaged away the member-level diversity that a hedge would have to exploit. "
                   "Reweighting can only rearrange what is left, and what is left is one signal.",
    },
    {
        "phase": "hedge",
        "id": "hedge_TEST_SIDE_DIVERSITY_COLLAPSES_BECAUSE_MEMBERS_SHARE_TRAINING_DATA",
        "kind": "EXPERIMENT",
        "claim": "Is an OOF-diverse hedge still diverse in its TEST predictions? This matters "
                 "because the hedge's value is realised on the 80 percent private split.",
        "finding": "Test-side diversity is systematically lower than OOF diversity for every "
                   "candidate, and for the family-diverse blend the gap is small but the absolute "
                   "level is already a clone. OOF versus TEST correlation with v3: H1 0.99993 "
                   "versus 0.99995, H0 0.99986 versus 0.99989, realmlp family 0.99900 versus "
                   "0.99920, tabm family 0.99725 versus 0.99798.",
        "evidence": "The mechanism is structural, not incidental: every TEST-time member is refit on "
                   "the SAME 100 percent of labels, whereas each OOF member saw a different 80 "
                   "percent subset. Models that differ on disjoint subsets converge when trained on "
                   "identical data. So OOF understates how similar the members are at scoring time, "
                   "and any diversity measured OOF is an over-estimate of what reaches the leaderboard.",
        "verdict": "This is a general warning about the whole private-hedge strategy as specified. A "
                   "genuine hedge cannot be built by reweighting existing members at all. It would "
                   "require a structurally different model trained under a different protocol, which "
                   "the store does not contain: the most independent artifact available, a tabm-only "
                   "prediction, still sits at 0.99798 and at OOF 0.960797, which is 71e-5 below v3 and "
                   "fails the stated eligibility bar of v3 minus 30e-5.",
    },
]


def main() -> int:
    p = Path("experiments/ledger.jsonl")
    existing = p.read_text(encoding="utf-8")
    n0 = sum(1 for line in existing.splitlines() if line.strip())
    with p.open("a", encoding="utf-8") as f:
        for r in ROWS:
            if f'"{r["id"]}"' in existing:
                print("already present, skipping:", r["id"])
                continue
            f.write(json.dumps({"ts": NOW, **r}) + "\n")
    n1 = sum(1 for line in p.open(encoding="utf-8") if line.strip())
    bad = 0
    for i, line in enumerate(p.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception as exc:
            bad += 1
            print("line", i, "INVALID:", exc)
    print(f"ledger lines {n0} -> {n1}; invalid: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())