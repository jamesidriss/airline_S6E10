"""Append the Phase 13 ledger entries."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()

ROWS = [
    {
        "phase": "p13",
        "id": "p13_AUX_TASK_EXPECTED_VALUE_IS_REAL_AND_THE_LARGEST_STANDALONE_EFFECT_MEASURED",
        "kind": "EXPERIMENT",
        "claim": "AUXILIARY-TASK EXPECTED-VALUE FEATURES on the champion extra_trees configuration. "
                 "13 auxiliary models, each predicting ONE rating from the OTHER columns with no "
                 "satisfaction label anywhere; the feature is the expected value sum_k k*P(k) of "
                 "that rating. Everything else identical to xt_xt_d127_s1.",
        "finding": "The largest standalone effect measured anywhere in this campaign, and the first "
                   "consistently positive ensemble marginal. Standalone vs the matched champion over "
                   "all five primary folds: +21.8, +16.4, +17.3, +3.9, +1.9e-5, mean +12.25e-5, "
                   "SE 3.94, t +3.11, 5/5 positive. Phases 11 and 12 produced at most +6e-5. "
                   "MARGINAL on v3, measured by splicing into the exact slot it replaces using real "
                   "vectors: +0.31, +0.25, +0.48, +0.14, +0.15e-5, mean +0.27e-5, SE 0.06, "
                   "t +4.24, 5/5 positive.",
        "evidence": "The champion spends 10.0-10.2 percent of its splits on the 13 aux columns, so "
                    "the features are genuinely used and not inert. The auxiliary models carry real "
                    "structure: mean out-of-fold accuracy 0.6998, ranging from 0.313 (Checkin "
                    "service, which the other columns barely constrain) to 0.954 (Inflight "
                    "entertainment). Each aux feature correlates 0.81-0.97 with the rating it "
                    "smooths with RMSE 0.33-0.75, i.e. they are genuine smoothers rather than "
                    "duplicates, and none reaches correlation 1.0, which confirms the cross-fitting "
                    "is doing its job.",
        "verdict": "Mechanism VALIDATED as real. Not ADMITTED: +0.27e-5 marginal is 5.6x short of "
                   "the +1.5e-5 gate, even though it passes the other two conditions (5/5 positive, "
                   "and mean 4.5x the paired SE). The other two conditions being met while the "
                   "effect size misses is exactly the pattern Phase 11R's gate B showed, and it is "
                   "why the effect size is the binding criterion. Phase 13B tests whether the gain "
                   "compounds across the six extra_trees slots in v3.",
    },
    {
        "phase": "p13",
        "id": "p13_EXTRA_TREES_STANDALONE_AUC_IS_CHAOTIC_UNDER_TINY_FEATURE_PERTURBATIONS",
        "kind": "METHODOLOGY",
        "claim": "A sensitivity probe reran the auxiliary build on fold 0 at 400 rounds / 5 inner "
                 "folds instead of 250 / 3, to check the result was not an artifact of the cheaper "
                 "settings used for the 5-fold run.",
        "finding": "The auxiliary models were IDENTICAL -- mean out-of-fold accuracy 0.6998 both "
                   "times, matching to three decimals for all 13 ratings -- yet the champion's "
                   "standalone AUC moved from 0.961320 (+21.76e-5) to 0.961047 (-5.55e-5). A 27e-5 "
                   "swing from features that were not meaningfully different. The features were "
                   "stable; the model was not.",
        "evidence": "extra_trees uses randomised thresholds and per-node feature subsampling, which "
                    "makes the fit chaotic under perturbations far below the level that changes any "
                    "feature's meaning. So a single fold's standalone delta for this family is NOT "
                    "reproducible, and the campaign's habit of reading standalone deltas is unsafe "
                    "for extra_trees members specifically.",
        "verdict": "Two consequences, both binding on how earlier results are read. (1) The Phase 13 "
                   "fold-0 standalone figure is NOT reproducible and the 5-fold standalone mean of "
                   "+12.25e-5 should be read as directionally supported but not precise; the 5/5 "
                   "sign consistency and t +3.11 survive the caveat because they are computed across "
                   "folds rather than within one. (2) The MARGINAL is unaffected -- +0.31e-5 and "
                   "+0.24e-5 at the two settings -- because pair ranking is robust to tree-level "
                   "chaos in a way a single model's AUC is not. The measured ensemble marginal is "
                   "therefore the trustworthy quantity for this family, which retrospectively "
                   "justifies Phase 12 having measured it rather than inferring it.",
    },
    {
        "phase": "p13",
        "id": "p13_LABEL_FREE_CHECK_THAT_COULD_NOT_FAIL_REPLACED_WITH_A_STRUCTURAL_GUARANTEE",
        "kind": "CORRECTION",
        "claim": "The first version of the runner asserted label-freeness by flipping every "
                 "satisfaction label in the frame and checking the aux features were unchanged.",
        "finding": "That check can never fail. build_aux receives only the view matrix, which is "
                   "constructed from input columns and contains no label, so relabelling the frame "
                   "cannot reach it. A check that cannot fail is worse than no check, because it "
                   "buys the appearance of verification. This is the third instance in this campaign "
                   "of the same defect, after the Phase 10A replication statistic that returned "
                   "r=1.000 for all 42 keys and the audit's '||' substring detector that reported a "
                   "working treatment as inert.",
        "verdict": "REPLACED with a structural guarantee that can actually fail: build_aux's "
                   "signature is asserted to contain no label argument and its body to reference "
                   "no label, so adding one later trips the runner. The label-free property is now "
                   "established by construction rather than by a vacuous probe, and tests/"
                   "test_phase13_aux.py pins it.",
    },
    {
        "phase": "p13",
        "id": "p13_AUX_FEATURES_WERE_MEMORISED_ON_FIT_ROWS_UNTIL_CROSS_FITTED",
        "kind": "CORRECTION",
        "claim": "The first implementation fitted the auxiliary models on the outer fit rows and "
                 "then asked them to predict those same fit rows.",
        "finding": "A model fitted on the fit rows and queried on those rows learns their ratings "
                   "almost perfectly, so aux_ev_j became a near-duplicate of rating_j during training "
                   "while remaining a genuinely smoothed estimate at validation time. That is a "
                   "train/serve mismatch: the champion calibrates its trust in aux_ev_j against the "
                   "memorised version and meets the smoothed version only at scoring time. Nothing "
                   "in the AUC alone would reveal it.",
        "evidence": "Fix: fit rows receive INNER CROSS-FITTED predictions and validation rows "
                    "receive predictions from a model fitted on all fit rows, so both sides are "
                    "out-of-sample. tests/test_phase13_aux.py demonstrates the gap is real on "
                    "structured data -- in-sample accuracy strictly exceeds cross-fitted -- and the "
                    "measurement corroborates the fix, because every aux feature lands at "
                    "correlation 0.81-0.97 with its rating rather than 1.0.",
        "verdict": "FIXED and pinned by test. The demonstration itself was wrong twice first: the "
                   "target was initially appended to the design matrix, giving 1.000 in-sample AND "
                   "1.000 cross-fitted (proving nothing while appearing to confirm the gap), and "
                   "then, with the target removed, above-chance accuracy was demanded on PURE NOISE "
                   "where the correct answer IS the 1/6 chance rate.",
    },
    {
        "phase": "p13",
        "id": "p13_CROSS_FOLD_AUX_DISPOSITION_STRICTER_THAN_NEEDED_BUT_DEFENSIBLE",
        "kind": "METHODOLOGY",
        "claim": "Auxiliary models are fitted on the OUTER FIT ROWS ONLY, so validation rows are "
                 "unseen by the model producing their features.",
        "finding": "The cheap alternative, fitting the auxiliary models once on train+test, uses no "
                   "competition label and so is fold-safe in the letter of the rule. It was rejected "
                   "anyway on a stricter ground: a validation row's own rating_j would have been in "
                   "its own auxiliary model's training set, which makes the OOF estimate optimistic "
                   "in a way that real test inference would not be. Cost is 13 fits per fold instead "
                   "of 13 total, roughly 7.5 minutes per fold.",
        "verdict": "KEPT. The public sources fit on all train+test rows; being stricter than the "
                   "sources makes a negative result here more informative, because it cannot be "
                   "explained away by their more permissive transductive choice.",
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