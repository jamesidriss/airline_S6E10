"""Append the Phase 12 ledger entries.

The findings are long and contain quotes and pipe characters, which a PowerShell here-string mangled
into a SyntaxError on the first attempt. Writing them from a file avoids re-quoting entirely.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()

ROWS = [
    {
        "phase": "p12",
        "id": "p12_LGBM_NATIVE_CATEGORICAL_AUDIT",
        "kind": "AUDIT",
        "claim": "Does LightGBM 4.7 honour native categorical splits inside the winning extra_trees "
                 "configuration, and is the treatment observable at all?",
        "finding": "Yes on both counts, and answering this was necessary rather than ceremonial. My "
                   "first audit detected categorical splits by counting the substring '||' and "
                   "reported ZERO for every configuration, including one where the declaration had "
                   "demonstrably worked. LightGBM's dump_model() renders both one-hot and subset "
                   "categorical splits as decision_type '==' with an integer threshold; only the "
                   "saved model TEXT file uses pipe-joined category lists. A wrong detector reported "
                   "a working treatment as inert, which would have closed Phase 12 before it "
                   "started, and would have been indistinguishable from a genuine negative.",
        "evidence": "With the corrected detector on a forced probe where the categorical is the ONLY "
                    "signal: undeclared gives 64 splits all '<='; declared gives 125 all '=='. "
                    "extra_trees=True and extra_trees=False give IDENTICAL categorical split counts "
                    "(125 and 125, same code set), so extra_trees randomises only the NUMERICAL "
                    "threshold search and the two mechanisms coexist -- which is what makes the test "
                    "possible inside the winning config. On the real 285-column view declaring META4 "
                    "produces 58 categorical splits against the control's 0, so the treatment is "
                    "observable. NaN in a declared categorical is routed as MISSING, proven from the "
                    "model: feature_infos carries LightGBM's internal missing marker -1, so a "
                    "missingness sentinel cannot become a real category code.",
        "verdict": "LightGBM 4.7 CPU build supports native categoricals by index and by name. "
                   "Audit re-run and committed as reports/lgb_native_cat_audit.json.",
    },
    {
        "phase": "p12",
        "id": "p12_TWO_FOLD_INDEX_COINCIDENCES_INVISIBLE_ON_FOLD_0",
        "kind": "CORRECTION",
        "claim": "Phase 12's first fold-0 screen reported L0 reproducing the stored champion to "
                 "+0.0001e-5, and was treated as a passing harness check.",
        "finding": "That check was passing for the wrong reason. TWO conventions in the champion's "
                   "runner are indexed by the fold, and fold 0 is exactly where my defaults "
                   "coincided with them. (a) run_xt_zoo.py:112 passes the zoo entry's single seed "
                   "and never adds k, so my seed+k matched on fold 0 (1+0 == 1) and diverged after "
                   "it. (b) run_xt_zoo.py:111 calls vb.assemble(..., inner_seed=k) -- the FOLD "
                   "INDEX seeds the inner cross-fit behind the fold-safe target encodings, so my "
                   "inner_seed=0 matched on fold 0 and diverged after it. Fixing (a) alone still "
                   "left folds 1 and 2 mismatched by +4.5e-5 and -6.6e-5, which is what exposed (b).",
        "evidence": "L0 vs stored xt_xt_d127_s1: fold 0 +0.0001e-5, fold 1 +4.4984e-5, fold 2 "
                    "-6.5733e-5 before the fix. After fixing both: +0.0001e-5, +0.0000e-5, "
                    "+0.0000e-5, +0.0000e-5, -0.0001e-5 across all five folds. So the harness now "
                    "IS the champion to within 1e-9 AUC.",
        "verdict": "The LESSON is the transferable part: any fold-indexed convention reproduces "
                   "correctly on fold 0 by default, so a fold-0-only screen cannot detect it. The "
                   "control-reproduction check therefore runs on EVERY fold, and it is now pinned by "
                   "tests asserting that run_xt_zoo passes a fixed seed and inner_seed=k, and that "
                   "no code line in the runner adds a fold index to either. All Phase 12 deltas are "
                   "from the post-fix harness; the earlier fold-0 screen is withdrawn.",
    },
    {
        "phase": "p12",
        "id": "p12_L2_META4_IRREGULAR_STANDALONE_BUT_NULL_MARGINAL",
        "kind": "EXPERIMENT",
        "claim": "L2 = champion extra_trees LightGBM + META4 and the three zero/N-A sensitive "
                 "surveys (Online boarding, Inflight wifi service, Gate location) declared as native "
                 "categorical features. Nothing else changed: same 285-column matrix, same 48 te_ "
                 "columns byte-identical, same seed, same folds, same ES policy.",
        "finding": "A REAL but small standalone effect that does not survive into the ensemble. "
                   "Standalone vs the matched champion: +5.5, +5.6, +6.5, -0.7, -7.5e-5, mean "
                   "+1.89e-5, SE 2.67, t +0.71, 3/5 positive. Marginal on v3 measured by splicing "
                   "the arm into the exact slot it replaces, using real vectors: -0.03, -0.02, "
                   "+0.15, +0.12, -0.04e-5, mean +0.04e-5, SE 0.04, t +0.88, 2/5 positive.",
        "evidence": "Logit correlation with the member it replaces is 0.99891-0.99919 and Spearman "
                    "0.9914-0.9933. 7,312 to 11,716 categorical splits are created across the folds, "
                    "so the treatment is emphatically not inert -- it simply produces a prediction "
                    "the blend already effectively has.",
        "verdict": "FAIL against the predeclared admission gate (+1.5e-5 marginal, >=4/5 folds, "
                   ">=2.5x paired SE): +0.04e-5 marginal is 37x short of the gate. Native LightGBM "
                   "categorical splits change the RANDOM-SPLIT BEHAVIOUR without adding information "
                   "the blend lacks. This is the direct answer to the checkpoint question, and it is "
                   "a mechanism-level negative rather than a null result: the extra_trees family "
                   "already randomises its split search, and the categorical subsets it finds are "
                   "smoothed target statistics of the same columns the 48 te_ features encode.",
    },
    {
        "phase": "p12",
        "id": "p12_L1_META4_NEGATIVE_OVER_FIVE_FOLDS",
        "kind": "EXPERIMENT",
        "claim": "L1 = champion extra_trees LightGBM + META4 only (Gender, Customer Type, Type of "
                 "Travel, Class) declared as native categorical features.",
        "finding": "NEGATIVE over five folds. Standalone: +6.0, -1.1, +5.6, +4.5, -25.4e-5, mean "
                   "-2.10e-5, SE 5.96, t -0.35, 3/5 positive. Marginal on v3: +0.01, -0.07, +0.10, "
                   "+0.02, +0.02e-5, mean +0.02e-5, SE 0.03, t +0.67, 4/5 positive.",
        "evidence": "Fold 4 selected 1,703 iterations against a typical 690-888, and scored -25.4e-5. "
                    "So the arm is not merely unhelpful, it is UNSTABLE: declaring four low-cardinality "
                    "columns categorical occasionally prevents early stopping from firing. Separately, "
                    "META4's 2-3 levels sit at or below LightGBM's max_cat_to_onehot default of 4, so "
                    "these columns receive one-hot rather than subset splits -- worth stating because "
                    "it changes what L1 actually tests, and it means L1 is not a subset of L2's test.",
        "verdict": "FAIL, and not rescued. No hyperparameter zoo. L3 (all 17 discrete) was gated on "
                   "L1 or L2 looking promising; L2's marginal was +0.04e-5, so L3 was NOT run and no "
                   "further arms were added. Branch closed.",
    },
    {
        "phase": "p12",
        "id": "p12_FOLD_0_2_SCREEN_OVERSTATED_THE_EFFECT_3X",
        "kind": "METHODOLOGY",
        "claim": "On folds 0-2 the predeclared promotion screen showed L2 at +5.5/+5.6/+6.5e-5, an "
                 "apparently clean and replicated-looking +5.87e-5.",
        "finding": "Completing the folds gave -0.7 and -7.5e-5 and a 5-fold mean of +1.89e-5. The "
                   "screen overstated the effect by 3.1x. The same thing happened to L1 in the "
                   "opposite direction: +3.47e-5 on folds 0-2 became -2.10e-5 over five folds.",
        "evidence": "Both statistics are printed side by side in reports/p12_final.json and in the "
                    "analysis output, so the gap is visible rather than forgotten.",
        "verdict": "Recorded because it is the campaign's second instance of the same failure mode "
                   "(Phase 11R's gate B: a criterion satisfied for a reason that carries no weight). "
                   "A three-fold screen on this competition is not sufficient to claim a standalone "
                   "effect, because the fold-level spread is several times the mean. Five folds are "
                   "the minimum for a standalone claim here.",
    },
    {
        "phase": "p12",
        "id": "p12_PUBLIC_NOTEBOOK_PULL_CORRUPTED_PROVENANCE",
        "kind": "CORRECTION",
        "claim": "scripts/pull_public_notebooks.py -- three kernels pulled into one shared directory "
                 "produced a single file.",
        "finding": "That single file contained a DIFFERENT notebook than the one already audited, "
                   "while still carrying the audited notebook's filename "
                   "s6e10-what-each-step-was-worth.ipynb. Had the audit been re-run, a notebook's "
                   "findings would have been attributed to the wrong author, and ledger entries "
                   "referencing that filename would have silently become wrong. This is the same "
                   "class as the earlier path-assumption bug -- trusting an output filename instead "
                   "of verifying identity -- but worse, because here the misfiled content was "
                   "referenced by existing ledger entries.",
        "evidence": "Each kernel now gets its own directory named from the ref, and identity is "
                    "cross-checked three ways: per-kernel directory, kernel-metadata.json id, and the "
                    "notebook slug in the filename. A mismatch is a hard failure. Verified: "
                    "sachith7/s6e10-what-each-step-was-worth 30 cells 29222 chars sha16=9fdae7c2e42ca6d526; "
                    "busyaprime/s6e10-route-or-distance-both-lb-0-96169 25 cells 46960 chars "
                    "sha16=4d3614bc2e42f414; "
                    "goodpjw2008/s6e10-auxiliary-task-features-lb-0-96129 34 cells 63975 chars "
                    "sha16=e55c9ef4b024fda8. The sachith7 hash and cell count match the original "
                    "audit exactly, so its recorded findings stand.",
        "verdict": "FIXED. All three notebooks are now individually identified with recorded hashes.",
    },
    {
        "phase": "p12",
        "id": "p12_PUBLIC_AUX_TASK_MECHANISM_IS_NEW_TO_US",
        "kind": "RESEARCH",
        "claim": "Which structural mechanisms have genuinely not been tested here?",
        "finding": "AUXILIARY-TASK EXPECTED-VALUE FEATURES. Two independent public sources converge "
                   "on them and we do not have them. goodpjw2008 trains 16 auxiliary models, each "
                   "predicting ONE rating from the other 20 columns with NO satisfaction label, and "
                   "uses the expected value sum_k k*P(k) of each rating as 13 new features. Measured "
                   "by that author on one XGBoost: 0.961078 -> 0.961208, +13e-5; on one LightGBM "
                   "0.96107 -> 0.96116, +9e-5; on the stack +5.9e-5 with all five folds up. "
                   "sachith7 independently uses 'aux expected ratings' computed once over train+test. "
                   "Our view has 48 te_ target encodings of satisfaction and 54 ogte_* and 2 "
                   "teach_* features, but NO per-rating auxiliary target: our teacher predicts the "
                   "satisfaction label from the original dataset, which is a different mechanism.",
        "evidence": "goodpjw2008 also reports a refinement that narrows the target: the SURPRISE "
                    "features built from those same auxiliary models do nothing, and only the plain "
                    "expected value helps. So the promising sub-mechanism is specific and small.",
        "verdict": "This is the single strongest remaining evidence-backed direction: two "
                   "independent sources, a mechanism absent from our 285-column view, a reported "
                   "effect (9-13e-5 standalone) an order of magnitude above anything Phase 11 or 12 "
                   "produced, and it is label-free so it can be fitted fold-safely with no new "
                   "infrastructure. Queued for Phase 13.",
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
        except Exception as exc:                                  # noqa: BLE001
            bad += 1
            print("line", i, "INVALID:", exc)
    print(f"ledger lines {n0} -> {n1}; invalid lines: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())