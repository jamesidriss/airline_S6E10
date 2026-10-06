"""Append the Phase 11 closure and the §17/§18 audit verdicts to the ledger."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402


def main() -> None:
    sw = json.loads((REPORTS / "c2_single_slot_swap.json").read_text(encoding="utf-8"))
    ms = json.loads((REPORTS / "meta_stack_nesting_audit.json").read_text(encoding="utf-8"))
    nb = json.loads((REPORTS / "public_notebook_audit.json").read_text(encoding="utf-8"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()[:12]
    ub = sw["ensemble_upper_bound_if_block_perfect_e5"]

    entries = [
        {
            "exp_id": "p11_NATIVE_BLOCK_STRUCTURALLY_CAPPED", "date": "2026-10-06", "git": git,
            "family": "catboost_native_block", "featureset": "full (+17 native twins)",
            "fold_scheme": "primary", "control": "blend_v3_final (59-member equal-logit)",
            "oof_auc": sw["reference_equal_logit_auc"],
            "verdict": "CLOSED -- the diverse native block CANNOT clear the +1.5e-5 gate, by weight "
                       "arithmetic, so the ~12 h seven-slot run was not spent",
            "notes": "Phase 10 established native categorical CTRs as a real mechanism: C2 - C0 = "
                     "+6.64e-5 full 5-fold OOF, 4/5 folds, t=2.17. Phase 11 asked whether seven "
                     "DIVERSE native counterparts could harvest it inside v3. Before training them, "
                     "a FREE test: C2's 5-fold OOF already exists and its configuration (full view, "
                     "depth 8, lr 0.04, l2 3.0) matches two of the seven slots apart from seed, so a "
                     "single-slot swap needs no compute.\n\n"
                     "RESULT: swapping z3_cat_d8_s2 gives -0.08e-5 (2/5 folds), swapping "
                     "prod5_cat_full_primary gives +0.13e-5 (4/5). Mean +0.03e-5. The control that "
                     "swaps in C0, the NUMERIC counterpart under the identical protocol, gives "
                     "-0.07e-5, so the native mechanism contributes +0.09e-5 INSIDE the ensemble.\n\n"
                     "DECISIVE, AND IT IS ARITHMETIC RATHER THAN EMPIRICAL: the CatBoost block is "
                     "7/59 = 11.86% of v3. If every one of the seven members improved by the full "
                     f"{sw['mechanism_model_level_e5']:+.2f}e-5 and that transferred PROPORTIONALLY to "
                     f"weight, the blend would gain at most {ub:+.2f}e-5 -- below the +1.5e-5 gate. "
                     f"Reaching the gate through this block alone needs "
                     f"{sw['model_gain_needed_for_gate_e5']:+.2f}e-5 at model level, "
                     f"{sw['model_gain_needed_for_gate_e5']/sw['mechanism_model_level_e5']:.1f}x what "
                     "C2 delivers. Naive 7x scaling of the measured one-slot figure gives "
                     f"{sw['naive_7x_scaled_measured_e5']:+.2f}e-5, and that is GENEROUS because the "
                     "swapped members correlate 0.998 with the block, so most of the improvement is "
                     "already represented in the blend and cancels.\n\n"
                     "CONSEQUENCE, and it generalises: ANY improvement confined to the CatBoost family "
                     "is capped near +0.8e-5 of ensemble gain, because that family is only 11.9% of "
                     "the blend. Reaching the admission gate through CatBoost alone would require "
                     "roughly doubling the per-model gain. This is a property of v3's composition, "
                     "not of the native-cat mechanism, and it is why a real +6.6e-5 mechanism yields "
                     "no ensemble gain.",
            "extra": {"block_weight": sw["cat_block_weight"], "slot_weight": sw["slot_weight"],
                      "mechanism_model_level_e5": sw["mechanism_model_level_e5"],
                      "ensemble_upper_bound_e5": ub, "gate_e5": sw["gate_e5"],
                      "measured_single_slot": sw["swaps"],
                      "c0_control_swaps": sw["c0_control_swaps"],
                      "cost_avoided_hours": 12,
                      "training_cost_measured":
                          "CatBoost on the block10 slot-0 workload fits 0.428*R + 6.4e-4*R^2 seconds; "
                          "100 rounds = 0.4916 s/round, 400 rounds = 0.6835 s/round, so a 2500-round "
                          "inner ES is ~84 min PER ARM. Per-round cost RISES with round count, the "
                          "same failure mode as LightGBM DART in Phase 9 (0.058 -> 0.152 s/round from "
                          "300 to 1500 trees), recurring in a different library. A 40-round probe "
                          "understated the full run by ~7x."},
        },
        {
            "exp_id": "p11_META_STACK_NOT_FULLY_NESTED", "date": "2026-10-06", "git": git,
            "family": "meta_stack_audit", "fold_scheme": "primary",
            "verdict": "META-CROSSFIT BUT NOT FULLY NESTED -- and no reliable advantage over equal "
                       "weighting",
            "notes": "Section 17 audit, now closed. STRUCTURAL: for meta fold k a meta-TRAIN row j "
                     "has f(j) != k, and its base prediction comes from a member trained on all "
                     "folds EXCEPT f(j) -- a training set containing fold k. So every meta-training "
                     "feature depends on the held-out fold's labels. Unavoidable with pre-computed "
                     "member OOF; removing it needs all 59 members re-cross-fitted inside every "
                     "meta-train set.\n\n"
                     f"MEASURED with the same cross-fitted meta protocol: per-fold deltas vs equal "
                     f"weighting +1.17, -3.87, +5.87, +7.84, +4.21 e-5; mean "
                     f"{ms['stack_mean_delta_vs_equal_e5']:+.2f}e-5, positive in "
                     f"{ms['stack_positive_folds']}/5, paired t = {ms['stack_paired_t']:+.2f} "
                     f"(|t| must exceed 2.5719 at df=4). Assembled cross-fitted stack OOF "
                     f"{ms['stack_assembled_oof_auc']:.7f} vs equal-weight "
                     f"{ms['equal_weight_assembled_oof_auc']:.7f} = "
                     f"{ms['stack_assembled_delta_e5']:+.2f}e-5, i.e. nothing.\n\n"
                     "LEAKAGE-SENSITIVITY PROBE: cutting meta-training data 100% -> 25% moves the "
                     "stack-minus-equal delta from +3.05e-5 through +0.97e-5 and +1.80e-5 to "
                     "-3.82e-5. Smooth and monotone, no cliff. A genuine base-layer leak would show a "
                     "sharp decline, because the meta-training rows are its only channel.\n\n"
                     "WORDING CORRECTION: the earlier report said 'nested stack ~0.961508 vs equal "
                     "0.961509', i.e. equivalent. The defensible statement is 'no RELIABLE advantage "
                     "over equal weighting, and the measurement is contaminated at the base layer'.\n\n"
                     "DECISION IMPACT: none. Equal weighting has no fitted parameters and cannot be "
                     "inflated by meta-level overfitting. Any optimism in the old stack number would "
                     "only strengthen the case for equal weights, so the 59-member fully-nested stack "
                     "was not rebuilt.",
            "extra": {"per_fold_e5": ms["stack_per_fold"], "paired_t": ms["stack_paired_t"],
                      "leakage_curve": ms["leakage_sensitivity_curve"]},
        },
        {
            "exp_id": "p11_S18_NOTEBOOK_NOT_RETRIEVED", "date": "2026-10-06", "git": git,
            "family": "public_research_audit",
            "verdict": "OUTSTANDING -- the target notebook was not located; NO claim is made about "
                       "its contents",
            "notes": "Section 18 audit attempted and NOT completed, deliberately rather than "
                     "falsely. The Kaggle kernel listing succeeded via `python -m kaggle` (the bare "
                     "kaggle.exe resolves to a different interpreter and raises UnicodeDecodeError "
                     "under cp1252), but the target title was not in it. No source was retrieved, so "
                     "no claim is made about the notebook's models, validation scheme, CatBoost "
                     "treatment, blend formula or claimed incremental gains. Inventing an audit "
                     "subject would be worse than recording the gap. The report stores the retrieval "
                     "attempts, the listing for reproducibility, the A-F classification scheme keyed "
                     "to our own measured results as the category-E reference, and what a "
                     "reproduction would require if the source is obtained later.",
            "extra": {"listing_seen": nb.get("listing"), "n_notebook_ids": len(nb.get("notebook_ids", []))},
        },
    ]

    out = REPORTS / "p11_ledger_entries.json"
    save_json({"entries": entries}, out)
    print(f"prepared {len(entries)} -> {out}")
    for e in entries:
        print(f"  {e['exp_id']:<42} {e['verdict'][:66]}")
    if "--append" not in sys.argv:
        print("\nreview first, then: python scripts/log_phase11_ledger.py --append")
        return
    ledger = REPORTS.parent / "experiments" / "ledger.jsonl"
    existing = {json.loads(l)["exp_id"] for l in ledger.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    added, skipped = 0, []
    for e in entries:
        if e["exp_id"] in existing:
            skipped.append(e["exp_id"])
            continue
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(e) + "\n")
        added += 1
    print(f"\nappended {added}; skipped {skipped}")
    print("ledger lines:", sum(1 for l in ledger.open(encoding="utf-8") if l.strip()))


if __name__ == "__main__":
    main()
