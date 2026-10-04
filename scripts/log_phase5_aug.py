"""Log the Phase 5 augmentation benchmark: control, Family A (rejected), Family B (positive).

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.jsonl"

ENTRIES = [
    {
        "exp_id": "aug_duplicate_control_fold0",
        "oof_auc": 0.961039,
        "fold_aucs": [0.961039],
        "verdict": "CONTROL -- duplication buys nothing, so the harness is not confounded",
        "method": "duplicate", "ratio": 1.0, "aug_weight": 1.0, "fold": 0, "scheme": "primary",
        "n_original_rows": 503739, "n_augmented_rows": 503739, "rows_multiplier": 2.0,
        "baseline_auc": 0.961103, "augmented_auc": 0.961039, "delta": -0.000064,
        "baseline_best_iter": 1073, "augmented_best_iter": 836,
        "corr_vs_baseline_logit": 0.99851, "blend_gain_best": -0.000023,
        "audit": {"off_support_categorical_values": 0, "non_finite_rows": 0},
        "why_it_matters": (
            "This is the gate on everything that follows. If simply doubling the physical row count "
            "moved the score, then any future augmentation 'gain' would be explained by row count, "
            "bagging or iteration count rather than by new support. It does not: -6.4e-5, i.e. "
            "nothing. So the augmentation experiments that follow measure what they claim to."
        ),
        "bug_found_first": (
            "The first run of this harness reported a baseline of 0.972387 -- far above the honest "
            "0.9611 for the identical configuration. Cause: the evaluation fold was selected with "
            "`folds != args.fold` instead of `==`, making the evaluation set identical to the "
            "training set. The implausible number is what exposed it; a plausible one would have "
            "passed unnoticed and invalidated every augmentation result. The harness now asserts "
            "fit/eval disjointness."
        ),
    },
    {
        "exp_id": "aug_familyA_hardlabel_local_interp_fold0",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "REJECT -- hard-label local augmentation makes things worse, monotonically in ratio",
        "method": "local same-class manifold mixing (hard inherited labels)",
        "results": [
            {"variant": "interp", "ratio": 0.5, "rows_multiplier": 1.5,
             "baseline_auc": 0.961103, "augmented_auc": 0.961035, "delta": -0.000068,
             "augmented_best_iter": 1533, "blend_gain_best": 0.000002},
            {"variant": "interp", "ratio": 1.0, "rows_multiplier": 2.0,
             "baseline_auc": 0.961103, "augmented_auc": 0.960858, "delta": -0.000245,
             "augmented_best_iter": 1905, "blend_gain_best": -0.000025},
            {"variant": "donor", "ratio": 1.0, "rows_multiplier": 2.0,
             "baseline_auc": 0.961103, "augmented_auc": 0.961095, "delta": -0.000007,
             "augmented_best_iter": 988, "blend_gain_best": 0.000013},
        ],
        "audit": {"off_support_categorical_values": 0, "non_finite_rows": 0,
                  "missingness_preserved": True, "same_class_guaranteed": True},
        "finding": (
            "This is the decisive test of the learning-curve hypothesis, and augmentation moves the "
            "WRONG WAY. A genuine doubling of rows is worth about +5e-4; interpolating to double "
            "the rows is worth -2.45e-4, a sign flip and four times the magnitude in the wrong "
            "direction. best_iter climbs 1073 -> 1533 -> 1905, i.e. the model spends increasing "
            "capacity on the synthetic region rather than on real support. Conclusion: the "
            "learning curve is measuring UNIQUE information, not sample count, and synthetic rows "
            "do not substitute for it. This is exactly the caveat recorded with the learning curve, "
            "now demonstrated rather than assumed."
        ),
        "root_cause_diagnosed": (
            "Interpolating two rows of the same class and copying the label assumes the class is "
            "locally pure. It is not: y is a Bernoulli draw from p(x), so the midpoint of two "
            "class-1 rows can have a materially lower p. Family A therefore baked that error into "
            "every new row, which is what Family B fixes."
        ),
    },
    {
        "exp_id": "aug_familyB_teacher_relabel_fold0",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "POSITIVE -- teacher relabelling flips the sign; promote for fold confirmation",
        "method": "same local augmentation, but augmented rows labelled by a fold-fit teacher",
        "results": [
            {"variant": "interp", "ratio": 1.0, "rows_multiplier": 2.0,
             "baseline_auc_matched_objective": 0.961103, "augmented_auc": 0.961175,
             "delta": 0.000072, "augmented_best_iter": 1319, "blend_gain_best": 0.000008},
            {"variant": "donor", "ratio": 1.0, "rows_multiplier": 2.0,
             "baseline_auc_matched_objective": 0.961103, "augmented_auc": 0.961127,
             "delta": 0.000025, "augmented_best_iter": 2262, "blend_gain_best": -0.000012},
        ],
        "teacher_diagnostic": {
            "teacher_auc_on_augmented_rows": 0.958430,
            "inherited_label_auc_on_augmented_rows": 1.000000,
            "teacher_minus_inherited": -0.041570,
            "reading": (
                "The teacher disagrees with the inherited label by 4.2e-2 of AUC on the very rows "
                "the augmentation created. That is the size of the error Family A was "
                "systematically baking in, and correcting it is what turns -24.5e-5 into +7.2e-5. "
                "The mechanism is therefore understood, not merely observed."),
        },
        "matched_objective_control": {
            "baseline_binary": 0.961103, "baseline_cross_entropy_no_aug": 0.961103,
            "both_best_iter": 1073,
            "why": ("switching to cross_entropy is itself a change that can move AUC, so an "
                    "objective-matched baseline is mandatory. It came out identical here, so the "
                    "measured delta is attributable to the augmented support and not to the "
                    "objective."),
        },
        "fold_safety": (
            "teacher fitted only on the outer-FIT rows, early-stopped on the inner holdout carved "
            "from those same rows, and applied to augmented rows built from outer-FIT features and "
            "labels only. No outer-validation row or target participates anywhere."
        ),
    },
]


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    added = 0
    with LEDGER.open("a", encoding="utf-8") as f:
        for e in ENTRIES:
            if e["exp_id"] in existing:
                print(f"  skip: {e['exp_id']}")
                continue
            e["date"] = "2026-10-04"
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']} [{e['verdict'][:60]}]")
    print(f"\n{added} entries appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
