"""Final Phase 5 record: augmentation rejected, with the mechanism isolated.

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.jsonl"

ENTRY = {
    "exp_id": "aug_familyB_teacher_relabel_donor_fold1",
    "oof_auc": 0.961380,
    "fold_aucs": [0.961380],
    "verdict": "REJECT -- on-manifold variant is flat, confirming the geometry diagnosis",
    "date": "2026-10-05",
    "method": "donor-mode local augmentation (all values inherited from ONE parent, so every "
              "synthetic row lies exactly on the observed support) with teacher labels",
    "fold": 1, "scheme": "primary", "rows_multiplier": 2.0,
    "baseline_auc_matched_objective": 0.961381, "augmented_auc": 0.961380,
    "delta": -0.000001,
    "baseline_best_iter": 920, "augmented_best_iter": 1245,
    "corr_vs_baseline_logit": 0.99928, "corr_vs_finalist_logit": 0.99925,
    "blend_gain_best": -0.000014,
    "teacher_diagnostic": {"teacher_auc_on_augmented_rows": 0.973690,
                           "inherited_label_auc_on_augmented_rows": 1.0,
                           "teacher_minus_inherited": -0.026310},
}


def append(e):
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    if e["exp_id"] in existing:
        print(f"  skip: {e['exp_id']}")
        return 0
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e) + "\n")
    print(f"  appended: {e['exp_id']}")
    return 1


CONCLUSION = {
    "exp_id": "phase5_augmentation_VERDICT",
    "oof_auc": None,
    "fold_aucs": [],
    "date": "2026-10-05",
    "verdict": ("AUGMENTATION REJECTED as a source of effective training signal. The learning "
                "curve's gain is real but is NOT reachable by manufacturing rows."),
    "all_arms_fold0_and_fold1": {
        "duplicate_control": {"f0": -0.000064,
                              "meaning": "row count alone buys nothing; harness not confounded"},
        "interp_hard_label": {"f0": -0.000245,
                              "meaning": "off-manifold rows actively harmful, worse with more rows"},
        "donor_hard_label": {"f0": -0.000007, "f1": None,
                             "meaning": "on-manifold rows are merely neutral"},
        "interp_teacher_label": {"f0": 0.000072, "f1": -0.000052, "mean": 0.000010,
                                 "meaning": "fixing labels recovers most of the harm, but signs flip"},
        "donor_teacher_label": {"f0": 0.000025, "f1": -0.000001, "mean": 0.000012,
                                "meaning": "on-manifold AND correctly labelled: flat, no gain"},
    },
    "mechanism_isolated": (
        "The two axes separate cleanly and sum to the observed behaviour. Geometry: hard-labelled "
        "interpolation costs -24.5e-5 while hard-labelled donor mixing costs only -0.7e-5, so being "
        "off-manifold accounts for roughly 24e-5 of the damage. Labels: replacing the inherited "
        "label with the teacher's probability recovers about 32e-5 of that on interpolation "
        "(-24.5e-5 -> +7.2e-5). What remains after both corrections is +1.0e-5 and +1.2e-5 "
        "respectively -- indistinguishable from zero, and with opposite signs across folds."),
    "teacher_diagnostic_replicates": (
        "The mechanism is confirmed even though the payoff is not. On the augmented rows the teacher "
        "scores 0.958430 (fold 0) and 0.958910 (fold 1) for interpolation, and 0.975380 / 0.973690 "
        "for donor mode, against 1.000000 for the inherited label. So hard labelling really was "
        "baking a 2.6e-2 to 4.2e-2 error into every synthetic row, and fixing it is worth a large "
        "amount of recovery -- just not a net gain."),
    "central_finding": (
        "The corrected learning curve says a genuine doubling of unique in-distribution rows is "
        "worth about +5e-4 to +7e-4, with AUC ~ a + b*n^(-1/5) at R^2 = 0.99894. Every attempt to "
        "manufacture that support failed, and the duplicate control proves the failures are not a "
        "row-count artefact. The limit is UNIQUE INFORMATION, not sample count. Synthetic rows add "
        "points the model must then spend capacity modelling, which is visible in best_iter rising "
        "1073 -> 1533 -> 1905 as the interpolation ratio increases."),
    "what_this_rules_out": (
        "Data augmentation of any form that does not introduce genuinely new independent labels: "
        "interpolation, donor mixing, duplication, and teacher-relabelled versions of the same. It "
        "also rules out the reading that the plateau is a data-quantity problem that more rows of "
        "any kind would solve."),
    "what_it_leaves_open": (
        "The learning curve measures UNIQUE LABEL information. The only ways to obtain more of that "
        "are (a) genuinely new labelled data in the competition domain, which is what external data "
        "attempts and which the original dataset is NOT (its conditional is 6e-3 worse and its delay "
        "distribution was rewritten by the generator), or (b) a better USE of the labels we have. "
        "That redirects effort away from row manufacture and toward the training objective: every "
        "model in the pool optimises logloss while the metric is ROC-AUC, which is the one remaining "
        "untested axis that changes error geometry rather than adding capacity."),
}


def main() -> int:
    n = append(ENTRY)
    n += append(CONCLUSION)
    print(f"\n{n} entries appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
