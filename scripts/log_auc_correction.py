"""Append-only CORRECTION to the Phase 6 (pairwise AUC surrogate) interpretation.

The empirical result stands. The MECHANISTIC EXPLANATION recorded with it was mathematically wrong
and is corrected here rather than rewritten, per append-only policy.

What was wrong
--------------
The original note said:

    "The pairwise surrogate is invariant to any monotone rescaling of the scores, so it has no
     incentive to produce well-separated scores, while ROC-AUC is decided entirely by score
     separation."

Both halves are wrong:

  1. ROC-AUC IS invariant to any strictly increasing transformation of the score. What decides
     ROC-AUC is the induced ORDERING of scores, and a strictly increasing transform preserves it.
     So AUC does not "depend on separation" in any absolute sense -- it depends on the ordering.

  2. The pairwise logistic loss log(1 + exp(-(s_pos - s_neg))) depends EXPLICITLY on pairwise score
     DIFFERENCES. It is invariant only to a GLOBAL ADDITIVE SHIFT, because a shift cancels in the
     difference. Multiplying scores, or applying a non-linear monotone map, generally CHANGES the
     pairwise loss -- so it is not invariant to arbitrary monotone rescaling.

  3. Consequently the claim that the pairwise loss "has no incentive to separate positives from
     negatives" is false: it maximises exactly a pos-vs-neg margin, and encourages separation
     directly.

What survives
-------------
Only the empirical finding, which was never in doubt:

    binary matched control                       0.961103
    pairwise lr 0.05 /  4000 rounds              0.956339   corr logit 0.98577  Spearman 0.91352
    pairwise lr 0.15 / 15000 rounds              0.959556   corr logit 0.99540  Spearman 0.97560

The apparent decorrelation at 4000 rounds was largely an undertraining artefact -- as training
proceeded, AUC rose AND correlation moved back toward the existing pool. The tested custom
pure-pairwise LightGBM branch is REJECTED: it stayed well below the matched control and its blend
contribution was negative at every weight tested.

What we do NOT know
-------------------
We have NO demonstrated explanation for why the pairwise surrogate underperforms here. Plausible
candidate causes, none of them established by our experiments:

  * optimisation mismatch between a pairwise margin loss and diagonal-Hessian tree boosting
  * the diagonal-Hessian approximation itself, which ignores the pair coupling between rows
  * pair sampling: only 4 negatives per positive, fixed across all boosting iterations
  * regularisation behaviour differing between objectives
  * statistical efficiency: BCE uses every row on every iteration, whereas the pairwise loss uses
    only sampled pairs
  * noise/variance injected by pair sampling

These are hypotheses, not findings. None was isolated experimentally, and no mechanism should be
attributed to this result.

SCOPE CORRECTION
----------------
The previous write-up implied the AUC-objective axis was closed. That overstates what was tested.
The accurate statement is: THE TESTED CUSTOM PURE-PAIRWISE LIGHTGBM BRANCH IS REJECTED. XGBoost's
native rank:pairwise and other AUC surrogates remain technically untested; current evidence makes
them LOW PRIORITY, but they are not excluded.

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
    "exp_id": "auc_objective_fold0_CORRECTION",
    "oof_auc": None, "fold_aucs": [],
    "date": "2026-10-05",
    "verdict": "CORRECTION -- the result stands, the mechanistic explanation was mathematically wrong",
    "corrects": "auc_objective_fold0",
    "empirical_result_unchanged": {
        "binary_matched_control": 0.961103,
        "pairwise_lr005_4000": {"auc": 0.956339, "corr_logit": 0.98577, "spearman": 0.91352},
        "pairwise_lr015_15000": {"auc": 0.959556, "corr_logit": 0.99540, "spearman": 0.97560},
        "conclusion": ("the apparent decorrelation at 4000 rounds was largely an undertraining "
                       "artefact; the tested custom pure-pairwise LightGBM branch is rejected"),
    },
    "wrong_claim": ("'the pairwise surrogate is invariant to any monotone rescaling of the scores, "
                    "so it has no incentive to produce well-separated scores, while ROC-AUC is "
                    "decided entirely by score separation'"),
    "corrections": [
        "ROC-AUC IS invariant to any strictly increasing transformation of the score; what decides "
        "it is the induced ORDERING, which such a transform preserves.",
        "The pairwise logistic loss log(1+exp(-(s_pos - s_neg))) depends EXPLICITLY on pairwise "
        "score DIFFERENCES. It is invariant only to a GLOBAL ADDITIVE SHIFT, since a shift cancels "
        "in the difference.",
        "Multiplying scores or applying a non-linear monotone map generally CHANGES the pairwise "
        "loss, so it is not invariant to arbitrary monotone rescaling.",
        "The pairwise loss therefore DOES encourage positive-versus-negative margin separation; the "
        "claim that it lacks such an incentive is false.",
    ],
    "unexplained": (
        "No mechanism was demonstrated for the pairwise branch's inferiority. Candidate causes, "
        "none isolated experimentally: optimisation mismatch between a margin loss and "
        "diagonal-Hessian boosting; the diagonal-Hessian approximation ignoring pair coupling; pair "
        "sampling at only 4 negatives per positive held fixed across iterations; differing "
        "regularisation behaviour; statistical efficiency (BCE uses every row every iteration, the "
        "pairwise loss only sampled pairs); and pair-sampling noise. These are hypotheses only."),
    "scope_correction": (
        "The earlier wording implied the AUC-objective axis was closed, which overstates what was "
        "tested. Correct scope: THE TESTED CUSTOM PURE-PAIRWISE LIGHTGBM BRANCH IS REJECTED. "
        "XGBoost's native rank:pairwise and other AUC surrogates remain technically untested; the "
        "current evidence makes them low priority but does not exclude them."),
}


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    if ENTRY["exp_id"] in existing:
        print("  skip: correction already logged")
        return 0
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ENTRY) + "\n")
    print(f"  appended: {ENTRY['exp_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())