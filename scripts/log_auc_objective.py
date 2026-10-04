"""Record the AUC-objective experiment: the pairwise surrogate converges back to logloss.

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
        "exp_id": "auc_objective_verification",
        "oof_auc": None, "fold_aucs": [],
        "verdict": "objective VERIFIED CORRECT before any result was trusted",
        "date": "2026-10-05",
        "checks": {
            "gradient_vs_central_finite_differences": {"max_abs": 3.143e-12, "max_rel": 1.286e-10,
                                                      "passes": True},
            "hessian_strictly_positive": {"min": 1.0e-06, "passes": True},
            "gradient_descent_raises_true_auc": {"before": 0.5000, "after": 0.9747,
                                                 "loss_before": 0.6931, "loss_after": 0.0723,
                                                 "passes": True},
            "lightgbm_lambdarank": {"applicable": False,
                                    "reason": "LightGBMError: Ranking tasks require query "
                                              "information",
                                    "note": "its NDCG-based per-query truncation is also the "
                                            "wrong surrogate for a single global ranking"},
            "xgboost_rank_pairwise": {"available": True},
        },
        "why_this_step": (
            "A hand-written pairwise loss is exactly the kind of thing that can be subtly and "
            "invisibly wrong: a sign error or a mis-scaled hessian still trains, still converges "
            "and still yields a plausible AUC. The gradient was therefore checked against finite "
            "differences and the loss shown to raise true AUC before it was allowed near an "
            "experiment."),
    },
    {
        "exp_id": "auc_objective_fold0",
        "oof_auc": None, "fold_aucs": [],
        "verdict": "REJECT -- pairwise AUC surrogate converges back toward the logloss solution",
        "date": "2026-10-05",
        "fold": 0, "scheme": "primary", "view": "full", "n_features": 285,
        "n_train_rows": 503739, "n_neg_per_pos": 4,
        "matched_control": {"arm": "binary_logloss", "auc": 0.961103, "best_iter": 1073,
                            "corr_finalist_logit": 0.99170, "spearman_finalist": 0.99214,
                            "blend_best_gain": 0.0},
        "arms": [
            {"arm": "pairwise", "lr": 0.05, "rounds": 4000, "auc": 0.956339,
             "delta_e5": -476.3, "best_iter": 4000, "hit_round_cap": True,
             "corr_finalist_logit": 0.98577, "spearman_finalist": 0.91352,
             "blend_best_gain": -0.000013},
            {"arm": "pairwise", "lr": 0.15, "rounds": 15000, "auc": 0.959556,
             "delta_e5": -155.5, "best_iter": 14999, "hit_round_cap": True,
             "corr_finalist_logit": 0.99540, "spearman_finalist": 0.97560,
             "blend_best_gain": -0.000014},
            {"arm": "binary", "auc": 0.961103, "note": "matched control, reproduces the champion "
                                                          "fold-0 number exactly"},
        ],
        "finding": (
            "The striking decorrelation at the first setting was an ARTEFACT OF UNDERTRAINING, not a "
            "different error geometry. Both pairwise runs hit their round cap, and the trajectory is "
            "monotone: more training raises AUC (0.9563 -> 0.9596) while simultaneously pushing the "
            "correlation back up (logit 0.98577 -> 0.99540, Spearman 0.91352 -> 0.97560). The "
            "apparently orthogonal model was simply an unconverged one. At its most decorrelated "
            "setting Spearman 0.91352 is the lowest ever measured in this campaign -- but that point "
            "came with a 4.8e-3 AUC deficit and still gave a NEGATIVE blend gain at every weight."),
        "why_pairwise_loses_here": (
            "The pairwise surrogate is invariant to any monotone rescaling of the scores, so it has "
            "no incentive to produce well-separated scores. ROC-AUC, however, is decided entirely by "
            "score SEPARATION. Logloss spends capacity on calibration that AUC does not reward, and "
            "that apparently acts as a useful regulariser: a well-calibrated, well-spread score "
            "vector ranks better than a rank-optimal but arbitrarily-scaled one. This is the mirror "
            "image of the earlier soft-target finding, where shrinking toward a prior also failed."),
        "untested": ("XGBoost's native rank:pairwise was capability-probed as available but not run "
                     "to a result; given that the custom surrogate already converges back to the "
                     "logloss solution, its expected value is low and it is recorded as untested "
                     "rather than implied to be covered."),
        "protocol": ("identical features, seed and inner early-stopping holdout across arms, so the "
                     "only difference is the training objective; early stopping uses the inner "
                     "holdout carved from outer-FIT rows; the outer fold is scored once."),
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
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']} [{e['verdict'][:56]}]")
    print(f"\n{added} entries appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
