"""Append-only ledger entries for the Phase 7 block. Idempotent; never rewrites history.

AGENTS.md rule: "Every claim needs a number. Record experiments in experiments/ledger.jsonl with the
paired fold deltas, correlation with the champion, and a verdict." This writes one entry per Phase 7
result, including the ones that failed and the two places where a first pass produced a wrong number
that had to be corrected.

Corrections are appended as their own entries rather than folded into the originals. That is the
whole point of an append-only ledger: the fact that the bagging harness first reported -586e-5 from
an index-space bug, and that the LB-noise estimate was first done by hand and came out three times
too large, are both information about how this campaign's results were produced.

Usage: python scripts/log_phase7.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEDGER = ROOT / "experiments" / "ledger.jsonl"

DATE = "2026-10-05"

ENTRIES = [
    {
        "exp_id": "p7_train_fraction_audit",
        "date": DATE,
        "method": "audit exact row counts in the training pipeline",
        "verdict": "FINDING -- every CV-stage OOF model trains on 72% of labels, not 80%",
        "numbers": {
            "total_labelled_rows": 699635,
            "primary_K5_outer_fit_rows": 559708, "primary_K5_frac": 0.80,
            "primary_K5_inner_es_rows": 55969,
            "primary_K5_ACTUAL_model_fit_rows": 503739, "primary_K5_actual_frac": 0.72,
            "block10_K10_ACTUAL_model_fit_rows": 566706, "block10_actual_frac": 0.81,
        },
        "notes": ("the inner early-stopping holdout costs a further 8 percentage points on top of the "
                  "outer-fit carve; a first audit asserted by ASSUMPTION that the test-time refit used "
                  "100% of rows, which was wrong and is corrected by p7_test_refit_correction"),
    },
    {
        "exp_id": "p7_test_refit_correction",
        "date": DATE,
        "corrects": "p7_train_fraction_audit",
        "verdict": "CORRECTION -- no test-time model is trained on 100% of the labels",
        "numbers": {
            "test_model_rows_K5": 559708, "test_model_frac_K5": 0.80,
            "test_model_rows_K10": 629672, "test_model_frac_K10": 0.90,
            "fraction_of_labels_unused_at_K5": 0.20,
            "fraction_of_labels_unused_at_K10": 0.10,
            "any_path_training_on_all_rows": False,
        },
        "method": "read which row index the test-time fit actually receives, not a grep for 'refit'",
        "notes": ("scripts/run_views.py lines 114-128 loop over the outer folds and fit each test "
                  "model on np.where(folds != k)[0], averaging over folds. So 20% (K=5) or 10% (K=10) "
                  "of the real labels are never shown to any member that votes on the test set, "
                  "although we hold them and test inference needs no held-out fold."),
    },
    {
        "exp_id": "p7_curve_validated_on_folds",
        "date": DATE,
        "verdict": "VALIDATION -- the learning-curve law predicts the 5->10 fold gain a priori",
        "method": "seed-matched pair from the prediction store; only the fold scheme varies",
        "numbers": {
            "k5_member": "xt_xt_sh_s1", "k10_member": "xt_xt_f10", "seed": 1,
            "predicted_gain_e5": 10.7, "observed_gain_e5": 9.9, "residual_e5": -0.8,
            "relative_error": 0.074,
            "seed_spread_shadow_e5": 4.4, "seed_spread_block10_e5": 6.4,
        },
        "notes": ("the residual is SMALLER than the seed-to-seed spread within either scheme, so the "
                  "law has predictive power on an axis it was not fitted to. Fold-count scaling is "
                  "not a mystery diversity effect; it is the same more-unique-labels-per-learner "
                  "effect the subsample curve measures."),
    },
    {
        "exp_id": "p7_inner_es_vs_full_outer_fit",
        "date": DATE,
        "verdict": "HELD -- real but small, and 4x smaller than the learning curve predicted",
        "method": ("champion extra_trees LightGBM, view full, primary folds; three arms differing "
                   "ONLY in the estimator of the iteration count"),
        "numbers": {
            "ctl_rows": 503739, "ctl_frac": 0.72, "full_fit_rows": 559708, "full_fit_frac": 0.80,
            "ctl_fold0": 0.961299, "ctl_fold1": 0.961396,
            "ctl_scaled_fold0_e5": 4.0, "ctl_scaled_fold1_e5": 0.4,
            "ctl_scaled_mean_e5": 2.2, "ctl_scaled_positive": "2/2",
            "innercv_raw_mean_e5": 1.6, "innercv_raw_positive": "1/2",
            "innercv_scaled_mean_e5": -2.9, "innercv_scaled_positive": "1/2",
            "learning_curve_prediction_e5": 9.6,
        },
        "over_iteration_evidence": {
            "iter_811_auc_fold0": 0.961338, "iter_863_auc_fold0": 0.961339,
            "iter_1191_auc_fold0": 0.961240, "iter_1191_delta_vs_ctl_e5": -5.9,
        },
        "notes": ("mean +2.2e-5 with 2/2 folds is below the +5e-5 promotion rule, so HELD not "
                  "escalated. Two lessons: the law OVER-EXTRAPOLATES at the top of the range, which "
                  "demotes the projected 80pct->100pct gain from +20.2e-5 to single-digit e-5; and "
                  "over-iteration is the failure mode, so the full-data policy must SHORTEN the "
                  "extrapolation rather than sharpen it."),
    },
    {
        "exp_id": "p7_group_conditional_calibration",
        "date": DATE,
        "verdict": "CLOSED -- best arm +0.8e-5, 5/5 folds, ~4x below the predeclared +3e-5 rule",
        "method": "nested meta-validation on primary folds over logit(blend_v3_final)",
        "numbers": {
            "G0_global_control_delta_e5": 0.0, "G0_folds_with_zero_movement": "5/5",
            "G2_union_best_mean_e5": 0.8, "G2_union_positive": "5/5", "G2_union_columns": 11,
            "G1_union_mean_e5": 0.6, "G2_class_mean_e5": 0.3, "G1_gender_mean_e5": -0.1,
        },
        "notes": ("the mandatory control is load-bearing: a global strictly-increasing Platt map moved "
                  "AUC by +0.0e-5 on all five folds, exactly as theory requires, so the harness is "
                  "trustworthy. There IS a small systematic cross-group rank bias and it is too "
                  "small to act on. Distinct from the rejected local-reliability gate, which changed "
                  "model weights by score region."),
    },
    {
        "exp_id": "p7_bagging_fraction",
        "date": DATE,
        "verdict": "REJECTED -- hypothesis refuted in both directions; signs flip",
        "hypothesis": ("extra_trees already randomises features AND thresholds, so subsample=0.8 "
                       "discards 20% of real labels per tree redundantly"),
        "numbers": {
            "subsample_0.8_fold0": 0.961299, "subsample_0.8_fold1": 0.961396,
            "subsample_0.9_fold0_e5": 2.8, "subsample_0.9_fold1_e5": -18.8,
            "subsample_0.9_mean_e5": -8.0, "subsample_0.9_positive": "1/2",
            "subsample_1.0_mean_e5": -4.2, "subsample_1.0_positive": "0/2",
            "iter_at_0.8": [797, 731], "iter_at_0.9": [860, 1082], "iter_at_1.0": [457, 757],
        },
        "notes": ("colsample_bytree left untouched because the mechanism that would justify touching "
                  "it was not established. The DIAGNOSTIC matters more than the verdict: changing only "
                  "subsample moved the early-stopped count 731->1082 on fold 1, and that fit cost "
                  "-18.8e-5. A single 55,969-row holdout cannot locate a flat optimum, so the "
                  "full-data iteration policy now takes the median of THREE independent 5pct "
                  "holdouts. Cross-check: this harness's control reproduces run_fullfit's control to "
                  "6 decimal places on both folds."),
    },
    {
        "exp_id": "p7_class_conditional_generative",
        "date": DATE,
        "verdict": "CLOSED -- standalone is strong and highly decorrelated, yet every weight is negative",
        "method": "binned class-conditional log-likelihood ratios over the 21 raw columns, nested",
        "numbers": {
            "gen_plain_standalone_auc": 0.930873, "gen_plain_best_gain_e5": -0.2,
            "gen_weighted_standalone_auc": 0.930595, "gen_top8_standalone_auc": 0.912040,
            "logit_corr_with_finalist": [0.87004, 0.87499, 0.82493],
            "blend_gain_at_w_0.005": -0.18, "blend_gain_at_w_0.2": -24.90,
        },
        "notes": ("the most decorrelated thing produced in this campaign (0.87 vs ~0.99 for every "
                  "previously rejected member) and still useless. Naive Bayes cannot represent the "
                  "Class x Type-of-Travel or rating-by-rating interactions that drive the target, so "
                  "its errors lie INSIDE the GBDT's error set rather than beside it. Decorrelation is "
                  "necessary for a useful blend member and demonstrably not sufficient -- the "
                  "cleanest available demonstration of that distinction."),
    },
    {
        "exp_id": "p7_pseudo_label_evaluation_rows",
        "date": DATE,
        "verdict": "REJECTED -- largest negative effect in the campaign, ~30x its own matched control",
        "method": ("held-out fold k treated as the unlabelled test set; M0 fitted on outer-fit only "
                   "predicts fold k, pseudo-labels the most confident fraction, fold k scored once "
                   "with true labels. Matched control: same row count via duplicated outer-fit rows "
                   "carrying their TRUE labels."),
        "numbers": {
            "frac_0.25_pseudo_minus_ctl_fold0_e5": -160.9, "frac_0.25_pseudo_minus_dup_fold0_e5": -156.1,
            "frac_0.50_pseudo_minus_ctl_fold0_e5": -242.0, "frac_0.50_pseudo_minus_dup_fold0_e5": -236.1,
            "frac_0.25_pseudo_minus_dup_fold1_e5": -148.9, "frac_0.50_pseudo_minus_dup_fold1_e5": -237.5,
            "pseudo_minus_dup_mean_frac025_e5": -152.5, "pseudo_minus_dup_mean_frac050_e5": -236.8,
            "positive_folds": "0/2 at both fracs",
            "dup_control_cost_e5": [-4.7, -5.9, -5.7, -16.6],
            "pseudo_positive_rate": [0.286, 0.321, 0.279, 0.329],
        },
        "notes": ("DISTINCT from Phase 5 and from it with a different mechanism. Phase 5 failed "
                  "because interpolated rows are off-manifold; this fails because a model's own labels "
                  "carry no information it did not already have, and selecting by most-confident "
                  "|p-0.5| deliberately concentrates its errors. Clean dose-response: damage nearly "
                  "doubles when the pseudo-labelled row count doubles. The duplicate control works "
                  "(-4.7e-5 to -16.6e-5, consistent with Phase 5's -6.4e-5), so row count is not the "
                  "cause. Together with Phase 5 this makes the unique-information conclusion much "
                  "stronger."),
    },
    {
        "exp_id": "p7_structural_probes",
        "date": DATE,
        "verdict": "FOUR negatives, one useful calibration",
        "numbers": {
            "train_test_exact_matches": 0,
            "distinct_train_feature_vectors": 699635, "total_train_rows": 699635,
            "train_rows_with_a_duplicate": 0,
            "domain_classifier_auc_train_vs_test": 0.500620,
            "raw_id_auc": 0.500073, "id_within_block_mean_abs_dev": 0.0019,
            "leave_one_out_bucket_only_auc": 0.933454,
            "label_pure_bucket_fraction": 0.688,
            "full_model_oof_auc": 0.961509,
            "added_by_all_modelling": 0.028055,
        },
        "notes": ("A: the campaign had checked only 21 matches against the ORIGINAL dataset, never "
                  "competition-train vs competition-test. All 699,635 train rows are distinct feature "
                  "vectors and no test row matches a train row, so no lookup feature exists. "
                  "B: train/test covariate shift confirmed negligible by a proper balanced 3-fold "
                  "domain classifier (early stopping fires at iteration 1). C: id confirmed useless, "
                  "so the fold protocol is not optimistic through id. D: 0.9335 of AUC comes free "
                  "from 11 raw columns and a lookup table; all accumulated modelling buys +0.028."),
    },
    {
        "exp_id": "p7_lb_noise_analysis",
        "date": DATE,
        "verdict": "FINDING -- the leaderboard gap IS real; a first hand estimate said the opposite",
        "numbers": {
            "n_public_rows": 59969, "n_test_rows": 299844,
            "our_public": 0.960980, "leader_public": 0.961760, "gap": 0.000780,
            "hanley_mcneil_se_range": [0.000990, 0.001242],
            "z_at_rho_0.99": 4.4, "z_at_rho_0.995_worst": 3.1,
            "gap_clears_2sigma_even_uncorrelated": True,
            "hand_estimate_se_range_first_pass": [0.0025, 0.0036],
        },
        "notes": ("the +-2e-4 paired floor in AGENTS.md is correct but narrow: it governs "
                  "near-identical submissions and is the WRONG instrument for asking whether we "
                  "differ from a DIFFERENT solution. A first pass, by hand, put the single-estimate SE "
                  "at 0.0025-0.0036 and concluded the gap might be 0.2 sigma; Hanley-McNeil gives "
                  "0.0011, three times smaller, so the gap is z=3.1-4.4. Consequence for spending: we "
                  "are genuinely behind, the top 20 are a converged pack 7.8e-4 up, and every gain "
                  "measurable here is single-digit e-5, so more GBDT-family polish will not close it."),
    },
    {
        "exp_id": "p7_index_space_guards",
        "date": DATE,
        "verdict": "GUARDS ADDED -- this bug class produced three confident fake results",
        "numbers": {
            "instances_caught": 3,
            "fake_results_produced": [-483.5, -586.3, -2086.0],
            "root_cause": ("assemble() returns a matrix indexed by POSITION WITHIN fit_idx, while y "
                           "and every helper on src.validation.folds is indexed by GLOBAL ROW NUMBER"),
            "new_tests": 5,
        },
        "notes": ("(1) run_fullfit inner-CV picker indexed by position-within-inner-train, early "
                  "stopping fired at 4/5/43 rounds and the run reported a firm REJECT at -483.5e-5. "
                  "(2) run_bagging_test swapped _inner_es_split's return order and reported "
                  "-586.3e-5. (3) run_bagging_test paired local features with global labels and "
                  "reported -2086.0e-5. tests/test_index_space.py now asserts the invariant directly: "
                  "correctly paired rows must exceed fold AUC 0.60 and mis-paired rows must sit within "
                  "0.02 of chance. Comparing MEANS was tried first and was far too weak -- the "
                  "mis-paired model 'separated' by 0.4442 vs 0.4441 and the guard passed."),
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
                continue
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']}")
    if not added:
        print("  nothing to append (all entries already logged)")
    print(f"  ledger lines now: {sum(1 for _ in LEDGER.open(encoding='utf-8'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
