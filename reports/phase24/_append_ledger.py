"""Append the Phase-24 measurements to the append-only experiment ledger (idempotent by exp_id)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments/ledger.jsonl"
GIT = "9648517"
TS = "2026-10-20T08:30:00+00:00"

ENTRIES = [
    {
        "kind": "CENSUS",
        "exp_id": "p24_20261020_E1_oof_store_inventory",
        "ts": TS,
        "git": GIT,
        "scope": "all 133 stored OOF vectors in artifacts/predictions vs certified v7",
        "method": "recomputed pooled AUC, per-fold paired deltas on primary, logit correlation, best single-weight blend over w in {0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50}; (AUC, rho) frontier envelope",
        "results": {
            "n_vectors": 133,
            "positive_mean_paired_delta": 0,
            "best_hindsight_blend_delta": 1.5367545388778048e-05,
            "best_hindsight_vector": "z5_rm_ens16_ogs",
            "envelope_at_rho_0.997": 1.3141192498178356e-05,
            "envelope_at_rho_0.998": 4.329603052410924e-06,
        },
        "finding": "No stored vector has a positive mean paired fold delta against v7. Even with hindsight selection over ~930 vector/weight comparisons the ceiling is +1.54e-5, and it collapses to +4.3e-6 once a rho>=0.998 floor is imposed. The store is exhausted.",
        "verdict": "STORE_EXHAUSTED_EVEN_UNDER_HINDSIGHT_SELECTION",
    },
    {
        "kind": "EXPERIMENT",
        "exp_id": "p24_20261020_E2_composition_geometry_lab",
        "ts": TS,
        "git": GIT,
        "prereg": "configs/phase24/composition_lab_prereg.json sha256 7f9c296ff8732ea5f9c460c5f974ce1dff5d9f7d322d0ac9a7e9e70639db32a9",
        "scope": "7 frozen composition rules vs the deployed 50/50 logit mean, two certified parents, 5 primary folds",
        "results": {
            "identity_max_abs": 5.586324780537666e-08,
            "C1_nested_weight": {"delta": 1.694e-06, "t": 0.95, "folds_positive": 3},
            "C5_platt_average": {"delta": 1.02e-07, "t": 1.52, "folds_positive": 4},
            "C4_confidence_logistic": {"delta": -9.38e-07, "t": -0.17},
            "C7_geometric_mean": {"delta": -6.18e-06, "t": -1.95},
            "C6_probability_mean": {"delta": -1.0068e-05, "t": -1.77},
            "C2_block_rank": {"delta": -1.0355e-05, "t": -1.48},
            "C3_fit_cdf": {"delta": -1.1039e-05, "t": -1.55},
            "holm_min": 1.0,
            "any_pass": False,
        },
        "finding": "Every zero-parameter nonlinearity is strictly worse by 6e-6 to 1.1e-5; the fitted rules recover at most +1.7e-6. The deployed 50/50 logit mean is the optimum of the whole low-variance composition family, so the 0.0056 aggregation-loss mass measured by the phase-19 atlas is not recoverable by reparameterising the two parent scores.",
        "verdict": "NO_ADMISSION_COMPOSITION_GEOMETRY_CLOSED",
    },
    {
        "kind": "DIAGNOSTIC",
        "exp_id": "p24_20261020_E3_parent_member_seed_diagnostics",
        "ts": TS,
        "git": GIT,
        "scope": "10 certified aux10 members (leave-one-out) and the TabPFN route seed pair on primary fold 0",
        "results": {
            "aux10_mean_auc": 0.9614187093375335,
            "aux10_reconstruction_max_abs": 2.980223934301307e-08,
            "best_leave_one_out_gain": 2.3791583687593487e-06,
            "worst_leave_one_out_gain": -1.974191984188689e-05,
            "route_seed1201_fold0": 0.9614576551,
            "route_seed1202_fold0": 0.9611372,
            "route_two_seed_average_fold0": 0.9615469,
            "route_seed_average_delta": 8.93e-05,
            "route_seed_logit_corr": 0.9973,
            "measured_parent_to_portfolio_conversion": 0.071,
            "implied_portfolio_gain": 6.3e-06,
        },
        "finding": "The aux10 equal mean is optimal to 2.4e-6. Seed averaging is real at the parent level (+8.93e-5 on fold 0) but the measured parent->portfolio conversion of 7.1% (independent axis: internal2->internal4, route +5.34e-5 -> portfolio +3.818e-6) turns it into ~6e-6. Parent-local improvements are structurally diluted.",
        "verdict": "PARENT_LOCAL_IMPROVEMENTS_DILUTED_BELOW_GATE",
    },
    {
        "kind": "EXPERIMENT",
        "exp_id": "p24_20261020_E4_segment_gate_lab",
        "ts": TS,
        "git": GIT,
        "prereg": "configs/phase24/segment_gate_prereg.json sha256 f2ab010904607a2aff01addb34e9f62a5f9b5ad46dffca14a745307918974835",
        "scope": "8 deterministic family representatives x 2 label-free segment definitions = 16 frozen tests",
        "results": {
            "segment_rows": {"S1_unique_key": 304464, "S2_sparse_key": 431005},
            "best": {"family": "tabm", "segment": "S2_sparse_key", "delta": 2.799e-06, "t": 0.85, "folds_positive": 3},
            "second": {"family": "cat", "segment": "S2_sparse_key", "delta": 2.502e-06, "t": 1.70, "folds_positive": 4},
            "tests_with_zero_gate_weight": 11,
            "holm_min": 1.0,
            "any_pass": False,
        },
        "finding": "The FIT-optimal gate weight inside the slice is exactly zero in 11 of 16 tests: every candidate family actively hurts inside the slice on FIT rows. The champion's deficit on sparse-key rows is not a composition problem and is not repaired by smoother families.",
        "verdict": "NO_ADMISSION_SEGMENT_GATING_CLOSED",
    },
    {
        "kind": "AUDIT",
        "exp_id": "p24_20261020_E5_duplicate_and_key_audit",
        "ts": TS,
        "git": GIT,
        "scope": "exact feature-vector overlap and 16-column rating-key concentration",
        "results": {
            "train_rows": 699635,
            "train_unique_vectors": 699635,
            "test_rows": 299844,
            "test_unique_vectors": 299844,
            "cross_split_exact_matches": 0,
            "train_unique_rating_keys": 386132,
            "test_rows_with_key_seen_in_train": 169156,
            "median_key_size": 1.0,
            "p90_key_size": 3.0,
            "max_key_size": 186,
            "share_rows_in_keys_ge_20": 0.1787532070293796,
        },
        "finding": "There is no exact duplicate leak anywhere, within or across splits. The only dense key structure is the 16-column rating key, with median size 1 and p90 3; that structure is already carried by the fold-safe TE blocks and by the original-data target statistics in the rich view.",
        "verdict": "NO_NEW_INFORMATION_SOURCE_IN_KEY_STRUCTURE",
    },
    {
        "kind": "EXPERIMENT",
        "exp_id": "p24_20261020_E6_source_anchor_and_determinism",
        "ts": TS,
        "git": GIT,
        "scope": "source-distribution CV anchor at 130k and 65k rows; matched leave-one-out profile determinism on the 16-column rating/categorical key in both populations; delay contribution and delay-block label rates",
        "results": {
            "source_cv5_auc_130k": 0.9888061518961919,
            "source_cv5_auc_65k": 0.9878729802976574,
            "source_doubling_gain": 0.0009331715985345657,
            "source_cv5_auc_without_delays": 0.9884781698085249,
            "delay_contribution": 0.00032798208766704473,
            "source_loo_profile_auc": 0.9709949084975641,
            "synthetic_loo_profile_auc": 0.9252854476040077,
            "source_pure_group_row_share": 0.2678318447797967,
            "synthetic_pure_group_row_share": 0.30821213918686174,
            "source_mean_group_size": 1.245086949019307,
            "synthetic_mean_group_size": 1.8119062911128836,
            "source_duplicate_feature_vectors": 0,
        },
        "finding": "The source population is nearly deterministic at a fixed discrete profile (LOO AUC 0.9710) while the synthetic population is not (0.9253), despite repeating MORE profiles (mean group 1.81 vs 1.25). The generator smoothed the label function. Two alternative explanations are refuted by measurement: delays carry only 3.28e-4 of the source model (0.988806 -> 0.988478), and the source has zero duplicate feature vectors, so neither the delay rewrite nor duplicate leakage explains the 0.027 gap. This is the mechanism behind the shared-error mass and the 0.9618-0.9620 plateau.",
        "verdict": "MECHANISM_IDENTIFIED_GENERATOR_SMOOTHS_LABEL_FUNCTION",
    },
    {
        "kind": "PHASE_CLOSE",
        "exp_id": "p24_20261020_PHASE24_CLOSE",
        "ts": TS,
        "git": GIT,
        "decision": "NO_UPLOAD_IN_PHASE_24",
        "finalists": [
            {"slot": "A", "name": "v7_sol_tabpfn_internal2_aux10", "ref": 57090508, "oof": 0.9618529228157235, "public": 0.96146},
            {"slot": "B", "name": "v6_sol_tabpfn_route_aux10", "ref": 56951906, "oof": 0.9617656138245796, "public": 0.96144},
        ],
        "reserve": {"name": "v4_fulldata", "ref": 56845975, "use": "structural-failure hedge only"},
        "submissions_used": 0,
        "gpu_hours_used": 0.0,
        "finding": "All score-level, segment-level, member-level and composition-level levers are measured at or below the gate. The only unmeasured levers are hardware-blocked. The champion is re-certified byte-for-byte. The programme closes on a measured plateau, not on an untested assumption.",
        "reopen_conditions": [
            ">=16 GiB usable accelerator (raw8, TabICLv2 rich view)",
            "a new information source on the 21 columns",
            "a rules change affecting the two final slots or the private split",
            "a reproducible public result above 0.9620 that contradicts the measured ceiling",
        ],
        "verdict": "PHASE_CLOSED_MEASURED_PLATEAU",
    },
]


def main() -> None:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except json.JSONDecodeError:
                    pass
    added = 0
    with LEDGER.open("a", encoding="utf-8", newline="\n") as f:
        for e in ENTRIES:
            if e["exp_id"] in existing:
                continue
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
            added += 1
    total = sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip())
    print(json.dumps({"added": added, "ledger_lines": total}))


if __name__ == "__main__":
    main()
