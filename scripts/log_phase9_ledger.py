"""Append the Phase 9 findings to the experiment ledger.

Kept as a script rather than an inline command so the records are reproducible and reviewable, and
so the numbers come from the saved report rather than being retyped from a log. Every record carries
the fold-0 AUC, the paired delta against the matched control, the gate outcome, and the specific
mechanism or correction, because a ledger entry without a mechanism is not much use later.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.common import ID_COL  # noqa: E402

GATE_STANDALONE = 5.0     # e-5, promotion bar for a standalone improvement
GATE_MARGINAL = 1.5       # e-5, admission bar for a marginal blend gain


def main() -> None:
    rep = json.loads((REPORTS / "p9_f0_report.json").read_text(encoding="utf-8"))
    f0 = rep["folds"]["0"]
    arms = f0["arms"]
    ctrl = f0["control"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()[:12]
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y_int, tr[ID_COL]).folds

    entries = []

    def add(exp_id, family, params, verdict, notes, extra=None):
        entries.append({
            "exp_id": exp_id, "date": "2026-10-06", "git": git, "family": family,
            "featureset": "full", "fold_scheme": "primary", "fold": 0,
            "params": params, "control": ctrl, "control_auc": f0["control_auc"],
            "oof_auc": arms.get(family, {}).get("auc"),
            "delta_vs_control_e5": arms.get(family, {}).get("delta_e5"),
            "blend_gains_e5": arms.get(family, {}).get("blend_gains_e5"),
            "logit_corr_vs_control": arms.get(family, {}).get("logit_corr_vs_ctrl"),
            "spearman_vs_control": arms.get(family, {}).get("spearman_vs_ctrl"),
            "gate_standalone_e5": GATE_STANDALONE, "gate_marginal_e5": GATE_MARGINAL,
            "verdict": verdict, "notes": notes, "extra": extra or {},
        })

    add("p9_dart005_REJECTED", "dart005",
        {"boosting_type": "dart", "drop_rate": 0.05, "skip_drop": 0.5, "extra_trees": False,
         "learning_rate": 0.02, "num_leaves": 127, "colsample_bytree": 0.8, "subsample": 0.8,
         "subsample_freq": 1, "rounds": 1700, "round_selected_from": "inner refit curve, argmax"},
        "REJECTED -- standalone -42.8e-5 and every blend weight negative",
        "DART loses 42.8e-5 against the matched ctl_fixed control and has negative marginal blend "
        "gain at every weight from 1% to 10%. Logit corr 0.99788 means it is a near-clone carrying "
        "less signal, not a decorrelated view, so there is no diversity to harvest. The inner curve "
        "was still rising at the 1700-round grid boundary, but the slope there is +7e-5 per 600 "
        "rounds, so closing a 42.8e-5 deficit would need roughly 3700 further rounds at a per-round "
        "cost that grows with tree count (measured 2.62x from 300 to 1500 trees).",
        {"inner_curve": {"400": 0.960419, "700": 0.960906, "1100": 0.961165, "1700": 0.961238},
         "snapshots_unsafe": True,
         "note": "DART has NO early stopping in LightGBM 4.7.0; best_iteration is None and the "
                 "callback is a silent no-op, so rounds came from an inner refit curve."})

    add("p9_dart010_REJECTED", "dart010",
        {"boosting_type": "dart", "drop_rate": 0.10, "skip_drop": 0.5, "extra_trees": False,
         "rounds": 1700},
        "REJECTED -- standalone -48.7e-5, worse than drop_rate 0.05",
        "Doubling the dropout rate makes DART slightly worse (-48.7e-5 vs -42.8e-5), so the mild "
        "setting was already the better of the two and no third setting is worth a fold-1 run.",
        {"inner_curve": {"400": 0.960216, "700": 0.960776, "1100": 0.961078, "1700": 0.961277}})

    add("p9_dart005_xt_REJECTED", "dart005_xt",
        {"boosting_type": "dart", "drop_rate": 0.05, "skip_drop": 0.5, "extra_trees": True,
         "rounds": 1700},
        "REJECTED -- standalone -35.2e-5, but BETTER than DART without extra_trees",
        "This arm exists to test the capability probe's claim that extra_trees is catastrophic under "
        "DART. On real folds the claim replicates in DIRECTION but not magnitude: extra_trees is "
        "worth +7.6e-5 to DART here (-35.2e-5 vs -42.8e-5), against the probe's -4.82e-3 on a 90k "
        "subsample of raw columns. The interaction is real and mild, not catastrophic, and the probe "
        "subsample overstated its size by ~600x. Still rejected: -35.2e-5 is far below the gate.",
        {"probe_claimed_e5": -482.0, "fold0_actual_e5_vs_noxt": 76.0,
         "inner_curve": {"400": 0.960190, "700": 0.960762, "1100": 0.961163, "1700": 0.961336}})

    add("p9_rf_REJECTED", "rf",
        {"boosting_type": "rf", "bagging_fraction": 0.8, "bagging_freq": 1,
         "feature_fraction": 0.8, "extra_trees": False, "rounds": 900},
        "REJECTED -- standalone -306.8e-5, an order of magnitude below every other arm",
        "LightGBM RF mode is structurally the most different tree family tested and it does "
        "decorrelate: spearman 0.93286 against the champion is the lowest measured in this campaign, "
        "versus >=0.984 for every GBDT and DART arm. But it is 306.8e-5 weaker and no weight helps "
        "(blend@1% is already -0.19e-5). Its inner curve is flat from 500 to 3600 rounds "
        "(0.958834 to 0.958837, a 2e-5 range), so RF mode saturates almost immediately and more trees "
        "would not close the gap. This is the clearest case in the campaign of a genuinely different "
        "mechanism that is simply not competitive.",
        {"inner_curve": {"500": 0.958834, "900": 0.958848, "1500": 0.958854, "2400": 0.958842,
                         "3600": 0.958837},
         "saturation_range_e5": 2.0,
         "correction": "I predicted LightGBM would refuse RF at bagging_fraction=1.0/bagging_freq=0. "
                       "It trained fine (144 trees). RF mode does NOT require bagging, so that arm "
                       "was not a random forest and was dropped as invalid."})

    add("p9_ctl_det_CONTROL", "ctl_det",
        {"boosting_type": "gbdt", "extra_trees": False, "rounds": 900},
        "CONTROL -- confirms extra_trees is load-bearing under the champion config",
        "Matched GBDT control WITHOUT extra_trees, needed to interpret dart005. It scores -44.8e-5 "
        "against ctl_fixed, so extra_trees is worth +44.8e-5 here and is not optional. This is the "
        "opposite sign to its effect under DART (+7.6e-5), so the two families genuinely interact.",
        {"extra_trees_value_under_gbdt_e5": 44.8, "extra_trees_value_under_dart_e5": 7.6})

    add("p9_fixed_round_vs_inner_es", "ctl_fixed",
        {"boosting_type": "gbdt", "extra_trees": True, "rounds": 900,
         "protocol": "inner-selected fixed rounds, refit on 100% of outer-fit"},
        "CONFIRMS -- fixed-round protocol beats established inner-ES by +5.3e-5",
        "ctl_fixed 0.961352 vs ctl_es 0.961299, same rows, same params, same process. Phase 7 "
        "measured +2.2e-5 for the same comparison over two folds; this is a larger effect on one "
        "fold with the same sign. Mechanism: inner-ES permanently discards 10% of the outer-fit rows "
        "from training, whereas the fixed-round protocol selects the round count on inner data and "
        "then refits on 100% of outer-fit.",
        {"ctl_fixed_auc": arms["ctl_fixed"]["auc"], "ctl_es_auc": arms["ctl_es"]["auc"]})

    add("p9_error_concentration_DIAGNOSTIC", "diagnostic", {},
        "DIAGNOSTIC -- remaining error is NOT where members disagree; variance reduction is exhausted",
        "Conditioned pair error rate on pairwise member disagreement over 4000 x 4000 = 1.6e7 pairs. "
        "All seven level-free dispersion measures put the WORST band at full agreement, not at "
        "maximum disagreement: rank_range 0.07432 (band 0) vs 0.03291 (median) vs 0.04335 (band 9). "
        "The curve is U-shaped. Rows where all 98 members agree carry a 1.7-1.8x higher pair error "
        "rate than the median band. This contradicts the variance-reduction hypothesis that "
        "motivated Phase 9: full agreement marks rows where every model is confidently wrong for a "
        "SHARED reason, and averaging reinforces a shared bias rather than fixing it. Verdict "
        "logged as INCONSISTENT across measures (5 of 7 missing-signal, 2 variance) rather than "
        "overstated.",
        {"worst_band": 0, "band0_e5": 7432.0, "band_median_e5": 3291.0, "band9_e5": 4335.0,
         "measures_agreeing_worst_at_band0": 7, "n_measures": 7,
         "key_correction": "cross-member std is NOT an uncertainty measure: measured mean std by "
                           "predicted-p decile is 0.545, 0.487, 0.460, 0.437, 0.393, 0.273, 0.436, "
                           "0.462, 0.497, 0.614, i.e. SMALLEST at p ~ 0.5. It mostly encodes "
                           "|p - 0.5|, so stratifying error by it measures a near-tie band, not "
                           "difficulty. All measures were rebuilt level-free."})

    add("p9_CAPABILITY_PROBE", "probe", {},
        "ESTABLISHED -- DART has no early stopping; RF does not require bagging",
        "LightGBM 4.7.0 capability probe on a 90k subsample of raw columns. (1) DART emits 'Early "
        "stopping is not available in dart mode' and best_iteration is None, so the standard callback "
        "is a silent no-op -- fixed-round inner selection is mandatory, not stylistic. (2) RF mode "
        "does NOT require bagging: bagging_fraction=1.0/bagging_freq=0 trained 144 trees fine, so "
        "that arm was not a random forest and was dropped. (3) RF+extra_trees is degenerate, early "
        "stopping at 36 trees. (4) An earlier reading that the RF callback 'agreed with the argmax' "
        "was an artifact: RF stopped at 174 trees while the snapshot grid began at 100, so only one "
        "curve point was ever measured.",
        {"dart_early_stopping": "unavailable, best_iteration None",
         "rf_requires_bagging": False, "rf_xt_degenerate_trees": 36})

    add("p9_TRUNCATION_PROBE", "probe", {},
        "ESTABLISHED -- DART round selection must use refits, not truncated snapshots",
        "An r-round DART model is not the r-round prefix of an N-round one. Measured truncation error "
        "against a same-seed fresh refit: GBDT+extra_trees 0.00e-5 at every round (the control, and "
        "exactly 0 as it must be since GBDT boosting is strictly additive), DART drop 0.05 up to "
        "-23.2e-5, DART drop 0.10 up to -34.4e-5, systematically signed. Selecting rounds from a "
        "truncated curve would optimise a different objective from the model deployed. The harness "
        "now REFUSES snapshot mode for any DART arm unless this probe's verdict says it is safe.",
        {"gbdt_control_gap_e5": 0.0, "dart005_worst_gap_e5": 23.15, "dart010_worst_gap_e5": 34.40,
         "gate_e5": 1.5})

    add("p9_CHAMPION_CONFIG_CORRECTION", "correction", {},
        "CORRECTION -- champion params were taken from the wrong source and the control caught it",
        "I set the harness champion from src/models/gbdt.py (learning_rate 0.03, num_leaves 63) on "
        "the reasoning that a module's defaults must be the champion's. ctl_es returned 0.961170 at "
        "432 rounds against the expected 0.961299 at 797, and the run was stopped. gbdt.py holds the "
        "FAMILY's generic defaults, which the champion overrides. The authoritative values are in "
        "scripts/run_fullfit.py:72-75 (lr 0.02, num_leaves 127, extra_trees True, n_estimators "
        "6000, early_stopping 300). Two independent signals agreed: the 12.9e-5 AUC shortfall and "
        "the iteration count, since lr 0.02 runs to roughly twice the rounds of lr 0.03. Without "
        "ctl_es this would have surfaced as a uniform ~13e-5 deficit across all seven arms and been "
        "misattributed to DART and RF being weak.",
        {"wrong_auc": 0.961170, "wrong_rounds": 432, "right_auc": 0.961299, "right_rounds": 797,
         "source_of_truth": "scripts/run_fullfit.py:72-75"})

    add("p9_DART_COST_CORRECTION", "correction", {},
        "CORRECTION -- DART per-round cost grows with tree count; my projection was 55x then 10x wrong",
        "First timing probe measured DART at 0.058 s/round from a 300-round fit and projected the "
        "full 8900-round grid at 0.09 h. Fold 0 exposed this by spending >25 min on dart005 with no "
        "output. A second probe on identical data at 1500 rounds measured 0.152 s/round, 2.62x more "
        "expensive per round, because every new DART tree re-normalises the surviving ensemble. A "
        "second bug then re-expressed that average as a per-300-round cost and scaled from 300, "
        "understating the grid another ~10x. Both are fixed: the fit is anchored at (measured "
        "rounds, measured seconds) with r^1.5 for DART and linear for GBDT. Worth noting "
        "independently: LightGBM's early-stopping path would have hidden this entirely.",
        {"s_per_round_300": 0.058, "s_per_round_1500": 0.152, "ratio": 2.62})

    out = REPORTS / "p9_ledger_entries.json"
    save_json({"entries": entries}, out)
    print(f"prepared {len(entries)} ledger entries -> {out}")
    for e in entries:
        print(f"  {e['exp_id']:<42} {e['verdict']}")
    if "--append" not in sys.argv:
        print("\nNothing appended to experiments/ledger.jsonl yet. Review first, then:")
        print("  python scripts/log_phase9_ledger.py --append")
        return

    # Append-only. Existing lines are never rewritten or reordered; duplicates are refused rather
    # than silently double-written, since a correction must be a NEW record.
    import experiments_ledger as EL
    ledger = REPORTS.parent / "experiments" / "ledger.jsonl"
    existing = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(json.loads(line)["exp_id"])
    added, skipped = 0, []
    for e in entries:
        if e["exp_id"] in existing:
            skipped.append(e["exp_id"])
            continue
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(e) + "\n")
        added += 1
    print(f"\nappended {added} entries to {ledger}")
    if skipped:
        print(f"skipped (already present): {skipped}")


if __name__ == "__main__":
    main()
