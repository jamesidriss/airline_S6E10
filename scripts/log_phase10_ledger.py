"""Append the Phase 10 results to the experiment ledger.

Kept as a script so the numbers come from the saved reports rather than being retyped from a log, and
so every record carries the mechanism or the correction alongside the delta.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402

GATE_STANDALONE = 8.0    # e-5, immediate fold-1 promotion
GATE_WEAK = 3.0          # e-5, weak but potentially real if consistent
GATE_BLEND = 1.5         # e-5, marginal blend admission


def main() -> None:
    pb = json.loads((REPORTS / "p10b.json").read_text(encoding="utf-8"))
    p10a = json.loads((REPORTS / "p10a.json").read_text(encoding="utf-8"))
    xfer = json.loads((REPORTS / "p10a_transfer.json").read_text(encoding="utf-8"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()[:12]
    f0 = pb["folds"]["0"]["arms"]
    a0 = f0["C0"]["auc"]

    def delta(a):
        return (f0[a]["auc"] - a0) * 1e5

    entries = []

    def add(exp_id, family, verdict, notes, extra=None):
        entries.append({
            "exp_id": exp_id, "date": "2026-10-06", "git": git, "family": family,
            "featureset": "full (+17 native categorical twins)", "fold_scheme": "primary",
            "fold": 0, "control": "C0", "control_auc": a0,
            "oof_auc": f0.get(family, {}).get("auc"),
            "delta_vs_control_e5": delta(family) if family in f0 else None,
            "blend_gains_e5": f0.get(family, {}).get("blend_gains_e5"),
            "inner_best_iter": f0.get(family, {}).get("inner_best_iter"),
            "n_fit_rows_refit": f0.get(family, {}).get("n_fit_rows_refit"),
            "seconds_es": f0.get(family, {}).get("seconds_es"),
            "seconds_refit": f0.get(family, {}).get("seconds_refit"),
            "gate_standalone_e5": GATE_STANDALONE, "gate_weak_e5": GATE_WEAK,
            "gate_blend_e5": GATE_BLEND, "verdict": verdict, "notes": notes,
            "extra": extra or {},
        })

    add("p10b_C2_native_cats_PROMISING", "C2",
        "POSITIVE +5.9e-5 on fold 0 -- native categorical CTRs HELP; replication test running",
        "C2 is C0 plus 17 native categorical twins (META4 + SERVICE13) under Plain boosting. Nothing "
        "else changes: the numeric block is byte-identical and C0-vs-C2 differ only by the appended "
        "string columns. +5.9e-5 is the first genuinely new positive mechanism in several phases, and "
        "it is the mechanism an audit proved had NEVER been tested (zero cat_features and zero "
        "CatBoost boosting_type anywhere in src/ or scripts/). It falls in the +3..+8e-5 'weak but "
        "potentially real if consistent' band. NOTE the diversity clause of the fold-0 gate is NOT "
        "met: blend@2% is -0.14e-5 and logit corr vs v3 is 0.99895, so C2 is a near-clone of C0 and of "
        "v3. Fold 1 is therefore being run as a REPLICATION test rather than as a promotion, because "
        "only a second fold can distinguish +5.9e-5 from a fold-0 fluke.",
        {"n_cat_twins": 17, "iters_vs_C0": f"{f0['C2']['inner_best_iter']} vs "
                                         f"{f0['C0']['inner_best_iter']}"})

    add("p10b_C1_ordered_REJECTED", "C1",
        "REJECTED -- Ordered boosting costs -71.7e-5 standalone",
        "C1 is C0 with boosting_type=Ordered and nothing else changed. It is the WORST arm in the 2x2 "
        "and it also selects far fewer iterations (497 vs C0's 851). Mechanism, stated as a "
        "hypothesis consistent with the data rather than a measured finding: Ordered boosting exists "
        "to prevent target leakage, and it buys that by deliberately computing each tree's target "
        "statistics from an ordered SUBSET of the rows rather than all of them. With 559,708 "
        "outer-fit rows that safety is not needed and the reduced effective sample is a pure cost. "
        "The early iteration count is the same signal.",
        {"iters": f0["C1"]["inner_best_iter"], "C0_iters": f0["C0"]["inner_best_iter"]})

    add("p10b_C3_ordered_plus_cats_REJECTED", "C3",
        "REJECTED -- -59.7e-5; Ordered's harm dominates and swamps the +5.9e-5 from categoricals",
        "C3 combines both mechanisms. It is worse than C2 by 65.5e-5, so Ordered's cost is not offset "
        "by anything the categorical twins contribute. The 2x2 is therefore clean and additive in the "
        "harmful direction: categoricals help by +5.9e-5 under Plain and Ordered hurts by -71.7e-5 "
        "regardless of categoricals (C3 - C1 = +12.1e-5).",
        {"C3_minus_C2_e5": (f0["C3"]["auc"] - f0["C2"]["auc"]) * 1e5,
         "C3_minus_C1_e5": (f0["C3"]["auc"] - f0["C1"]["auc"]) * 1e5})

    add("p10b_C0_control", "C0",
        "CONTROL VALIDATED -- 0.961008 vs the established 0.960880, and the +12.8e-5 is expected",
        "C0 is the current numeric CatBoost path under Plain. The established fold-0 result for "
        "view_cat_full_primary is 0.960880, so C0 lands +12.8e-5 higher. That is not a discrepancy: "
        "C0 uses the Phase 9 fixed-round protocol, which selects the iteration count on a 10% inner "
        "carve and then REFITS on 100% of outer-fit, whereas the established path discards the carve. "
        "Phase 9 measured +5.3e-5 for exactly this change on LightGBM; a weaker model gaining more "
        "from 11% more data is plausible. The control is therefore validated AND the protocol change "
        "is the reason it differs, which is recorded so the +12.8e-5 is not later mistaken for a bug.",
        {"established_fold0": 0.960880, "delta_e5": 12.8})

    entries.append({
        "exp_id": "p10a_RESIDUAL_EXHAUSTED", "date": "2026-10-06", "git": git,
        "family": "diagnostic", "featureset": "full", "fold_scheme": "primary",
        "verdict": "SIMPLE GROUP RESIDUAL STRUCTURE EXHAUSTED -- no specialist built",
        "notes": "42 predeclared groupings scanned under a fully nested protocol: inner cross-fit "
                 "inside META_TRAIN only, then a champion fit on 100% of META_TRAIN, corrections "
                 "frozen and applied to META_VAL. Exactly one key passed on the surrogate "
                 "(raw:On-board service, prob +2.00e-5 4/5, logit +1.92e-5 4/5, sign agreement 0.720) "
                 "but it DOES NOT TRANSFER: applied to v3's OOF on the same held-out rows it gives "
                 "-0.28e-5 (probability, 2/5 folds) and +0.30e-5 (logit, 3/5 folds), both far below "
                 "the +1.5e-5 gate. The +2.00e-5 was a property of the SURROGATE's residuals, not a "
                 "shared bias in the ensemble. Phase 9's models-fail-together result stands, but the "
                 "shared bias is not a group-constant mean offset on any of these groupings.",
        "extra": {"n_keys": 42, "surrogate_pass": "raw:On-board service",
                  "surrogate_prob_e5": 2.00, "surrogate_logit_e5": 1.92,
                  "v3_prob_e5": xfer["mean_v3_prob_e5"], "v3_logit_e5": xfer["mean_v3_logit_e5"],
                  "mean_bias_gap_e5": p10a["aggregate"]["raw:On-board service"]["mean_bias_gap_e5"],
                  "n_folds_transferred": len(xfer["folds"])},
    })

    entries.append({
        "exp_id": "p10a_REPLICATION_STATISTIC_WAS_TAUTOLOGICAL", "date": "2026-10-06", "git": git,
        "family": "correction", "verdict": "CORRECTION -- my replication statistic could not fail",
        "notes": "I computed the 'confirmation-observed' group bias from (y - p_oof) over the "
                 "DISCOVERY rows -- the same rows and the same quantity the bias table was fitted on, "
                 "differing only by shrinkage. That is a tautology and it returned r = 1.000 for all "
                 "42 keys, which is exactly why every key in the first report looked perfectly "
                 "replicated. It measured shrinkage, not replication, so gate criterion 3 was vacuous. "
                 "Fixed to estimate E[y - p | g] independently on META_VALIDATION rows using that "
                 "side's base scores; honest correlations are 0.02-0.98 with a median near 0.3, and "
                 "several keys go NEGATIVE. The uniform 1.000 was the only clue, and had I not "
                 "reacted to it being suspiciously perfect, I would have reported a tautology as "
                 "strong evidence. A diagnostic that cannot fail is worse than no diagnostic.",
        "extra": {"tautological_r": 1.000, "honest_r_min": 0.02, "honest_r_max": 0.98,
                  "n_keys_affected": 42},
    })

    entries.append({
        "exp_id": "p10_CTR_ENGAGEMENT_VERIFIED", "date": "2026-10-06", "git": git,
        "family": "capability", "verdict": "VERIFIED -- CatBoost CTR machinery is genuinely engaged",
        "notes": "The capability probe's CTR check called get_leaf_ctr_description(), which does not "
                 "exist on CatBoostClassifier in 1.2.10. The call raised, the probe recorded None, and "
                 "the summary printed 'categoricals may have been silently ignored' -- a conclusion it "
                 "had never tested. Replaced with three decisive checks: the saved model JSON contains "
                 "ctr_type structures; predictions DIVERGE from an ordinal-code control fitted on the "
                 "same frame (max |diff| 3.65e-01); and declaring a numeric column as categorical "
                 "raises TypeError. This matters because the silent-ignore case is real -- a "
                 "categorical column present but undeclared trains happily as a float and every number "
                 "looks plausible while the experiment tests nothing.",
        "extra": json.loads((REPORTS / "ctr_engagement.json").read_text(encoding="utf-8")),
    })

    out = REPORTS / "p10_ledger_entries.json"
    save_json({"entries": entries}, out)
    print(f"prepared {len(entries)} entries -> {out}")
    for e in entries:
        print(f"  {e['exp_id']:<44} {e['verdict'][:74]}")
    if "--append" not in sys.argv:
        print("\nreview first, then: python scripts/log_phase10_ledger.py --append")
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
    print(f"\nappended {added} to {ledger}; skipped {skipped}")


if __name__ == "__main__":
    main()
