"""Append the final Phase 10B aggregate entry from the swap report.

Why a separate script: scripts/log_phase10_ledger.py read reports/p10b.json for per-arm deltas, but
each `--folds` invocation OVERWRITES that file with only its own folds, so by the end it held folds 2-4
and the fold-0 arm records were gone (KeyError: '0'). The authoritative 5-fold record now lives in
reports/p10b_swap.json, which is written once from the per-fold .npy files. This script reads that,
so it does not depend on any run-scoped report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402


def main() -> None:
    sw = json.loads((REPORTS / "p10b_swap.json").read_text(encoding="utf-8"))
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()[:12]

    entries = [{
        "exp_id": "p10b_native_cats_FULL5FOLD_FINAL",
        "date": "2026-10-06", "git": git, "family": "catboost_native_cats",
        "featureset": "full (+17 native categorical twins)", "fold_scheme": "primary",
        "control": "C0 (current numeric CatBoost path)", "control_auc": sw["oof_C0"],
        "oof_auc": sw["oof_C2"], "oof_auc_v3": sw["oof_v3"],
        "delta_vs_control_e5": sw["delta_C2_minus_C0_e5"],
        "per_fold_e5": sw["per_fold_e5"], "paired_fold_se_e5": sw["paired_se_e5"],
        "t_stat": sw["t_stat"], "folds_positive": sum(1 for x in sw["per_fold_e5"] if x > 0),
        "n_folds": len(sw["per_fold_e5"]),
        "blend_gains_e5": sw["marginal_gains_e5"],
        "gate_standalone_e5": 8.0, "gate_blend_e5": sw["gate_e5"],
        "verdict": "REAL, REPLICATED mechanism at the model level (+6.64e-5 full OOF, t=2.17, 4/5 "
                   "folds) but NOT harvestable by the ensemble: marginal add +0.00e-5, single-model "
                   "swap -0.39e-5. NO SUBMISSION.",
        "notes": "C2 = C0 + 17 native categorical twins (META4 + SERVICE13) under Plain boosting, "
                 "numeric block byte-identical. Fold deltas +5.87, +11.58, -2.37, +14.44, +3.02 e-5: "
                 "mean +6.51e-5, paired SE 3.00e-5, t=2.17, 4/5 positive. Fold 2 is NEGATIVE, and the "
                 "two-fold mean of +8.75e-5 observed before folds 2-4 ran overstated the effect by "
                 "35% -- a direct vindication of the predeclared multi-fold gate.\n\n"
                 "C2 (OOF 0.961121) is individually stronger than ALL SEVEN of v3's existing "
                 "CatBoost members, by +4.9e-5 (z3_cat_f10) to +25.7e-5 (z3_cat_ogs). So the "
                 "mechanism genuinely improves the CatBoost family and every current CatBoost member "
                 "is strictly dominated.\n\n"
                 "Yet the ensemble gains nothing: marginal admission of C2 as a new member is "
                 "+0.00e-5 at w=1% and negative beyond, and giving the CatBoost block's 7/59 = 0.119 "
                 "weight to C2 instead moves the blend 0.961509 -> 0.961505 (-0.39e-5, 2/5 folds). The "
                 "explanation is DIVERSITY. v3 rebuilt from its 59 stored members reproduces the "
                 "stored blend to logit corr 1.000000 and AUC 0.961509 exactly, so the arithmetic is "
                 "trustworthy, and C2's logit correlation with each existing CatBoost member is "
                 "0.997-0.998. Replacing seven correlated-but-distinct members with one individually "
                 "stronger model destroys the averaging that made the block worth 11.9% of the "
                 "ensemble. Individual member quality and block contribution are different quantities "
                 "and here they point opposite ways.\n\n"
                 "LIVE UNTESTED THREAD: a DIVERSE native-categorical CatBoost block, several members "
                 "mirroring the existing configs (depth 6/8/10, feature_fraction, core3 and ogs views, "
                 "seed variation), each individually better AND retaining the diversity the block "
                 "depends on. The single-model swap is pessimistic about diversity and is explicitly "
                 "an approximation, not a submission.",
        "extra": {"swap": sw["swap"], "per_cat_member": sw["per_member"],
                  "cat_members": sw["cat_members"], "verdict_detail": sw["verdict"]},
    }]

    out = REPORTS / "p10_final_ledger_entry.json"
    save_json({"entries": entries}, out)
    print(f"prepared {len(entries)} entry -> {out}")
    if "--append" not in sys.argv:
        print("review first, then: python scripts/log_phase10_final.py --append")
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
    print(f"appended {added}; skipped {skipped}")
    print("ledger lines now:", sum(1 for l in ledger.open(encoding="utf-8") if l.strip()))


if __name__ == "__main__":
    main()
