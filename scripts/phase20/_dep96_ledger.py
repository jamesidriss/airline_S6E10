"""DEP 96 (finalist validation) ledger + catalogue. Prereg acd3986c frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/dep96_finalist_validation_prereg.json"
SHA = "acd3986c07101d5c43c7164b72ff9893e7197917f7ce13b3fc517560e9fe592c"


def main() -> int:
    entry = {
        "exp_id": "p20_dep96_finalist_validation",
        "hypothesis_id": "H96-BLOCK10-INFRASTRUCTURE",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "DEP",
        "experiment_number": 96,
        "fold_scheme": "block10 (10-fold, seed 20261010) — independent of primary 5-fold",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "LGBM prod5_lgbm_full_primary (xt_xt_f10) on 285-feature rich view, 10-fold CV",
        "results": {
            "block10_oof_auc": 0.961050,
            "champion_primary_oof": 0.961853,
            "delta_vs_primary": -0.0008034,
            "per_fold_aucs": [0.960313, 0.961463, 0.960478, 0.962049, 0.961183,
                              0.961007, 0.961103, 0.959694, 0.962286, 0.960983],
            "mean_fold_auc": 0.961056,
            "std_fold_auc": 0.000737,
            "artifact": "reports/phase20/dep96_finalist_validation/dep96_finalist_validation.json",
        },
        "finding": (
            "Block10 (10-fold, seed 20261010) fold scheme generated and champion LGBM "
            "validated. 10-fold OOF AUC 0.961050 vs primary 5-fold 0.961853 (delta -8.03e-4). "
            "Per-fold AUCs range 0.959694–0.962286 (std 7.37e-4). The champion is robust "
            "across fold schemes (delta < 1e-3). Block10 folds saved for finalist B "
            "validation when selected. Infrastructure ready for independent finalist "
            "validation per AGENTS.md final-submission strategy."),
        "verdict": "INFRASTRUCTURE_READY",
        "closes_rows": [96],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Block10 (10-fold, seed 20261010) fold scheme generated and registered in "
        "data/cache/folds/. Champion LGBM 10-fold OOF 0.961050 vs primary 5-fold "
        "0.961853 (delta -8.03e-4). Per-fold AUCs 0.959694–0.962286 (std 7.37e-4). "
        "Champion robust across fold schemes. Infrastructure ready for finalist B "
        "validation (DEP 96 complete). Fold scheme independent of primary/shadow, "
        "per AGENTS.md final-submission strategy."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 96:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F1"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 96: finding recorded (F1)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())