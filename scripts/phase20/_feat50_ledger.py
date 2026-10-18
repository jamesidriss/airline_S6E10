"""FEAT 50 (learned interactions) ledger + catalogue. Prereg 7c5e79bb frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat50_learned_interactions_prereg.json"
SHA = "7c5e79bb063c6ce33ce10fa5c67a604678ed2a4d16c3236a094d786f934ebe3a"


def main() -> int:
    entry = {
        "exp_id": "p20_feat50_learned_interactions",
        "hypothesis_id": "H50-LEARNED-INTERACTIONS",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 50,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "FM interaction layer (rank 8) on 285-feature rich view, cross-fit 5-fold. Interaction features added to LGBM.",
        "results": {
            "base_rich_auc": 0.960953,
            "augmented_rich_fm_auc": 0.960843,
            "augmentation_delta": -0.000111,
            "spearman_aug_vs_champion_proxy": 0.979942,
            "equal_blend_auc_with_champion_proxy": 0.961547,
            "blend_delta_vs_aug": 0.000694,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.001010,
            "promising": False,
            "artifact": "reports/phase20/feat50_learned_interactions/feat50_learned_interactions.json",
        },
        "finding": (
            "FM interaction layer (rank 8) on rich view DEGRADES performance by -1.11e-4 "
            "(0.960953 -> 0.960843). High correlation with champion proxy (Spearman 0.9799). "
            "Blend with champion proxy +6.94e-4 but standalone degradation. NEGATIVE."),
        "verdict": "NEGATIVE_DEGRADATION",
        "closes_rows": [50],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "FM interaction layer (rank 8) on rich view: augmentation delta -1.11e-4 "
        "(0.960953 -> 0.960843). High correlation with champion (Spearman 0.9799). "
        "Blend +6.94e-4 but standalone degradation. NEGATIVE."
    )
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 50:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 50: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())