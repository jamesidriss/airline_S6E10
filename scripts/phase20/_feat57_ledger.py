"""FEAT 57 (low-rank interactions) ledger + catalogue. Prereg e815517d frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat57_lowrank_interactions_prereg.json"
SHA = "e815517dd97b9df2aea328cacc1c257a1a7ae5802f8f12a6ea5e9bf1df8a76cf"


def main() -> int:
    entry = {
        "exp_id": "p20_feat57_lowrank",
        "hypothesis_id": "H57-LOWRANK-INTERACTIONS",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 57,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "Cross-fit SVD rank 8 of 13 ratings (5-fold). 8 latent dims added to 21raw. LGBM on 21raw+8svd vs 21raw.",
        "results": {
            "base_21raw_auc": 0.953423,
            "augmented_21raw_8svd_auc": 0.956610,
            "augmentation_delta": 0.003190,
            "svd_only_auc": 0.925331,
            "spearman_aug_vs_champion_proxy": 0.928457,
            "equal_blend_auc_with_champion_proxy": 0.960420,
            "blend_delta_vs_aug": 0.003810,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.005243,
            "promising": True,
            "artifact": "reports/phase20/feat57_lowrank_interactions/feat57_lowrank_interactions.json",
        },
        "finding": (
            "Low-rank SVD interactions (rank 8, 5-fold cross-fit): augmentation delta "
            "+3.19e-3 (21raw 0.953423 → 0.956610). SVD features alone 0.925. Decorrelated "
            "with champion proxy (Spearman 0.928). Equal-logit blend with champion proxy "
            "+3.81e-3. PROMISING by +5e-5 bar, between FEAT 51 residuals (+3.94e-3) and "
            "FEAT 48 hierarchical (+3.94e-4). The SVD captures pairwise rating interactions "
            "effectively. PROMISING but FEAT 51 residuals remain the strongest lever."),
        "verdict": "PROMISING_INTERMEDIATE",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Cross-fit SVD rank-8 of 13 ratings: augmentation delta +3.19e-3 (0.953423 → "
        "0.956610). SVD features alone 0.925331. Decorrelated (Spearman 0.928). Blend "
        "with champion proxy +3.81e-3. PROMISING by +5e-5 bar, intermediate between "
        "FEAT 51 residuals (+3.94e-3) and FEAT 48 hierarchical (+3.94e-4). The SVD "
        "captures pairwise rating interactions effectively. PROMISING but FEAT 51 "
        "residuals remain the strongest feature block."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 57:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "PROMISING"
            print("row 57: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())