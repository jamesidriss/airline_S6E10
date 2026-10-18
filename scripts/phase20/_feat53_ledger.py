"""FEAT 53 (conditional clustering) ledger + catalogue. Prereg 13d0c3e2 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat53_conditional_clustering_prereg.json"
SHA = "13d0c3e2d21ffe6045e6023a01af1259e137f15d3517e45ce9728330e2ee9922"


def main() -> int:
    entry = {
        "exp_id": "p20_feat53_conditional_clustering",
        "hypothesis_id": "H53-CONDITIONAL-CLUSTERING",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 53,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "K-means (8 clusters) on 21 covariates, cross-fit 5-fold. Cluster distances as features. LGBM on 21raw+8cluster vs 21raw.",
        "results": {
            "base_21raw_auc": 0.953423,
            "augmented_21raw_8cluster_auc": 0.953465,
            "augmentation_delta": 4.23e-05,
            "spearman_aug_vs_champion_proxy": 0.909725,
            "equal_blend_auc_with_champion_proxy": 0.959888,
            "blend_delta_vs_aug": 0.00642,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.008388,
            "promising": False,
            "artifact": "reports/phase20/feat53_conditional_clustering/feat53_conditional_clustering.json",
        },
        "finding": (
            "K-means clustering (8 clusters, cross-fit): augmentation delta +4.23e-5 "
            "(0.953423 -> 0.953465). Below +5e-5 threshold. Decorrelated (Spearman 0.910). "
            "Blend with champion proxy +6.42e-3 but standalone below threshold. NEGATIVE."),
        "verdict": "NEGATIVE_BELOW_THRESHOLD",
        "closes_rows": [53],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "K-means clustering (8 clusters, cross-fit): augmentation delta +4.23e-5 "
        "(0.953423 -> 0.953465). Below +5e-5 threshold. Decorrelated (Spearman 0.910). "
        "Blend with champion proxy +6.42e-3 but standalone below threshold. NEGATIVE."
    )
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 53:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 53: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())