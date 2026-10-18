"""FEAT 46 (learned similarity) ledger + catalogue. Prereg f416934a frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat46_learned_similarity_prereg.json"
SHA = "f416934a53c5c0c540715ed9902120d85e9af901278571896b89c427a3bd6d69"


def main() -> int:
    entry = {
        "exp_id": "p20_feat46_learned_similarity",
        "hypothesis_id": "H46-LEARNED-SIMILARITY",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 46,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "Contrastive Siamese network (embed_dim=8) on 21 covariates, cross-fit 5-fold. Learned embeddings added to LGBM.",
        "results": {
            "base_21raw_auc": 0.953423,
            "augmented_21raw_8emb_auc": 0.953174,
            "augmentation_delta": -0.000249,
            "spearman_aug_vs_champion_proxy": 0.904236,
            "equal_blend_auc_with_champion_proxy": 0.959904,
            "blend_delta_vs_aug": 0.006726,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.008679,
            "promising": False,
            "artifact": "reports/phase20/feat46_learned_similarity/feat46_learned_similarity.json",
        },
        "finding": (
            "Contrastive Siamese network (embed_dim=8) DEGRADES base LGBM by -2.49e-4 "
            "(0.953423 -> 0.953174). Embeddings decorrelated (Spearman 0.904) but "
            "standalone degradation. Blend with champion proxy +6.73e-3 but standalone "
            "dominates. The learned similarity metric does not extract LGBM-useful "
            "signal beyond raw covariates. NEGATIVE."),
        "verdict": "NEGATIVE_DEGRADATION",
        "closes_rows": [46],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Learned similarity metric (contrastive Siamese, embed_dim=8, cross-fit): "
        "augmentation delta -2.49e-4 (0.953423 -> 0.953174). NEGATIVE - degradation. "
        "Decorrelated (Spearman 0.904) but standalone degradation dominates. "
        "Blend with champion proxy +6.73e-3 but standalone dominates. NEGATIVE."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 46:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 46: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())