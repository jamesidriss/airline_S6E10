"""FEAT 52 (rating factorization) ledger + catalogue. Prereg c53890fd frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat52_rating_factorization_prereg.json"
SHA = "c53890fd05de98a989d599d1a55463556d480cb145fb110dd18de75dd64b0acc"


def main() -> int:
    entry = {
        "exp_id": "p20_feat52_factorization",
        "hypothesis_id": "H52-FACTORIZATION-SIGNAL",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 52,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "Supervised low-rank factorization (rank 8) of 13 ratings, cross-fit 5-fold, "
                "PyTorch/AdamW logistic head. Passenger embeddings (8 dims) added to LGBM.",
        "results": {
            "base_21raw_auc": 0.953423,
            "augmented_21raw_8emb_auc": 0.950533,
            "augmentation_delta": -0.00289,
            "spearman_aug_vs_champion_proxy": 0.884804,
            "equal_blend_auc_with_champion_proxy": 0.958611,
            "blend_delta_vs_aug": 0.00808,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.01132,
            "promising": False,
            "artifact": "reports/phase20/feat52_rating_factorization/feat52_rating_factorization.json",
        },
        "finding": (
            "Supervised low-rank factorization (rank 8) DEGRADES the base LGBM by -2.89e-3 "
            "(0.953423 → 0.950533). Embeddings are decorrelated (Spearman 0.885) and the "
            "equal-logit blend with champion proxy gains +8.08e-3, but the standalone "
            "degradation makes this NEGATIVE for the feature block. The embeddings may "
            "be too low-dimensional or the supervised factorization objective doesn't "
            "extract the right signal for LGBM. The decorrelation + blend potential "
            "suggests they could be a second-family member, but the feature block itself "
            "is NEGATIVE. Closed."),
        "verdict": "NEGATIVE_DEGRADATION",
        "closes_rows": [52],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Supervised low-rank factorization (rank 8): augmentation delta -2.89e-3 "
        "(0.953423 → 0.950533). NEGATIVE — the embeddings degrade LGBM performance. "
        "Decorrelated (Spearman 0.885) and blend with champion proxy +8.08e-3, but "
        "standalone degradation dominates. The supervised factorization objective "
        "does not extract LGBM-useful signal from the ratings. CLOSED."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 52:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 52: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())