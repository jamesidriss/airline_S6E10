"""FEAT57 supervised low-rank ledger + catalogue. Prereg 495cf036 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat57_supervised_lowrank_prereg.json"
SHA = "495cf0366b1d17168f3a0632ea2ea0ac5c834ef6389d54aa48db97b7d4b12bfd"


def main() -> int:
    entry = {
        "exp_id": "p20_feat57_supervised_lowrank",
        "hypothesis_id": "H57-SUPERVISED-LOWRANK",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 57,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": "495cf0366b1d17168f3a0632ea2ea0ac5c834ef6389d54aa48db97b7d4b12bfd"},
        "arch": "SUPERVISED low-rank factorization (rank 8) of 13 ratings with satisfaction supervision, cross-fit 5-fold, PyTorch/AdamW. Distinct from unsupervised SVD (FEAT57-LOWRANK).",
        "results": {
            "base_21raw_auc": 0.953423,
            "augmented_21raw_8emb_auc": 0.951334,
            "augmentation_delta": -0.002089,
            "emb_only_auc": 0.717451,
            "spearman_aug_vs_champion_proxy": 0.878839,
            "equal_blend_auc_with_champion_proxy": 0.958611,
            "blend_delta_vs_aug": 0.007277,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.010519,
            "promising": False,
            "artifact": "reports/phase20/feat57_supervised_lowrank/feat57_supervised_lowrank.json",
        },
        "finding": (
            "SUPERVISED low-rank factorization (rank 8) DEGRADES base LGBM by -2.09e-3 "
            "(0.953423 -> 0.951334). Embeddings alone very weak (AUC 0.717). Decorrelated "
            "(Spearman 0.879) and blend with champion proxy gains +7.31e-3, but standalone "
            "degradation dominates. The supervised factorization objective does not extract "
            "LGBM-useful signal from the ratings. The unsupervised SVD (FEAT57-LOWRANK, "
            "+3.19e-3) outperforms this supervised mechanism. The missing mechanism is "
            "now measured and NEGATIVE."),
        "verdict": "NEGATIVE_DEGRADATION (supervised mechanism inferior to unsupervised SVD)",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "SUPERVISED low-rank factorization (rank 8, cross-fit): augmentation delta "
        "-2.09e-3 (0.953423 -> 0.951334). Embeddings alone AUC 0.717. Decorrelated "
        "(Spearman 0.879). Blend with champion proxy +7.31e-3 but standalone "
        "degradation dominates. NEGATIVE - the supervised factorization objective "
        "does not extract LGBM-useful signal. The unsupervised SVD (FEAT57-LOWRANK, "
        "+3.19e-3) outperforms this supervised mechanism. Both mechanisms now measured; "
        "unsupervised SVD is the promising low-rank approach."
    )
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 57:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 57: finding recorded (F2) - supervised mechanism NEGATIVE")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())