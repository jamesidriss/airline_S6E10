"""FEAT 54 (smoothed categorical) ledger + catalogue."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat54_smoothed_categorical_prereg.json"
SHA = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()

def main() -> int:
    entry = {
        "exp_id": "p20_feat54_smoothed_categorical",
        "hypothesis_id": "H54-SMOOTHED-CATEGORICAL",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 54,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": "configs/phase20/feat54_smoothed_categorical_prereg.json", "sha256": SHA},
        "arch": "Bayesian shrinkage target encoding for 4 categorical features, cross-fit 5-fold. LGBM on 21raw+4smooth vs 21raw.",
        "results": {
            "base_21raw_auc": 0.953423,
            "augmented_21raw_4smooth_auc": 0.953508,
            "augmentation_delta": 8.49e-05,
            "spearman_aug_vs_champion_proxy": 0.911794,
            "equal_blend_auc_with_champion_proxy": 0.959911,
            "blend_delta_vs_aug": 0.00640,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.008345,
            "promising": True,
            "artifact": "reports/phase20/feat54_smoothed_categorical/feat54_smoothed_categorical.json",
        },
        "finding": (
            "Smoothed categorical (Bayesian shrinkage, 4 features): augmentation delta +8.49e-5 "
            "(0.953423 -> 0.953508). PROMISING by +5e-5 threshold. Decorrelated (Spearman 0.912). "
            "Blend with champion proxy +6.40e-3. PROMISING as feature block for integration."
        ),
        "verdict": "PROMISING_FEATURE_BLOCK",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Smoothed categorical (Bayesian shrinkage, 4 features): augmentation delta +8.49e-5 "
        "(0.953423 -> 0.953508). PROMISING by +5e-5 threshold. Decorrelated (Spearman 0.912). "
        "Blend with champion proxy +6.40e-3. PROMISING as feature block for integration."
    )
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 54:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "PROMISING"
            print("row 54: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
