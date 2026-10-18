"""FEAT54 v1 -> INVALID_PROTOCOL (append-only), then freeze strict v2 prereg.

Step 1 (this script): append correction field to the v1 report; never rewrite.
Step 2: v2 prereg is hashed + committed BEFORE any v2 result exists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "reports" / "phase20" / "feat54_smoothed_categorical" / "feat54_smoothed_categorical.json"
PREREG_V2 = ROOT / "configs" / "phase20" / "feat54_strict_prereg.json"


def main() -> int:
    rep = json.loads(V1.read_text(encoding="utf-8"))
    prereg_sha_v1 = hashlib.sha256(
        (ROOT / "configs" / "phase20" / "feat54_smoothed_categorical_prereg.json").read_bytes()
    ).hexdigest()
    rep["status"] = "INVALID"
    rep["correction_20261018_d"] = {
        "classification": "INVALID_PROTOCOL (historical result preserved, not deleted)",
        "defect_1_protocol": (
            "build_smoothed_features() cross-fit encodings for every row k over folds != k. "
            "Consequently the outer-FIT rows (folds 1-4, used as LGBM training rows for the "
            "fold-0 screen) received encodings whose statistics were computed from folds "
            "{0,2,3,4}, {0,1,3,4}, ... i.e. INCLUDING the outer evaluation fold's (fold-0) "
            "satisfaction labels. The evaluation fold's labels leaked into the training "
            "features -> contaminated fold-0 evaluation."),
        "defect_2_provenance": "prereg_sha256 was 'TODO'; corrected here to the verified value.",
        "defect_3_decision_rule": (
            "Report applied augmentation_delta >= +5e-5 (raw 21-model delta) instead of the "
            "frozen prereg rule: operational port_delta >= +5e-5 AND corr_vs_portfolio < 0.995."),
        "prereg_sha256_verified_v1": prereg_sha_v1,
        "reported_augmentation_delta": rep.get("fold0_results", {}).get("augmentation_delta"),
        "superseded_by": "configs/phase20/feat54_strict_prereg.json (strict outer-FIT-only, "
                         "inner-cross-fitted re-measurement, not yet run at time of this note)",
        "promoted": False,
    }
    V1.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("v1 report marked INVALID (append-only correction_d)")
    print(f"verified v1 prereg sha256 = {prereg_sha_v1}")
    if PREREG_V2.exists():
        print(f"v2 prereg already present: {PREREG_V2.name} (leave untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
