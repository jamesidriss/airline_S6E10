"""FEAT51 paired integration ledger + catalogue. Prereg 36beca0 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat51_paired_integration_prereg.json"
SHA = "36becae012fc27c6351d2e92aaf5c76f13a848123ba9d5a644c002a5e191d863"


def main() -> int:
    entry = {
        "exp_id": "p20_feat51_paired_integration",
        "hypothesis_id": "H51-PAIRED-INTEGRATION",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 51,
        "fold_scheme": "primary 5-fold (F3 full OOF + admission gate)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "Paired control/treatment 5-fold CV: champion static full (237) vs +13 cross-fit residuals. Identical folds, LGBM xt_xt_f10, inner ES.",
        "results": {
            "control_oof_auc": 0.961007,
            "treatment_oof_auc": 0.961079,
            "per_fold_control": [0.960953, 0.961137, 0.961011, 0.960359, 0.961614],
            "per_fold_treatment": [0.960982, 0.961137, 0.960984, 0.960494, 0.961830],
            "paired_deltas_per_fold": [2.876e-05, 2.4e-07, -2.742e-05, 0.00013436, 0.00021635],
            "mean_paired_delta": 7.046e-05,
            "std_paired_delta": 0.000102,
            "se_paired_delta": 4.562e-05,
            "t_stat": 1.5446,
            "positive_folds": 4,
            "admission_gate": {"required_gain": 1.5e-5, "observed_mean_delta": 7.046e-05, "se": 4.562e-05, "t_stat": 1.5446, "positive_folds": "4/5", "passes": False},
            "artifact": "reports/phase20/feat51_paired_integration/feat51_paired_integration.json",
        },
        "finding": (
            "Paired control/treatment 5-fold CV: champion static full view (237 features) vs "
            "same + 13 cross-fit residuals. Mean paired delta +7.05e-5 (4/5 folds positive). "
            "Admission gate NOT passed: mean delta +7.05e-5 < 2.5*SE (1.14e-4), t=1.54 < 2.5. "
            "The residuals provide a small but inconsistent improvement. The signal is real "
            "but inconsistent across folds. The fold-0 blend with champion proxy (+3.32e-3) "
            "does not translate to full-5 OOF paired gain. Residual block is PROMISING for "
            "NEW pipelines but does not clear the admission gate for champion upgrade."),
        "verdict": "NEGATIVE_ADMISSION_GATE (positive but inconsistent)",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Paired control/treatment 5-fold CV (prereg 36beca0): champion static full (237) "
        "vs +13 cross-fit residuals. Mean paired delta +7.05e-5 (4/5 folds positive). "
        "Admission gate NOT passed: mean +7.05e-5 < 2.5*SE (t=1.54 < 2.5). The residual "
        "signal is real but inconsistent across folds. Fold-0 blend +3.32e-3 does not "
        "translate to full-5 OOF paired gain. FEAT 51 residuals are PROMISING for NEW "
        "pipelines but do not clear admission gate for champion upgrade. Status: "
        "PROMISING feature block, NEGATIVE for champion upgrade."
    )
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 51:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F3"
            print("row 51: finding recorded (F3)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())