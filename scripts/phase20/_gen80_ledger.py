"""GEN 80 (AUC ceiling) ledger + catalogue. Prereg 82af441e frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/gen80_auc_ceiling_prereg.json"
SHA = "82af441ed577d5cdf2dade569516568de529108a29be6c63b16e688ac39fcc24"


def main() -> int:
    entry = {
        "exp_id": "p20_gen80_auc_ceiling",
        "hypothesis_id": "H80-ORACLE-BLEND",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "GEN",
        "experiment_number": 80,
        "fold_scheme": "n/a (analysis on saved primary OOF arrays)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "n/a",
        "results": {
            "route_reference_auc": 0.961766,
            "rich_blend_auc": 0.961509,
            "equal_logit_blend_auc": 0.961756,
            "oracle_ceiling_auc": 0.964906,
            "headroom_oracle_minus_champion": 0.003140,
            "stretch_target": 0.96274,
            "stretch_gap": 0.000974,
            "stretch_reachable_by_oracle_blend": True,
            "hardest_decile_auc": 0.557065,
            "artifact": "reports/phase20/gen80_auc_ceiling/gen80_auc_ceiling.json",
        },
        "finding": (
            "The oracle (label-cheating) blend ceiling of the route reference and the "
            "59-member rich blend is 0.964906 -- headroom +3.14e-3 above the champion, "
            "far above the stretch gap (+0.97e-4). So the stretch is THEORETICALLY "
            "reachable by a better blender over these two candidates. But the oracle "
            "is not a deployable model: a real blender must generalize from "
            "covariates, not labels. The hardest decile (AUC 0.557) shows errors "
            "concentrate on few rows -- exactly where a covariate-gated blender "
            "(ENS 72) could help. All tried real blenders (equal, softmax, "
            "family-balanced) were <= champion -> the binding constraint is blender "
            "generalization, not blend headroom."),
        "verdict": "DESCRIPTIVE_STRETCH_REACHABLE_VIA_BLEND",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "AUC ceiling analysis (prereg P20-GEN80-AUC-CEILING): oracle (label-cheating) "
        "blend of route reference (0.961766) + 59-member rich blend (0.961509) = "
        "0.964906, headroom +3.14e-3 >> stretch gap +0.97e-4 -> stretch THEORETICALLY "
        "reachable by a better blender. Real blenders (equal/softmax/family-balanced) "
        "all <= champion -> constraint is blender generalization, not headroom. "
        "Hardest decile AUC 0.557 (errors concentrate on few rows). Next lever: "
        "covariate-gated blender (ENS 72), fold-safe, could generalize."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 80:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F1"
            print("row 80: finding recorded (F1)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
