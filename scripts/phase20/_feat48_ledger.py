"""FEAT 48 (hierarchical ratings) ledger + catalogue. Prereg 84795f40 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat48_hierarchical_ratings_prereg.json"
SHA = "84795f40f0fa7031e3eb04627b6b36780ec5c5be3e7d156c5aef986ac461a356"


def main() -> int:
    entry = {
        "exp_id": "p20_feat48_hierarchical",
        "hypothesis_id": "H48-HIERARCHICAL-SIGNAL",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 48,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "Cross-fit multi-output HistGB (13 ratings from 8 non-rating covariates). "
                "Captures shared structure across ratings. LGBM on 21raw+13hier vs 21raw.",
        "results": {
            "base_21raw_auc": 0.953423,
            "augmented_21raw_13hier_auc": 0.953817,
            "augmentation_delta": 0.000394,
            "hier_only_auc": 0.872179,
            "spearman_aug_vs_champion_proxy": 0.916087,
            "equal_blend_auc_with_champion_proxy": 0.959957,
            "blend_delta_vs_aug": 0.00614,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.008044,
            "promising": True,
            "artifact": "reports/phase20/feat48_hierarchical_ratings/feat48_hierarchical_ratings.json",
        },
        "finding": (
            "Multi-output HistGB hierarchical features (13 columns) give a SMALL "
            "augmentation delta of +3.94e-4 (21raw 0.953423 → 0.953817). The hierarchical "
            "features alone are weak (AUC 0.872). Decorrelated with champion proxy "
            "(Spearman 0.916). Equal-logit blend with champion proxy gains +6.14e-3. "
            "PROMISING by the +5e-5 bar, but FAR weaker than FEAT 51 residuals "
            "(+3.94e-3). The multi-output approach captures some shared structure but "
            "the predictive power is limited. FEAT 51 (residuals) dominates."),
        "verdict": "PROMISING_BUT_WEAKER_THAN_FEAT51",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Multi-output HistGB hierarchical features (13 columns): augmentation delta "
        "+3.94e-4 (0.953423 → 0.953817). Hierarchical features alone weak (0.872). "
        "Decorrelated (Spearman 0.916). Blend with champion proxy +6.14e-3. PROMISING "
        "by +5e-5 bar but 10x weaker than FEAT 51 residuals (+3.94e-3). The "
        "multi-output shared-structure approach is dominated by the residual approach. "
        "PROMISING but not the primary lever."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 48:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "PROMISING"
            print("row 48: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())