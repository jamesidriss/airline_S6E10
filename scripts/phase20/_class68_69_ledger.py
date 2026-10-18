"""CLASS 68/69 ledger + catalogue. Preregs a2767481 / 65dfca82 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"


def main() -> int:
    entries = [
        {
            "exp_id": "p20_class68_mixed_factor",
            "hypothesis_id": "H68-MIXED-FACTOR",
            "date": "2026-10-18",
            "ts": datetime.now(timezone.utc).isoformat(),
            "family": "CLASS",
            "experiment_number": 68,
            "fold_scheme": "primary fold-0 (F2 screen)",
            "prereg": {"path": "configs/phase20/class68_mixed_factor_prereg.json", "sha256": "a27674818508d36e42449f1ccd3e1ffe7d13d3fec772e4ea1770fe193937f7c2"},
            "arch": "Factorization Machine: 8-dim embeddings for 4 categorical features + linear terms on 281 numeric features. Cross-fit 5-fold on 285-feature rich view.",
            "results": {
                "auc": 0.958391, "spearman_vs_champion_proxy": 0.942157,
                "equal_blend_auc": 0.960831, "blend_delta": 0.00244,
                "champion_ref": 0.961853, "port_delta_vs_champion": -0.003462,
                "promising": False,
                "artifact": "reports/phase20/class68_mixed_factor/class68_mixed_factor.json",
            },
            "finding": (
                "Factorization Machine (8-dim embeddings for 4 categoricals + linear): fold-0 AUC "
                "0.958391, port_delta -3.46e-3 vs champion. Decorrelated (Spearman 0.942). "
                "Equal-logit blend with champion proxy +2.44e-3 but standalone is weaker. "
                "NEGATIVE for fold-0 screen. The FM embeddings don't capture sufficient "
                "new signal beyond the rich view's existing categorical encodings."),
            "verdict": "NEGATIVE_STANDALONE",
            "closes_rows": [68],
        },
        {
            "exp_id": "p20_class69_rule_ensembles",
            "hypothesis_id": "H69-RULE-ENSEMBLES",
            "date": "2026-10-18",
            "ts": datetime.now(timezone.utc).isoformat(),
            "family": "CLASS",
            "experiment_number": 69,
            "fold_scheme": "primary fold-0 (F2 screen)",
            "prereg": {"path": "configs/phase20/class69_rule_ensembles_prereg.json", "sha256": "65dfca8266ccf58ef62b53924788494c46ee75dbea63ae14eb69d79d581a59f4"},
            "arch": "Conditional rule ensembles: 50 shallow DecisionTrees (max_depth=3) on random 30% feature subspaces, weighted by inner-val AUC. Cross-fit 5-fold on rich view.",
            "results": {
                "auc": 0.954065, "spearman_vs_champion_proxy": 0.883995,
                "equal_blend_auc": 0.960085, "blend_delta": 0.00602,
                "champion_ref": 0.961853, "port_delta_vs_champion": -0.007788,
                "promising": False,
                "artifact": "reports/phase20/class69_rule_ensembles/class69_rule_ensembles.json",
            },
            "finding": (
                "Conditional rule ensembles (50 shallow trees, max_depth=3, 30% subspaces): "
                "fold-0 AUC 0.954065, port_delta -7.79e-3 vs champion. Decorrelated (Spearman 0.884). "
                "Equal-logit blend with champion proxy +6.02e-3 but standalone is significantly "
                "weaker. NEGATIVE for fold-0 screen. The rule ensemble doesn't extract sufficient "
                "signal from the rich view."),
            "verdict": "NEGATIVE_STANDALONE",
            "closes_rows": [69],
        },
    ]
    with LEDGER.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    print("appended 2 ledger entries")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    findings = {
        68: (
            "Factorization Machine (8-dim embeddings for 4 categoricals + linear): fold-0 AUC "
            "0.958391, port_delta -3.46e-3 vs champion. Decorrelated (Spearman 0.942). "
            "Equal-logit blend with champion proxy +2.44e-3 but standalone is weaker. NEGATIVE."
        ),
        69: (
            "Conditional rule ensembles (50 shallow trees, max_depth=3, 30% subspaces): "
            "fold-0 AUC 0.954065, port_delta -7.79e-3 vs champion. Decorrelated (Spearman 0.884). "
            "Equal-logit blend +6.02e-3 but standalone degradation. NEGATIVE."
        ),
    }
    for num, finding in findings.items():
        for row in cat["experiments"]:
            if row.get("experiment_number") == num:
                row["phase20_finding"] = finding
                row["fidelity_reached"] = "F2"
                row["status"] = "COMPLETE_NEGATIVE"
                print(f"row {num}: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())