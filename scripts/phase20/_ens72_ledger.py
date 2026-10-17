"""ENS 72 (covariate gating) ledger + catalogue. Prereg ae9c07e5 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/ens72_covariate_gating_prereg.json"
SHA = "ae9c07e5d05dad9ab2e5d7224e1e98b5db1edcab1a0063a70c2f3b19f06e58be"


def main() -> int:
    entry = {
        "exp_id": "p20_ens72_covariate_gating",
        "hypothesis_id": "H72-GATE-BEATS-EQUAL",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "ENS",
        "experiment_number": 72,
        "fold_scheme": "primary fold-0 (F2 screen, saved OOF arrays)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "Cross-fit L2 logistic gate on 21 raw covariates (ordinal-encoded categoricals, median-imputed); target=route_correct; train folds 1-4, apply fold-0",
        "results": {
            "route_auc": 0.961727,
            "rich_auc": 0.961501,
            "equal_blend_auc": 0.961725,
            "gated_blend_auc": 0.961735,
            "gated_minus_route": 7.99e-6,
            "gated_minus_equal": 1.01e-5,
            "promising": False,
            "decision_rule": "PROMISING if gated >= route + 1e-4",
            "artifact": "reports/phase20/ens72_covariate_gating/ens72_covariate_gating.json",
        },
        "finding": (
            "Cross-fit covariate gate beats the equal blend by only +1.01e-5 and route "
            "alone by +7.99e-6 -- far below the +1e-4 bar. NEGATIVE: covariates cannot "
            "predict when route vs rich is correct. The errors are concentrated on the "
            "same hard rows (GEN 80 hardest decile AUC 0.557), so no covariate signal "
            "separates the candidates. Consistent with the structural finding."),
        "verdict": "NEGATIVE_GATE_CANNOT_GENERALIZE",
        "closes_rows": [72],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "Covariate-gated blender (prereg P20-ENS72-COVARIATE-GATING): cross-fit L2 "
        "logistic gate on 21 raw covariates (ordinal-encoded, median-imputed), target "
        "= route-correct, trained folds 1-4, applied fold-0. Gated blend 0.961735 vs "
        "route 0.961727 (+7.99e-6) vs equal blend 0.961725 (+1.01e-5). NEGATIVE: far "
        "below the +1e-4 bar. Covariates cannot predict when route vs rich is correct "
        "-- errors concentrate on the same hard rows. Covariate gating closed; the "
        "+3.14e-3 oracle headroom (GEN 80) is not capturable this way."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 72:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["next_action"] = "CLOSED by direct fold-0 measurement"
            print("row 72: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
