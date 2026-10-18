"""FEAT 55 (leaf embeddings) ledger + catalogue. Prereg 7dc3beb4 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/feat55_leaf_embeddings_prereg.json"
SHA = "7dc3beb48a3069de4bcd845de2a70292c5805c8420ba723e6f9963cb16509456"


def main() -> int:
    entry = {
        "exp_id": "p20_feat55_leaf_embeddings",
        "hypothesis_id": "H55-LEAF-EMBEDDINGS",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "FEAT",
        "experiment_number": 55,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "LGBM leaf embeddings (xt_xt_f10) -> second family (LogReg/MLP on leaf indices). "
                "Cross-fit: LGBM on folds 1-4, leaf indices for fold-0, second family trained folds 1-4.",
        "results": {
            "base_rich_lgbm_auc": 0.960905,
            "leaf_logreg_auc": 0.957962,
            "leaf_mlp_auc": 0.958501,
            "best_second_family": "mlp",
            "best_second_auc": 0.958501,
            "delta_vs_base": -0.002404,
            "spearman_vs_champion_proxy": 0.945036,
            "equal_blend_auc_with_champion_proxy": 0.962984,
            "blend_delta_vs_second": 0.002083,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.003352,
            "promising": False,
            "artifact": "reports/phase20/feat55_leaf_embeddings/feat55_leaf_embeddings.json",
        },
        "finding": (
            "LGBM leaf embeddings (6000 trees, xt_xt_f10) fed into LogReg/MLP second family "
            "DEGRADES performance: best second family (MLP) AUC 0.958501 vs base rich LGBM "
            "0.960905 (delta -2.40e-3). DECORRELATED (Spearman 0.945) but standalone "
            "degradation dominates. Equal-logit blend with champion proxy gains +2.08e-3 "
            "but standalone is weaker. The leaf embedding representation does not extract "
            "LGBM-useful signal beyond what the LGBM already captures internally. NEGATIVE "
            "for this second-family approach."),
        "verdict": "NEGATIVE_DEGRADATION",
        "closes_rows": [55],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "LGBM leaf embeddings (6000 trees, xt_xt_f10) -> LogReg/MLP second family: "
        "best second family (MLP) AUC 0.958501 vs base rich LGBM 0.960905 (delta -2.40e-3). "
        "NEGATIVE — second family is weaker than the base LGBM. Decorrelated (Spearman 0.945) "
        "but standalone degradation dominates. Equal-logit blend with champion proxy +2.08e-3 "
        "doesn't compensate. The leaf embedding representation adds no value beyond the "
        "LGBM's own internal representation. CLOSED."
    )
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 55:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 55: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())