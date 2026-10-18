"""CLASS 62 (GAM splines) ledger + catalogue. Prereg e9ec6d74 frozen before results."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
PREREG = "configs/phase20/class62_gam_splines_prereg.json"
SHA = "e9ec6d74fe322086211bae373e9fe8f8912079438335fd497b07891e66c10ca8"


def main() -> int:
    entry = {
        "exp_id": "p20_class62_gam_splines",
        "hypothesis_id": "H62-GAM-SPLINES",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "CLASS",
        "experiment_number": 62,
        "fold_scheme": "primary fold-0 (F2 screen)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "GAM: SplineTransformer (degree=3, n_knots=10, 2607 features) -> LogisticRegression(C=1.0). Cross-fit 5-fold on 285-feature rich view.",
        "results": {
            "auc": 0.958893,
            "spearman_vs_champion_proxy": 0.94849,
            "equal_blend_auc": 0.960953,
            "blend_delta": 0.002063,
            "champion_ref": 0.961853,
            "port_delta_vs_champion": -0.002960,
            "promising": False,
            "artifact": "reports/phase20/class62_gam_splines/class62_gam_splines.json",
        },
        "finding": (
            "GAM with spline basis (SplineTransformer degree=3, n_knots=10 -> 2607 features) "
            "+ LogisticRegression: fold-0 AUC 0.958893, port_delta −2.96e-3 vs champion. "
            "Decorrelated (Spearman 0.948). Equal-logit blend with champion proxy gains "
            "+2.06e-3 but standalone is weaker. NEGATIVE for fold-0 screen. The spline "
            "basis on 285 features creates 2607 features but the non-linear univariate "
            "effects don't add sufficient signal beyond the rich view's existing TE/token/"
            "external blocks. NEGATIVE for fold-0 screen."),
        "verdict": "NEGATIVE_STANDALONE",
        "closes_rows": [62],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended 1 ledger entry")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    finding = (
        "GAM splines (SplineTransformer degree=3, n_knots=10, 2607 features) + "
        "LogisticRegression: fold-0 AUC 0.958893, port_delta −2.96e-3 vs champion "
        "0.961853. Decorrelated (Spearman 0.948) but standalone degradation. Blend "
        "+2.06e-3 doesn't compensate. The 2607 spline features on the rich view don't "
        "extract sufficient new signal. NEGATIVE for fold-0 screen."
    )
    for row in cat["experiments"]:
        if row.get("experiment_number") == 62:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
            print("row 62: finding recorded (F2)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())