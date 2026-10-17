"""Append items 91/94 (finalist-readiness audit) to ledger + update catalogue.

Prereg P20-FINALIST-READINESS sha 304179ee (frozen before results).
H91: seed variance ~0.3-1.2% of subset variance -> pipeline seed-stable.
H94: no preprocessing variant improves equal-logit blend (negative for blend
augmentation); champion v7 OOF not archived (documentation gap).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
REPORT = ROOT / "reports" / "phase20" / "finalist_readiness" / "finalist_readiness_audit.json"
PREREG = "configs/phase20/finalist_readiness_prereg.json"
SHA = "304179ee1bdec11361971261c427784ae9c77297fbdda9ebe99ddfcb8e29d574"

ENTRIES = [
    {
        "exp_id": "p20_finalist_h91_variance",
        "hypothesis_id": "H91-SEED-VS-SUBSET",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "DEP",
        "experiment_number": 91,
        "fold_scheme": "n/a (audit on saved primary OOF arrays)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "n/a",
        "results": {
            "lgbm_xt_f10": {"seed_var": 7.20e-9, "subset_var": 6.04e-7, "ratio": 0.012},
            "realmlp_full": {"seed_var": 1.80e-9, "subset_var": 6.32e-7, "ratio": 0.003},
            "conclusion": (
                "Seed variance is 0.3-1.2% of subset (FIT-composition) variance. "
                "The certified pipeline is seed-stable; fold-to-fold AUC variation "
                "(~6e-7 variance, ~8e-4 std) is data-composition noise, not a seed "
                "lottery. For the private split this means: do not expect a "
                "different seed to move the score; the data composition of the "
                "private rows is the dominant uncertainty."),
            "artifact": "reports/phase20/finalist_readiness/finalist_readiness_audit.json",
        },
        "finding": "H91: seed << subset variance (ratio 0.003-0.012). Pipeline is seed-stable.",
        "verdict": "DESCRIPTIVE",
        "closes_rows": [],
    },
    {
        "exp_id": "p20_finalist_h94_stability",
        "hypothesis_id": "H94-STABILITY-AUDIT",
        "date": "2026-10-18",
        "ts": datetime.now(timezone.utc).isoformat(),
        "family": "DEP",
        "experiment_number": 94,
        "fold_scheme": "n/a (audit on saved primary OOF arrays)",
        "prereg": {"path": PREREG, "sha256": SHA},
        "arch": "n/a",
        "results": {
            "reference": "sol_route_aux10_primary_final (OOF 0.961766)",
            "champion_oof_archived": False,
            "champion_oof_gap": (
                "Champion v7_sol_tabpfn_internal2_aux10 (OOF 0.961853) is NOT in the "
                "prediction store; only its test prediction is archived. A true "
                "champion-stability audit (DEP 94/96) requires archiving the champion "
                "OOF first."),
            "variant_result": (
                "No preprocessing/selection variant improves the equal-logit blend "
                "with the route+aux10 reference on any fold (min blend delta -1.02e-4 "
                "to -6.32e-4; min Spearman 0.952-0.984). NEGATIVE for blend "
                "augmentation -- consistent with the structural finding that rich-view "
                "members are decorrelated-but-weak."),
            "artifact": "reports/phase20/finalist_readiness/finalist_readiness_audit.json",
        },
        "finding": (
            "H94: equal-logit blend with the route+aux10 reference is not improvable "
            "by any saved preprocessing variant (negative). Champion v7 OOF not "
            "archived -- documentation gap to close before finalist lock."),
        "verdict": "NEGATIVE_BLEND_AUGMENTATION + DOC_GAP",
        "closes_rows": [],
    },
]


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["prereg_sha256"] == SHA

    with LEDGER.open("a", encoding="utf-8") as fh:
        for e in ENTRIES:
            fh.write(json.dumps(e) + "\n")
    print(f"appended {len(ENTRIES)} ledger entries")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    findings = {
        91: (
            "Variance decomposition on saved multi-seed OOF arrays (prereg "
            "P20-FINALIST-READINESS): seed variance 1.8e-9..7.2e-9 vs subset "
            "(FIT-composition) variance 6.0e-7..6.3e-7 -> seed/subset ratio "
            "0.003..0.012. The pipeline is seed-stable; fold-to-fold variation is "
            "data-composition noise (~8e-4 std), not a seed lottery. Descriptive, "
            "no go/no-go."
        ),
        94: (
            "Stability audit (prereg P20-FINALIST-READINESS): equal-logit blend of "
            "the route+aux10 reference with any saved preprocessing/selection "
            "variant degrades on at least one fold (min delta -1.0e-4..-6.3e-4, "
            "min Spearman 0.95-0.98) -> NEGATIVE for blend augmentation. "
            "Documentation gap: champion v7_sol_tabpfn_internal2_aux10 OOF is not "
            "in the prediction store (only test pred archived); a true champion "
            "stability audit needs it archived first. DEP 96 deferred."
        ),
    }
    for row in cat["experiments"]:
        num = row.get("experiment_number")
        if num in findings:
            row["phase20_finding"] = findings[num]
            row["fidelity_reached"] = "F1"
            print(f"row {num}: finding recorded (F1 audit)")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
