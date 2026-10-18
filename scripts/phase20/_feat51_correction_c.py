"""Append-only correction: FEAT51 blend is NOT a champion improvement.

Run after feat51_paired_integration_v2.json exists. Never rewrites history.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "phase20" / "feat51_paired_integration" / "feat51_paired_integration_v2.json"
CORR = ROOT / "reports" / "phase20" / "CORRECTIONS_GEN80_FEAT51_FEAT57_DEP96.md"

CHAMPION = 0.9618529228157235

def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    blend = rep["results"]["blend_with_champion_proxy"]["equal_blend_auc"]
    treatment = rep["results"]["treatment_oof_auc"]
    delta = rep["results"]["blend_with_champion_proxy"]["blend_delta_vs_treatment"]

    rep["correction_20261018_c"] = {
        "issue": "The +0.00059667 blend delta is measured against the FEAT51 "
                 "treatment standalone model (0.961079), NOT against champion A v7.",
        "blend_auc": blend,
        "treatment_standalone_auc": treatment,
        "champion_v7_oof": CHAMPION,
        "blend_minus_champion": round(blend - CHAMPION, 10),
        "correct_statement": (
            "Equal-logit blend of FEAT51-treatment with the route reference reaches "
            f"{blend:.6f}, which is BELOW champion A v7 ({CHAMPION:.10f}) by "
            f"{blend - CHAMPION:+.6f}. The +5.97e-4 gain is relative to the weaker "
            "FEAT51 standalone model only. FEAT51 does NOT improve the champion "
            "portfolio and is NOT promoted on this basis."),
        "promoted": False,
    }
    REPORT.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")

    note = ("\n## FEAT51 blend correction (2026-10-18, append-only, c)\n\n"
            f"- Corrected blend AUC: **{blend:.6f}** (FEAT51-treatment + route reference, equal logit).\n"
            f"- Champion A v7 OOF: **{CHAMPION:.10f}**. Blend minus champion: **{blend - CHAMPION:+.6f}**.\n"
            f"- The reported +{delta:.8f} delta is vs the FEAT51 standalone model "
            f"({treatment:.6f}), not vs v7. FEAT51 is NOT a champion improvement and is not promoted.\n")
    with CORR.open("a", encoding="utf-8") as fh:
        fh.write(note)
    print(f"appended correction_c to {REPORT.relative_to(ROOT).as_posix()}")
    print(f"appended section to {CORR.relative_to(ROOT).as_posix()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
