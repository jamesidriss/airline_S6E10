"""Close H18-B at wave 1 per the preregistered advance bar (CPU-only)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import file_sha256  # noqa: E402
from src.phase18 import registry  # noqa: E402
from src.phase18.fingerprint import git_head  # noqa: E402


def main() -> None:
    r = json.loads((ROOT / "reports/phase18/h18b_f0.json").read_text(encoding="utf-8"))
    assert r["status"] == "COMPLETE" and not r["advance_pass"]
    assert r["portfolio_delta"] < 0, "delta is not negative; do not close with this record"
    corr = r["stump_correlation"]["champion_portfolio"]["logit"]
    entry = {
        "exp_id": "phase18_h18b_close",
        "hypothesis_id": "H18-B",
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_head(),
        "status": "NEGATIVE_WAVE1_APPEND_HURTS",
        "portfolio_delta_f0": r["portfolio_delta"],
        "aux10_delta_via_append_f0": r["aux10_delta_via_append"],
        "stump_auc_f0": r["stump_auc"],
        "aux10_control_auc_f0": r["aux10_auc_control"],
        "champion_portfolio_logit_corr": corr,
        "advance_bar": r["advance_bar"],
        "evidence_report": "reports/phase18/h18b_f0.json",
        "evidence_report_sha256": file_sha256(ROOT / "reports/phase18/h18b_f0.json"),
        "note": (
            "H18-B CLOSED AT WAVE 1. The additive member (LightGBM boosted stumps, "
            "num_leaves=2/max_depth=1 = Friedman additive gradient boosting; the honest "
            "EBM/GAM stand-in on certified machinery) is WEAKER standalone "
            f"({r['stump_auc']:.6f} vs aux members ~0.9612) and APPENDING it to the "
            "mean-logit aux arm HURTS: aux10 "
            f"{r['aux10_delta_via_append']:+.3e}, portfolio {r['portfolio_delta']:+.3e} "
            "(below the predeclared +3.0e-5 bar in the negative direction). This "
            "quantitatively confirms the S6E7 cross-season lesson (additive structures "
            "add nothing once interactions are modeled) for this dataset and portfolio "
            "architecture: the mean-logit aggregate cannot absorb a weaker member even "
            "when its errors are somewhat less correlated (logit corr 0.9952 vs "
            "champion, Spearman 0.9682). Diversity clause: 0.9952 > 0.99, so NO "
            "champion-B evidence is claimed either. No folds 1-4, no promotion, "
            "champion v7 untouched."),
        "verdict": "CLOSE_NO_FURTHER_FITS",
    }
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")
    if not registry.lookup("phase18_h18b_close"):
        registry.append_record({
            "exp_id": "phase18_h18b_close", "fingerprint": "H18-B",
            "status": "NEGATIVE", "git": git_head(),
            "evidence": "reports/phase18/h18b_f0.json",
        })
    print(json.dumps({"closed": "H18-B", "portfolio_delta_f0": r["portfolio_delta"],
                      "aux10_delta_via_append_f0": r["aux10_delta_via_append"],
                      "verdict": entry["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
