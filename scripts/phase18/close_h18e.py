"""Close H18-E at wave 1 with the measured admission-level evidence (CPU-only)."""
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
    ev = json.loads((ROOT / "reports/phase18_h18e_f0_evaluation.json").read_text(encoding="utf-8"))
    b2 = json.loads((ROOT / "reports/phase17_B2_f0_evaluation.json").read_text(encoding="utf-8"))
    b1 = json.loads((ROOT / "reports/phase17_B1_f0_evaluation.json").read_text(encoding="utf-8"))
    cross = b2["portfolio_delta"] - b1["portfolio_delta"]
    assert abs(cross - ev["portfolio_delta"]) < 1e-15, "two-path verification failed"

    route_delta = ev["standalone_delta"]
    port_delta = ev["portfolio_delta"]
    conversion = port_delta / route_delta
    note = (
        "H18-E CLOSED AT WAVE 1 (fold 0, primary, certified artifacts; no GPU beyond existing). "
        f"route4-route2 standalone = {route_delta:+.6e} AUC (real, consistent with the member-level "
        f"audit), but the admission-level portfolio4-v7 delta = {port_delta:+.6e} on fold 0 — "
        f"conversion ratio {conversion:.3f}. The 0.5 route + 0.5 aux10 logit blend dilutes "
        "route-only improvements almost entirely because route and aux10 share 0.9977 correlation: "
        "the portfolio ranking is dominated by their shared component. Expected mean portfolio gain "
        "even under perfect route-delta replication is ~0.4e-5, an order of magnitude below the "
        "1.5e-5 admission gate; the phase17 continuation gate (mean of first 3 folds >= 3e-5) "
        "would also block fold 3. Folds 1-4 would cost ~4.4h GPU for a near-certain gate failure. "
        "Verified by two independent paths: (1) direct recomposition vs champion fold portfolio, "
        "(2) difference of the two certified phase17 vs-v6 fold evaluations "
        f"(B2 {b2['portfolio_delta']:+.6e} - B1 {b1['portfolio_delta']:+.6e} = {cross:+.6e}). "
        "Lesson recorded: route-level gains must be re-measured at the PORTFOLIO level before "
        "any GPU replication budget is granted; the conversion ratio is change-dependent (7% here, "
        "~27% for the v7 internal2 change)."
    )
    entry = {
        "exp_id": "phase18_h18e_close",
        "hypothesis_id": "H18-E",
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_head(),
        "status": "NEGATIVE_ADMISSION_LEVEL_CLOSE",
        "fold0_standalone_route_delta": route_delta,
        "fold0_portfolio_delta_vs_v7": port_delta,
        "conversion_ratio": conversion,
        "cross_check_delta": cross,
        "evidence_report": "reports/phase18_h18e_f0_evaluation.json",
        "evidence_report_sha256": file_sha256(ROOT / "reports/phase18_h18e_f0_evaluation.json"),
        "gpu_burned_this_closure": 0.0,
        "note": note,
        "verdict": "CLOSE_NO_FURTHER_GPU_FOLDS_1_4_NEVER_RUN",
    }
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")
    prior = registry.lookup("phase18_h18e_close")
    if not prior:
        registry.append_record({
            "exp_id": "phase18_h18e_close", "fingerprint": "H18-E",
            "status": "NEGATIVE", "git": git_head(),
            "evidence": "reports/phase18_h18e_f0_evaluation.json",
        })
    print(json.dumps({"closed": "H18-E", "fold0_route_delta": route_delta,
                      "fold0_portfolio_delta": port_delta,
                      "conversion": conversion, "verdict": entry["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
