"""Close H18-A at wave 2 per the preregistered S3-launch rule (CPU-only)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import file_sha256  # noqa: E402
from src.phase18 import registry  # noqa: E402
from src.phase18.fingerprint import git_head  # noqa: E402

OUT = ROOT / "reports" / "phase18"
S3_LAUNCH_MIN = 5.6e-5
GATE_MIN = 1.5e-5
CONVERSION_BAND = (0.071, 0.27)  # measured: H18-E change 7.1%, v7 internal2 change ~27%


def main() -> None:
    comps = [json.loads((OUT / f"h18a_s1_f{k}_comparison.json").read_text(encoding="utf-8"))
             for k in (0, 1)]
    assert all(c["discovery_pass"] for c in comps)
    deltas = [c["delta"] for c in comps]
    mean_delta = float(np.mean(deltas))
    assert mean_delta < S3_LAUNCH_MIN, "rule did not fail; do not close"

    lo = mean_delta * CONVERSION_BAND[0]
    hi = mean_delta * CONVERSION_BAND[1]
    entry = {
        "exp_id": "phase18_h18a_close",
        "hypothesis_id": "H18-A",
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_head(),
        "status": "NEGATIVE_INSUFFICIENT_EXPECTED_PORTFOLIO_CONVERSION",
        "fold_deltas_route_level": deltas,
        "control_aucs": [c["auc_control"] for c in comps],
        "treatment_aucs": [c["auc_treatment"] for c in comps],
        "mean_route_delta": mean_delta,
        "s3_launch_rule": f"mean >= {S3_LAUNCH_MIN}",
        "s3_launch_rule_passed": False,
        "expected_portfolio_band": [lo, hi],
        "admission_gate_min": GATE_MIN,
        "gate_clearable_at_optimistic_conversion": hi >= GATE_MIN,
        "evidence_reports": ["reports/phase18/h18a_s1_f0_comparison.json",
                             "reports/phase18/h18a_s1_f1_comparison.json"],
        "gpu_burned_hours": 2.6,
        "gpu_saved_by_rule_hours": 13.5,
        "note": (
            "H18-A CLOSED AT WAVE 2 per the predeclared S3-launch rule. The coverage "
            "mechanism is REAL and transfers from S6E9 to TabPFN: 80%->90% context "
            f"coverage improved the route on both matched folds (+{deltas[0]:.4e} on "
            f"block10 f0, +{deltas[1]:.4e} on block10 f1, identical eval rows, exact "
            "pair-credit identity verified). But the mean route gain "
            f"({mean_delta:.4e}) is 1.5x below the predeclared 5.6e-5 launch threshold, "
            f"and its expected portfolio conversion ({lo:.2e}-{hi:.2e} using the "
            "measured 7.1-27% band from H18-E and v7's own change) does not reach the "
            "1.5e-5 admission gate even at the optimistic end. The 10-context S3 build "
            "(~5.5h) and S4 test contexts (~8h) NEVER RUN. Fold heterogeneity "
            "(f0 2.4x f1) confirms single-fold route deltas must not drive GPU "
            "commitments. S6E9 cross-lesson recorded: deployment-coverage gains are "
            "model-family-transferable at the MODEL level but are diluted 4-14x by the "
            "0.5 route + 0.5 aux10 portfolio architecture; only changes to the "
            "portfolio's shared-signal structure survive the blend."),
        "verdict": "CLOSE_NO_S3_S4_PROMOTION_NONE",
    }
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")
    if not registry.lookup("phase18_h18a_close"):
        registry.append_record({
            "exp_id": "phase18_h18a_close", "fingerprint": "H18-A",
            "status": "NEGATIVE", "git": git_head(),
            "evidence": "reports/phase18/h18a_s1_f0_comparison.json; "
                        "reports/phase18/h18a_s1_f1_comparison.json",
        })
    # append the S1/S2 stage records for completeness
    for k, c in enumerate(comps):
        rid = f"phase18_h18a_s1_f{k}"
        if rid not in {json.loads(ln).get("exp_id") for ln in
                       (ROOT / "experiments" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
                       if ln.strip()}:
            with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8",
                                                              newline="\n") as f:
                f.write(json.dumps({
                    "exp_id": rid, "kind": "PHASE18_H18A_MATCHED_ARM",
                    "ts": c.get("utc", entry["ts"]),
                    "paired_fold_deltas": [c["delta"]], "auc": c["auc_treatment"],
                    "complete_folds": [k], "full_oof": False,
                    "report": f"reports/phase18/h18a_s1_f{k}_comparison.json",
                    "verdict": "MATCHED_ROUTE_DELTA_MEASURED"}) + "\n")
    print(json.dumps({"closed": "H18-A", "fold_deltas": deltas,
                      "mean_route_delta": mean_delta,
                      "expected_portfolio_band": [lo, hi],
                      "verdict": entry["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
