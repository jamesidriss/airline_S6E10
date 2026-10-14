"""Close H18-C at wave 1 per the preregistered advance bar (CPU-only)."""
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
    r = json.loads((ROOT / "reports/phase18/h18c_f0.json").read_text(encoding="utf-8"))
    assert r["status"] == "COMPLETE" and r["fidelity_control_replay"]["bit_identical"]
    assert not r["advance_pass"]
    deltas = {m: v["delta"] for m, v in r["members"].items()}
    entry = {
        "exp_id": "phase18_h18c_close",
        "hypothesis_id": "H18-C",
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_head(),
        "status": "NEGATIVE_WAVE1_INSUFFICIENT",
        "portfolio_delta_f0": r["portfolio_delta"],
        "aux10_delta_f0": r["aux10_delta"],
        "member_deltas": deltas,
        "advance_bar": r["advance_bar"],
        "fidelity": "all gates passed: certified feature hashes, library versions, EV gap 1e-10, bit-identical control replay of xt_xt_d127_ss06",
        "evidence_report": "reports/phase18/h18c_f0.json",
        "evidence_report_sha256": file_sha256(ROOT / "reports/phase18/h18c_f0.json"),
        "cpu_burned_minutes": r["total_seconds"] / 60.0,
        "note": (
            "H18-C CLOSED AT WAVE 1. The 17-column rating-tuple-key TE is real but "
            "UNSTABLE across member families: LGBM deltas +16.5e-5..-12.2e-5 (3 of 6 "
            "positive), XGB deltas -5.3e-5..-21.1e-5 (0 of 3 positive), Cat +1.5e-5. "
            f"The mean-logit aggregation cancels the scatter: aux10 +{r['aux10_delta']:.2e}, "
            f"portfolio +{r['portfolio_delta']:.2e} — below the predeclared +3.0e-5 bar "
            "by 4.4x. This closes the last cheap CPU feature axis consistent with the "
            "prior evidence chain (all21 single-column TE -13.3e-5; key-TE standalone "
            "0.818-0.868; unique-key rows carry the residual error). No folds 1-4, no "
            "shadow, no promotion."),
        "verdict": "CLOSE_NO_FURTHER_FITS",
    }
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry) + "\n")
    if not registry.lookup("phase18_h18c_close"):
        registry.append_record({
            "exp_id": "phase18_h18c_close", "fingerprint": "H18-C",
            "status": "NEGATIVE", "git": git_head(),
            "evidence": "reports/phase18/h18c_f0.json",
        })
    print(json.dumps({"closed": "H18-C", "portfolio_delta_f0": r["portfolio_delta"],
                      "verdict": entry["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
