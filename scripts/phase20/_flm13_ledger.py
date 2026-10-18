"""Ledger + catalogue for FLM13 verification (PARTIAL_VALID: checkpoints verified,
matched GPU inference deferred behind champion-OOF reproduction)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
REPORT = ROOT / "reports" / "phase20" / "flm13_checkpoint_verify" / "flm13_checkpoint_verify.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    entry = {
        "exp_id": "p20_flm13_checkpoint_verify",
        "hypothesis_id": "H13-TABPFN-3-VS-2.6-VERIFICATION",
        "date": "2026-10-18", "ts": utcnow(), "family": "FLM", "experiment_number": 13,
        "fold_scheme": "n/a (checkpoint availability + licensing verification, CPU-only)",
        "prereg": {"path": None, "sha256": None,
                   "note": "verification task, not a predictive experiment; no result to bias"},
        "arch": rep["installed_tabpfn"],
        "results": {
            "v2_6_exists": True, "v3_exists": True,
            "v2_6_repo": "Prior-Labs/tabpfn_2_6 @ 24148873a9a5 (1 clf ckpt, ungated)",
            "v3_repo": "Prior-Labs/tabpfn_3 @ 24a16a89d245 (4 clf ckpts, ungated)",
            "champion_local_v35_exists": rep["champion_local_checkpoint"]["exists"],
            "champion_local_v35_sha16": str(rep["champion_local_checkpoint"]["sha256"])[:16],
            "gpu_seconds_spent": 0,
        },
        "finding": (
            "VERIFIED: TabPFN-2.6 (Prior-Labs/tabpfn_2_6 @ 24148873a9a5, single "
            "classifier checkpoint, ungated) and TabPFN-3 (Prior-Labs/tabpfn_3 @ "
            "24a16a89d245, 4 classifier checkpoints incl. dated binary variant, ungated) "
            "both exist and are pinnable. Licensing: license:other, ungated, identical "
            "posture to the v3.5 checkpoint already in champion v7 - no new exposure. "
            "Matched-protocol GPU inference vs our v3.5 DEFERRED: v3.5 is the newest "
            "checkpoint and already champion, so v2.6/v3 can only enter as diversity "
            "members, and any member test requires the champion route recipe + OOF "
            "reproduction that is the finalist-lock blocker. Verification portion of "
            "FLM13 is complete at F1+ (exact files, revisions, licensing pinned)."),
        "verdict": "VERIFICATION_COMPLETE_MATCHED_INFERENCE_DEFERRED",
        "closes_rows": [],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == 13:
            row["phase20_finding"] = (
                "Checkpoints verified (CPU): v2.6 = Prior-Labs/tabpfn_2_6 @ 24148873a9a5 "
                "(1 clf ckpt), v3 = Prior-Labs/tabpfn_3 @ 24a16a89d245 (4 clf ckpts); both "
                "ungated, license:other (same posture as champion's v3.5). Matched GPU "
                "inference deferred behind champion-OOF reproduction (v3.5 already newest "
                "and already champion; v2.6/v3 only possible as diversity members).")
            row["status"] = "PARTIAL_VALID"
            row["fidelity_reached"] = "F1"
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    now = utcnow()
    j = sched["jobs"]["J013"]
    j["history"].append({"ts": now, "event": "reconcile", "from": j["state"], "to": "DONE",
                         "why": "verification portion complete; matched inference deferred "
                                "with rationale (reserve preserved)"})
    j["state"] = "DONE"
    j["result"] = {"verdict": "PARTIAL_VALID",
                   "report": "reports/phase20/flm13_checkpoint_verify/flm13_checkpoint_verify.json"}
    sched["updated_utc"] = now
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")
    print("ledger + catalogue row 13 -> PARTIAL_VALID; scheduler J013 -> DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
