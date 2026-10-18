"""Ledger + catalogue for CLASS61 (genuine EBM) and CLASS66 (hierarchical kernels)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main() -> int:
    specs = [
        (61, "p20_class61_genuine_ebm", "H61-GENUINE-EBM",
         "configs/phase20/class61_genuine_ebm_prereg.json",
         "reports/phase20/class61_genuine_ebm/class61_genuine_ebm.json",
         "ExplainableBoostingClassifier (interpret 0.7.8), 16 real pairwise interactions, "
         "outer-FIT only, outer_bags=4"),
        (66, "p20_class66_hierarchical_kernels", "H66-HIERARCHICAL-KERNELS",
         "configs/phase20/class66_hierarchical_kernels_prereg.json",
         "reports/phase20/class66_hierarchical_kernels/class66_hierarchical_kernels.json",
         "per-segment (Type of Travel x Class) Nadaraya-Watson kernel regression, K=50, "
         "bandwidth = in-segment median kNN distance, outer-FIT only"),
    ]
    findings = {
        61: "NEGATIVE. Genuine EBM with 16 real interactions: fold0 0.954200 vs portfolio "
            "0.961727; equal-logit blend 0.959281; port_delta -2.447e-3 (need >= +5e-5), "
            "corr 0.9069. EBM decorrelates well (corr limb passes easily) but is far too "
            "weak standalone to add portfolio value; interactions did not close the gap "
            "(fit 63s, outer-FIT only). Frozen rule FAILS on the port limb.",
        66: "NEGATIVE. Per-segment kernel regression: fold0 0.950981 vs portfolio 0.961727; "
            "equal-logit blend 0.958031; port_delta -3.696e-3, corr 0.8699 (lowest "
            "correlation of any classical candidate measured - genuine diversity, but the "
            "kernel predictor is far too weak standalone). Frozen rule FAILS on the port "
            "limb. Local kernel smoothing does not add value on this dataset.",
    }
    for num, exp_id, hid, prereg, report, arch in specs:
        rep = json.loads((ROOT / report).read_text(encoding="utf-8"))
        entry = {
            "exp_id": exp_id, "hypothesis_id": hid, "date": "2026-10-18",
            "ts": utcnow(), "family": "CLASS", "experiment_number": num,
            "fold_scheme": "primary fold-0 (F2 screen, frozen rule)",
            "prereg": {"path": prereg, "sha256": sha(prereg)},
            "arch": arch,
            "results": rep["metrics"],
            "finding": findings[num],
            "verdict": "COMPLETE_NEGATIVE", "closes_rows": [num],
        }
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        n = row.get("experiment_number")
        if n in findings:
            row["phase20_finding"] = findings[n]
            row["fidelity_reached"] = "F2"
            row["status"] = "COMPLETE_NEGATIVE"
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    now = utcnow()
    for num in (61, 66):
        j = sched["jobs"]["J%03d" % num]
        if j["state"] != "CLOSED_NEGATIVE":
            j["history"].append({"ts": now, "event": "reconcile", "from": j["state"],
                                 "to": "CLOSED_NEGATIVE",
                                 "why": "measured NEGATIVE this session (frozen rule)"})
            j["state"] = "CLOSED_NEGATIVE"
            j["result"] = {"verdict": "COMPLETE_NEGATIVE", "report": None}
    sched["updated_utc"] = now
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")
    print("appended 2 ledger entries; catalogue rows 61/66 -> COMPLETE_NEGATIVE; scheduler synced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
