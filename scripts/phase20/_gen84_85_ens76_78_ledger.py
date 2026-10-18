"""Ledger + catalogue for GEN84, GEN85, ENS76, ENS78 (all measured this session)."""
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
        (84, "p20_gen84_original_transfer", "H84-ORIGINAL-CONDITIONAL-TRANSFER",
         "configs/phase20/gen84_original_transfer_prereg.json",
         "reports/phase20/gen84_original_transfer/gen84_original_transfer.json",
         "source LGBM on 129,880 original rows; kNN(20)-distance label-free gate; "
         "zero competition labels"),
        (85, "p20_gen85_source_specialists", "H85-SOURCE-SPECIALISTS",
         "configs/phase20/gen85_source_specialists_prereg.json",
         "reports/phase20/gen85_source_specialists/gen85_source_specialists.json",
         "per-(Type of Travel x Class) source-trained specialists; zero competition labels"),
        (76, "p20_ens76_feature_subspace", "H76-FEATURE-SUBSPACE",
         "configs/phase20/ens76_feature_subspace_prereg.json",
         "reports/phase20/ens76_feature_subspace/ens76_feature_subspace.json",
         "5 LGBM members on 50% random feature subspaces of the 237-feature static view; "
         "equal-logit blend"),
        (78, "p20_ens78_crossfold_robust", "H78-CROSS-FOLD-ROBUST",
         "configs/phase20/ens78_crossfold_robust_prereg.json",
         "reports/phase20/ens78_crossfold_robust/ens78_crossfold_robust.json",
         "greedy member selection maximizing min per-fold AUC on folds 1-4 (fold-0 held out); "
         "saved OOF only, 92-member pool"),
    ]
    findings = {
        84: "NEGATIVE. Source-only model fold0 0.949888; conditional-gated candidate 0.956732; "
            "equal blend 0.954884; portfolio 0.961727. port_delta -4.995e-3, corr 0.8983. "
            "The kNN gate beats the equal blend (+1.85e-3 mechanism gain) but the source "
            "transfer is far too weak standalone. Frozen rule FAILS.",
        85: "NEGATIVE. Specialist stitched prediction fold0 0.948785; candidate 0.954862; "
            "port_delta -6.865e-3, corr 0.8860. Per-segment source training is WORSE than the "
            "global source model (GEN84 equal-blend contrast -2.2e-5 incremental). "
            "Original-label features (ogte_*) already absorb the usable source signal. "
            "Frozen rule FAILS.",
        76: "NEGATIVE. Member AUCs [0.960598, 0.959406, 0.959666, 0.960597, 0.959810] "
            "(mean 0.960015); blend 0.960629 (+6.14e-4 subspace-ensemble gain) but "
            "port_delta -1.098e-3 vs portfolio 0.961727, corr 0.9788. Feature-subspace "
            "diversity helps within-family but cannot reach the portfolio. Frozen rule FAILS.",
        78: "NEGATIVE. Robust (min-fold, folds 1-4) 8-member selection reaches fold0 "
            "0.961448 vs portfolio 0.961727: port_delta -2.789e-4, corr 0.9906 (corr limb "
            "passes at < 0.995; port limb fails). Diagnostic fold-0-greedy cheat selection "
            "reaches 0.961620, so the robustness-vs-peak gap is only +2.0e-4 - even "
            "fold-0-greedy selection would NOT clear the portfolio bar. Closest ENS "
            "candidate measured so far; frozen rule FAILS.",
    }
    for num, exp_id, hid, prereg, report, arch in specs:
        rep = json.loads((ROOT / report).read_text(encoding="utf-8"))
        entry = {
            "exp_id": exp_id, "hypothesis_id": hid, "date": "2026-10-18",
            "ts": utcnow(),
            "family": "ENS" if num in (76, 78) else "GEN", "experiment_number": num,
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
    for num in (76, 78, 84, 85):
        j = sched["jobs"]["J%03d" % num]
        if j["state"] != "CLOSED_NEGATIVE":
            j["history"].append({"ts": now, "event": "reconcile", "from": j["state"],
                                 "to": "CLOSED_NEGATIVE",
                                 "why": "measured NEGATIVE this session (frozen rule)"})
            j["state"] = "CLOSED_NEGATIVE"
            j["result"] = {"verdict": "COMPLETE_NEGATIVE", "report": None}
    sched["updated_utc"] = now
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")
    print("appended 4 ledger entries; catalogue rows 76/78/84/85 -> COMPLETE_NEGATIVE; scheduler synced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
