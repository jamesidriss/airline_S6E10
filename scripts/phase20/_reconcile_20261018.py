"""FEAT54 strict-v2 ledger entry + catalogue/scheduler reconciliation.

Part A: append FEAT54 v2 ledger entry; catalogue row 54 PROMISING -> COMPLETE_NEGATIVE
        (v1 finding preserved verbatim in a v1 field, superseded by v2).
Part B: reconcile catalogue <-> scheduler states (append history events, never
        clobber), and write an exact-count reconciliation report.

No historical report content is rewritten here; only statuses/findings that were
verifiably executed but never tracked get corrected, with the correction noted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
REPORT = ROOT / "reports" / "phase20" / "feat54_smoothed_categorical" / "feat54_strict_v2.json"
OUT = ROOT / "reports" / "phase20" / "reconciliation_20261018"
PREREG_SHA = "4d200f7db408b058c490505e32b488d3f8e4502f342d508db73b1f7d80894fd6"

STATUS_MAP = {
    "COMPLETE_NEGATIVE": "CLOSED_NEGATIVE",
    "COMPLETE_CERTIFIED": "PROMOTED",
    "HISTORICAL_EQUIVALENT": "DONE",
    "PARTIAL_VALID": "DONE",
    "LICENSE_BLOCKED": "LICENSE_BLOCKED",
    "NOT_STARTED": "PENDING",
}
DEP_REASONS = {
    "blocked-on-base", "blocked-on-arch", "blocked-on-acc", "no-candidate",
    "no-strong-candidate", "not-warranted", "blocked-identifiability",
}
# Catalogue status corrections for rows executed this session but left NOT_STARTED,
# and rows whose finding was recorded but status not advanced. (row, new_status, why)
STATUS_FIXES = [
    (72, "COMPLETE_NEGATIVE", "ENS72 executed (ledger p20_ens72_covariate_gating): "
        "NEGATIVE_GATE_CANNOT_GENERALIZE; finding was recorded but status left NOT_STARTED."),
    (80, "PARTIAL_VALID", "GEN80 executed (ledger p20_gen80_auc_ceiling): descriptive oracle "
        "analysis with appended correction_20261018; finding recorded but status left NOT_STARTED."),
    (91, "PARTIAL_VALID", "DEP91 executed (ledger p20_finalist_h91_variance): F1 descriptive "
        "variance decomposition; finding recorded but status left NOT_STARTED."),
    (94, "PARTIAL_VALID", "DEP94 executed (ledger p20_finalist_h94_stability): audit with doc "
        "gap (champion v7 OOF not archived); finding recorded but status left NOT_STARTED."),
    (54, "COMPLETE_NEGATIVE", "FEAT54 v1 INVALID_PROTOCOL (correction_20261018_d) superseded by "
        "strict v2 (prereg 4d200f7d): port_delta -1.912e-3 < +5e-5 -> frozen rule NEGATIVE."),
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    # ---------- Part A: ledger entry for FEAT54 strict v2 ----------
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    m = rep["metrics"]
    entry = {
        "exp_id": "p20_feat54_strict_v2",
        "hypothesis_id": "H54-SMOOTHED-CATEGORICAL-STRICT",
        "date": "2026-10-18",
        "ts": utcnow(),
        "family": "FEAT",
        "experiment_number": 54,
        "fold_scheme": "primary fold-0 (F2 screen, strict)",
        "prereg": {"path": "configs/phase20/feat54_strict_prereg.json", "sha256": PREREG_SHA},
        "arch": "Bayesian-shrinkage smoothed target encoding (4 cat features), strict "
                "outer-FIT-only EVAL stats + inner cross-fit within outer-FIT, ES labels "
                "excluded from all encoding sources. LGBM xt_xt_f10 on 21raw+4 vs 21raw.",
        "results": {
            "control_fold0_auc": m["control_fold0_auc"],
            "treatment_fold0_auc": m["treatment_fold0_auc"],
            "augmentation_delta_descriptive_only": m["augmentation_delta_descriptive_only"],
            "portfolio_ref": m["portfolio_ref"],
            "portfolio_fold0_auc": m["portfolio_fold0_auc"],
            "port_delta": m["port_delta"],
            "candidate_spearman_vs_portfolio": m["candidate_spearman_vs_portfolio"],
            "promising_frozen_rule": m["promising_frozen_rule"],
            "artifact": "reports/phase20/feat54_smoothed_categorical/feat54_strict_v2.json",
        },
        "finding": (
            "STRICT re-measurement under frozen rule (port_delta >= +5e-5 AND corr < 0.995). "
            "v1 was INVALID (eval-fold labels leaked into outer-FIT training features; "
            "augmentation delta +8.49e-5 collapsed to +6.11e-6 when the contamination was "
            "removed -> the v1 signal was the leak). Strict results: treatment 0.953429 vs "
            "control 0.953423 (+6.11e-6 descriptive); port_delta -1.912e-3 (blend with route "
            "reference 0.9599 vs portfolio 0.9617) and corr 0.9103. Frozen rule FAILS on both "
            "limbs. NEGATIVE; row 54 PROMISING -> COMPLETE_NEGATIVE."),
        "verdict": "COMPLETE_NEGATIVE",
        "closes_rows": [54],
        "supersedes": "p20_feat54_smoothed_categorical (INVALID_PROTOCOL, preserved in ledger)",
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("appended ledger: p20_feat54_strict_v2")

    # ---------- Part A2: catalogue ----------
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    by_row = {x["experiment_number"]: x for x in cat["experiments"]}

    r54 = by_row[54]
    old_finding = r54.get("phase20_finding")
    if old_finding and "v1_invalid" not in r54:
        r54["v1_finding_invalidated"] = old_finding
    r54["phase20_finding"] = (
        "v1 INVALID_PROTOCOL (eval-fold labels in FIT features; prereg sha was TODO; wrong "
        "decision rule) - superseded by strict v2 (prereg 4d200f7d): aug delta collapses to "
        "+6.11e-6; port_delta -1.912e-3 (need >= +5e-5), corr 0.9103 (need < 0.995). "
        "Frozen rule NEGATIVE -> COMPLETE_NEGATIVE.")
    r54["status"] = "COMPLETE_NEGATIVE"
    r54["fidelity_reached"] = "F2"
    print("row 54: PROMISING -> COMPLETE_NEGATIVE (v1 finding preserved in v1_finding_invalidated)")

    for num, new_status, why in STATUS_FIXES:
        if num == 54:
            continue
        row = by_row[num]
        if row["status"] == "NOT_STARTED":
            row["status"] = new_status
            row["status_correction_20261018"] = why
            print(f"row {num}: NOT_STARTED -> {new_status}")
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    # ---------- Part B: scheduler sync ----------
    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    jobs = {v["exp_row"]: v for v in sched["jobs"].values()}
    synced = []
    for row in cat["experiments"]:
        num = row["experiment_number"]
        job = jobs[num]
        cs = row["status"]
        br = (row.get("blocking_reason") or "").strip()
        if cs == "NOT_STARTED":
            want = "DEPENDENCY_BLOCKED" if br in DEP_REASONS else "PENDING"
        elif cs == "PROMISING":
            continue  # screen passed; integration work still pending -> stays as-is
        else:
            want = STATUS_MAP[cs]
        if job["state"] != want:
            prev = job["state"]
            job["history"].append({"ts": utcnow(), "event": "reconcile",
                                   "from": prev, "to": want,
                                   "why": f"catalogue status {cs} synced"})
            job["state"] = want
            job["result"] = {"verdict": cs if want in (
                "DONE", "CLOSED_NEGATIVE", "PROMOTED", "LICENSE_BLOCKED") else None,
                "report": job.get("result", {}).get("report")}
            synced.append((num, prev, want))
    # Register FEAT54 strict-v2 execution on J054 (ran this session outside `run`)
    j54 = jobs[54]
    if j54["state"] == "PENDING" or j54.get("cmd") is None:
        j54["cmd"] = ["C:\\Users\\James\\airline_S6E10\\.venv\\Scripts\\python.exe",
                      "scripts/phase20/feat54_strict_v2.py"]
        j54["kind"] = "cpu"
        j54["prereg"] = {"path": "configs/phase20/feat54_strict_prereg.json",
                         "sha256": PREREG_SHA}
        j54["fingerprint"] = "manual-run-20261018"
        j54["attempts"] = 1
        j54["result"] = {"verdict": "COMPLETE_NEGATIVE",
                         "report": "reports/phase20/feat54_smoothed_categorical/feat54_strict_v2.json"}
        j54["history"].append({"ts": utcnow(), "event": "reconcile",
                               "note": "strict v2 executed this session outside scheduler run(); "
                                       "cmd/prereg/report recorded, fingerprint pinned manual",
                               "from": "PENDING", "to": "CLOSED_NEGATIVE"})
        j54["state"] = "CLOSED_NEGATIVE"
        synced.append((54, "PENDING", "CLOSED_NEGATIVE"))
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")
    for s in synced:
        print(f"scheduler row {s[0]}: {s[1]} -> {s[2]}")

    # ---------- Part B2: exact counts report ----------
    import collections
    cat_counts = dict(collections.Counter(x["status"] for x in cat["experiments"]))
    sch_counts = dict(collections.Counter(v["state"] for v in sched["jobs"].values()))
    not_started = [x for x in cat["experiments"] if x["status"] == "NOT_STARTED"]
    dep_blocked = [x for x in not_started
                   if (x.get("blocking_reason") or "").strip() in DEP_REASONS]
    executable = [x for x in not_started if x not in dep_blocked]
    license_blocked = [x for x in cat["experiments"] if x["status"] == "LICENSE_BLOCKED"]
    promising = [x for x in cat["experiments"] if x["status"] == "PROMISING"]
    executed = [x for x in cat["experiments"] if x["status"] not in
                ("NOT_STARTED", "LICENSE_BLOCKED")]
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {
        "date": "2026-10-18",
        "issue": "Catalogue showed 52 NOT_STARTED / scheduler 56 PENDING / status report claimed "
                 "37 remaining - three different, partially stale views.",
        "root_causes": [
            "13 rows executed with measured results never synced to scheduler (stayed PENDING).",
            "4 rows (72/80/91/94) executed with ledger findings but left NOT_STARTED in catalogue.",
            "row 54 PROMISING from INVALID v1 run.",
            "9 rows NOT_STARTED but DEPENDENCY_BLOCKED in scheduler (by design, counted as pending "
            "by naive NOT_STARTED/PENDING tallies).",
            "'37 remaining' in an earlier status report was an undercount of the true executable "
            "slice and overcounted completed items; recomputed below.",
        ],
        "counts": {
            "total_rows": len(cat["experiments"]),
            "catalogue_status": cat_counts,
            "scheduler_state": sch_counts,
            "executed_with_result": len(executed),
            "dependency_blocked_not_started": len(dep_blocked),
            "dependency_blocked_rows": [x["experiment_number"] for x in dep_blocked],
            "license_blocked": len(license_blocked),
            "license_blocked_rows": [x["experiment_number"] for x in license_blocked],
            "promising_screen_passed_pending_integration": len(promising),
            "promising_rows": [x["experiment_number"] for x in promising],
            "truly_executable_not_started": len(executable),
            "truly_executable_rows": [x["experiment_number"] for x in executable],
        },
        "reconciliations_applied": [
            {"catalogue_status_fixes": [{"row": r, "to": s, "why": w}
                                        for r, s, w in STATUS_FIXES]},
            {"scheduler_synced": [{"row": r, "from": a, "to": b} for r, a, b in synced]},
        ],
        "statement": (
            f"96 total = {len(executed)} executed-with-result + {len(dep_blocked)} dependency-blocked "
            f"(not started by design) + {len(license_blocked)} license-blocked + "
            f"{len(executable)} truly executable NOT_STARTED."),
    }
    path = OUT / "reconciliation.json"
    path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(rec["statement"])
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
