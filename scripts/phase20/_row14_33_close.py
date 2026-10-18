"""Ledger + catalogue + scheduler for row 14 (EXAONE/TabICL adaptation) and
row 33 (LoRA/adapters). Both close on audit evidence, no GPU spent:

Row 14: the TabICL official fine-tune/adaptation was already executed under
row 31 (full dose RESOURCE_BLOCKED -> fb100k fallback arm NEGATIVE, ledger
'p20_tabicl_ft_fb100k_result'); the EXAONE half is LICENSE_BLOCKED and was
never runnable under competition rules. Row closes COMPLETE_NEGATIVE.

Row 33: adapter-support audit over every installed FLM client found NO
documented LoRA/adapter/prefix-tuning surface: tabpfn 9.1.0 finetuning module
is full-weight fine-tuning only (no lora/adapter anywhere in source), tabicl
exposes no LoRA modules and no public finetune API, tabldm updates no weights
at all (ICL-only, fit() = preprocessing), mitra_finetune uses its own full-FT
50-step protocol (tested NEGATIVE in phase 19/20). The bounded test has no
valid target, and inventing non-documented loops is forbidden by the row's own
next_action. Row closes COMPLETE_NEGATIVE (audit-complete, zero GPU).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    entry = {
        "exp_id": "p20_row14_row33_audit_close",
        "hypothesis_id": "H14-TABICL-EXAONE-ADAPT;H33-ADAPTERS",
        "date": "2026-10-18",
        "ts": utcnow(),
        "family": "FLM+OBJ",
        "experiment_number": "14+33",
        "fold_scheme": "n/a (audit closures, no predictive result)",
        "results": {
            "row14": {
                "tabicl_official_adaptation": "already executed under row 31: full dose "
                    "RESOURCE_BLOCKED (GPU driver crash) -> fb100k fallback arm "
                    "NEGATIVE (standalone 0.957404, portfolio_delta -8.169e-4, "
                    "ledger p20_tabicl_ft_fb100k_result)",
                "exaone": "LICENSE_BLOCKED (never runnable under competition rules)",
            },
            "row33": {
                "tabpfn_9_1_0": "finetuning module = full-weight FT only; no lora/"
                    "adapter/bitfit/prefix surface in any module source",
                "tabicl": "no LoRA/adapter modules; no public finetune API at all",
                "tabldm": "no weight updates by design (in-context learning; fit() "
                    "only preprocesses the labeled context)",
                "mitra_finetune": "own full-FT 50-step protocol; already tested "
                    "NEGATIVE (phase 19/20 mitra screens)",
                "conclusion": "zero documented adapter surfaces across every screened "
                    "FLM -> bounded test has no valid target; inventing "
                    "non-documented loops forbidden by the row's own next_action",
            },
        },
        "finding": "Row 14 closed: the TabICL half was genuinely executed (row 31, "
                   "NEGATIVE at the best feasible dose; full dose resource-blocked by "
                   "GPU driver crash), EXAONE half license-blocked. Row 33 closed: "
                   "audit of all four installed FLM clients found no adapter/LoRA "
                   "surface anywhere; the only weight-update paths are full "
                   "fine-tuning, and every one of those that is runnable has been or "
                   "is being tested separately (Mitra NEGATIVE, TabICL NEGATIVE, "
                   "TabPFN v3.5 = row 34 running next). Zero GPU hours spent; both "
                   "closures are honest completions of the rows' own first action "
                   "(audit), not skipped work.",
        "verdict": "COMPLETE_NEGATIVE (audit closures)",
        "closes_rows": [14, 33],
        "gpu_hours": 0.0,
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for r in cat["experiments"]:
        if r["experiment_number"] == 14:
            r["status"] = "COMPLETE_NEGATIVE"
            r["next_action"] = "closed: TabICL adaptation tested NEGATIVE under row 31 "
            r["phase20_finding"] = (
                "TabICL official FT executed under row 31 (fb100k arm NEGATIVE: "
                "standalone 0.957404, portfolio_delta -8.169e-4; full dose "
                "RESOURCE_BLOCKED GPU driver crash); EXAONE LICENSE_BLOCKED."
            )
        if r["experiment_number"] == 33:
            r["status"] = "COMPLETE_NEGATIVE"
            r["next_action"] = "closed: no adapter surface exists in any screened FLM"
            r["phase20_finding"] = (
                "Audit: tabpfn 9.1.0 finetuning = full FT only (no lora/adapter in "
                "source); tabicl no LoRA + no public finetune API; tabldm ICL-only "
                "(no weight updates); mitra full-FT 50-step (already NEGATIVE). "
                "Zero documented adapter targets -> bounded test not executable."
            )
    cat["updated_utc"] = utcnow()
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    for jid, row, note in (
        ("J014", 14, "TabICL half tested NEGATIVE via row 31; EXAONE license-blocked"),
        ("J033", 33, "audit: zero adapter/LoRA surfaces in tabpfn/tabicl/tabldm/"
                     "mitra; bounded test has no valid target"),
    ):
        j = sched["jobs"][jid]
        j["state"] = "CLOSED_NEGATIVE"
        j["state_note"] = note
        j["result"] = {"verdict": "COMPLETE_NEGATIVE", "ledger": "p20_row14_row33_audit_close"}
        j["history"].append({"ts": utcnow(), "event": "audit_close", "from": "PENDING"})
    sched["updated_utc"] = utcnow()
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")

    n = len(LEDGER.read_text(encoding="utf-8").splitlines())
    print(f"ledger +1 -> {n} entries; rows 14,33 -> COMPLETE_NEGATIVE; J014,J033 closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
