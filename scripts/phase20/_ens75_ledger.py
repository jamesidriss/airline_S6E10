"""Ledger + catalogue + scheduler for ENS75 (hard-pair stacking corrector)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
REPORT = ROOT / "reports" / "phase20" / "ens75_hard_pair_corrector" / "ens75_hard_pair_corrector.json"
PREREG = "configs/phase20/ens75_hard_pair_corrector_prereg.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    m = rep["metrics"]

    entry = {
        "exp_id": "p20_ens75_hard_pair_corrector",
        "hypothesis_id": "H75-HARD-PAIR-CORRECTOR",
        "date": "2026-10-18",
        "ts": utcnow(),
        "family": "ENS",
        "experiment_number": 75,
        "fold_scheme": "primary fold-0 (F2 screen, frozen rule)",
        "prereg": {"path": PREREG, "sha256": sha(PREREG)},
        "arch": "stacking corrector: rich-view LGBM, uniform weights, portfolio OOF logit "
                "appended as input feature (no reweighting, no gating); candidate = full "
                "fold-0 vector; equal-logit blend with PORTFOLIO_REF",
        "results": m,
        "finding": "NEGATIVE (port limb). Corrector standalone fold0 0.961344 vs portfolio "
                   "0.961727; equal-logit blend 0.961596; port_delta -1.312e-4 (need >= +5e-5); "
                   "Spearman 0.99433 (corr limb passes, narrowly). Even GIVEN the portfolio's "
                   "own logit as an input feature -- the most direct possible correction "
                   "signal -- a uniform-weight rich-view LGBM cannot learn residual "
                   "corrections that survive the blend. Mechanism falsified on its own terms: "
                   "on the bottom-30% portfolio-margin fold-0 subset (hard pairs, n=41647) the "
                   "corrector is WORSE than the portfolio (0.900086 vs 0.901471). Converging "
                   "evidence across three independent ENS mechanisms (ENS74 margin-reweight "
                   "-1.093e-4, ENS75 stacking corrector -1.312e-4, ENS78 cross-fold robust "
                   "-2.789e-4): the champion portfolio is at a local optimum on this data; "
                   "every rich-view LGBM variant lands 0.9611-0.9616 standalone and none "
                   "survives equal-logit blending with the reference. Runtime 225s CPU. "
                   "Caveat (recorded, no false-admission risk since NEGATIVE): FIT-side "
                   "portfolio logit column from standard OOF (second-order fold-0 influence); "
                   "a future positive stacking variant must use FIT-internal cross-fitted "
                   "logits (H19-A protocol). Verdict: stacking correction on the portfolio "
                   "closed for this feature set.",
        "verdict": "COMPLETE_NEGATIVE",
        "closes_rows": [75],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for r in cat["experiments"]:
        if r["experiment_number"] == 75:
            r["status"] = "COMPLETE_NEGATIVE"
            r["fidelity_reached"] = "F2"
            r["next_action"] = ("closed: stacking correction with portfolio logit as feature "
                                "cannot beat the portfolio on its own hard pairs")
            r["phase20_finding"] = (
                f"F2 fold-0: corrector 0.961344, blend 0.961596, port_delta "
                f"{m['port_delta']:+.3e}, Spearman {m['candidate_spearman_vs_portfolio']}. "
                f"NEGATIVE; hard-pair subset corrector 0.900086 vs port 0.901471 "
                f"(mechanism falsified). Third ENS mechanism converging on -1e-4..-3e-4: "
                f"portfolio at local optimum for LGBM-family variants."
            )
    cat["updated_utc"] = utcnow()
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    j = sched["jobs"]["J075"]
    j["state"] = "CLOSED_NEGATIVE"
    j["state_note"] = (f"F2: corrector 0.961344, blend 0.961596, port_delta {m['port_delta']:+.3e}; "
                       f"hard-pair subset worse than portfolio (0.900086 vs 0.901471)")
    j["prereg"] = {"path": PREREG, "sha256": sha(PREREG)}
    j["result"] = {"verdict": "COMPLETE_NEGATIVE",
                   "report": "reports/phase20/ens75_hard_pair_corrector/ens75_hard_pair_corrector.json"}
    j["history"].append({"ts": utcnow(), "event": "f2_screen_done", "from": "NOT_STARTED"})
    sched["updated_utc"] = utcnow()
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")

    print(f"ledger +1 -> {len(LEDGER.read_text(encoding='utf-8').splitlines())} entries; "
          f"row 75 -> COMPLETE_NEGATIVE; J075 -> CLOSED_NEGATIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
