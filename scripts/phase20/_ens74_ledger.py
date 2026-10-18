"""Ledger + catalogue + scheduler for ENS74 (near-tie specialist)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
REPORT = ROOT / "reports" / "phase20" / "ens74_near_tie_specialist" / "ens74_near_tie_specialist.json"
PREREG = "configs/phase20/ens74_near_tie_specialist_prereg.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(p: str) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    m = rep["metrics"]

    entry = {
        "exp_id": "p20_ens74_near_tie_specialist",
        "hypothesis_id": "H74-NEAR-TIE",
        "date": "2026-10-18",
        "ts": utcnow(),
        "family": "ENS",
        "experiment_number": 74,
        "fold_scheme": "primary fold-0 (F2 screen, frozen rule)",
        "prereg": {"path": PREREG, "sha256": sha(PREREG)},
        "arch": "rich-view LGBM, sample_weight 3.0 on rows with |port_OOF - 0.5| <= tau "
                "(tau = 30% quantile of FIT margins from PORTFOLIO_REF OOF); candidate = "
                "specialist proba on all fold-0 rows; equal-logit blend with PORTFOLIO_REF",
        "results": m,
        "finding": "NEGATIVE (port limb). Specialist standalone fold0 0.961113 vs portfolio "
                   "0.961727; equal-logit blend 0.961618; port_delta -1.093e-4 (need >= +5e-5); "
                   "Spearman 0.9825 (corr limb passes). NOTABLE: smallest |port_delta| of any "
                   "candidate this phase and the strongest standalone member measured in phase 20 "
                   "-- margin weighting buys real per-row skill (0.961113, above every other "
                   "single member screened) but its errors remain too correlated with the "
                   "portfolio's to survive the blend. Mechanism falsified on its own terms: on "
                   "the near-tie fold-0 subset (29.8% of rows) the blend is if anything worse "
                   "(0.901327 vs 0.901471) -- upweighting small-margin rows does not preferentially "
                   "fix small-margin ranking. PROTOCOL CAVEAT (recorded, no false-admission risk "
                   "since NEGATIVE): FIT-side margins came from PORTFOLIO_REF standard OOF whose "
                   "producers trained INCLUDING fold 0, so the weights are second-order "
                   "fold-0-influenced; a future positive margin-weighted candidate must be "
                   "re-verified with FIT-internal inner cross-fitted margins (H19-A protocol). "
                   "Runtime 50s, CPU only. Verdict: near-tie margin weighting closed.",
        "verdict": "COMPLETE_NEGATIVE",
        "closes_rows": [74],
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for r in cat["experiments"]:
        if r["experiment_number"] == 74:
            r["status"] = "COMPLETE_NEGATIVE"
            r["fidelity_reached"] = "F2"
            r["next_action"] = ("closed: margin weighting yields strongest standalone member "
                                "(0.961113) but blend still below portfolio; tie-subset mechanism "
                                "falsified")
            r["phase20_finding"] = (
                f"F2 fold-0: specialist 0.961113, blend 0.961618, port_delta "
                f"{m['port_delta']:+.3e}, Spearman {m['candidate_spearman_vs_portfolio']}. "
                f"NEGATIVE; tie-subset blend 0.901327 vs port 0.901471 (mechanism falsified). "
                f"Margins from standard OOF (second-order fold-0 influence) -- a future positive "
                f"variant must use FIT-internal cross-fitted margins."
            )
    cat["updated_utc"] = utcnow()
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    j = sched["jobs"]["J074"]
    j["state"] = "CLOSED_NEGATIVE"
    j["state_note"] = (f"F2: specialist 0.961113, blend 0.961618, port_delta {m['port_delta']:+.3e}, "
                       f"Spearman {m['candidate_spearman_vs_portfolio']}; smallest |delta| of phase 20")
    j["prereg"] = {"path": PREREG, "sha256": sha(PREREG)}
    j["result"] = {"verdict": "COMPLETE_NEGATIVE",
                   "report": "reports/phase20/ens74_near_tie_specialist/ens74_near_tie_specialist.json"}
    j["history"].append({"ts": utcnow(), "event": "f2_screen_done", "from": "NOT_STARTED"})
    sched["updated_utc"] = utcnow()
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")

    print(f"ledger +1 -> {len(LEDGER.read_text(encoding='utf-8').splitlines())} entries; "
          f"row 74 -> COMPLETE_NEGATIVE; J074 -> CLOSED_NEGATIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
