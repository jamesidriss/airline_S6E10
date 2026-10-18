"""Ledger + catalogue + scheduler for FEAT49 (conditional ordinal embed)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"
SCHED = ROOT / "research" / "phase20_exhaustive" / "scheduler.json"
REP = ROOT / "reports" / "phase20" / "feat49_conditional_ordinal_embed" / "feat49_conditional_ordinal_embed.json"
PREREG = "configs/phase20/feat49_conditional_ordinal_embed_prereg.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    rep = json.loads(REP.read_text(encoding="utf-8"))
    m = rep["metrics"]
    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()

    entry = {
        "exp_id": "p20_feat49_conditional_ordinal_embed",
        "hypothesis_id": "H49-COND-ORDINAL",
        "date": "2026-10-18",
        "ts": utcnow(),
        "family": "FEAT",
        "experiment_number": 49,
        "fold_scheme": "primary fold-0 (F2 screen, frozen rule vs portfolio ref)",
        "prereg": {"path": PREREG, "sha256": sha},
        "results": {
            "control_fold0_auc": m["control_fold0_auc"],
            "treatment_fold0_auc": m["treatment_fold0_auc"],
            "augmentation_delta_descriptive": m["augmentation_delta_descriptive_only"],
            "portfolio_fold0_auc": m["portfolio_fold0_auc"],
            "port_delta": m["port_delta"],
            "spearman_vs_portfolio": m["candidate_spearman_vs_portfolio"],
            "promising_frozen_rule": m["promising_frozen_rule"],
            "n_context_cells": rep["n_context_cells"],
            "n_rating_columns": rep["n_rating_columns"],
            "eval_rows": rep["eval_rows"],
        },
        "finding": "NEGATIVE on the frozen rule (port_delta -6.494e-04 < +5e-5) despite "
                   "the LARGEST descriptive augmentation measured this phase: the 13 "
                   "rating-value x context-cell encodings lift the paired raw21 LGBM "
                   "control by +4.94e-03 AUC (0.953423 -> 0.958366) with strong "
                   "decorrelation (Spearman 0.9407, the second-lowest of the phase). "
                   "The mechanism is real -- how each service rating maps to "
                   "satisfaction genuinely depends on traveler segment -- but the "
                   "resulting candidate's equal-logit blend with the portfolio still "
                   "lands 6.5e-4 BELOW the portfolio, so it earns no admission. "
                   "Consistent with the phase-wide pattern: route-view standalone "
                   "candidates top out ~0.958 vs portfolio 0.961727. Recorded as a "
                   "future-direction note only (not a promotion): the block carries "
                   "signal on the 21-feature view and could be re-tested as a fold-safe "
                   "TE block inside the FULL 285-feature view under a new prereg; the "
                   "current screen's verdict is negative.",
        "verdict": "COMPLETE_NEGATIVE",
        "closes_rows": [49],
        "gpu_hours": 0.0,
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for r in cat["experiments"]:
        if r["experiment_number"] == 49:
            r["status"] = "COMPLETE_NEGATIVE"
            r["fidelity_reached"] = "F2"
            r["next_action"] = "closed: port_delta -6.494e-04; strongest descriptive "
            r["phase20_finding"] = (
                "F2 fold-0: +4.94e-03 descriptive augmentation (0.953423 -> 0.958366), "
                "Spearman 0.9407, but port_delta -6.494e-04 fails the frozen rule; "
                "mechanism real (rating->satisfaction mapping depends on segment) yet "
                "candidate blends below portfolio. Future note: re-test block inside "
                "the FULL view under a new prereg; not a promotion."
            )
    cat["updated_utc"] = utcnow()
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")

    sched = json.loads(SCHED.read_text(encoding="utf-8"))
    j = sched["jobs"]["J049"]
    j["state"] = "CLOSED_NEGATIVE"
    j["state_note"] = ("largest descriptive augmentation of the phase (+4.94e-03 raw21) "
                       "but port_delta -6.494e-04 fails frozen rule; candidate blends "
                       "below portfolio")
    j["prereg"] = {"path": PREREG, "sha256": sha}
    j["result"] = {"verdict": "COMPLETE_NEGATIVE",
                   "report": "reports/phase20/feat49_conditional_ordinal_embed/feat49_conditional_ordinal_embed.json"}
    j["history"].append({"ts": utcnow(), "event": "f2_screen_done", "from": "PENDING"})
    sched["updated_utc"] = utcnow()
    SCHED.write_text(json.dumps(sched, indent=1) + "\n", encoding="utf-8")

    n = len(LEDGER.read_text(encoding="utf-8").splitlines())
    print(f"ledger +1 -> {n} entries; row 49 -> COMPLETE_NEGATIVE; J049 closed")
    print(f"  port_delta={m['port_delta']:+.3e} corr={m['candidate_spearman_vs_portfolio']} "
          f"aug_delta={m['augmentation_delta_descriptive_only']:+.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
