"""Append-only ledger correction: the upward best_iter size correction is WITHDRAWN.

Why this entry exists
---------------------
`scripts/iteration_scaling.py` fitted best_iter ~ n^0.7527 to the fold-0 SUBSAMPLE curve and that
law was used to justify scaling an iteration count measured at 95% of the labels up by 1.039x for a
fit on 100%. A direct measurement at 95% of the labels then contradicted the SIGN of that law, so the
correction was withdrawn before any full-data member was fitted under it.

This is appended rather than folded into `p7_inner_es_vs_full_outer_fit` because the fact that a
plausible-looking law had to be withdrawn on new evidence is itself information about how this
campaign's conclusions were reached.

Usage: python scripts/log_iteration_correction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEDGER = ROOT / "experiments" / "ledger.jsonl"

ENTRY = {
    "exp_id": "p7_iteration_size_correction_WITHDRAWN",
    "date": "2026-10-05",
    "corrects": "p7_inner_es_vs_full_outer_fit",
    "verdict": "WITHDRAWN -- the upward best_iter size correction rested on a law whose sign is "
               "contradicted by direct measurement",
    "withdrawn_claim": {
        "law": "best_iter = 0.0459 * n^0.7527  (OLS on log-log, three largest subsample points, "
               "R^2 0.98371, reproduces the observed 504k count to 2.3%)",
        "implication": "a count measured at 72% of the labels should be scaled UP by 1.281x for a "
                       "100% fit, or up by 1.039x for a count measured at 95%",
        "was_used_for": "the first draft of scripts/refit_members_fulldata.py",
    },
    "contradicting_measurement": {
        "rows": {"503739": [797, 731], "664655": [613, 846, 641]},
        "sources": {"503739": "fold-0 / fold-1 controls, 10pct inner-ES holdout",
                    "664655": "full-data iteration measurement, three independent 5pct holdouts"},
        "reading": ("the LARGEST training size yields a LOWER count, not a higher one. best_iter is "
                    "flat-to-decreasing in n in this regime, so extrapolating upward is the wrong "
                    "direction."),
        "competing_fold0_point": "the subsample curve's frac=1.0 point gave 879 at the same 503,739 "
                                 "rows where the fold-0 control early-stopped at 797",
    },
    "mechanism": ("with extra_trees plus colsample_bytree=0.8 and subsample=0.8, every tree already "
                  "sees about 64pct of the columns and 80pct of the rows. More data therefore makes "
                  "split statistics less noisy and the validation curve FLATTER, which both lowers "
                  "the optimum and makes the argmax less well determined. The 38pct spread across "
                  "three holdouts at one size ([613, 846, 641]) is that flatness showing up "
                  "directly."),
    "resolution": {
        "policy_now": ("take the MEDIAN of three independent 5pct early-stopping holdouts measured on "
                       "95pct of the labels, and apply NO size correction at all"),
        "why_this_is_safe": ("the measurement sits at 95pct of the target size so there is almost no "
                             "extrapolation, and over-iteration is the one failure mode that has "
                             "actually been measured (1191 rounds cost -5.9e-5 against control; a "
                             "change of subsample alone moved the argmax 731 -> 1082 and cost "
                             "-18.8e-5). Mild under-iteration is the safe residual."),
        "materiality": ("small -- 666 vs 641 rounds is a 4pct difference, and fold 0 measured 811 vs "
                        "863 rounds (6pct apart) as worth exactly 0.0e-5, so the AUC surface is "
                        "flat there. The correction was withdrawn for correctness of provenance, not "
                        "because it would have moved the score materially."),
        "action": ("the refit was restarted so the recorded manifest reflects the policy actually "
                   "used; a job already in memory was killed rather than allowed to write an "
                   "inaccurate provenance record"),
    },
    "lesson": ("a fitted scaling law validated out-of-sample on the FOLD-COUNT axis (it predicted the "
               "5->10 fold gain to 7.4pct) still did not transfer to the ITERATION-COUNT axis. "
               "Predictive power on one axis is not evidence on another, and a five-point fit from a "
               "different regime is a hypothesis rather than a constant."),
}


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    if ENTRY["exp_id"] in existing:
        print("  skip: already logged")
        return 0
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ENTRY) + "\n")
    print(f"  appended: {ENTRY['exp_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
