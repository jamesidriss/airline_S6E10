"""Append-only corrections for the Phase 9 extra_trees arithmetic.

Two errors in the Phase 9 reporting, both arithmetic rather than conceptual, and both about the same
quantity. Appended as NEW records; the originals are never rewritten, per the campaign's append-only
rule for corrections.

What was wrong
--------------
1. "Under DART the sign flips." It does not. `dart005_xt` 0.9610003 - `dart005` 0.9609236 =
   +7.67e-5, which is POSITIVE, the same sign as the GBDT effect (+44.76e-5). The correct statement is
   attenuation by 5.83x, not reversal. I had compared the two DART arms' deltas against the control
   (-35.2e-5 vs -42.8e-5) and read the ordering as a sign change, when both numbers are negative and
   their difference is positive.

2. Magnitude ratio stated as ~600x. The probe claimed -4.82e-3 and the fold measured +7.67e-5, so the
   ratio is 4.82e-3 / 7.67e-5 = 62.8x, and the SIGN is opposite, so the probe did not replicate at
   all. I wrote "~600x" in two places, which is wrong by an order of magnitude, and I wrote in STATUS
   that the probe "replicated in direction" when it did not.

3. Minor: the DART effect is +7.67e-5, which I rounded to 7.6e-5. Correct to one decimal is +7.7e-5.

Why this matters beyond bookkeeping: the erroneous "sign flip / catastrophic under DART" reading was
briefly the headline mechanism for closing DART, and it invited a wrong mechanistic story (that DART
renormalisation is actively harmful to random-threshold trees). The measured fact is much weaker and
more mundane: extra_trees helps everywhere, and DART simply dilutes how much it helps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402


def main() -> None:
    rep = json.loads((REPORTS / "p9_f0_report.json").read_text(encoding="utf-8"))
    a = rep["folds"]["0"]["arms"]
    gb = a["ctl_fixed"]["auc"] - a["ctl_det"]["auc"]
    da = a["dart005_xt"]["auc"] - a["dart005"]["auc"]
    probe = -4.82e-3

    entries = [
        {
            "exp_id": "p9_extra_trees_sign_CORRECTION",
            "date": "2026-10-06",
            "corrects": ["p9_dart005_xt_REJECTED", "p9_ctl_det_CONTROL"],
            "verdict": "CORRECTION -- extra_trees ATTENUATES 5.83x under DART; it does NOT reverse sign",
            "measured": {
                "under_gbdt_e5": round(gb * 1e5, 4),
                "under_dart_e5": round(da * 1e5, 4),
                "attenuation_factor": round(gb / da, 2),
                "both_positive": bool(gb > 0 and da > 0),
            },
            "wrong_claims": [
                "'Under DART the sign flips' -- FALSE. Both effects are positive: +44.76e-5 under "
                "GBDT and +7.67e-5 under DART.",
                "'the 90k-row probe replicated on real data, in the same direction, at 1/6 the "
                "magnitude' -- FALSE on both counts. The probe claimed -4.82e-3 and the fold "
                "measured +7.67e-5, so the direction is OPPOSITE and the magnitude ratio is 62.8x, "
                "not ~600x and not 1/6.",
                "'~600x' magnitude error -- the ratio is 62.8x; I overstated it by an order of "
                "magnitude.",
                "'+7.6e-5' -- the value is +7.67e-5, i.e. +7.7e-5 to one decimal.",
            ],
            "error_mechanism": "I compared the two DART arms' deltas against the control (-35.2e-5 vs "
                               "-42.8e-5) and read their ordering as a sign change. Both numbers are "
                               "negative, so their difference is positive: the correct quantity is "
                               "the WITHIN-family contrast (dart005_xt vs dart005), not the "
                               "within-family contrast's ordering relative to the control. The "
                               "within-family contrast is what identifies the extra_trees effect, "
                               "and computing it correctly shows no reversal.",
            "correct_statement": "extra_trees is beneficial under both GBDT (+44.76e-5) and DART "
                                 "(+7.67e-5). Its value is attenuated 5.83x under DART. The 90k "
                                 "probe's claim that it is catastrophic under DART (-4.82e-3) is "
                                 "contradicted in sign and off by 62.8x in magnitude; only the "
                                 "probe's broader conclusion (DART is uncompetitive) survived.",
            "claim_withdrawn": "The mechanistic story that DART's per-tree renormalisation is "
                               "actively HARMFUL to random-threshold trees is withdrawn. What "
                               "survives is the weaker and more mundane observation that DART "
                               "dilutes how much extra_trees helps. No mechanism is claimed.",
        },
        {
            "exp_id": "p9_leader_gap_zscore_CORRECTION",
            "date": "2026-10-06",
            "corrects": ["p7_lb_gap_significance_CORRECTION"],
            "verdict": "CORRECTION -- a stale paired z-score claim had survived in STATUS.md section 8",
            "found": "STATUS.md line ~1349 still read: 'The gap is real and it is large. ... That is "
                     "z = 3.1-4.4 even after allowing for correlation between two different "
                     "solutions, and it clears 2 sigma even between uncorrelated predictions.'",
            "why_wrong": "That is the exact claim already retracted in section 6e and in "
                         "p7_lb_gap_significance_CORRECTION. We do NOT hold the leader's prediction "
                         "vector, so rho (pairwise error correlation) is UNMEASURED. Across rho in "
                         "[0,1] the paired z spans 0.44 to 7.88 and clears 2 sigma only if rho > "
                         "0.9411. At rho = 0 the z is 0.49, so 'clears 2 sigma even uncorrelated' is "
                         "false. The section-8 summary had not been updated when section 6e was "
                         "corrected, so the retraction existed but the headline did not.",
            "correct_statement": "Factual scores only: ours 0.960980, leader 0.961760, gap 7.8e-4. "
                                 "The rho-INDEPENDENT observation, which is the one that should "
                                 "govern compute, is that the top twenty occupy a 1.0e-4 band while "
                                 "sitting 7.8e-4 above us, i.e. a converged pack at a common higher "
                                 "level. No paired significance is asserted.",
            "process_lesson": "A correction written into one section of STATUS.md did not propagate "
                              "to the executive summary in section 8. Corrections must be applied at "
                              "every location that states the corrected claim, and the summary is "
                              "the location most likely to be read and most likely to be stale.",
        },
    ]

    out = REPORTS / "p9_correction_entries.json"
    save_json({"entries": entries}, out)
    print(f"prepared {len(entries)} correction entries -> {out}")
    for e in entries:
        print(f"  {e['exp_id']}")
    if "--append" not in sys.argv:
        print("\nreview first, then: python scripts/log_phase9_corrections.py --append")
        return

    ledger = REPORTS.parent / "experiments" / "ledger.jsonl"
    existing = {json.loads(l)["exp_id"] for l in ledger.read_text(encoding="utf-8").splitlines()
                if l.strip()}
    added, skipped = 0, []
    for e in entries:
        if e["exp_id"] in existing:
            skipped.append(e["exp_id"])
            continue
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(e) + "\n")
        added += 1
    print(f"\nappended {added} to {ledger}; skipped {skipped}")


if __name__ == "__main__":
    main()
