"""Append-only corrections to two claims made after the v4 public result. Idempotent.

Correction A -- v4 does NOT bound the full-data effect
-----------------------------------------------------
Written earlier: "It also BOUNDS the mechanism: training on 100% of the labels cannot be worth much
more than +2e-5 here, whatever the fold arithmetic suggested."

That was an overstatement. The observed public delta is +2e-5 and the paired public-LB noise scale
is ~+-2e-4, so the observation sits an order of magnitude BELOW the resolution of the leaderboard.
It is compatible with the cross-validated estimate, and it is equally compatible with the true effect
being several times larger or several times smaller. One such observation places essentially no
constraint on the underlying effect.

The evidence for a SMALL full-data gain is the fold-level measurement, not the public score:

    fold 0  +4.0e-5
    fold 1  +0.4e-5
    mean    +2.2e-5        (2/2 folds)

Correct wording: "v4 is directionally consistent with the cross-validated estimate, but the public
delta is unresolved at leaderboard precision."

The second-order consequence is still fair and is retained: the realised board delta gives no reason
to expect more from this mechanism than the fold arithmetic already implied, so nothing should be
tuned upward on the strength of +2e-5. But "gives no reason to expect more" is not "bounds it".

Correction B -- no paired significance is claimed for the gap to the leader
---------------------------------------------------------------------------
Written earlier: "the gap is REAL, z = 3.1-4.4" and "clears 2 sigma even between uncorrelated
predictions".

The score gap is factual: 0.960980 vs 0.961760. But significance of a difference between two models
is conditional on the correlation between their rankings of the public rows, and we do not hold the
leader's prediction vector, so that correlation is UNMEASURED. Across rho in [0,1] the z ranges from
0.44 to 7.88; the gap clears 2 sigma only above rho = 0.941.

Worse, the specific claim "clears 2 sigma even between uncorrelated predictions" was FALSE, and false
because of a bug in my own script: the correlation needed for a 2-sigma gap was computed as
`1 - (min_z/2)**2/2`, which is not the inversion of the z formula. It returned a negative number,
which I clamped to zero and then rendered as a confident statement. At rho = 0 the z is 0.49. The
script had printed that number; I did not check it.

Both corrections are appended rather than edited, because the sequence -- hand estimate too large,
then a "correction" in the opposite direction -- is itself the most useful record here: it is exactly
the shape of error that produces a confident number nobody checks.

Usage: python scripts/log_corrections_phase8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEDGER = ROOT / "experiments" / "ledger.jsonl"

ENTRIES = [
    {
        "exp_id": "p7_v4_bounds_claim_CORRECTION",
        "date": "2026-10-05",
        "corrects": "p7_v4_fulldata_public_result",
        "verdict": "CORRECTION -- the public delta does NOT bound the full-data effect",
        "wrong_claim": ("'It also BOUNDS the mechanism: training on 100% of the labels cannot be "
                        "worth much more than +2e-5 here, whatever the fold arithmetic suggested.'"),
        "why_wrong": ("the observation is +2e-5 and the paired public-LB noise scale is ~+-2e-4, so it "
                      "sits an order of magnitude BELOW the resolution of the leaderboard. A single "
                      "such observation is compatible with the true effect being several times larger "
                      "or several times smaller and therefore places essentially no constraint on it."),
        "correct_wording": ("'v4 is directionally consistent with the cross-validated estimate, but "
                            "the public delta is unresolved at leaderboard precision.'"),
        "evidence_for_a_small_gain_is_the_fold_measurement": {
            "fold_0_e5": 4.0, "fold_1_e5": 0.4, "mean_e5": 2.2, "folds_positive": "2/2",
            "public_delta_e5": 2.0, "public_paired_noise_e5": 20.0,
        },
        "retained": ("the second-order consequence survives: the board result gives no reason to "
                     "expect MORE from this mechanism than the fold arithmetic already implied, so "
                     "nothing should be tuned upward on the strength of +2e-5. 'Gives no reason to "
                     "expect more' is not 'bounds it'."),
        "finalist_decision_unchanged": ("v3_final remains the primary finalist on public score and "
                                        "v4_fulldata the complementary second on methodology. That "
                                        "decision never rested on the size of the delta, only on v4 "
                                        "being a one-directional inference-policy change."),
    },
    {
        "exp_id": "p7_lb_gap_significance_CORRECTION",
        "date": "2026-10-05",
        "corrects": "p7_lb_noise_analysis",
        "verdict": "CORRECTION -- score gap is factual; no paired significance is claimed",
        "factual": {"our_public": 0.960980, "leader_public": 0.961760, "gap": 0.000780},
        "wrong_claims": [
            "'the gap is REAL, z = 3.1-4.4' -- presented without stating that rho is unmeasured",
            "'it clears 2 sigma even between two UNCORRELATED predictions' -- FALSE",
        ],
        "bug_in_our_own_script": (
            "the correlation needed for a 2-sigma gap was computed as 1 - (min_z/2)**2/2, which is "
            "NOT the inversion of the z formula. It returned a negative number; I clamped it to zero "
            "and rendered it as a confident sentence. At rho = 0 the z is 0.49, so the claim it "
            "supported is the opposite of the truth. The script had already printed 0.49 and the "
            "number was not checked against the sentence built from it."),
        "correct_arithmetic": (
            "z = 2 => se_diff = gap/2 and se_diff = se*sqrt(2(1-rho)) => "
            "rho = 1 - (gap/2)^2 / (2*se^2) = 0.9411"),
        "correct_statement": (
            "across rho in [0,1] the z ranges from 0.44 to 7.88. The gap clears 2 sigma only if rho "
            "exceeds 0.941. We do not hold the leader's prediction vector, so rho is UNMEASURED and "
            "no single z can be asserted. 'Probably real' is a reasonable working belief for two "
            "strong tabular solutions on the same data and metric, and is recorded as a belief."),
        "rho_independent_conclusion_retained": (
            "the top 20 span 1.0e-4 while sitting 7.8e-4 above us, and every gain this campaign has "
            "actually measured is single-digit e-5. Whether or not the gap is statistically real, more "
            "GBDT-family polish will not close it. That conclusion does not depend on rho."),
    },
]


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    added = 0
    with LEDGER.open("a", encoding="utf-8") as f:
        for e in ENTRIES:
            if e["exp_id"] in existing:
                continue
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']}")
    if not added:
        print("  nothing to append")
    print(f"  ledger lines now: {sum(1 for _ in LEDGER.open(encoding='utf-8'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
