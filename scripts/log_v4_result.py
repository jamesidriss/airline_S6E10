"""Record the v4 public-LB outcome and interpret it correctly. Idempotent.

The measurement
---------------
    v3_final     public 0.96098
    v4_fulldata  public 0.96100
    delta        +0.00002  (+2e-5)

How this must be read
---------------------
+2e-5 is FAR inside the measured paired public-LB noise floor of +-2e-4. It is NOT a demonstration
that the full-data policy helps. A single observation of that size is exactly what the noise produces
by itself, and claiming otherwise would be the same error as reading the 7.8e-4 leaderboard gap as
noise a few hours ago -- in the opposite direction.

What it IS worth: it is consistent, in both sign and size, with the cross-validated prediction, which
was low-single-digit e-5 after discounting the learning-curve extrapolation. So the first direct
check of whether the fold-level training-fraction finding transfers to the actual competition test set
came back the way the mechanism said it would, rather than the way a broken pipeline would.

That is a sanity check on the whole Phase 7 chain -- the train-fraction audit, the leakage-safe
full-fit protocol, the iteration policy, and the full-data assembly verification -- because any of
those being wrong could easily have produced a large NEGATIVE number instead.

It also bounds the gain: the mechanism cannot be worth much more than +2e-5 here, whatever the
fold-level arithmetic suggested. Anything claiming a bigger test-time effect from this change is
contradicted by the board.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CSV = ROOT / "submissions" / "kaggle_submissions.csv"

REF = "56845975.0"
PUBLIC = 0.96100
V3_PUBLIC = 0.96098

ENTRY = {
    "exp_id": "p7_v4_fulldata_public_result",
    "date": "2026-10-05",
    "verdict": "MEASURED -- +2e-5 on the public board; consistent with prediction, NOT a "
               "demonstration of effect",
    "submission": {
        "name": "v4_fulldata", "ref": REF, "public_score": PUBLIC,
        "v3_final_public_score": V3_PUBLIC,
        "delta": PUBLIC - V3_PUBLIC, "delta_e5": (PUBLIC - V3_PUBLIC) * 1e5,
        "n_members_refit_on_100pct": 32, "n_members_total": 59,
        "sha16": "9dd695800ca04b7d",
    },
    "paired_public_noise_floor_e5": 20.0,
    "interpretation": {
        "not_significant": True,
        "why": ("+2e-5 is an order of magnitude inside the +-2e-4 paired floor. One observation that "
                "size is what the noise produces by itself. It must not be reported as proof that the "
                "policy helps -- that would be the same error as reading the 7.8e-4 leaderboard gap "
                "as noise, in the opposite direction."),
        "what_it_does_support": ("sign AND size match the cross-validated prediction, which was "
                                 "low-single-digit e-5 after discounting the learning-curve "
                                 "extrapolation. So the fold-level training-fraction finding "
                                 "transferred to the real test set rather than reversing."),
        "why_it_is_a_useful_sanity_check": ("the audit, the leakage-safe full-fit protocol, the "
                                            "median-of-3 iteration policy and the full-data feature "
                                            "verification all feed this number. Any of them being "
                                            "wrong could easily have produced a LARGE NEGATIVE delta "
                                            "instead of a small positive one."),
        "upper_bound": ("the mechanism cannot be worth much more than +2e-5 on this competition. Any "
                        "claim of a larger test-time effect from training on 100pct of the labels is "
                        "contradicted by the board, and the +20.2e-5 raw extrapolation was rightly "
                        "discounted."),
        "rank_effect": ("rank did not move (215 of 759), which is the expected consequence of a gain "
                        "this small against a converged top-20 band only 1.0e-4 wide."),
    },
    "decision_after": ("v3_final remains the primary finalist on public score; v4_fulldata is retained "
                       "as the complementary second finalist on METHODOLOGY, since it uses a "
                       "different inference policy (100pct-of-labels members) at identical blend "
                       "geometry. Both were reproduced from recorded provenance."),
}


def patch_csv() -> None:
    if not CSV.exists():
        return
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    hit = False
    for r in rows:
        if r.get("file") == "v4_fulldata.csv":
            r["publicScore"] = f"{PUBLIC:.5f}"
            r["ref"] = REF
            r["status"] = "submitted"
            hit = True
    if hit:
        with CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  patched {CSV.name}: v4_fulldata public={PUBLIC:.5f} ref={REF}")


def main() -> int:
    patch_csv()
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
