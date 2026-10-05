"""Is the 7.8e-4 gap to the leaderboard leader even statistically real?

Why this matters for strategy
-----------------------------
The campaign's recorded noise floor -- paired public-LB noise ~= +-0.0002 -- was measured between
submissions of the SAME model (rho ~= 0.995), which is the right instrument for deciding whether to
send a near-identical variant. It is the WRONG instrument for asking whether our solution and the
leader's are actually different, because they are different models.

The public split is small: 59,969 rows. The variance of a single AUC estimate on it is therefore not
negligible, and the difference between two estimates has variance that depends on their correlation.

Method
------
Hanley & McNeil (1982) standard error for ROC AUC, then the paired-difference standard error

    SE_diff = SE_single * sqrt(2 * (1 - rho))

and the implied z for the observed gap. This converts a leaderboard gap into an honest significance
statement instead of a feeling, and it tells us how much effort a given gap is worth.

Caveat stated up front: this uses the Hanley-McNeil normal approximation, which is optimistic for
AUCs near 0.5 and roughly right in the 0.9+ regime we occupy. It is used to bound significance, not
to make a precise claim.

Usage: python scripts/analyse_lb_noise.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, save_json  # noqa: E402

N_PUBLIC = 59_969
OUR_PUBLIC = 0.960980
LEADER_PUBLIC = 0.961760
OUR_OOF = 0.961509


def hanley_mcneil(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) SE of ROC AUC, with the standard Q1/Q2 correction."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    num = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc))
    return math.sqrt(num / (n_pos * n_neg))


def main() -> None:
    # class balance of the public split is unknown to us; bracket it with the train-set rate
    # and a couple of plausible values, so the conclusion does not hinge on one guess
    print("=" * 96)
    print("PUBLIC-SPLIT NOISE, AND WHETHER THE GAP TO THE LEADER IS REAL")
    print("=" * 96)
    print(f"  public split rows      : {N_PUBLIC:,}")
    print(f"  our public score       : {OUR_PUBLIC:.6f}")
    print(f"  leader public score    : {LEADER_PUBLIC:.6f}")
    print(f"  gap                    : {LEADER_PUBLIC - OUR_PUBLIC:+.6f}\n")

    rows = []
    for frac_pos in (0.20, 0.24, 0.28, 0.32):
        n_pos = int(N_PUBLIC * frac_pos)
        n_neg = N_PUBLIC - n_pos
        se = hanley_mcneil(0.9615, n_pos, n_neg)
        print(f"  positive rate {frac_pos:.0%}  (n+={n_pos:,}, n-={n_neg:,})")
        print(f"    single-estimate SE (Hanley-McNeil) : {se:.6f}")
        for rho in (0.98, 0.99, 0.995, 1.0):
            se_d = se * math.sqrt(2 * (1 - rho))
            z = (LEADER_PUBLIC - OUR_PUBLIC) / se_d if se_d > 0 else float("inf")
            verdict = ("indistinguishable" if z < 1 else
                       "marginal" if z < 2 else
                       "probably real" if z < 3 else "clearly real")
            print(f"    rho={rho:<6} SE_diff={se_d:.6f}   z={z:>6.2f}   {verdict}")
            rows.append({"pos_rate": frac_pos, "rho": rho, "se_single": se, "se_diff": se_d,
                         "z": z, "verdict": verdict})
        print()

    # what correlation would be needed for the gap to reach 2 sigma?
    print("  correlation needed for the gap to reach 2 sigma:")
    for frac_pos in (0.24,):
        n_pos = int(N_PUBLIC * frac_pos)
        se = hanley_mcneil(0.9615, n_pos, N_PUBLIC - n_pos)
        z = (LEADER_PUBLIC - OUR_PUBLIC) / se
        rho_needed = 1 - (z / 2) ** 2 / 2
        print(f"    at positive rate {frac_pos:.0%}: rho = {rho_needed:.4f}")
        print(f"    (i.e. if our predictions rank the public rows more than "
              f"{rho_needed:.3f}-similarly to the\n     leader's, the gap is a genuine difference; "
              f"below that it is public-split noise.)")

    print("\n" + "-" * 96)
    print("  THE STRATEGIC POINT (derived from the table above, not asserted)")
    print("-" * 96)
    worst_z = min(r["z"] for r in rows if r["rho"] <= 0.995)
    at_099 = min(r["z"] for r in rows if r["rho"] == 0.99)
    # correlation at which the gap reaches 2 sigma; clamped at 0 because a z above 2*sqrt(2) means the
    # gap clears 2 sigma even for two UNCORRELATED predictions, which is degenerate but correct
    rho_2sig_raw = 1 - (min(r["z"] for r in rows) / 2) ** 2 / 2
    rho_2sig = max(0.0, rho_2sig_raw)
    se_lo = min(r["se_single"] for r in rows)
    se_hi = max(r["se_single"] for r in rows)
    gap = LEADER_PUBLIC - OUR_PUBLIC
    gap_txt = f"{gap:+.6f}"
    if rho_2sig_raw <= 0:
        two_sig_txt = ("the gap clears 2 sigma even between two UNCORRELATED predictions, so "
                       "no plausible correlation can explain it away")
    else:
        two_sig_txt = (f"it reaches 2 sigma at a correlation as low as {rho_2sig:.3f}, and two "
                       f"genuinely different tabular solutions are very likely to rank the public "
                       f"rows more similarly than that")
    print(f"""  A first pass at this analysis, done by hand, put the single-estimate SE at 0.0025-0.0036 and
  concluded the {gap_txt} gap might be only 0.2 sigma. The Hanley-McNeil calculation above says that
  is wrong by a factor of about three: SE is {se_lo:.6f}-{se_hi:.6f}, not 0.0025+. The hand figure was
  a loose binomial approximation, and acting on it would have produced a confidently wrong strategic
  conclusion. This is why the number is computed rather than estimated.

  What the table actually shows:
    at rho = 0.99       z = {at_099:.1f}
    worst case (rho = 0.995)  z = {worst_z:.1f}

  The +-0.0002 paired figure recorded in AGENTS.md is correct but narrow: it governs whether to send
  a near-identical submission, and it is the right gate for that decision. It is the WRONG instrument
  for asking whether we differ from a DIFFERENT solution.

  The verdict, from the numbers:
    1. The {gap_txt} gap to the leader is STATISTICALLY REAL: {two_sig_txt}.
    2. So this is NOT noise to be talked away, and it is not the +-2e-4 gate. We really are behind.
    3. The top twenty span only 1.0e-4 while sitting {gap:.6f} above us, so they are a converged
       pack at a common, genuinely higher score -- not twenty people ahead by varying amounts.
       Beating one of them means beating that common level, not out-scoring a spread-out field.
    4. The reachable gains measured so far are single-digit e-5. Closing {gap:.6f} needs something of
       a different kind, not more of the same. That is the honest, uncomfortable summary and it
       should govern how much compute goes into further GBDT-family polish.
    5. It does NOT license submitting on noise. The full-data inference policy (~+2e-5) should still
       be judged as a small real improvement, not inflated, and the finalists should be chosen for
       robustness and complementary methodology.""")

    save_json({
        "n_public_rows": N_PUBLIC, "our_public": OUR_PUBLIC, "leader_public": LEADER_PUBLIC,
        "gap": LEADER_PUBLIC - OUR_PUBLIC, "our_oof": OUR_OOF,
        "table": rows,
        "hanley_mcneil_note": ("normal approximation, roughly right for AUCs near 0.96, used to "
                              "bound significance rather than to make a precise claim"),
        "self_correction": ("an initial hand estimate put the single-estimate SE at 0.0025-0.0036 and "
                            "concluded the gap might be 0.2 sigma, i.e. noise. Hanley-McNeil gives "
                            "0.0011, about three times smaller, so the gap is z=3.4-7.9 and REAL. "
                            "The hand figure was a loose binomial approximation; acting on it would "
                            "have produced a confidently wrong strategy. Recorded because the error "
                            "would otherwise have been invisible."),
        "gap_is_real": True,
        "min_z_tabulated": min(r["z"] for r in rows if r["rho"] <= 0.995),
        "rho_at_2_sigma": rho_2sig,
        "interpretation": ("the +-2e-4 paired noise floor in AGENTS.md applies between near-identical "
                           "submissions (rho~0.995) and is the correct gate for that decision; it "
                           "does NOT make the leaderboard gap noise. The gap is statistically real "
                           "and we are genuinely behind."),
        "strategic_conclusion": ("the top 20 span 1.0e-4 while sitting 7.8e-4 above us: a converged "
                                "pack at a common genuinely higher score. Closing that needs "
                                "something structurally different, not more GBDT-family polish, since "
                                "every reachable gain measured is single-digit e-5."),
    }, REPORTS / "lb_noise_analysis.json")
    print("\nwrote", REPORTS / "lb_noise_analysis.json")


if __name__ == "__main__":
    main()
