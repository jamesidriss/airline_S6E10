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
        for rho in (0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995):
            se_d = se * math.sqrt(2 * (1 - rho))
            z = (LEADER_PUBLIC - OUR_PUBLIC) / se_d if se_d > 0 else float("inf")
            verdict = ("indistinguishable" if z < 2 else
                       "marginal" if z < 3 else
                       "probably real" if z < 5 else "real")
            print(f"    rho={rho:<6} SE_diff={se_d:.6f}   z={z:>6.2f}   {verdict}")
            rows.append({"pos_rate": frac_pos, "rho": rho, "se_single": se, "se_diff": se_d,
                         "z": z, "verdict": verdict})
        print()

    # Correlation at which the gap reaches 2 sigma, solved directly from z = 2:
    #   se_diff = gap/2  and  se_diff = se * sqrt(2(1-rho))
    #   =>  rho = 1 - (gap/2)^2 / (2 se^2)
    # rho = 0 means the two models rank the public rows INDEPENDENTLY, in which case a 7.8e-4
    # realised gap is entirely ordinary. We do NOT hold the leader's prediction vector, so rho is
    # UNMEASURED and no single z can be asserted for 'us vs the leader'.
    se_mid = next(r["se_single"] for r in rows if r["pos_rate"] == 0.24)
    gap = LEADER_PUBLIC - OUR_PUBLIC
    rho_2sig = 1 - (gap / 2) ** 2 / (2 * se_mid ** 2)
    z_at_rho0 = gap / (se_mid * math.sqrt(2))
    print(f"  correlation needed for the gap to reach 2 sigma : rho = {rho_2sig:.4f}")
    print(f"  z at rho = 0 (independent rankings)             : {z_at_rho0:.2f}")
    print("  We do not have the leader's prediction vector, so rho is UNMEASURED. Any single z-score")
    print("  for 'us vs the leader' would be a guess dressed as a statistic.")

    print("\n" + "-" * 96)
    print("  THE STRATEGIC POINT (derived from the table above, not asserted)")
    print("-" * 96)
    se_lo = min(r["se_single"] for r in rows)
    se_hi = max(r["se_single"] for r in rows)
    zs = [r["z"] for r in rows]
    print(f"""  Two claims made in this file were wrong, and both erred toward overstating certainty. They
  are corrected here rather than edited out, because the arithmetic that produced them was checked
  and looked plausible.

  1. A first pass, by hand, put the single-estimate SE at 0.0025-0.0036 and called the {gap:.6f} gap
     'maybe 0.2 sigma, i.e. noise'. Hanley-McNeil gives {se_lo:.6f}-{se_hi:.6f}, about three times
     smaller. That hand figure was simply wrong.

  2. Having corrected one overstatement, the script introduced the OPPOSITE one. It computed the
     correlation needed for a 2-sigma gap as `1 - (min_z/2)**2 / 2`, which is not the inversion of
     the z formula, got a negative number, clamped it to zero, and printed

         'the gap clears 2 sigma even between two UNCORRELATED predictions'

     That is FALSE and the script's own numbers said so: at rho = 0 the z is {z_at_rho0:.2f}, not above
     2. The correct inversion is rho = 1 - (gap/2)^2 / (2 se^2) = {rho_2sig:.4f}.

  Where that leaves the gap, stated no more strongly than the evidence allows:

    * The scores are factual: ours {OUR_PUBLIC:.6f}, leader {LEADER_PUBLIC:.6f}, gap {gap:.6f}.
    * Significance of the difference between two models is conditional on how similarly they rank the
      public rows -- on rho -- which is UNMEASURED because we do not hold the leader's vector.
    * Across rho in [0, 1] the z ranges from {min(zs):.2f} to {max(zs):.2f}. The gap clears 2 sigma
      only once rho exceeds {rho_2sig:.3f}.
    * Two strong tabular solutions on the same 699k rows and the same metric would plausibly sit well
      above that threshold, so 'probably real' is a reasonable WORKING BELIEF. It is a belief, not a
      measurement, and it is recorded as one.

  What does NOT depend on rho, and is therefore the part that should govern compute:
    * the top twenty span 1.0e-4 while sitting {gap:.6f} above us, so they are a converged pack at a
      common level, not twenty teams spread out ahead of us;
    * every gain this campaign has actually MEASURED is single-digit e-5, and the one mechanism that
      was mechanistically sound delivered +2e-5 on this very board -- itself unresolved at
      leaderboard precision. Whether or not the gap is statistically real, more GBDT-family polish
      will not close 7.8e-4.""")
    save_json({
        "n_public_rows": N_PUBLIC, "our_public": OUR_PUBLIC, "leader_public": LEADER_PUBLIC,
        "gap": LEADER_PUBLIC - OUR_PUBLIC, "our_oof": OUR_OOF,
        "table": rows,
        "hanley_mcneil_note": ("normal approximation, roughly right for AUCs near 0.96; used to bound "
                              "the plausible range of z, never as a point estimate"),
        "self_correction": (
            "TWO overstatements, in OPPOSITE directions, both corrected here. (1) An initial hand "
            "estimate put the single-estimate SE at 0.0025-0.0036 and called the gap 'maybe 0.2 "
            "sigma, i.e. noise'; Hanley-McNeil gives 0.0011, about three times smaller, so that hand "
            "figure was simply wrong. (2) Having fixed that, the script introduced the opposite error: "
            "it computed the correlation needed for a 2-sigma gap as 1-(min_z/2)^2/2, which is NOT the "
            "inversion of the z formula, obtained a negative number, clamped it to zero, and printed "
            "that the gap 'clears 2 sigma even between two UNCORRELATED predictions'. That is false "
            "and the script's own table said so: at rho=0 the z is 0.49. The correct inversion is "
            "rho = 1-(gap/2)^2/(2*se^2) = 0.941."),
        "gap_is_significant": False,
        "rho_is_measured": False,
        "z_range_over_rho": [min(r["z"] for r in rows), max(r["z"] for r in rows)],
        "rho_at_2_sigma": rho_2sig,
        "interpretation": (
            "the +-2e-4 paired floor in AGENTS.md applies between near-identical submissions and is "
            "the correct gate for that decision. For a DIFFERENT solution the pairwise error "
            "correlation is UNMEASURED, because we do not hold the leader's prediction vector. The gap "
            "is therefore recorded as a score difference, NOT as a significance claim. It clears 2 "
            "sigma only if rho > 0.941; treating that as established was an error."),
        "strategic_conclusion": (
            "rho-independent, and therefore the part that should govern compute: the top 20 span 1.0e-4 "
            "while sitting 7.8e-4 above us (a converged pack, not a spread field), and every gain "
            "actually MEASURED in this campaign is single-digit e-5. More GBDT-family polish will not "
            "close 7.8e-4, regardless of whether the gap is statistically real."),
    }, REPORTS / "lb_noise_analysis.json")
    print("\nwrote", REPORTS / "lb_noise_analysis.json")


if __name__ == "__main__":
    main()
