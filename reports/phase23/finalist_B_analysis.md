# Phase 23 — Finalist B analysis (private-first)

Updated 2026-10-20 ~00:40 UTC. Analysis only: **zero submissions**, nothing here touches the
champion. Selection is a Kaggle web-UI action the user must perform before the deadline
(2026-10-31 23:59 UTC); this report supplies the refs and the evidence behind them.

## Question

The rules allow **up to 2** final submissions to be chosen for private judging (re-check the
Rules page near the deadline — AGENTS.md requirement). Finalist A is fixed: the champion
`v7_sol_tabpfn_internal2_aux10` (ref **57090508**, OOF `0.9618529228157235`). What should the
second slot be — or should it stay empty?

## Candidate table (all measured on the immutable primary folds, seed 20261010)

| candidate | store id | ref | OOF AUC | public | Δ = v7 − cand | paired SE | folds v7>cand | ρ(logit) OOF / test vs v7 |
|---|---|---|---|---|---|---|---|---|
| **v7 champion** | `v7_sol_tabpfn_internal2_aux10` | 57090508 | **0.961852922816** | 0.96146 | — | — | — | 1 / 1 |
| v6 frozen route aux10 | `sol_route_aux10_primary_final` | 56951906 | 0.961765613825 | 0.96144 | +8.731e-5 | 1.34e-5 | 5/5 | 0.999824 / 0.999896 |
| v3 59-member classical | `blend_v3_final` | 56805526 | 0.961508578580 | 0.96098 | +3.443e-4 | 3.41e-5 | 5/5 | 0.998703 / 0.998883 |
| v4 59-member, 32 refit on 100 % labels | `blend_v4_fulldata` | 56845975 | 0.961508578580¹ | 0.96100 | +3.443e-4 | 3.41e-5 | 5/5 | 0.998703 / 0.998658 |
| v2 34-member blend | `blend_v2_equal34` | 56793823 | 0.961438403316 | 0.96092 | +4.145e-4 | 3.48e-5 | 5/5 | 0.998674 / 0.998859 |
| v1 equal-4 | `blend_v1_equal4` | 56788249 | 0.961273344330 | 0.96093 | +5.796e-4 | 4.14e-5 | 5/5 | 0.965371² / 0.998298 |

¹ v4 shares v3's OOF vector by construction (same members, same weights; only the 32 test models
were refit on 100 % of labels). Its OOF is therefore *not* an independent measurement — it is the
same 0.961508578580, and the training-fraction effect is estimated, not measured, at +2.2e-5
(phase-22 `run_fullfit`, 2/2 folds, 72 %→80 % analogue).
² v1's OOF vector in the store is not the submitted blend's OOF (0.961273); treat its OOF
correlation as unreliable and use the test correlation.

The ten `endgame_*` submissions of 2026-10-19 were checked too: their OOF values are
0.961184–0.961502 (best: `endgame_01_cand_family_balanced` 0.9615021), i.e. all below v6 and
v7. The table above is the complete alternative set — no submitted candidate has a higher
OOF than v6.

## Blend scan: does adding a classical ensemble to v7 help?

Logit blend `expit((1−w)·logit(v7) + w·logit(cand))`, primary OOF, paired deltas vs v7
(full table in `reports/phase23/finalist_B_scan.json`):

| candidate | w | OOF AUC | Δ vs v7 | SE | folds + | gate (≥1.5e-5 and ≥2.5×SE) |
|---|---|---|---|---|---|---|
| v6 | 0.05 | 0.961851430 | −1.49e-6 | 6.6e-7 | 0/5 | fail (negative) |
| v3 / v4 | 0.05 | 0.961855683 | +2.76e-6 | 1.7e-6 | 5/5 | fail (needs ≥1.5e-5 and ≥2.5 SE = 4.3e-6) |
| v3 / v4 | 0.10 | 0.961856300 | +3.38e-6 | 3.5e-6 | 4/5 | fail (0.97 SE) |
| v3 / v4 | 0.25 | 0.961845398 | −7.5e-6 | 8.7e-6 | 1/5 | fail |
| v2 | 0.05 | 0.961852575 | −3.5e-7 | 1.8e-6 | 2/5 | fail |

**No blend qualifies.** The best observed blend gain (+3.4e-6 at w=0.10) is an order of magnitude
below the predeclared admission threshold and below the measurement noise floor of the public
board (±2e-4 paired). It is not a candidate and it is not a reason to spend a submission.

## Expected value of the second slot (measured, not assumed)

Under the competition's IID random test split, the private score of candidate *i* is
`OOF_i + b_i + ε_i`, where `b_i` is a family-specific OOF→test bias (unknown) and `ε_i` the
split noise. Because the private score is the **max** over the selected submissions,
selecting B alongside A can never lower the score — it is free insurance whose value is
`E[max(0, (OOF_B − OOF_A) + (b_B − b_A) + (ε_B − ε_A))]`.

Simulation: 200 random private subsets of 239,875 rows (the prereg's private size), scoring
each candidate against v7 on the same subset (`finalist_B_private_sim.json`):

| candidate | mean Δ vs v7 on the subset | sd | P(beats v7) | best of 200 |
|---|---|---|---|---|
| v6 | −9.08e-5 | 2.00e-5 | **0.000** | −2.10e-5 |
| v4 | −3.50e-4 | 4.73e-5 | **0.000** | −2.28e-4 |
| v3 | −3.42e-4 | 4.77e-5 | **0.000** | −2.30e-4 |

**Sampling noise alone never flips the ranking** — the margin is larger than any 80 %-subset
fluctuation. The slot can only pay if the OOF→test bias differs between families. Under a
bias-difference model with sd `σ_b`, `P(B wins) = Φ((OOF_B − OOF_A)/√(2σ_b² + 2·(paired sd)²))`:
v6 needs `σ_b ≈ 1.2e-4` to reach a 25 % chance; v4 needs `σ_b ≈ 4.6e-4` for the same.
Family-level biases of ~1e-4 are plausible; ~5e-4 are not. **v6 is the better hedge.**

## Recommendation

1. **Finalist A = v7**, ref **57090508** (unchanged; protected).
2. **Finalist B = v6 `sol_route_aux10_primary_final`, ref 56951906.** Highest OOF of every
   other submitted candidate (0.9617656, −8.7e-5 vs v7, ρ_test = 0.99990), so it is the
   closest alternative and the most likely to win under any plausible family bias. Selecting
   it costs nothing (max-of-two), so it is pure insurance.
3. **v4_fulldata (ref 56845975) is the alternative if methodological independence is
   preferred**: classical 59-member GBDT ensemble with 32 members refit on 100 % of labels,
   ρ_test 0.99866. It is *quantitatively dominated* as a hedge (−3.5e-4, P(win) ≈ 0.01 under
   a 1e-4 bias model), but it is the only candidate whose failure mode is genuinely
   independent of TabPFN. Choose it only if the goal is insurance against a TabPFN-specific
   collapse rather than expected score.
4. **Do not** spend a submission on any new blend: no blend passes the gate, and the public
   board cannot resolve differences this small (AGENTS.md: submit only on an OOF gain
   ≳ +3e-4).

If the raw8 campaign (phase 23) certifies a candidate with a genuine OOF gain ≳ +3e-4 over v7,
that candidate displaces this recommendation and becomes Finalist B — a separate committed
`SUBMISSION_DECISION` would be required first.

## What the user must do (UI, no API)

1. Re-check the competition Rules page for the current two-selection rule.
2. In *My Submissions*, select the two finalists for private judging before 2026-10-31
   23:59 UTC: **A = 57090508 (v7)** and **B = 56951906 (v6)** — or 56845975 (v4) if
   insurance against a TabPFN-specific collapse is preferred over expected score.
3. No other action is needed; nothing in this report authorizes an upload.
