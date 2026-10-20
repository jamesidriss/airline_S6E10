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

## Expected value of the second slot

Under the competition's IID random test split, the private score of candidate *i* is
`OOF_i + b_i + ε_i`, where `b_i` is a family-specific OOF→test bias (unknown) and `ε_i` the
split noise. Because the private score is the **max** over the selected submissions, the second
slot is worth

`E[max(0, (OOF_B − OOF_A) + (b_B − b_A) + (ε_B − ε_A))]`.

With `OOF_B − OOF_A = −3.4e-4` (classical) or `−8.7e-5` (v6) and a paired split-noise of a few
e-5, the slot's expected value is ≈ 0 unless the OOF→test bias differs between families by more
than ~1e-4. The second slot is therefore a **hedge against systematic OOF bias**, not an
expected-value play — and the one family whose bias could plausibly differ from TabPFN's is the
classical GBDT ensemble.

## Recommendation

1. **Finalist A = v7**, ref **57090508** (unchanged; protected).
2. **Finalist B (recommended) = `v4_fulldata`, ref 56845975.** Rationale: largest training
   fraction of any submitted candidate (32 of 59 members refit on 100 % of labels), classical
   methodology (GBDT/RealMLP/TabM, no TabPFN), ρ_test = 0.9987 vs v7 — the most methodologically
   independent candidate we have. Its OOF deficit (+3.4e-4) is real but the slot is a free hedge
   against family-specific OOF bias.
3. **Acceptable alternative: v6**, ref 56951906 (smallest OOF deficit, +8.7e-5, ρ_test 0.9999).
   It hedges almost nothing (nearly identical to A) but never hurts.
4. **Do not** spend a submission on any new blend: no blend passes the gate, and the public board
   cannot resolve differences this small (AGENTS.md: submit only on an OOF gain ≳ +3e-4).

If the raw8 campaign (phase 23) certifies a candidate with a genuine OOF gain ≳ +3e-4 over v7,
that candidate displaces this recommendation and becomes Finalist B — a separate committed
`SUBMISSION_DECISION` would be required first.

## What the user must do (UI, no API)

1. Re-check the competition Rules page for the current two-selection rule.
2. In *My Submissions*, select the two finalists (A = 57090508, B = 56845975 as recommended) for
   private judging before 2026-10-31 23:59 UTC.
3. No other action is needed; nothing in this report authorizes an upload.
