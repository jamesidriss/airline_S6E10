# STATUS_p14.md — Phase 14 complete, S1 submitted

## Headline

**v5_aux_cross scored 0.96103 — the highest public score of all four submissions** (v3 0.96098,
v4 0.96100). OOF predicted +1.49e-5; public moved +5e-5. Same sign, larger magnitude, inside the
±2e-4 paired noise floor. Transfer is **consistent with CV**, not contradictory.

## Phase 14 model table (5 primary folds, matched ctrl/aux pairs, deltas are aux-minus-own-control)

| family | config | folds | base AUC | aux AUC | Δ standalone | O/A corr | single-slot v3 Δ | runtime |
|---|---|---|---|---|---|---|---|---|
| xgb | X0 `prod5_xgb_full_primary` | 0-4 | — | — | **+9.8e-5** | 0.9975-0.9978 | **+0.286e-5** (5/5) | 8-11 s |
| xgb | X1 `xt_xgb_lossguide` | 0-4 | — | — | **+8.7e-5** | 0.9976-0.9979 | +0.228e-5 (4/5) | 22-30 s |
| xgb | X2 `zoo_xgb_d6` | 0-4 | — | — | **+10.2e-5** | 0.9962-0.9978 | **+0.332e-5** (4/5) | 6-12 s |
| cat | C0 numeric + aux | 0-4 | — | — | **+10.3e-5** | 0.9970-0.9981 | +0.106e-5 (3/5) | 95-176 s |
| cat | C1 native-cat + aux | 0-4 | — | — | **+8.2e-5** | 0.9966-0.9982 | +0.203e-5 (5/5) | 426-939 s |

Fold-0 base→aux AUCs: X0 0.960991→0.960924; X1 0.960852→0.960930; X2 0.960828→0.961011;
C0 0.960841→0.961055; C1 0.960945→0.961003.

**Controls reproduce their stored members** where the runner is shared (X0 +0.0e-5), which is what
licenses the deltas.

## Cross-family table

| candidate | OOF AUC | Δ vs v3 | folds+ | corr v3 | Spearman | rescue | damage |
|---|---|---|---|---|---|---|---|
| AUX_cat1 (1) | 0.961510 | +0.15e-5 | 5/5 | — | 0.99999 | 0.0025 | 0.0001 |
| AUX_xt6 (6) | 0.961516 | +0.72e-5 | 5/5 | — | 0.99996 | 0.0061 | 0.0002 |
| AUX_xgb3 (3) | 0.961517 | +0.80e-5 | 5/5 | — | 0.99997 | 0.0056 | 0.0002 |
| AUX_xgb3_cat1 (4) | 0.961518 | +0.93e-5 | 5/5 | — | 0.99996 | 0.0067 | 0.0002 |
| **AUX_CROSS (10)** | **0.961523** | **+1.491e-5** | **5/5** | — | 0.99989 | **0.0103** | 0.0003 |

Per fold: **+0.92, +2.80, +1.54, +0.64, +1.78e-5**, paired SE 0.376, **t = +3.97**.

Per-slot value: **0.12e-5 for extra_trees, 0.27e-5 for XGBoost** — corrections stack better across
families than within one, because a member can only fix pairs its near-clones cannot.

## Aux + native CatBoost do **not** compound

numeric 0.960841 · native-cat 0.960945 · numeric+aux 0.961055 · native-cat+aux 0.961003.
native+aux < numeric+aux. Aux is **3.7× weaker** on native-cat. **No synergy claimed.**

## Private hedge — the conclusion reverses, pessimistically

| candidate | OOF | gap vs v3 | corr v3 (OOF) | corr v3 (TEST) | rescue | role |
|---|---|---|---|---|---|---|
| H1_family_diverse | 0.961496 | −1.3e-5 | **0.99993** | **0.99995** | 0.0190 | **not a hedge** |
| H0_nonXT | 0.961468 | −4.0e-5 | **0.99986** | 0.99989 | 0.0270 | **not a hedge** |
| tabm alone | 0.960797 | −71e-5 | 0.99725 | 0.99798 | — | fails eligibility |

Two structural findings:
1. **My earlier hedge correlations were a sign function.** `base` is the mean *logit*; `lg()` clips to
   [1e-6, 1−1e-6] first — correct for a probability, catastrophic for a logit. **97.87%** of values
   saturated. True correlations are 0.999+, not 0.966.
2. **Test-side diversity collapses.** Every test-time member is refit on the *same* 100% of labels,
   whereas each OOF member saw a different 80% subset. So OOF *overstates* the diversity that
   reaches the leaderboard.

**Consequence: no reweighting of v3's members is a private hedge.** The 59-member equal-logit average
has already averaged away the diversity a hedge would exploit. S2 was **built then DECLINED** — its
CSV has test corr 0.99995 with v3. Submitting it would have spent one of two authorised submissions
on a proven non-hedge. The slot is preserved.

## Submission table

| name | role | OOF evidence | test corr vs v3 | public | Δ public | ref | interpretation |
|---|---|---|---|---|---|---|---|
| v3_final | previous A | 0.961509 | 1.0 | 0.960980 | — | 56805526 | baseline |
| v4_fulldata | previous B | 0.961509 (unchanged) | ~0.9999 | 0.961000 | +2e-5 | 56845975 | full-data policy |
| **v5_aux_cross** | **Champion A** | **+1.491e-5, 5/5, t+3.97** | 0.99999 | **0.96103** | **+5e-5** | **56918343** | **transferred, same sign** |

Expectation was predeclared as UNKNOWN/likely below resolution. No score was predicted, none will be
tuned toward, and no weight variants will be submitted.

## Answers

1. **Aux help XGBoost?** Yes — +8.7 to +10.2e-5 standalone, per-slot marginal up to +0.332e-5.
2. **Aux help numeric CatBoost?** Yes — +10.3e-5 standalone, +0.106e-5 marginal (3/5).
3. **Aux + native CatBoost compound?** **No.** Aux is 3.7× weaker on native-cat; native+aux <
   numeric+aux. No synergy claimed.
4. **Can a cross-family block improve v3?** **Yes** — +1.491e-5, 5/5, t +3.97.
5. **Champion A now?** `v5_aux_cross`.
6. **Best private hedge?** **None exists** among current artifacts.
7. **Is v4 still the second-final candidate?** **Yes**, but not as a hedge — nothing better exists.
8. **v5_aux6 public?** n/a — **v5_aux_cross** submitted instead (the stronger validated mechanism);
   **0.96103**.
9. **v6_private_hedge public?** **Not submitted.** Built, measured at test corr 0.99995, declined.
10. **Did either contradict CV?** No. v5 moved +5e-5 vs CV's +1.49e-5 — same sign, inside noise.
11. **HEAD** `fabbb6c` (+ the ledger commit below). **Ledger 184 lines.**
12. **Tests** 8 suites, **705 assertions, 0 failures**.
13. **Two candidates for the final private board:** **`v5_aux_cross` (A)** and **`v4_fulldata` (B)** —
    with the explicit caveat that B is not a genuine hedge.

## Five defects found, all mine

1. CatBoost ES model handed the early-stopping **carve** as its fit frame with train labels — a length
   mismatch, and once length-matched, early stopping on rows it trained on (the Phase 11R defect).
2. Column labels reused from the 285-name view for the 298-column aux variant.
3. `delta_vs_stored` conflated harness reproduction gap with treatment effect (X1/X2 gaps of
   +1.1e-5 / −5.0e-5 would have been charged to the treatment).
4. **The per-fold table was fabricated** — a full-OOF AUC subtracted from per-fold AUCs, which made the
   *control* non-zero (+0.79, −14.15, −0.35, +71.48, −61.28e-5). The aggregate was unaffected. The
   control now asserts its per-fold delta is exactly zero.
5. **Two index-space errors** in the test builder: `ViewBuilder` keeps `static_tr` and `static_te`
   separately; test rows come through the 4th argument, not the `val` apply set.

Plus the run-scoped overwrite for a third time — refit lengths recovered from logs with a guard that
refuses a median over fewer than five folds.