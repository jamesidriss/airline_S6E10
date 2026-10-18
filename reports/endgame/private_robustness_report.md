# Private robustness report — champion v7 as the private-LB bet

**Generated:** 2026-10-19 UTC · **Champion:** `v7_sol_tabpfn_internal2_aux10`
**Raw artifacts:** `candidate_comparison.json`, `private_robustness_report.json`

## 1. The governing question

The private leaderboard scores ~80 % of the test rows. We cannot observe it. The
only honest inputs are (a) how well the model transfers from OOF to test, and
(b) how stable its ranking is under resampling and population shifts. Everything
below is a *validation-based estimate*, never an observation of the private split.

## 2. Is the test population the same as the training population?

Yes — measured, not assumed.

| covariate audit | value |
|---|---|
| max standardized mean difference (numeric: Age, Flight Distance, delays) | **0.0033** |
| max categorical total-variation distance (17 columns incl. all 13 ratings) | **0.0029** |

For a synthetic dataset (S6E10 is generated), this is the expected picture: train and
test are draws from one generator. **Consequence: an OOF-validated ranking is the
best available estimator of private performance.** There is no distribution-shift
correction to apply, and no segment that is over- or under-represented privately.

## 3. Does the champion behave the same way on test as on OOF?

| statistic | OOF | test |
|---|---|---|
| mean score | 0.444194 | 0.444115 |
| std | 0.433338 | 0.433048 |
| fraction > 0.5 | 0.433569 | 0.433876 |
| fraction > 0.9 | 0.384092 | 0.383946 |

Member agreement on the test set (archived route vs aux10 member vectors):
Spearman **0.977**, logit-Pearson ≈ 0.98 — the same relationship the equal-logit
blend geometry sees on OOF. Nothing about the blend's behaviour changes off the
OOF sample.

## 4. How precise is the champion's OOF estimate?

| estimate | value |
|---|---|
| OOF AUC | 0.9618529228157235 |
| within-fold row bootstrap (B=1000) | 0.961854 ± 2.5e-4, 95 % CI [0.961349, 0.962325] |
| per-fold spread | 0.961099 … 0.962496 (std ≈ 5.9e-4) |
| paired public-LB noise floor (historical) | ≈ ±2e-4 |
| stretch target | 0.962740 — at the extreme upper tail of the bootstrap distribution |

**Honest reading:** any candidate whose true gain is below ~2.5e-4 is not
distinguishable from noise by our own resampling. This is why the campaign refused
to promote sub-1e-4 "improvements" and why the ending is champion A rather than a
coin-flip marginal blend.

## 5. Rank stability and blend fragility (H94, true champion reference)

Every one of nine saved preprocessing/selection variants, blended equal-logit with
champion A, **degraded the portfolio on at least one fold** (min Δ −1.7e-4 … −7.0e-4).
The champion therefore ships **unblended**: its internal geometry is not robust to
additive perturbations, so adding "diversity" is a measured loss, not a hedge.

## 6. Segment view

Per-segment AUCs for the champion and every candidate were computed in
`candidate_comparison.json` (segments from the phase-18 error atlas: Class, travel
type, customer type, gender, age bands, distance quartiles, delay patterns, rating
extremes, zero-rating and constant-rating rows). No candidate beat the champion on a
segment basis while losing globally — the segments agree with the pooled verdict.

## 7. What this report does NOT claim

* It does not predict a private score, a rank, or a probability of winning.
* It does not treat the public LB (0.96146) as evidence for any decision.
* It does not claim that any synthetic resample equals the private 80 %.
* It does not claim the stretch target is reachable — only that it sits at the top
  tail of our estimate distribution.