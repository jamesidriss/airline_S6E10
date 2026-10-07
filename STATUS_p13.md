# STATUS_p13.md — Phase 13 complete

## Verdict

**NO ADMISSION. NO SUBMISSION.** The auxiliary-task mechanism is **validated as real** — it is the
first genuinely positive result in this campaign — and it is **still not admissible**, falling short on
effect size alone. Both facts are reported without rounding toward either.

`v3_final` (OOF 0.961509, public 0.960980) and `v4_fulldata` (public 0.961000) remain the immutable
finalists. 4 submissions used; none today.

## What the mechanism is

Every model here predicts *satisfaction* from the columns. This asks a different question of the same
data: fit a model predicting **one rating from the other columns**, with no satisfaction label
anywhere, then feed `Σ_k k·P(k)` — the expected value of that rating — as a feature.

Absent from our 285-column view. We have 48 `te_` + 54 `ogte_*` + 2 `teach_*`, but our teacher predicts
*satisfaction* from the original dataset, which is a different mechanism: there is **no per-rating
auxiliary target** here. Not the closed `teacher` branch, not the closed all21-TE branch.

Two independent public sources report it, and that convergence is why it was worth the compute.

## Results — five primary folds, from real prediction vectors

| configuration | per fold (e-5) | mean | SE | t | positive |
|---|---|---|---|---|---|
| **single slot** standalone | +21.8, +16.4, +17.3, +3.9, +1.9 | **+12.25e-5** | 3.94 | +3.11 | 5/5 |
| **single slot** marginal on v3 | +0.31, +0.25, +0.48, +0.14, +0.15 | **+0.27e-5** | 0.06 | +4.24 | 5/5 |
| **all 6 extra_trees slots** marginal | +0.706, +1.164, +0.468, +0.224, +1.160 | **+0.745e-5** | 0.187 | +3.99 | 5/5 |

**+12.25e-5 standalone is the largest standalone effect measured anywhere in this campaign** — Phases
11 and 12 produced at most +6e-5. The marginal is the first consistently positive *and* statistically
clean one (t +4.24 single-slot, t +3.99 all-slot, both 5/5).

**Gate:** mean ≥ +1.5e-5 → **2.0× short, FAIL**. ≥4/5 positive → 5/5, PASS. mean ≥ 2.5×SE → 3.99×,
PASS.

Six slots buy **2.8×**, not 6× — the effect compounds but sub-linearly. Extrapolating to the remaining
extra_trees members is **refused**: Phase 11R withdrew exactly that arithmetic, and the measured
cumulative curve is non-monotonic in slot order (fold 0: +0.31, +0.63, +0.69, +0.93, +0.79, +0.71), so
extrapolating would invent precision.

## The chaos finding — how to read any extra_trees number

A sensitivity probe reran the aux build on fold 0 at 400 rounds/5 inner folds instead of 250/3. The
auxiliary models were **identical** — mean OOF accuracy 0.6998 both times, matching to three decimals
for all 13 ratings — yet the champion's standalone AUC moved 0.961320 → 0.961047, a **27e-5 swing**.

The features were stable; the model was not. `extra_trees` uses randomised thresholds and per-node
feature subsampling, making the fit **chaotic under perturbations far below the level that changes any
feature's meaning**.

Two consequences:
1. A single fold's standalone delta for this family is **not reproducible**. The fold-0 figures above
   are directionally supported, not precise. The 5/5 sign consistency and t +3.11 survive because they
   are computed *across* folds.
2. The **marginal is unaffected** (+0.31 vs +0.24e-5 at the two settings). Pair ranking is robust to
   tree-level chaos in a way one model's AUC is not.

This retrospectively justifies Phase 12 having measured its marginal rather than inferring it, and it
means the measured ensemble marginal is the trustworthy quantity for this family.

## Evidence the features are real, not inert

- Champion spends **10.0–10.2%** of its splits on the 13 aux columns.
- Aux OOF accuracy mean **0.6998**, from 0.313 (Checkin service) to 0.954 (Inflight entertainment) —
  the mechanism carries structure, and varies sensibly by how constrained each rating is.
- Each `aux_ev_j` correlates **0.81–0.97** with its rating (RMSE 0.33–0.75): genuine smoothers, and
  **none reaching 1.0**, which confirms the cross-fitting is doing its job.

## Five defects found, all mine, all pinned

1. **A label-free check that could not fail.** Flipping every satisfaction label and asserting the aux
   features were unchanged — impossible either way, since `build_aux` receives only the label-free view
   matrix. Third instance in this campaign (after the Phase 10A statistic that returned r=1.000 for
   all 42 keys, and the audit's `"||"` detector that called a working treatment inert). Replaced with a
   **structural** guarantee that can fail: the signature is asserted to contain no label argument.
2. **The aux features were memorised on fit rows.** A model queried on its own training rows learns
   their ratings nearly perfectly, so `aux_ev_j` was a near-duplicate of `rating_j` during training and
   smoothed at validation — a train/serve mismatch invisible in AUC. Fixed with inner cross-fitting.
   The demonstration of the gap was itself wrong twice first (target left in the design matrix → 1.000
   both ways; then above-chance demanded on pure noise, where 1/6 *is* correct).
3. **A silently wrong slot seed.** Parsing seeds from `exp_id` crashed on one slot and silently gave
   `xt_xt_d255` seed 2 instead of 4. Now explicit, cross-checked against the ledger, aborts on
   disagreement.
4. **The headline was overwritten.** `p13b_runs.json` held only folds 3–4 after the second
   invocation. `p13b_final.py` recomputes from the `.npy` vectors; the recomputed **+0.745e-5 matches
   the logs**, so no number changed — only the record was made durable.
5. **Phase 12's own**, from earlier in this session: two fold-index coincidences (fixed seed vs
   `seed+k`, `inner_seed=0` vs `inner_seed=k`) that both hold on fold 0 and diverge after.

## Where this leaves the campaign

The 7.8e-4 leaderboard gap remains unexplained. But this session produced the first mechanism in a
long while that is **both** real **and** positively signed at the ensemble level, and the reason it
cannot be harvested is now measured rather than assumed: a single extra_trees slot is 1/59 of the
blend, and six of them buy 2.8× rather than 6×, because each new prediction correlates 0.998–0.999
with the member it replaces.

That is a specific, falsifiable statement about where the ceiling is, and it implies the next thing
worth trying is not more slots of this model but a mechanism that changes the *pair ranking* rather
than one model's score.

## Tests, ledger, git

- 8 suites, **705 assertions, 0 failures** (`run_tests` 55, stochastic 69, native_cat 101,
  native_block 56, residual_nested 62, phase11r 90, phase12 224, phase13 48).
- Ledger **176** append-only lines, including every withdrawn claim.
- Durable records: `reports/p13_runs.json`, `reports/p13b_final.json`, `reports/p12_final.json`,
  `reports/lgb_native_cat_audit.json`, `reports/public_notebook_pull_manifest.json`.