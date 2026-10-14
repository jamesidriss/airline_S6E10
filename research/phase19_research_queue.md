# Phase 19 — ranked research queue

Generated 2026-10-14, after the shared-error Atlas (H19-ATLAS) and the two audits
(neural/classical gap, public sources + zero-rating probe). Registry:
`research/phase19_hypothesis_registry.jsonl` (preregistered, committed before results).

Champion A unchanged: `v7_sol_tabpfn_internal2_aux10` = 0.961852923 OOF (hash-verified
byte-exact this session). Champion B remains **UNRESOLVED**. Stretch target
0.962100000 requires **+2.47e-4** over champion A (exact: 0.962100000 − 0.9618529228157235 = 0.0002470771842765).

## Atlas headline numbers (all asserted exact; `reports/phase19/shared_error_atlas.json`)

| quantity | value |
|---|---|
| residual AUC mass (= 1 − champion AUC) | 0.038147 |
| **shared fail pairs** (route+aux fail same pair) | **3.932e9 = 85.3 % of portfolio fail** |
| shared AUC mass | 0.032547 |
| route-only / aux-only fail mass | 0.005937 / 0.006034 |
| aggregation-loss mass (one half right, average wrong) | 0.005600 (15 % of residual) |
| **shared-difficulty predictability (covariates only)** | **AUC 0.9318** (folds 0.9310–0.9323, SE 0.0002) |
| satisfaction-context AUC of same model | 0.534 (it locates *difficulty*, not the label) |
| teacher-confident-wrong enrichment, top-burden quintile | **5.0×** (0.273 vs 0.0547) |
| unique_rating_key subset | 254,260 rows (36 %), port AUC 0.95032 |

Reading: 85 % of the residual is *shared* — unreachable by reweighting or averaging the
two halves (both closed axes anyway). Hard rows are covariate-locatable (0.9318) but
locating ≠ fixing; the 5× teacher enrichment suggests hard rows deviate from
original-distribution structure (partly irreducible). The reachable levers are new
inductive biases that are *right where both halves are wrong*, plus the 15 %
aggregation-loss pool.

## Ranked hypotheses (at most 6; top 3 executed)

| rank | id | one-line | prior | status |
|---|---|---|---|---|
| 1 | **H19-A** | difficulty-targeted reweighted member; right on shared-fail pairs | moderate-low | **EXECUTE** (fold 0 first) |
| 2 | **H19-B** | aux10 deployment policy: bag-of-fold vs full-fit, pseudo-test | low (S6E9 +2.6e-4 precedent, non-transferable) | **EXECUTE** (CPU, pseudo-test standard) |
| 3 | **H19-C** | smoothed nearest-key generalization on unique-key rows | very low | **EXECUTE** (fold-0 screen only) |
| 4 | H19-D | aggregation-loss tiebreaker / champion B | unresolved | reserve, no candidate |
| 5 | H19-E | neural reserve | strongly negative | reserve, do not spend GPU |

### Rank 1 — H19-A: difficulty-targeted reweighted member
The only admissible mechanism that directly targets the 85 % shared residual using the
Atlas's strongest new finding (difficulty AUC 0.9318). Reweighting the member's training
rows by an inner-cross-fitted difficulty score is a new inductive bias (training-distribution
reweighting), not in the closed list. The negative-leaning prior: the 5× teacher
enrichment suggests hard rows may be label noise (irreducible). The fold-0 falsifying
experiment settles it — the specialist must be *right* on shared-fail pairs, not merely
trained on them. **Fold-safe note:** the Atlas outer `pred_a` must NOT be reused as
training weights (its training folds include the eval fold → leak); recompute with an
inner cross-fit inside the FIT fold, mirroring TE.

### Rank 2 — H19-B: aux10 deployment-policy pseudo-test
The aux10 test policy (single full-label fit, 100 %) is asymmetric with the OOF
validation (fold-fit, 80 %). The route side already matches (5 contexts at 80 % both
OOF and test); the aux side does not. S6E9's strongest transferable lesson is that a
deployment bagging-policy change moved both finalists +2.6e-4 private. Measured on a
pseudo-test (carve a holdout from train). **Cannot pass the OOF admission gate** — it
changes test predictions only, OOF is unchanged. Treated as a robustness signal for the
2 final submissions, with the S6E9 caveat that pseudo-test deltas do not transfer cleanly.

### Rank 3 — H19-C: smoothed nearest-key on unique-key rows
The unique-rating-key subset is the largest contiguous weak region (36 % of rows,
AUC 0.95032). H18-C closed *exact* key-TE (+6.8e-6, insufficient). Nearest-key smoothing
is a different mechanism but the prior is very low: members already see all 13 ratings
and generalize across similar keys via splits. Fold-0 screen only.

### Rank 4 — H19-D: aggregation-loss tiebreaker / champion B (reserve, no candidate)
Aggregation-loss mass 0.0056 (15 % of residual). A third member strong AND decorrelated
enough on these pairs could shift the average. But all measured members are ≥0.995
logit-correlated and weaker; Phase 7 showed decorrelation (0.87) is insufficient if the
member is weak. Champion B remains UNRESOLVED — do not invent a hedge.

### Rank 5 — H19-E: neural reserve (strong negative prior)
RealMLP full view 0.96079–0.96088, TabM 0.96069, corrected RealMLP f0 0.96076 — all
~1e-3 below champion. H18-B: a weaker member hurts the portfolio. The audit disproven
"neural lacked the rich view" (historical members already used it). Reopen only with a
concrete architecture hypothesis that plausibly closes a 1e-3 gap. Do not spend GPU.

## Closed this cycle (do not reopen)
- **Zero-rating / not-offered semantics**: already implemented in `src/features/s6e10.py`
  (`na_*` indicators, zero-aware aggregates `rate_*`, zero×segment crosses). The public
  notebook's insight is real but absorbed. Not a new feature block.
- **Conditional-density / class-conditional generative ratio**: closed by Phase 7
  `p7_class_conditional_generative` (standalone 0.931, all blend weights negative,
  logit-corr 0.87 — the most decorrelated member ever, still useless).
- **Residual group-bias offsets**: closed by Phase 10 (+2.00e-5 did not transfer to v3).
- Standing closed branches (unchanged): generic GBDT sweeps, DART, RF, Ordered CatBoost,
  key/all-column TE, TabR, original appending, distillation, pseudo-labels, SSL32,
  TabPFN Fast, 2→4 estimators, 80→90 % context, H18-C key-TE, H18-B stump,
  probability/rank averaging, imported stacks.

## Decision rules (frozen)
- Admission gate: mean paired fold gain ≥ 1.5e-5 AUC, ≥ 2.5× paired SE, positive ≥ 4/5
  folds, vs the current stack; prefer pooled ≥ 3e-5.
- Submission: at most 2 new justified submissions (0 of 2 used); pre-result manifest +
  hashes + commit before upload; only on OOF gain ≳ +3e-5 with a written reason
  (H19-B is pseudo-test standard and cannot satisfy this — it is finalist-selection
  evidence, not an OOF promotion).
- Never fit to the public leaderboard. Public LB is a measurement.
