# STATUS — Kaggle Playground S6E10 (Airline Satisfaction)

_Last updated: 2026-10-03 21:10 UTC._

## 1. Verified competition facts

| Item | Value |
|---|---|
| Metric | ROC-AUC (maximize), 5 dp truncation |
| Deadline | **2026-10-31 23:59 UTC** |
| Submissions | 10/day; **up to 2 final submissions** |
| Team size | 3 (merges allowed) |
| **External data** | **ALLOWED** (Rules §2.6) if public + free + equally accessible |
| Public LB share | 20 % (299,844 test rows) |
| Prizes | Kaggle merchandise, top 3 |
| Data licence | CC BY 4.0 |
| Public LB leader | 0.96158 (rank 1), 0.96105 at rank 100 |
| Our rank after submission #1 | **114 / 3624** at 0.96093 |

## 2. Validation scheme

- **primary** — `StratifiedKFold(5, shuffle, seed=20261010)`. Immutable, registered on disk in
  `data/cache/folds/` and hash-checked. All discovery CV.
- **shadow** — `StratifiedKFold(5, seed=777001)`. Independent confirmation only.
- **block10** — 10 folds, seed 20261010, for finalists.
- Early stopping uses an **inner 10 % split carved from the FIT rows**, never the evaluation fold.
  This makes our OOF ≈ 1e-4 *less* optimistic than the field's convention.
- Test predictions use a **second pass**: each fold model is refitted on all fold-fit rows at the
  median CV-selected iteration count. More data, no held-out label consulted.

## 3. Current results

### Submissions
| # | File | Members | OOF AUC | Public LB | Rank |
|---|---|---|---|---|---|
| 1 | `v1_equal4` | 4 (xgb, cat, lgbm, realmlp) | 0.961273 | 0.960930 | 114 |
| 2 | `v2_equal34` | 34, quality-weighted logit | 0.961438 | 0.960920 | — |
| 3 | `v3_final` | 59, equal-logit (**finalist candidate**) | **0.961509** | 0.960980 | 144 |

**Submissions used today: 3 / 10.**

All three land within 0.00006 of each other on the board despite a 0.00024 spread in OOF — a
direct confirmation of the ±0.0002 noise floor. The leaderboard has moved from 0.96158 to 0.96165
over the day; rank 100 is 0.96123, rank 25 is 0.96152.

### The LB-resolution calibration result (most important operational fact)
Submission #2 gained **+0.000165 OOF** over #1 and moved the board by **−0.00001**.
Test Spearman between the two files is 0.9954, and the *OOF* Spearman is also 0.9954 — OOF and
test orderings move together, so the CV is a faithful guide.

Paired public-LB noise is therefore **≈ ±0.0002** (public split = 59,969 rows ⇒ standalone AUC
SE ≈ 0.0015; paired SE ≈ 0.0002 at ρ ≈ 0.995). **Only OOF gains ≳ +0.0003 are resolvable on the
board.** Stop hill-climbing the public LB; reserve submissions for real steps and the finals.

### Best models and ensemble
| Item | OOF AUC (primary 5-fold) |
|---|---|
| best single (`z4_xt_f10_s4`, 10-fold extra_trees LightGBM) | **0.961260** |
| best single (`xt_xt_f10`) | 0.961242 |
| best RealMLP (`z5_rm_ens32`, n_ens=32) | 0.961017 |
| best CatBoost (`z3_cat_d10`) | 0.961056 |
| **59-member equal-logit blend** | **0.961509** |
| nested logit-LR stack (honest, out-of-sample w.r.t. the stacker) | 0.961508 |

Blend schemes at 59 members: equal-weight **0.961509** · family-balanced 0.961502 ·
quality-softmax 0.961493 · nested logit-LR 0.961508. All within 1e-5 — equal weighting is both
the best and the one with zero degrees of freedom, so it is the choice.

The ensemble is **saturated within the existing families**: a new member buys **+1e-6 to +5e-6**,
far below the 1.5e-5 admission gate.

### Feature-view ablation ladder (LGBM, primary folds)
| View | Blocks | n_feat | OOF AUC | Δ vs raw |
|---|---|---|---|---|
| raw | 21 raw columns | 21 | 0.959029 | — |
| raw_teacher | + original-only teacher (AUC 0.9550) | 23 | 0.959064 | +0.000035 |
| raw_ext | + original-data smoothed target stats | 75 | 0.960044 | **+0.001015** |
| raw_trans | + counts / route profile / digits / N-A masks | 148 | 0.960484 | **+0.001455** |
| raw_ogsurf | + original conditional surfaces | 189 | 0.960362 | **+0.001333** |
| core3 | raw + cat twins + trans + external + teacher | 225 | 0.960879 | **+0.001850** |
| core3_ogsurf | + original surfaces | 393 | 0.960888 | +0.001859 |
| core3_te | + fold-safe target encoding | 273 | 0.960827 | +0.001798 |
| **full** | + GPT-2 BPE token keys | 285 | **0.960904** | **+0.001875** |

## 4. Shadow-fold confirmation (independent seed 777001)

| claim | deterministic | extra_trees | verdict |
|---|---|---|---|
| extra_trees beats deterministic, `full` view (285 feat) | 0.960868 | **0.961143** (+0.000275) | **REPRODUCED** |
| extra_trees beats deterministic, `raw` view (21 feat) | 0.958959 | 0.953388 (**−0.005571**) | **NOT reproduced** |
| extra_trees beats deterministic, `raw_ext` (75 feat) | 0.960067 | 0.958506 (**−0.001561**) | **NOT reproduced** |
| external block helps | +0.001108 | +0.005118 | **REPRODUCED both** |
| transductive block helps | +0.001426 | +0.006595 | **REPRODUCED both** |

**Conclusion: `extra_trees` is an interaction with feature dimensionality, not a free win.**
Randomly-chosen features and thresholds need many candidate columns to land on a good split; with
21 features they cannot. Every `extra_trees` member in the ensemble is on a rich view, so the
ensemble is safe — but the flag must never be applied to a minimal view. This is recorded because
it is the most likely way to silently break this solution later.

## 5. Key discoveries

1. **Source dataset identified (VERIFIED)**: `arseniyshutko/binary-aviation-satisfaction-129k`
   (129,880 × 22). Exact schema fingerprint — 3 `Class` levels (no Premium Economy), 5
   `Baggage handling` levels (never 0), 75 `Age` levels, 100 % of synthetic values present in the
   original. Only **21** of 999,479 competition rows match an original row exactly, so there is no
   row-level answer key; those 21 are excluded from the teacher anyway.
2. **The original dataset is valuable as *knowledge*, not as *rows*.** Smoothed `P(y | exact
   value)` per column: **+0.001015**. Conditional surfaces anchored on Flight Distance / Class /
   travel type / online boarding: **+0.001333** alone. An original-only teacher reaches AUC 0.9550
   on competition train but adds only +0.000035 on top of the target statistics — its information
   is largely subsumed, so it is kept only for ensemble diversity.
3. **Duplicate feature groups carry NO recoverable label signal (decisive).** 48.96 % of rows
   share an (13 ratings + 4 categoricals) key, yet a fold-safe target encoding on those exact keys
   scores only AUC 0.8677 against 0.9609 for the full model. The generator drew labels **i.i.d.
   per row** from p(x): duplicate rows share p(x) but not their noise draw. Therefore there is no
   leak through duplicates, target encoding cannot help (−0.00005 measured), and the problem is
   bounded by how well p(x) can be estimated from ~700k noisy labels.
4. **`extra_trees=True` is the largest single-model lever found: +2.3e-4** over a deterministic
   LightGBM of the same size (0.960833 → 0.961175), reproduced on shadow folds at +0.000275.
   **Not used anywhere in the public S6E10 field.** Mechanism: with i.i.d. label noise a
   deterministic GBDT grows splits until the *training* labels are nearly pure, fitting the noise
   draw; extremely randomised trees are much weaker learners whose errors decorrelate, so averaging
   cancels more noise while keeping the p(x) component.
5. **Raising RealMLP's internal averaging budget works the same way**: `n_ens` 8 → 32 gives
   0.960820 → **0.961017** (+0.0002). Same mechanism, and the only other lever of that size.
6. **10-fold beats 5-fold by ~+1.1e-4 and additionally decorrelates the member** (different 90 %
   subsets): 0.961136 (5-fold xt) → 0.961242 (10-fold xt).
7. **`id` carries no signal.** Every digit / modulo / divisor probe gives AUC 0.5000–0.5009; 20
   equal-width id bins all have positive rate 0.440–0.450. Adversarial validation without `id`:
   **AUC 0.50027**. All 21 marginal KS tests p ≥ 0.084. **No covariate shift** — the community's
   reported `Cleanliness` p = 0.0376 is one false positive out of 21 tests, exactly as expected.
8. **`Flight Distance` behaves as a route ID** (3,474 levels, ~200 rows/value). Route-profile group
   statistics keyed on it are a large part of the `trans` block gain, and 100 % of synthetic
   Flight Distance values occur in the original.
9. **Survey `0` is an N/A sentinel, not the bottom of the scale**: `Online boarding == 0` → 62.0 %
   satisfied vs 10.5 % at 1; `Inflight wifi == 0` → 88.7 %. The original dataset contains exactly
   one `Baggage handling` zero. Encoded as explicit `na_*` masks; the standalone effect is not
   separable because the ratings are massively redundant (leave-one-out on `Online boarding`
   costs only 0.0006).
10. **Raising RealMLP's internal averaging budget works the same way**: `n_ens` 8 → 32 gives
    0.960820 → **0.961017** (+0.0002). Same mechanism, and the only other lever of that size.
11. **The `enrich` block is an `extra_trees` artifact, not new signal — verified in both
    directions.** Adding 94 label-free candidate columns (route spread/profile, more digit and
    token decompositions, rating differences, extra aggregates) to the `full` view:

    | model | `full` | `full_enrich` | Δ | folds + | bootstrap95 |
    |---|---|---|---|---|---|
    | `extra_trees=True` | 0.961092 | **0.961158** | **+0.000066** | **5/5** | [+0.000001, +0.000140] |
    | deterministic | 0.961092 | 0.960982 | **−0.000110** | 1/5 | [−0.000207, −0.000005] |

    A deterministic tree finds *no* extra signal in the new columns (it is hurt by their variance)
    while random splits gain from having more candidate columns. Both effects are significant, in
    opposite directions — exactly the predicted interaction, and direct evidence that the columns
    carry no label information. Ensemble marginal value: **−0.000001** (the 23 existing
    `extra_trees` members already absorb it). Block rejected.
12. **The original survey's own p(x) is learnable to AUC 0.9949 with duplicate-free grouped folds**
    (0.994876 grouped vs 0.994855 stratified — the naive 0.9948 was *not* duplicate inflation).
    On the synthetic data the best model reaches 0.9612.

    > **SCIENTIFIC CORRECTION (2026-10-04).** An earlier version of this note reasoned that
    > because i.i.d. label noise cannot lower an AUC *without changing the ordering of p(x)*,
    > the 0.9949-vs-0.9612 gap implied the synthetic generator learned a weaker p(x), and treated
    > 0.9615 as close to a ceiling. **That inference is wrong.** If labels were drawn as
    > Bernoulli(p(x)) samples, extra label noise reduces the *observed* AUC while leaving the
    > ordering of p(x) intact. So the gap is equally consistent with "p_syn = p_orig, plus a lot
    > of label noise" — in which case the *ordering* is fully recoverable and better
    > probability estimation should still pay.
    >
    > **0.9615 is therefore only the current modelling plateau.** Noise reduction and better
    > probability estimation are high-priority unexplored directions, and are pursued from here
    > (see the soft-target / denoising campaign below). The one thing the 0.9949 number *does*
    > still establish is unaffected: original **rows** teach the wrong conditional, which
    > discovery 14 measures directly and monotonically.
14. **The whole mechanism story closes: original *rows* teach the wrong function.** Measured
    directly — `full` view with `extra_trees`, appending the 129,859 leak-audited original rows to
    the training set:
    | original-row sample weight | OOF AUC | Δ vs excluded |
    |---|---|---|
    | 0.0 (excluded) | **0.961078** | — |
    | 0.3 | 0.960493 | −0.000585 |
    | 1.0 | 0.959975 | −0.001103 |
    The harm is **monotonic in weight**, exactly as the p-mismatch story predicts. Together with
    discovery 2 this gives one coherent account: the original rows carry a different, far sharper
    conditional than the synthetic labels, so training on them biases the model — but their
    *conditional target statistics* still transfer, because they encode feature-level structure
    (`Online boarding = 0` rows are far more satisfied) that survives even when the overall
    dependence is much weaker.
    **Original data: use it as knowledge, never as rows — now measured, not assumed.**

15. **Community notebook audit: the current leaderboard-leading technique does not survive an
    honest test.** `kozykappa/S6E10 | Local-Reliability Residual Blend` reports public LB 0.96152
    — above ours — by (a) consuming a *shared external OOF library* (`najiama/s6e10-oof`) plus four
    other notebooks, and (b) post-processing it: measure each model's local ROC-AUC inside
    equal-frequency score regions, then gate a correction by that reliability. The correction is
    fitted and then evaluated on the same OOF vector, so its number is not adoptable.
    We extracted the underlying question and tested it ourselves under nested validation
    (`scripts/local_reliability_gate.py`): region weights for a fold are computed only from the
    *other* folds' rows, so no gate is ever fitted on the rows it scores.
    | bin count | 5 | 10 | 20 | 40 |
    |---|---|---|---|---|
    | Δ vs equal weighting | −0.000000 | −0.000000 | −0.000000 | −0.000000 |
    **No member beats the equal-weight blend in any score region** — its local-AUC gain is negative
    in every bin (e.g. −0.0137 and −0.0065 mean gain for two sampled members). An oracle control
    (a member equal to `y`) was correctly detected at +0.434, so the machinery is not silently
    broken. Region gating is a fitting artefact; **rejected with evidence, and we keep our own
    reproducible models rather than importing someone else's OOF.**

## 6. Rejected hypotheses — do not re-spend

| Idea | Measured | Verdict |
|---|---|---|
| Exact-row lookup into the original dataset | +0.000038 | rejected |
| Fold-safe TE on duplicate feature keys | AUC 0.8677 standalone; −0.00005 in-model | rejected |
| **118 pairwise/triple original-data target tables (+236 features)** | **−0.000136, bootstrap95 [−0.000240, −0.000017], 1/5 folds positive** | **significantly worse** |
| Original conditional surfaces *on top of* `external` | +0.000009 | redundant |
| Original-only teacher *on top of* `external` | +0.000035 | diversity only |
| `extra_trees` on a **minimal** view (`raw`, `raw_ext`) | −0.0056 / −0.0016 on shadow folds | rejected |
| `id` digit / modulo / batch features | AUC ≈ 0.500 | rejected |
| RealMLP `batch_size` 256 vs 4096 (hypothesis: closes a 3.5e-4 gap) | +0.00004 — not the gap | rejected |
| **`enrich` block (94 extra label-free columns)** | **+6.6e-5 for extra_trees but −11.0e-5 for a deterministic tree; ensemble marginal −1e-6** | **rejected — artifact, not signal** |
| Appending original rows as training data | **−0.000585 at weight 0.3, −0.001103 at weight 1.0** (monotonic) | measured and rejected |
| TabM as a member | 0.960693 standalone; +0.000003 marginal | weak; diversity only |
| GPT-2 BPE token keys | +0.000025 | kept (cheap, non-negative) |
| Blend-geometry search (prob / logit / rank / LR / greedy) | spread ≤ 3e-5 | no gain available |
| Family-balanced / quality-softmax weighting | −0.000007 / −0.000016 vs equal | equal weighting wins |
| Pseudo-labelling | reported harmful by the field | not attempted |
| GOSS with bagging parameters | invalid LightGBM combination | dropped |
| sklearn `ExtraTreesClassifier` (1000 trees) | too slow, marginal value | dropped |
| `base_margin` residual boosting on the teacher logit | probe returned AUC 0.163 — the init/predict score spaces were mismatched, so the number is meaningless | **untested, not rejected** (low priority: the duplicate-key result implies little learnable residual) |
| **Region-local reliability gating** (a public notebook's 0.96152 technique, tested honestly) | **delta 0.000000 at 5/10/20/40 bins; no member beats the equal-weight blend in *any* region** | **rejected — see below** |
| Public-LB hill climbing | ±0.0002 paired noise | wasted submission |

## 7. Software quality

15/15 tests in `tests/run_tests.py` pass, including four automated leakage audits:
`test_fold_safe_te_never_sees_apply_rows`, `test_crossfit_row_never_sees_own_label`,
`test_external_features_use_no_competition_label`, `test_original_overlap_audit_is_small`.

`scripts/external_block_audit.py`: **14/14 pass**. Proves the external blocks are label-free by
runtime proof (bit-identical output when the competition target is randomised, and when it is
zeroed), an AST check that every `TARGET` subscript is rooted at the original frame, and the
exact-overlap audit on the original rows.

The submission builder has a hard pre-flight gate (column order, row count, id equality,
finite, in-range) which has already rejected one bad artefact (an out-of-range logit blend).

### Finalist selection
Two candidates were built and compared:
| candidate | OOF AUC | max single-family weight | role |
|---|---|---|---|
| **`equal_all` (59 members, equal weight)** | **0.961509** | 0.017 | **the finalist** |
| `family_balanced` (equal weight per family) | 0.961502 | 0.083 | structural hedge |

Their test predictions are **rank-identical (Spearman 0.999685)**, so banking both would add no
diversity — the honest conclusion is that **one finalist is correct**, and it is
`submissions/v3_final.csv` (already submitted, public 0.960980).

Blend vs its own best single member: **+0.000247, positive in 5/5 folds, +24.7 units against a
fold SE of 2.04, bootstrap95 [+0.000192, +0.000322]** — the ensemble gain is real and large
relative to its uncertainty.

## 8. Current state and next steps

- **Champion / finalist**: 59-member equal-logit blend, OOF **0.961509** (primary 5-fold),
  submitted as `v3_final` → public 0.960980, rank 144.
- **Honest position**: rank 1 is 0.96165; rank 100 is 0.96123. The gap is ≈ 5e-4 of OOF-equivalent
  and we have not been able to attribute it to any mechanism — see the plateau note below.
- **Submission budget**: 7 remaining today; plan ≤ 3/day. Two final selections are allowed at the
  deadline; one is the right choice here.
- **Plateau note**: 0.9615 is the **current system plateau, not a proven ceiling.** Evidence that
  the synthetic p(x) is weaker than the real survey's (0.9949 vs 0.9612) explains why the ceiling
  feels close, but it is an inference about the generator, not a bound on what is achievable.
  Concretely still open: an architecture genuinely decorrelated from the GBDT/RealMLP pool
  (every family tried — LightGBM, XGBoost, CatBoost, RealMLP, TabM — has logit-correlation
  0.995–0.999 with the others), and a representation richer in a way that helps *deterministic*
  trees, since the `enrich` experiment showed added columns only help random splits.
- **Next**:
  1. Re-check the final-submission selection rules on the competition Rules page near the deadline.
  2. Re-run `scripts/reproduce_finalist.py` from clean to confirm the submission is byte-reproducible.
  3. Continue probing for a genuinely decorrelated family; do **not** re-tune existing families.

## 9. Reproducing the submission

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py            # idempotent
.\.venv\Scripts\python.exe scripts\reproduce_finalist.py --name v3_final
```
This verifies data hashes, runs the 15 unit tests and the 14-check leakage audit, retrains any
missing member (idempotent), rebuilds the equal-logit blend, and writes
`submissions/v3_final.csv` through the pre-flight gate. The manifest with member list, weights,
prediction hashes and git commit lands in `reports/finalist_v3_final.json`.

## 10. Git

`main` — see `git log -1`. Repo: <https://github.com/jamesidriss/airline_S6E10>