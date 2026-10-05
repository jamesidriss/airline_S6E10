# STATUS — Kaggle Playground S6E10 (Airline Satisfaction)

_Last updated: 2026-10-04._

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

## 6b. TabR (Phase 4) — the orthogonal-family attempt

**Why.** Every model in the pool is a GBDT or an MLP and they sit at logit-correlation 0.995-0.999
with the finalist, so new near-clones buy ~1e-6..5e-6 each. TabR is retrieval-based: it retrieves
the k nearest training rows in a *learned* embedding space and classifies from a
parameter-efficient ensemble of heads conditioned on that context. No axis-aligned partitioning
(GBDTs), no single global parametric function (MLPs) — an explicitly local, retrieval-based decision
rule, which is the strongest available candidate for decorrelated errors. The success criterion is
**marginal ensemble value, not standalone AUC**.

**Infrastructure that had to be built first.** aiss-cpu is the only CPython 3.11 wheel for
Windows and has neither GpuIndexFlatConfig nor GpuIndexFlatL2, while pytabkit's TabrModel
requires the GPU symbol; a CPU IndexFlatL2 over 503k x 128 per batch step is not viable. So
src/models/tabr_retrieval.py implements TorchExactL2Index (chunked torch matmuls, GPU-native,
exact squared-L2) behind the identical faiss API, and install_faiss_gpu_shim() supplies the two
missing symbols so **pytabkit's own code path runs unchanged** rather than being monkey-patched.

Retrieval is validated, not assumed: exact agreement with an independent **float64** reference
(relative error 2.4e-07), top-k neighbour-**set** overlap exact (4800/4800) on a 600k x 265 problem,
memory bounded by the query chunk (222 MiB peak vs a 586 MiB full distance matrix), 
eset()
provably preventing cross-batch contamination, and **no label parameter anywhere** in the retrieval
path — the mechanical fold-safety guarantee.

**Four bugs found; the earlier TabR results are invalid, not negative.**

1. **memory_efficient=False was catastrophic.** 	abr.py:281-283 reads
   with torch.set_grad_enabled(torch.is_grad_enabled() and not self.memory_efficient). With the
   flag false, autograd stayed ENABLED while all ~503k candidate rows were encoded on *every*
   training step, building a full autograd graph over the entire retrieval database each time. That
   is both the host-RAM exhaustion (silent death, no Python traceback, pagefile peak 10.1 GB) and
   essentially all of the ~9 min/epoch cost. memory_efficient=True encodes candidates under
   
o_grad and recomputes gradients only for the retrieved context rows.
2. **--d-main was never wired into the params dict** — the run labelled itself d128 while
   pytabkit's default d_main=265 was actually in force. A ledger that disagrees with the trained
   model is worse than a crash. Now asserted at two levels, including the live retrieval index's
   dim, which proves the width retrieval actually used.
3. **Validation/test features were read from b.static_tr[val]** (237 static columns) while
   training used the assembled fold-safe view — a pandas shape error at predict time. Measured gap:
   static 237 vs assembled 285. All of {inner-train, inner-ES, outer-val, test} now come from a
   single ssemble() call over the outer-FIT block, which is also the only fold-safe ordering.
4. **OOM recovery could not actually shrink the retrieval query chunk** — the shim factory closed
   over query_chunk at install time. Two-part fix, plus a subtlety our own test caught: the
   factory was a *nested* function, so _SHIM_QUERY_CHUNK (assigned in the enclosing scope) became
   a function-local there and it silently closed over that local instead of the module global.

**Also learned.** pytabkit wires Lightning's logger to DummyLogger, so *all* epoch metrics are
discarded: there was no learning curve at all, and a run dying at epoch 20 would leave no evidence.
src/models/tabr_epochlog.py attaches an on_validation_epoch_end hook (TabrLightning does not
define it, so nothing is shadowed) and flushes each epoch to disk immediately. Separately, pytabkit
early-stops on **al_accuracy**, not AUC — fold-safe, but a weaker proxy for our metric than the
inner-ES AUC curve we now persist.

**Verdict: REJECTED for the ensemble. Stopped.** Measured at two context sizes on the immutable
primary fold 0:

| ctx | fold-0 AUC | train | logit corr vs finalist | Spearman |
|---|---|---|---|---|
| 32 | 0.960282 | 550 s | 0.99501 | 0.97647 |
| 48 | **0.960329** | 1355 s | 0.99502 | 0.97727 |

TabR **did** achieve what it was chosen for -- it is the most decorrelated model in the pool, sitting
just below the 0.99577 floor of the existing pairs' logit correlation, with the lowest Spearman ever
measured here. But at both context sizes **every predeclared fixed blend weight was negative**
(2% -> -1e-6, 5% -> -3e-6, 10% -> -8e-6), and 50% more neighbours bought only +4.7e-5 for 2.46x the
compute without improving decorrelation at all. Its 1.2e-3 standalone deficit is simply larger than
its decorrelation is worth. Tuning further would cost hours per fold on a family whose marginal value
is measurably zero.

## 6b. k-NN target encoding -- rejected

The cheapest genuinely-local estimator available, reusing the validated GPU exact-L2 retrieval:
distance-weighted neighbour target mean at k in {16, 64, 256}, plus unweighted mean, mean neighbour
distance, and neighbour-label dispersion. Metric space = the 21 raw survey variables (standardised
numerics + one-hot categoricals, so no fabricated ordering of Class); retrieval fp32/exact and
validated against a float64 reference. Fold discipline: the database is outer-FIT rows only, every
fit row is encoded by a 5-fold inner cross-fit that excludes it, inner-ES/val/test rows are encoded
against the full outer-FIT-minus-ES database. Built in **29 seconds** against TabR's 22 minutes.

Result on fold 0, champion configuration, identical rows and seed -- only the view differs:

| | n_feat | AUC |
|---|---|---|
| base | 285 | 0.961103 |
| + kNN block | 297 | 0.960953 (**-1.5e-4**) |

A pure k-NN with no model at all reaches 0.9468 on this fold, so the neighbourhood metric is
genuinely informative and the protocol is sound -- the block is simply **redundant** with what the
285-feature view already extracts, and it displaces better features under colsample_bytree. It is
*more* correlated with the champion than TabR was (Spearman 0.989 vs 0.977), i.e. the opposite of
what this campaign needed.

## 6c. Learning curve -- the campaign is still DATA-limited, not representation-limited

This is the most strategically important measurement taken. Champion configuration (LightGBM +
extra_trees on ull), retrained on stratified subsamples of the outer-FIT rows, scored on the same
untouched outer-validation rows, 3 seeds per point so model variance is separable from noise:

| fraction | n train | AUC | +/- seed std | best iter |
|---|---|---|---|---|
| 0.125 | 69,963 | 0.959318 | 0.000139 | 377 |
| 0.25 | 139,927 | 0.960020 | 0.000060 | 395 |
| 0.50 | 279,854 | 0.960696 | 0.000204 | 525 |
| 0.75 | 419,781 | 0.960956 | 0.000133 | 753 |
| 1.00 | 559,708 | **0.961212** | 0.000108 | 879 |

Gain per doubling of data: **+70.2e-5, +67.5e-5** (0.125->0.25 and 0.25->0.50), and **+51.6e-5** for
0.50->1.00. The two half-steps are smaller, as expected for sub-doublings: +26.1e-5 for 1.5x and
+25.6e-5 for 1.333x.

> **CORRECTION (2026-10-04).** These per-doubling figures were initially written into this file and
> the ledger **10x too small** (+7.0/+6.8/+5.2e-5). The measurement was always correct -- the script
> printed +51.6e-5 -- and the error was a hand-transcription slip when copying the numbers across.
> Corrected figures are recomputed directly from 
eports/lcurve_fold0.json and re-appended to the
> append-only ledger as lcurve_fold0_CORRECTION. The corrected conclusion is **stronger**, not
> weaker: a genuine doubling of unique in-distribution rows is worth ~+5e-4 to +7e-4.

Fitted scaling law (five points, order of magnitude only -- **not** a Bayes ceiling):

- **AUC ~ a + b*n^(-1/5)**, R^2 = **0.99894** (best of the residual forms tried)
- AUC linear in log2(n): slope **+62.9e-5 per doubling**, R^2 = 0.99539, residuals <= +-7.7e-5

Extrapolated value of more effective data: **1.5x -> +3.7e-4, 2x -> +6.3e-4**. The entire
rank-1-to-rank-45 spread is ~3e-4, so a genuine doubling of training support would be worth about
**twice the whole competitive gap** -- an order of magnitude more than any ensemble-level gain
measured in this campaign (the 59-member blend buys +2.5e-4 over the best single model).

**This overturns the campaign's self-assessment.** We had been treating ourselves as
representation-limited and had stopped looking for gains. The evidence says the scarce resource is
still *effective training signal per leaf*, which implies two things that have NOT been tried here:

- **in-distribution data augmentation**, which manufactures exactly the resource that is scarce
  (appending the original 129,880 rows does not count -- measured harmful, because they are a
  different conditional, not more of this one);
- continued **variance reduction by averaging**, though note this is partly spent: the 59-member
  blend already captures +2.5e-4 over the best single model.

**Important caveat.** The learning curve proves that *unique genuine* training signal still pays. It
does **not** prove that synthetic augmentation delivers the same benefit -- duplicating rows, or
interpolating between neighbours, creates no new information. Whether augmentation tracks the real
learning curve is exactly the hypothesis now under test, and a null result there is itself
informative: it would mean the limit is unique information, not sample count.

This is a directional diagnostic, **not** a Bayes ceiling and not claimed as one.

## 6d. CORRECTION -- the Phase 6 pairwise explanation was mathematically wrong

The **empirical result** of Phase 6 was never in doubt and is unchanged:

| arm | AUC | corr logit vs pool | Spearman |
|---|---|---|---|
| binary matched control | 0.961103 | -- | -- |
| pairwise lr 0.05 / 4000 rounds | 0.956339 | 0.98577 | 0.91352 |
| pairwise lr 0.15 / 15000 rounds | 0.959556 | 0.99540 | 0.97560 |

The spectacular apparent decorrelation at 4000 rounds was largely an **undertraining artefact**: as
training progressed the AUC rose *and* the correlation moved back toward the existing pool. The
tested custom pure-pairwise LightGBM branch is **rejected**.

The **mechanistic explanation** recorded with it was wrong on both halves:

- ❌ "pairwise loss is invariant to any monotone rescaling" — **false**. The pairwise logistic loss
  `log(1 + exp(-(s_pos - s_neg)))` depends *explicitly* on pairwise score **differences**. It is
  invariant only to a **global additive shift**, because a shift cancels in the difference.
  Multiplying scores, or applying a non-linear monotone map, generally *changes* the loss.
- ❌ "ROC-AUC depends on score separation" — **false as stated**. ROC-AUC **is** invariant to any
  strictly increasing transformation of the score; what decides it is the induced **ordering**,
  which such a transform preserves.
- ❌ "pairwise loss has no incentive to separate positives from negatives" — **false**. It maximises
  exactly a pos-versus-neg margin and encourages separation directly.

**What we do not know.** We have *no demonstrated explanation* for the branch's inferiority.
Candidate causes, none isolated experimentally: optimisation mismatch between a margin loss and
diagonal-Hessian tree boosting; the diagonal-Hessian approximation ignoring pair coupling; pair
sampling at only 4 negatives per positive, held fixed across boosting iterations; differing
regularisation behaviour; statistical efficiency (BCE uses every row every iteration, the pairwise
loss only sampled pairs); and pair-sampling noise. These are **hypotheses, not findings**, and no
mechanism may be attributed to this result.

**Scope correction.** The earlier wording implied the AUC-objective axis was closed, which
overstates what was tested. Accurate scope: **the tested custom pure-pairwise LightGBM branch is
rejected.** XGBoost's native `rank:pairwise` and other AUC surrogates remain technically untested;
current evidence makes them **low priority**, but does not exclude them.

Recorded append-only in `experiments/ledger.jsonl` as `auc_objective_fold0_CORRECTION` via
`scripts/log_auc_correction.py`. Git history was **not** rewritten.

## 6e. PHASE 7 -- audit: our models train on 72% of the labels, not 80%

`scripts/audit_train_fractions.py` measures the **actual** row counts rather than assuming them
from "5-fold means 80%". The inner early-stopping holdout removes a further 8 percentage points on
top of the outer-fit carve:

| scheme | K | outer-fit rows | % of all labels | inner-ES rows | **ACTUAL model-fit rows** | **% of labels** |
|---|---|---|---|---|---|---|
| `primary` | 5 | 559,708 | 80.00% | 55,969 | **503,739** | **72.00%** |
| `block10` | 10 | 629,672 | 90.00% | 62,966 | **566,706** | **81.00%** |
| `shadow` | 5 | 559,708 | 80.00% | 55,969 | **503,739** | **72.00%** |

Total labelled competition rows: **699,635**. Every CV-stage OOF model in the 5-fold protocol sees
**72%**, not 80%.

Projected through our own measured learning curve (**+62.9e-5 per doubling**):

| transition | doublings | projected gain |
|---|---|---|
| 72% -> 80% (recover the inner-ES holdout) | 0.152 | **~ +9.5e-5** |
| 72% -> 90% | 0.322 | **~ +2.0e-4** |
| 72% -> 100% | 0.474 | **~ +3.0e-4** |

This is larger than anything else measured in the campaign and it is grounded in our own data.

### The learning curve predicts a fold-count effect a priori -- and then over-extrapolates
`scripts/validate_curve_on_folds.py`. If the measured 5→10 fold gain is nothing but the
training-fraction change, the curve must predict it **before** the fold experiment was run. Using
the seed-matched pairs from the prediction store (identical view, family, `extra_trees` and seed;
only the fold scheme differs):

| K=5 member | K=10 member | observed | predicted a priori | residual |
|---|---|---|---|---|
| `xt_xt_sh_s1` (shadow, seed 1) | `xt_xt_f10` (block10, seed 1) | **+9.9e-5** | **+10.7e-5** | −0.8e-5 |

ratio 566,706/503,739 = 1.125 = 0.1699 doublings × 62.9e-5. A **7.4%** relative error, and the
residual (−0.8e-5) is *smaller* than the seed-to-seed spread within either scheme (shadow 4.4e-5,
block10 6.4e-5). Two consequences:

1. **Fold-count scaling is not a mysterious diversity effect.** It is the same "more unique labels
   per learner" effect the subsample curve measures. That explains why 5→10 helped, and predicts
   saturation as the fraction approaches 1.
2. The law has out-of-sample predictive power on an axis it was not fitted to — which is the only
   thing that can justify a full-data refit, an intervention no CV can score.

**But the law over-extrapolates at the top of the range.** `scripts/run_fullfit.py` measured
72%→80% directly and got **+4.0e-5 on fold 0** where the curve predicts +9.6e-5 (or +7.8e-5 using
the last segment's local slope rather than the mean). That is ~40–50% of prediction. So the
80%→100% figure of **+20.2e-5 is an optimistic bound, not a point estimate**; a realistic estimate
is single-digit e-5. This is recorded as a correction to our own extrapolation, and it is the reason
the full-data members are not sold as a +2e-4 win.

**The asymmetry that matters — and it is larger than we thought.** An earlier reading of this file
asserted that "a second-pass refit on all labelled rows is the documented test-time policy, so a
submitted model already trains on 100% of the labels." **That was wrong**, and it was wrong because
it was inferred from a substring search for the word "refit" rather than from reading which row
index the test-time fit actually receives. `scripts/run_views.py` does the second pass like this:

```python
n_it = int(np.median(iters))                    # median best_iteration from the CV folds
for k in sorted(set(folds.tolist())):
    fit = np.where(folds != k)[0]                # <-- the fold's OUTER-FIT rows, 80% of labels
    Xf, Xa, _ = vb.assemble(fit, y, val, np.arange(ntr, ntr + nte))
    pr_te = _fit_full_predict_lgbm(Xf, y[fit], Xtest, params, args.seed, n_it)
    test += pr_te / len(set(folds.tolist()))     # average over the K folds
```

| pipeline stage | rows per model | % of 699,635 labels |
|---|---|---|
| CV-stage OOF model (K=5, inner-ES protocol) | 503,739 | **72.00%** |
| CV-stage OOF model (K=10, inner-ES protocol) | 566,706 | **81.00%** |
| **test-time model (K=5)** | 559,708 | **80.00%** |
| **test-time model (K=10)** | 629,672 | **90.00%** |
| test-time model on 100% of labels | — | **not implemented anywhere** |

**No code path in this repository trains a test model on 100% of the labelled rows.** Every
submitted prediction is an average over K models, each fitted on K−1/K of the labels, so **20%
(K=5) or 10% (K=10) of the real labels are never shown to any member that votes on the test set** —
even though we already hold those labels and the fold protocol exists only to produce OOF
predictions, which test-time inference does not need.

Two separable, independently valuable consequences:

1. **Fold models (measurable now).** Recovering the inner-ES holdout takes a fold model from 72% to
   80% of the labels under a leakage-safe fixed iteration count. This *is* measurable, honestly,
   because the held-out fold stays untouched. Predicted ≈ **+9.5e-5**. Validated by
   `scripts/run_fullfit.py`.
2. **Test-time models (larger, and not directly measurable).** A full-data refit takes a test model
   from 80% to 100%. Predicted ≈ **+2.0e-4** for a single learner (0.322 doublings × 62.9e-5). This
   cannot be measured by any cross-validation, because a model that has seen all the labels cannot
   be scored on labels it has seen. It must be justified by (a) the measured learning-curve law,
   whose R^2 is 0.99894 across five points, and (b) the fresh 72%→80% measurement from point 1 as
   an out-of-sample validation of that law. It is reported as a **cross-validated training-policy
   gain, never as fabricated OOF.**

### Iteration-count scaling with training size
`scripts/iteration_scaling.py`. The learning curve gives five (n, best_iter) points. **Do not fit
them blindly**: the smallest point is off-trend (377 rounds at 63k rows, then only 395 at 126k), so
an all-points power law **extrapolates backwards** -- it predicts *fewer* rounds at 560k than were
actually observed at 504k.

Predeclared choice: OLS of `log(best_iter)` on `log(n)` over the **three largest** points,
`best_iter = 0.0459 * n^0.7527`, R^2(loglog) = 0.98371. It passes the one available check --
reproducing the observed 504k count to within 2.3% (predicts 900 vs observed 879) -- as do the
4-point fit (0.982) and the 5-point fit (0.918).

| training set | rows | size correction vs control | implied iterations |
|---|---|---|---|
| current inner-ES (control) | 503,739 | 1.000 | 879 |
| 5-fold full outer-fit | 559,708 | 1.083 | 952 |
| 10-fold full outer-fit | 629,672 | 1.183 | 1040 |
| full refit on all labels | 699,635 | 1.281 | 1126 |

**Caveat, stated plainly:** five points spanning under one order of magnitude is a weak law, and
**its sign turned out to be wrong in the regime that mattered** — see below.

### The iteration-scaling law was withdrawn, and that is the more useful result
A direct measurement at **95% of the labels** contradicted the law's direction:

| training rows | early-stopping argmax | source |
|---|---|---|
| 503,739 (72%) | 797, 731 | fold-0 / fold-1 controls, 10% holdout |
| 503,739 (72%) | 879 mean over 3 seeds | fold-0 subsample curve, frac = 1.0 |
| **664,655 (95%)** | **[613, 846, 641]** | full-data measurement, three independent 5% holdouts |

The **largest** training size yields a **lower** count. `best_iter` is flat-to-*decreasing* in `n`
here, so extrapolating upward is the wrong direction and the ×1.039 correction was **withdrawn before
any full-data member was fitted under it** (the refit job was killed and restarted so its recorded
manifest reflects the policy actually used).

**Mechanism.** With `extra_trees` plus `colsample_bytree=0.8` and `subsample=0.8`, every tree already
sees ~64% of the columns and 80% of the rows. More data makes split statistics less noisy and the
validation curve **flatter**, which both lowers the optimum and makes the argmax less well determined.
The **38% spread across three holdouts at a single size** — [613, 846, 641] — is that flatness showing
up directly, and it is the same phenomenon that cost −18.8e-5 when `subsample` moved 0.8→0.9.

**Policy finally used:** median of three independent 5% early-stopping holdouts measured on 95% of the
labels, **no size correction at all**. The measurement sits at 95% of the target size so there is
almost no extrapolation, and mild under-iteration is the safe residual when over-iteration is the one
failure mode actually measured. Materiality is small — 641 vs 666 rounds is 4%, and fold 0 measured
811 vs 863 (6% apart) as worth exactly 0.0e-5.

**The lesson, recorded in the ledger as `p7_iteration_size_correction_WITHDRAWN`:** a scaling law
validated out-of-sample on the *fold-count* axis (it predicted the measured 5→10 fold gain to 7.4%)
still did not transfer to the *iteration-count* axis. **Predictive power on one axis is not evidence on
another**, and a five-point fit from a different regime is a hypothesis, not a constant.

### PHASE 7 RESULT 1 -- inner-ES vs full outer-fit (`scripts/run_fullfit.py`)

Champion extra_trees LightGBM, view `full`, primary folds, identical features/params/seed; the
control is the existing protocol and the full-fit arm trains on 100% of outer-fit. Three arms differ
**only** in the estimator of the iteration count, so agreement between them is evidence about the
estimator rather than a search over it.

| arm | iteration source | fold 0 | fold 1 | mean | positive |
|---|---|---|---|---|---|
| **control** (503,739 rows = 72%) | inner early stopping | 0.961299 | 0.961396 | — | — |
| `ctl_scaled` (559,708 rows = 80%) | control's ES × 1.083 | **+4.0e-5** | **+0.4e-5** | **+2.2e-5** | **2/2** |
| `innercv_raw` | median inner-CV, no correction | +3.9e-5 | −0.8e-5 | +1.6e-5 | 1/2 |
| `innercv_scaled` | median inner-CV × 1.47 | −5.9e-5 | +0.2e-5 | **−2.9e-5** | 1/2 |

Three findings, all of which matter more than the headline number:

1. **Recovering the inner-ES holdout is real but small: +2.2e-5, positive in 2/2 folds.** That is
   below the +5e-5 promotion threshold, so it is *held*, not escalated. Critically it is **~4×
   smaller than the +9.6e-5 the learning curve predicted**, i.e. the law over-extrapolates at the top
   of the range (Section 6e above).
2. **Over-iteration is the real risk, not under-iteration.** `innercv_scaled` extrapolates the count
   over a 1.47× range and lands at 1191 rounds; AUC falls to 0.961240, **−5.9e-5 against the
   control**. Meanwhile 811 vs 863 rounds differ by +0.0e-5. The model is flat in iteration count
   over 811–863 and degrades beyond it. Therefore the full-data policy must **shorten the
   extrapolation**, not sharpen it.
3. The most defensible estimator is the one taken from the **closest available size measurement**
   (the control's own early stopping at 90% of the final fit), which is also the only arm positive
   in both folds.

The predeclared full-data iteration policy therefore became: measure the count by early stopping on
a **5%** holdout carved from all rows (fit on 95%), correct by only `(1.00/0.95)^0.7527 = 1.039`, then
fit on 100% at that fixed count. At 1.039× the policy is nearly insensitive to which exponent is
used, which is exactly the robustness the fold-0 failure demanded.

**Harness note.** The first run of this script returned a fake **−483e-5 "REJECT"** caused by an
index-space bug: the inner-CV picker indexed the design matrix by position-within-inner-train rather
than position-within-outer-fit, so it early-stopped at 5 and 43 rounds. Fixed, plus a hard guard
that now **aborts** if the inner-CV median differs from the control's early-stopping count by more
than 3×, so this failure class can never again be reported as a result.

### PHASE 7 RESULT 2 -- group-conditional calibration: CLOSED (`scripts/run_group_calibration.py`)

Nested meta-validation on the primary folds: fit the calibrator on four folds of `logit(v3_final)`
OOF, apply to the fifth, never the same rows.

| arm | design | AUC | mean delta | folds positive |
|---|---|---|---|---|
| `G0_global` | 1 col — **mandatory control** | 0.961505 | **+0.0e-5** | 0/5 |
| `G2_union` | base + 4 group intercepts + slopes (11 cols) | 0.961512 | +0.8e-5 | **5/5** |
| `G1_union` | base + 4 group intercepts (6 cols) | 0.961511 | +0.6e-5 | 4/5 |
| `G1_travel` | base + Type-of-Travel intercepts | 0.961510 | +0.5e-5 | 3/5 |
| `G1_class` | base + Class intercepts | 0.961509 | +0.4e-5 | 4/5 |
| `G2_class` | base + Class intercepts + slopes | 0.961508 | +0.3e-5 | 4/5 |
| `G3_class` | shared 5-knot piecewise-linear base + Class | 0.961505 | +0.3e-5 | 3/5 |
| `G1_custtype` | base + Customer-Type intercepts | 0.961506 | +0.0e-5 | 3/5 |
| `G1_gender` | base + Gender intercepts | 0.961504 | −0.1e-5 | 2/5 |

The **control check is the load-bearing part**: `G0` (a global Platt map, which is strictly
increasing) moves AUC by **+0.0e-5 on all five folds**, exactly as theory requires — ROC-AUC is
invariant to any strictly increasing transform of the score. The harness is therefore trustworthy,
and every other arm is interpretable.

**Verdict: closed.** The best arm is consistent (5/5 positive) but worth **+0.8e-5**, roughly 4×
below the predeclared +3e-5 rule and an order of magnitude below the admission gate. There *is* a
small systematic cross-group rank bias in the finalist, and it is too small to act on. Two harness
bugs were found and fixed on the way (G0 initially omitted the base column entirely; group one-hot
blocks were not subset per fold, which surfaced as a reshape error rather than a wrong number).

### PHASE 7 RESULT 3 -- row bagging is NOT redundant (`scripts/run_bagging_test.py`)

Not generic tuning: a stated mechanism. `subsample=0.8, subsample_freq=1` means every tree sees only
80% of rows, while `extra_trees=True` already randomises features *and* thresholds. If that
double-randomisation already regularises enough, the bagging is redundant and is discarding 20% of
real labels per tree.

| `subsample` | fold 0 (iter) | fold 1 (iter) | mean delta | positive |
|---|---|---|---|---|
| 0.8 (control) | 0.961299 (797) | 0.961396 (731) | — | — |
| 0.9 | 0.961327 (860) | 0.961208 (**1082**) | **−8.0e-5** | 1/2, signs flip |
| 1.0 | 0.961269 (457) | 0.961342 (757) | −4.2e-5 | 0/2 |

**Hypothesis refuted in both directions.** Removing bagging does not help monotonically, so
`extra_trees` and `subsample` are not redundant; `colsample_bytree` was left untouched because the
mechanism that would justify touching it was not established.

**The diagnostic is worth more than the verdict.** Look at the iteration counts: changing *only*
`subsample` from 0.8 to 0.9 moved the early-stopped count 797→860 on fold 0 (harmless) but
**731→1082 on fold 1, which cost −18.8e-5**. A single 55,969-row holdout cannot reliably locate a
flat optimum, and **over-iteration is the dominant failure mode in this setup**. That is why the
full-data iteration policy measures the count on **three** independent 5% holdouts and takes the
median, rather than trusting one argmax.

Cross-check of the harnesses: after the index-space fixes, this script's control arm reproduces
fold-0 AUC 0.961299 at iter 797 and fold-1 0.961396 at iter 731, matching `run_fullfit.py`'s
independently written controls exactly.

### PHASE 7 RESULT 4 -- class-conditional GENERATIVE score: CLOSED (`scripts/run_generative_probe.py`)

The last genuinely orthogonal direction available: estimate p(x|y) instead of p(y|x). Binned
class-conditional log-likelihood ratios over the 21 raw columns, three arms (plain / discriminatively
weighted / top-8), fully nested on the primary folds.

| arm | standalone AUC | best blend gain | logit corr with finalist |
|---|---|---|---|
| `gen_plain` | **0.930873** | **−0.2e-5** | 0.87004 |
| `gen_weighted` | 0.930595 | −0.2e-5 | 0.87499 |
| `gen_top8` | 0.912040 | −0.2e-5 | 0.82493 |

The predeclared kill rule was "standalone very weak **and** marginal gain ~0". Neither half held, and
the outcome is a *stronger* kill than the rule anticipated:

- The standalone score is **not weak** — 0.9309 from 21 raw columns is a respectable classifier.
- It is the **most decorrelated thing produced in this campaign** (logit corr 0.87, against ~0.99 for
  every previously rejected member). Decorrelation was not the problem.
- Yet **every blend weight is negative, and monotonically worsening**: −0.2e-5 at w=0.005 down to
  −24.9e-5 at w=0.2.

**Mechanism.** Naive Bayes cannot represent the interactions that actually drive this target — `Class`
× `Type of Travel`, and the survey ratings conditioned on one another. Its errors therefore lie
*inside* the GBDT's error set rather than beside it: decorrelated in the logit-correlation sense
while being strictly worse ordered on the same rows. Decorrelation is necessary for a useful blend
member and demonstrably not sufficient, and this is the cleanest available demonstration of the
distinction.

### PHASE 7 RESULT 5 -- pseudo-labelled evaluation rows: REJECTED, and badly (`scripts/run_pseudo_label.py`)

The one thing the rejected Phase 5 augmentation work did not cover. Phase 5 rejected *synthetic*
rows; pseudo-labels add **real covariate rows from the actual evaluation distribution**, so they are
on-manifold by construction and the mechanism that sank Phase 5 does not apply.

Held-out fold k plays the role of the unlabelled test set: M0 is fitted on outer-fit only, predicts
fold k's covariates (never seeing a fold-k label), pseudo-labels are attached to the most confident
fraction, and fold k is then scored **once** with its true labels. **Row-count-matched control:** a
`dup` arm adding the same number of *duplicated outer-fit rows with their true labels*, so
`pseudo − dup` isolates the information rather than the row count.

| frac | fold | `pseudo` AUC | `dup` AUC | pseudo − ctl | **pseudo − dup** |
|---|---|---|---|---|---|
| 0.25 | 0 | 0.959690 | 0.961251 | −160.9e-5 | **−156.1e-5** |
| 0.50 | 0 | 0.958879 | 0.961240 | −242.0e-5 | **−236.1e-5** |
| 0.25 | 1 | 0.959851 | 0.961339 | −154.6e-5 | **−148.9e-5** |
| 0.50 | 1 | 0.958855 | 0.961231 | −254.1e-5 | **−237.5e-5** |

Means: **pseudo − dup = −152.5e-5** (frac 0.25) and **−236.8e-5** (frac 0.50), **0/2 folds positive in
both**. Every finalist blend weight is negative too.

This is the **largest negative effect measured in this campaign**, ~30× worse than its own
row-count-matched duplicate control. Three things make it a strong result rather than a shrug:

1. **The control works.** `dup` costs only −4.7e-5 to −16.6e-5, consistent with Phase 5's duplicate
   control (−6.4e-5). So the 35k–70k extra rows are *not* what hurts.
2. **Clean dose–response.** Damage nearly doubles (−157.7e-5 → −248.1e-5 against the control) when the
   number of pseudo-labelled rows doubles. That is the signature of a causal mechanism, not a
   coincidence.
3. **The mechanism is identifiable.** Selecting by *most confident* `|p − 0.5|` deliberately picks
   the rows the model is most sure about, including the ones it is most wrong about. The pseudo-
   positive rate (0.279–0.329) is well above the base rate, so the added block is systematically
   skewed toward one class. Training on it lets the model reproduce its own confident errors. Classic
   confirmation bias, now quantified at ~1.5e-3 to 2.5e-3 of AUC.

Note this is a **different verdict from Phase 5 with a different mechanism**, and that distinction
matters: Phase 5 failed because interpolated rows are off-manifold; this fails because a model's own
labels carry no information it did not already have, and selecting for confidence actively concentrates
its errors. Together they make the "unique information" conclusion considerably stronger.

### PHASE 7 RESULT 6 -- structural probes (`scripts/structural_probes.py`)

Four cheap probes run because the LB gap is real (below) and every measured gain is single-digit e-5.

**A — train↔test duplicate rows: none.** The campaign had verified only 21 matches against the
*original* dataset, never competition-train against competition-test. Result: **0 distinct-reduced
matches**, and, more striking, **all 699,635 train rows are distinct feature vectors** — the
generator emitted no exact duplicates anywhere, and no test row exactly matches a train row. No
lookup feature is available. (This also made the cleanest form of Probe D impossible.)

**B — train/test covariate shift: none, confirmed properly.** A LightGBM domain classifier on the 21
raw columns, balanced 299,844 vs 299,844, 3-fold cross-fitted: **AUC 0.500620** (folds 0.500372 /
0.500709 / 0.500777). Early stopping fires at iteration 1 because nothing improves after the first
tree — the domains are genuinely indistinguishable. This is a stronger version of the previously
recorded "essentially zero".

**C — `id` is uninformative, confirmed.** AUC of raw id against the label **0.500073**; within ten
contiguous id blocks, mean |AUC − 0.5| = **0.0019**. The fold protocol is not optimistic through id.

**D — where the signal actually lives.** Probe A showed no duplicates, so label consistency was
measured in coarse buckets over the 11 columns that dominate the model (Class, Type of Travel, and the
nine service ratings): 161,248 buckets, 61,460 with ≥2 rows.

| quantity | value |
|---|---|
| **leave-one-out bucket-only AUC (11 raw columns)** | **0.933454** |
| fraction of buckets that are label-pure | **68.8%** |
| campaign's full model OOF AUC | 0.961509 |
| **total added by 285 features + 59-member blend** | **+0.028055** |
| within-bucket MSD observed / binomial-corrected expected | 0.05522 / 0.05013 (excess +0.00509) |

This is a useful calibration and mildly deflating: **0.9335 of AUC comes free from 11 columns and a
lookup table**, and the entire accumulated modelling effort of this campaign — 285 engineered
features, a 59-member blend, five rejected augmentation families — buys the remaining **+0.028**.
The residual +0.00509 excess over binomial says p(y|x) does still vary inside a coarse bucket, which
is expected and not a ceiling claim.

Two harness bugs were caught *by the assertions and sanity checks added alongside them*, both worth
recording: an indexing error produced a bucket-only AUC of **0.404** (impossible for a bucket-mean
predictor — a bucket-mean AUC below 0.5 is a reliable smell), and the finite-sample binomial
expectation was wrong by a factor of n (5.19 vs an observed 0.055), which a first pass reported as an
"excess" of −5.13.

### Is the leaderboard gap real? Unknown � and my own z-score was wrong twice
`scripts/analyse_lb_noise.py`. The +-0.0002 paired noise floor in `AGENTS.md` is correct but narrow:
it governs whether to send a *near-identical* submission, and it is the wrong instrument for asking
whether we differ from a *different* solution.

**This file got it wrong twice, in opposite directions, and both errors are worth recording.**

1. A first pass, done by hand, put the single-estimate SE at 0.0025-0.0036 and concluded the 7.8e-4
   gap might be only ~0.2 sigma, i.e. noise. **Hanley-McNeil says SE is 0.00099-0.00124**, about
   three times smaller, so that hand figure was simply wrong.
2. Having fixed that, the script introduced the **opposite** error and I reported it. It computed the
   correlation needed for a 2-sigma gap as `1 - (min_z/2)^2/2`, which is *not* the inversion of the
   z formula. It returned a negative number, which I clamped to zero and rendered as the sentence

   > "the gap clears 2 sigma even between two **uncorrelated** predictions"

   **That is false, and the script's own table said so � at rho = 0 the z is 0.49, not above 2.** I
   built a confident sentence from a number without checking the number against it. The correct
   inversion is `rho = 1 - (gap/2)^2 / (2*se^2)` = **0.9411**.

**Where that leaves the gap, stated no more strongly than the evidence allows:**

| | |
|---|---|
| scores (factual) | ours 0.960980, leader 0.961760, **gap 7.8e-4** |
| z across rho in [0,1] | **0.44 to 7.88** |
| clears 2 sigma only if | **rho > 0.941** |
| is rho measured? | **NO � we do not hold the leader's prediction vector** |

So no single z is asserted. Two strong tabular solutions on the same 699k rows and the same metric
would plausibly sit well above rho = 0.941, so **"probably real" is a reasonable working belief** �
recorded as a belief, not a measurement.

**What does not depend on rho, and is therefore the part that should govern compute:** the top twenty
occupy a 1.0e-4 band while sitting 7.8e-4 above us, so they are a converged pack at a common level,
not a spread field. And every gain this campaign has actually *measured* is single-digit e-5, with the
one mechanistically sound mechanism delivering +2e-5 on this very board. Whether or not the gap is
statistically real, **more GBDT-family polish will not close 7.8e-4**.

## 6f. PHASE 8 -- TARGET ENCODING: BOTH NEW MECHANISMS REJECTED, AND THE OLD BLOCK IS BETTER

### The audit that motivated this, stated exactly
`src/features/s6e10.py::TE_KEYS_DEFAULT` target-encodes **12 keys**, and only **4 of the 21 raw
columns get a direct single-column exact-value TE** (`te_fd`, `te_age`, `te_online`, `te_ent`).
`Class`, `Type of Travel` and `Customer Type` appear only inside crosses. The other **14 columns get no
exact-value target statistic at all**, and 13 of those 14 have at most 6 distinct values:

| column | distinct values | | column | distinct values |
|---|---|---|---|---|
| Inflight wifi service | 6 | | On-board service | 6 |
| Departure/Arrival time convenient | 6 | | Leg room service | 6 |
| Ease of Online booking | 6 | | Baggage handling | 5 |
| Gate location | 6 | | Checkin service | 6 |
| Food and drink | 6 | | Cleanliness | 6 |
| Seat comfort | 6 | | Gender | 2 |
| Online boarding | 6 | | Departure Delay | 186 |
| Inflight entertainment | 6 | | Arrival Delay | 180 |

So "TE already failed here" does **not** apply to an all-21 block, and the older negatives for
`core3_te` / `full` say nothing about it. That was worth checking before spending compute, and the
answer changed what got tested.

### 8A -- `te_all21` (all 21 raw columns, exact values, sklearn `TargetEncoder`): REJECTED

`scripts/run_te_all21.py`, fold 0, matched arms — same code path, same inner-ES split, same seed, same
hyper-parameters, same rows; only the view and `extra_trees` differ. The control reproduces the
recorded fold-0 baseline **exactly** (0.961299 at 797 rounds), which is what makes the rest readable.

| arm | view | xt | nfeat | iter | AUC | Δ vs ctrl | logit corr | blend@0.2 |
|---|---|---|---|---|---|---|---|---|
| base | full | ✓ | 285 | 797 | 0.961299 | control | — | +0.7e |
| stack | full_te21 | ✓ | 306 | 530 | 0.961296 | **−0.3e-5** | 0.99903 | +0.3e |
| replace | full_all21te | ✓ | 258 | 556 | 0.961166 | **−13.3e-5** | 0.99840 | −0.5e |
| det_base | full | ✗ | 285 | 458 | 0.960979 | control | — | −3.6e |
| det_stack | full_te21 | ✗ | 306 | 574 | 0.960918 | **−6.2e-5** | 0.99910 | −4.2e |
| det_replace | full_all21te | ✗ | 258 | 670 | 0.960888 | **−9.2e-5** | 0.99829 | −2.6e |

Section 8 decision table: xt flat, deterministic negative → **REJECT**, no smoothing sweep.

**The informative part.** Replacing our 48-column selective `te` block with 21 `te_all21` columns
**costs −13.3e-5** under extra_trees and −9.2e-5 deterministic. The old block wins on four axes at
once, and `te_all21` varies only the first:

1. **crosses** (`te_fd_class`, `te_age_class_trip_cust`, …) — `te_all21` has none
2. a **shrinkage spectrum** (smooth 10 / 20 / 100) — `te_all21` has one setting
3. **log counts** per key — none
4. **binned / modulo Flight Distance** variants — none

**So the value of target encoding here is interactions plus a shrinkage spectrum, not column
coverage** — the opposite of the public solution's headline mechanism. Iteration counts also fall
sharply when `te_all21` is added (797 → 530): the new columns give a strong, quickly-saturating signal
and the model early-stops before extracting the rest, which is what a redundant rather than a missing
feature system looks like.

### 8B -- `te_conditional` (13 rating × traveller-segment + 5 service-pair × trip/customer): REJECTED

Built deliberately to the two axes 8A identified as the ones that work: 18 exact-value **cross** keys
through `FoldSafeTE`, so each key gets smooth 10/20/100 **and** a count column (72 columns). Keys are
well-estimated: 53–133 distinct values, median group 674–4,495 rows, **0% of rows in a group of ≤5**.

| arm | view | xt | nfeat | iter | AUC | Δ vs ctrl | blend@0.2 (f0) | blend@0.2 (f1) |
|---|---|---|---|---|---|---|---|---|
| base | full | ✓ | 285 | 797/731 | 0.961299/0.961396 | control | +0.7e / −1.0e | — |
| tec_stack | full_tec | ✓ | 357 | 914/624 | 0.961299/0.961302 | −0.0e / −9.4e | **+2.6e** | **−1.2e** |
| tec_swap | full_tec_swap | ✓ | 309 | 872/611 | 0.961092/0.961138 | −20.7e / −25.8e | +2.0e | +0.5e |
| det_stack | full_tec | ✗ | 357 | 655 | 0.960975 | −0.5e | +0.1e | — |
| det_swap | full_tec_swap | ✗ | 309 | 548 | 0.960861 | −11.8e | +0.8e | — |

Standalone: **rejected**. Fold 0's `tec_stack` also showed the highest blend gain of any arm in either
experiment (+2.6e-5, against +0.7e-5 for the base model itself), which looked like the
"equally good, more complementary" member this campaign has been hunting. **Fold 1 did not replicate
it** (−1.2e-5); the 2-fold mean is +0.7e-5, far under the +1.5e-5 admission gate. Rejected on
complementarity as well as on standalone.

### Fold-safety: the inner split must not depend on y

sklearn 1.9.1 deprecates `TargetEncoder(shuffle=, random_state=)`; passing `cv=5` silently yields
`StratifiedKFold(shuffle=True, random_state=None)`. That is **not** a leak — `fit_transform` genuinely
fits each fold's table on the complement, verified as `crossfit Z[j] == fit(other folds).transform(X[j])`.
But `StratifiedKFold.split` stratifies **on y**, so perturbing one label can move that row to a
different fold:

```
KFold(shuffle=True, random_state=0)           splits y-independent   max own-change 0.000e+00
StratifiedKFold(shuffle=True, random_state=0)  splits y-DEPENDENT    max own-change 2.217e-02
```

With `KFold(shuffle=True, random_state=seed)` the leave-out property is **exactly zero** and therefore
directly testable. `tests/test_te_all21.py` asserts exact equality rather than a loose bound. The first
version of that test asserted `own < 1e-3` and failed at 5.7e-2; the cause was the stratified splitter
moving folds, and the fix was to change the splitter and tighten the assertion — loosening it would
have hidden the real cause.

A second contamination bug, caught by arithmetic rather than by any check: a duplicated `te_all21`
body was left **nested inside** the `if "te_cond"` branch, so `full_tec` silently received 21 extra
columns (378 assembled vs 237+48+72=357 expected). It looked entirely plausible — no NaN, every
block healthy — and would have made Phase 8B measure `full + te + te_cond + te_all21` and report it
as conditional TE. `tests/test_view_composition.py` now pins the composition of every view and asserts
that no view contains a block it does not declare.

### Phase 8C/8D: not built, and why
`te_numgroups` (Age × segment, FD × segment) is already largely covered — the existing block has
`te_age_class_trip_cust`, `te_fd_class`, `te_fd_trip` — so its prior is low after 8A/8B. `te_shift`
(competition TE − original TE) estimates how the generator rewrote the conditional, but for predicting
the competition target `p_comp(y|x)` is the whole quantity and we already estimate it from 700k
competition labels; the delta is redundant with the `external`/`ogte` blocks that are worth +1.0e-3
and are already in the champion view. Neither is worth compute on current evidence.

## 6g. PHASE 8 -- EXTERNAL LABELLED DATA: NONE INDEPENDENT EXISTS, AND THE ONE ANOMALY IS DEFINITIVELY DISQUALIFIED

`scripts/audit_external_datasets.py`. Eight files across six Kaggle datasets, audited by hashing
shared columns rather than by name.

| dataset | rows | rating overlap | positives | verdict |
|---|---|---|---|---|
| binaryjoker | 129,880 | 100% | 56,428 | MIRROR |
| mysarahmadbhat | 129,880 | 100% | 56,428 | MIRROR |
| nilanjansamanta1210 | 129,880 | 100% | 56,428 | MIRROR |
| teejmahal20 train | 103,904 | verbatim subset | 45,025 | MIRROR |
| teejmahal20 test | 25,976 | verbatim subset | 11,403 | MIRROR |
| **raminhuseyn** | 129,880 | **3.4%** | **71,087** | different label vector |
| **yakhyojon** | 129,880 | **3.4%** | **71,087** | byte-identical to raminhuseyn |

**Answer to §20: no genuinely independent, independently collected external labelled dataset exists
for this schema.** Five of seven data files are verbatim mirrors or subsets of the same 129,880-row
survey. Three bugs in this audit each produced a confidently wrong answer before being fixed:

* the label mapping understood only the string vocabulary, so the source's **boolean** labels became
  all-NaN and agreement printed as exactly 0.0000 for every candidate — reading as "every label
  disagrees" when the comparison had simply never happened;
* column-name matching rejected 5 of 8 files; four merely rename columns, and two carry
  **"Online support"** where the competition carries **"Gender"** — a different survey question, which
  aliasing together silently corrupts every hash;
* the verdict keyed on full-feature overlap then called four files INDEPENDENT at 0%. They are not:
  their 18-column overlap is low only because Age/Flight Distance/segment were perturbed while the
  survey ratings overlap 100%. On a matched rating tuple, Age reads 66 vs 47, FD 1576 vs 300, Class
  Eco Plus vs Economy.

### The one anomaly, tested properly — and rejected (`scripts/run_external_probe.py`)

**Step 1, domain transfer.** A model trained on competition fold-0 fit rows scores **0.953366** in
domain and **0.798237** on raminhuseyn's own labels. The ranking transfers (0.798 ≫ 0.5), so the
predeclared 0.60 threshold said run step 2 — and that is worth recording as the point where a naive
reader would already have declared success.

**Step 2, the append with a row-count-matched duplicate control** (all arms on the raw view, 129,880
extra rows each):

| arm | rows | iter | AUC | Δ vs ctl |
|---|---|---|---|---|
| ctl | 503,739 | 1877 | 0.953366 | — |
| dup (duplicated competition rows, true labels) | 633,619 | 896 | 0.952312 | −105.4e-5 |
| append (raminhuseyn rows, own labels) | 633,619 | 2894 | 0.952584 | **−78.2e-5** |

`append − dup = +27.2e-5` is positive — the candidate's labels do carry real directional information,
consistent with the 0.798 transfer AUC. **But `append − ctl = −78.2e-5`: appending is worse than not
appending, so it is REJECTED.** The first version of this script tested only `append − dup` and printed
"escalate to more folds" for a treatment 78e-5 worse than doing nothing. The duplicate control exists
to separate information from row count; it is not itself the bar.

**Mechanism, and it is the same one that closed two earlier branches.** raminhuseyn asks a **binary**
`satisfied` / `dissatisfied` question at 54.73% positive, where the competition collapses a
**three-class** survey to 44.36%. Its target is not a re-sample of the same question — it is a
different definition, so the appended rows teach a foreign objective. That is the identical mechanism
behind the closed original-row append and the Phase 1–2 external-teacher result (−4.2e-4 at λ=0.2,
monotonically harmful). The duplicate control also loses (−105.4e-5), which confirms the loss is not
a row-count effect.

## 6h. PHASE 9 -- STOCHASTIC TREE CONSTRUCTION: CAPABILITY PROBES (`scripts/probe_dart_rf.py`)

Phase 8 is closed. Phase 9 asks a different question: not what features to add, but how the trees
themselves are built. The motivation is the only mechanism still standing -- randomisation plus
averaging helps (`extra_trees` +2.3e-4, RealMLP 8→32 members +2.0e-4, 5→10 folds +1.0e-4, 59-member
averaging +2.5e-4) -- and member proliferation is saturated, so the randomness has to come from
elsewhere. LightGBM 4.7.0 offers two construction modes never measured here: `dart` and `rf`.

Everything below was **measured on a 90k-row subsample of the raw 21 columns**, deliberately
separate from the champion view and from any eval fold, so it informs design without touching
validation. Numbers are inner-validation AUC on 30k held-out rows.

| arm | mode | `extra_trees` | inner AUC | note |
|---|---|---|---|---|
| G0 | GBDT | yes | 0.952463 | reference |
| D1 | DART drop .05 | yes | 0.952167 | |
| D2 | DART drop .10 | yes | 0.952238 | |
| D3 | **DART drop .05** | **no** | **0.956984** | **+4.82e-3 over D1** |
| R1 | RF bag .8 | no | 0.953159 | |
| R2 | RF bag .8 | yes | — | degenerate, stopped at 36 trees |
| R3 | RF bag 1.0 / freq 0 | no | 0.953204 | invalid, see below |

### Finding 1 -- DART has NO early stopping. Confirmed, and the warning was justified
LightGBM emits `"Early stopping is not available in dart mode"` and `best_iteration` comes back
`None`. The `early_stopping` callback is a **silent no-op**: training runs to `num_boost_round` with
no round selection whatsoever. Anything that reported a DART "best iteration" from a standard
callback was reporting a fiction. The fixed-round inner-selection protocol is mandatory, not
stylistic.

### Finding 2 -- under DART, `extra_trees` is CATASTROPHIC, the inverse of the champion finding
D3 − D1 = **+4.82e-3**. Everywhere else in this campaign `extra_trees` is the single largest lever
(+2.3e-4). Under DART it is catastrophic. Plausible mechanism: DART repeatedly drops already-added
trees and renormalises the survivors, so the damage from dropping is proportional to how much signal
each tree carries. `extra_trees` trees are individually much weaker (random thresholds), so a fixed
drop rate removes a disproportionate share of the signal while the renormalisation spreads what
remains thinner. This is a **hypothesis from one subsample**, not a measured mechanism — it is
exactly what the fold-level `extra_trees` control arm exists to test.

This inverted the planned matrix. The brief led with DART **+** `extra_trees`; running those first
would have spent hours confirming a large deficit, so the arms that can win are the ones run, and
`ctl_det` (GBDT **without** `extra_trees`) was added as the matched control — without it, a
D3-vs-champion delta confounds DART with the removal of `extra_trees`.

### Finding 3 -- CORRECTION: I predicted LightGBM would refuse RF without bagging. It does not.
R3 (`bagging_fraction=1.0, bagging_freq=0`) trained 144 trees without complaint. So LightGBM 4.7.0
RF mode does **not** require bagging, and R3 had no row subsampling at all — it was not a random
forest. R3 is dropped as invalid, and the harness now sets bagging explicitly rather than relying on
a default it does not get for free.

### Finding 4 -- my earlier "RF callback is usable" reading was an artifact
RF stopped at 174 trees while the snapshot grid began at 100, so the measurement loop broke after
**one** point and "callback agrees with the argmax" was never actually tested. Retracted as untested.

### The control caught a wrong champion config in this very harness — record kept
The first fold-0 launch set the champion parameters from `src/models/gbdt.py:27-31`
(`learning_rate=0.03`, `num_leaves=63`) on the reasoning that a module's defaults must be the
champion's. `ctl_es` returned **0.961170 @ 432 rounds** against the expected **0.961299 @ 797**, and
the run was stopped. Cause: `gbdt.py` holds the **family's generic defaults**, which the champion
**overrides**. The authoritative values are `scripts/run_fullfit.py:72-75`, which states it
reproduces `run_views.py::_fit_lgbm_es` bit-for-bit and which produced the recorded 0.9612988:
**lr 0.02, num_leaves 127, `extra_trees=True`**, with `n_estimators=6000` and
`early_stopping(300)` (also not the 4000/200 in `gbdt.py`).

Two independent signals agreed — the AUC shortfall of 12.9e-5 *and* the iteration count, since lr
0.02 naturally runs to roughly twice the rounds of lr 0.03 — so the control was believed over my
reading of the source. The failed launch's log is kept. Lesson recorded because it will recur:
**"the model module's defaults" and "the champion's parameters" are different objects in this repo,
and only the second is a valid baseline.** Had I skipped `ctl_es` and compared only against a
historical number, this would have surfaced as a mysterious uniform deficit across all seven arms.

## 6i. DART round selection MUST use refits, not truncated snapshots (`scripts/probe_dart_truncation.py`)

Because DART drops trees and renormalises on a schedule that depends on the full round count, an
r-round model need not be the r-round prefix of an N-round one. Measured, same seed, paired:

| arm | r=200 | 400 | 700 | 1100 | 1600 | worst |
|---|---|---|---|---|---|---|
| GBDT `extra_trees` | 0.00e-5 | 0.00e-5 | 0.00e-5 | 0.00e-5 | 0.00e-5 | **0.00e-5** |
| DART drop .05 | −23.2e-5 | −15.2e-5 | −2.7e-5 | +5.1e-5 | 0.00e-5 | **23.2e-5** |
| DART drop .10 | −34.4e-5 | −30.5e-5 | −0.6e-5 | −4.1e-5 | 0.00e-5 | **34.4e-5** |

The GBDT row is the **measurement control**: GBDT boosting is strictly additive, so truncation must
be exact there, and it is — to the last bit. The DART errors are an order of magnitude above the
+1.5e-5 admission gate and systematically signed (truncation *under*-states early, over-states
late). Selecting rounds from a truncated curve would optimise a different objective from the model
deployed. `run_stochastic_boosting.py` therefore **refuses** `--curve-mode snapshot` for any DART
arm unless this probe's verdict says it is safe, and the verdict is `false`.

### Second correction from the same probe: the DART curve TURNS OVER
DART drop .05 refit curve: 0.956081 (200) → 0.956847 (400) → **0.956969 (700)** → 0.956799 (1100) →
0.956538 (1600). My earlier reading — "all DART arms still climbing at 900" — was taken from the
**+`extra_trees`** arms and does not apply to the −`extra_trees` arm, which peaks near 700 and
degrades after. An unbounded grid would have spent its compute in the over-iterated region and could
have selected a tail point. The grid was shortened, then widened once timing allowed.

### Third correction: my compute projection was 55x too high
Extrapolating from probe scale put one DART arm at ~4 hours. The real timing probe measured
**0.058 s/round** at 504k rows × 285 features — the full 8,900-round grid costs ~9 minutes. At 60k
rows fixed overhead dominated and LightGBM could not fill 8 threads. Cheap compute meant the round
grid could be widened to `[500, 900, 1500, 2400, 3600]` rather than trimmed to a compromise.

## 6j. WHERE OUR REMAINING ERROR LIVES — and a methodological trap worth more than the answer

Two independent results here, and the second is why the first is trustworthy.

### TRAP: cross-member std is NOT an uncertainty measure. It is a p-level proxy.
My first version bucketed rows by the standard deviation of member logits and concluded that 82% of
errors sat in the *lowest*-disagreement decile. Before accepting that, I cross-tabulated dispersion
against predicted probability:

| predicted-p decile | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean member std | 0.545 | 0.487 | 0.460 | 0.437 | 0.393 | **0.273** | 0.436 | 0.462 | 0.497 | 0.614 |
| positive rate | .020 | .029 | .036 | .046 | .083 | .423 | .916 | .946 | .961 | .976 |

Dispersion is **smallest at p ≈ 0.5** — the genuinely uncertain rows — and largest at both extremes.
So `std_logit` mostly encodes `|p − 0.5|`. Its "least disagreement" decile was a mid-probability band
with 33% positives, i.e. nearly balanced and therefore holding the most pos/neg pairs of any stratum.
The 82% figure was a **pair-count artifact of a near-tie band**, not a difficulty measurement. All
measures were rebuilt level-free (rank spread, std/gap, family-rank spread), with the naive ones kept
under `NAIVE_*` so the confound stays visible in the report.

### RESULT: error vs disagreement is U-shaped, and the WORST band is FULL AGREEMENT
Conditioning the pair error rate on pairwise disagreement (4,000 pos × 4,000 neg = 1.6e7 pairs, so no
stratification artifact is possible):

| measure | band 0 (agree) | band 4 | band 9 (disagree) | worst band | shape |
|---|---|---|---|---|---|
| `rank_range` | 0.07432 | 0.03291 | 0.04335 | **0** | U-shaped |
| `rank_std` | 0.07589 | 0.03331 | 0.04408 | **0** | U-shaped |
| `family_rank_std` | 0.07429 | 0.03410 | 0.04379 | **0** | U-shaped |
| `std_over_gap` | 0.07149 | 0.03101 | 0.06749 | **0** | U-shaped |
| `abs_logit_gap` | 0.08560 | 0.03192 | 0.02415 | **0** | monotone down |

**All seven measures put the worst band at full agreement, not at maximum disagreement.** Rows where
all 98 members agree carry a 1.7–1.8x higher pair error rate than the median band, and 3.5x higher
than the most-disputed band under `abs_logit_gap`.

This **contradicts the variance-reduction hypothesis** that motivated Phase 9. The intuition that
disagreement marks unresolved variance is wrong here: full agreement marks rows where every model is
confidently wrong for a *shared* reason. Averaging and stochastic construction cannot fix a shared
bias — by construction they reinforce it.

Verdict recorded as **INCONSISTENT across measures** (5 of 7 say missing-signal, 2 say variance), so
this instrument alone does not close the question. But the one thing it does establish robustly is the
negative: **error is not concentrated in high-disagreement rows**, so the specific mechanism Phase 9
chases has little room to work. That is consistent with every Phase 7–8 result returning ≤ +3e-5.

## 6k2. DART cost per round GROWS with tree count — measured, not assumed

The first timing probe measured DART at **0.058 s/round** from a 300-round fit and projected a full
8,900-round grid at 0.09 h. That projection was wrong, and the fold-0 run showed it by taking over
25 minutes on `dart005` alone with no output. A second probe on identical data at 1,500 rounds:

| probe | rounds | s/round |
|---|---|---|
| early | 300 | 0.058 |
| late | 1500 | **0.152** |

**2.62x more expensive per round at 1,500 trees than at 300.** The mechanism is structural: every
new DART tree re-normalises the surviving ensemble, so per-round cost grows with the accumulated tree
count. GBDT's s/round was flat and its curve decays after ~900, so GBDT is genuinely linear. The
harness now extrapolates DART with `r^1.5` and GBDT linearly, and records which it used. A single-fit
linear projection is not valid for DART — worth remembering independently of this project, since
LightGBM's own early-stopping path would have hidden the problem entirely.

## 6k. `extra_trees` is confirmed load-bearing under the champion config (fold 0)

| arm | rounds | fold-0 AUC | vs `ctl_fixed` | blend w=2% vs v3 |
|---|---|---|---|---|
| `ctl_es` (established, ES) | 797 | 0.961299 | — | — |
| `ctl_fixed` (GBDT **+xt**) | 900 | **0.961352** | control | +0.22e-5 |
| `ctl_det` (GBDT **−xt**) | 900 | 0.960904 | **−44.8e-5** | −0.13e-5 |

`ctl_es` reproduced the recorded 0.961299 **exactly**, validating the harness. Under the champion
config `extra_trees` is worth **+44.8e-5** and is not optional — so the Phase 9A probe finding that
`extra_trees` is catastrophic under DART is a genuine *interaction*, not a general property. Note also
that the fixed-round protocol beats the ES control by **+5.3e-5**, larger than the +2.2e-5 measured in
Phase 7 but the same sign.

## 6l. Earlier version of this section, retracted as confounded

## 6j-OLD. WHERE OUR REMAINING ERROR LIVES — SUPERSEDED BY 6j, conclusion retracted as confounded

The instrument, the motivation and the gate below all stand. The conclusion reached with the naive
`std_logit` measure did **not**, because that measure is a p-level proxy rather than an uncertainty
measure. See 6j for the confound and the corrected level-free measurement.

This was the fork in Phase 9's strategy, and it is answerable from artifacts already on disk — no
training. If most remaining ranking error sits where members **disagree**, then variance reduction
is still the lever and DART/RF is the right medicine. If it sits where members **agree**, the models
are confidently wrong together and no amount of averaging can help.

Dispersion of the 98 single-model members' logits (members at OOF ≥ 0.955, `blend_*` excluded so
constituents are not double-counted), bucketed into deciles, scored with `blend_v3_final`:

| decile | rows | positives | mean p | AUC | share of errors |
|---|---|---|---|---|---|
| 0 (least disagreement) | 69,964 | see report | see report | **0.740375** | **40.76%** |
| 1 | 69,963 | | | 0.930521 | 11.09% |
| 2 | 69,963 | | | 0.949137 | 8.63% |
| … | | | | | |
| 8 | 69,964 | | | 0.974151 | 4.57% |
| 9 (most disagreement) | 69,964 | | | 0.981492 | 3.11% |

**The error is concentrated at the OPPOSITE end from what the variance-reduction hypothesis
predicts.** The single most-agreeing decile of rows produces 40.8% of all pairwise ranking errors
while the single most-disagreeing decile produces 3.1% — a 13x inversion. Each decile contributes
almost exactly the same number of pairs (~1.1–1.2e9), so this is not a pair-count artifact.

The decile table is confounded, though, and the confound is not small: a row's member-logit
dispersion is mechanically small when its predicted probability sits near 0 or 1, and such rows are
overwhelmingly the easy majority class. So decile 0 may be error-dense simply because it is a
near-tie stratum of one class. The script therefore also measures the **confound-free** version —
error rate conditioned on *pairwise* mean disagreement rather than on rows — which needs no
assumption that dispersion and difficulty are separable. Both numbers are in
`reports/error_concentration.json`.

## 7. Software quality

`tests/run_tests.py`: **55/55 pass**, including the four original leakage audits
(`test_fold_safe_te_never_sees_apply_rows`, `test_crossfit_row_never_sees_own_label`,
`test_external_features_use_no_competition_label`, `test_original_overlap_audit_is_small`), **5
index-space guards** and **9 `te_all21` / view-composition guards** added this session.

The guards were not decorative — each one was written after that failure class actually bit:


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

- **Champion / finalist**: 59-member equal-logit blend, OOF **0.961509** (primary 5-fold), submitted
  as `v3_final`, public **0.960980**, **rank 215 of 759** (refreshed 2026-10-05).
- **Second finalist (`v4_fulldata`, ref 56845975)**: public **0.961000**, i.e. **+2e-5** over v3.
  Chosen on **methodology**, not score — see below for why +2e-5 must not be called a win.
- **Submission budget**: 1 used today, 10/day cap, 4 used in total. Two final selections are allowed
  at the deadline.
- **What v4 actually tells us.** v4 keeps v3's 59 members, weights and OOF, but refits 32 of the
  LightGBM members on **100% of the labels** instead of that fold's outer-fit subset, with the
  iteration count set to the median of three independent 5% early-stopping holdouts.
  - It is **directionally consistent with the cross-validated estimate, but the public delta is
    unresolved at leaderboard precision**: +2e-5 against a **±2e-4** paired floor, i.e. an order of
    magnitude below the board's resolution. One such observation places essentially no constraint on
    the true effect — it is compatible with an effect several times larger or several times smaller.
    It must not be quoted as measuring the full-data gain.
  - **The evidence for a small full-data gain is the fold-level measurement**, not the public score:
    fold 0 **+4.0e-5**, fold 1 **+0.4e-5**, mean **+2.2e-5**, 2/2 folds.
  - What the board result *does* support is a sanity check on the whole chain — the train-fraction
    audit, the leakage-safe full-fit protocol, the median-of-3 iteration policy and the full-data
    feature verification all feed it, and any of them being wrong could easily have produced a large
    *negative* delta instead. It came back the way the mechanism predicted rather than reversed.
  - **No upside tuning**: the result gives no reason to expect *more* from this mechanism than the
    fold arithmetic already implied, so nothing should be scaled up on the strength of +2e-5. (That
    is "gives no reason to expect more", not "bounds it" — an earlier draft of this file conflated
    the two; see `p7_v4_bounds_claim_CORRECTION`.)
  - Rank did not move (215), the expected consequence of a change this small against a top-20 band
    only 1.0e-4 wide.
- **The gap is real and it is large.** Leader 0.961760, our public 0.960980, gap **7.8e-4**. That is
  **z = 3.1-4.4** even after allowing for correlation between two *different* solutions, and it clears
  2 sigma even between uncorrelated predictions (Hanley-McNeil SE ~ 0.0011 on the 59,969-row public
  split, Section 6e). The top twenty occupy a 1.0e-4 band *while sitting 7.8e-4 above us*, so they
  are a converged pack at a common higher level, not a spread field.
- **Every gain measurable in this campaign is single-digit e-5.** Phase 7 measured six further axes
  and the best was +2.2e-5. For calibration: **0.9335 of AUC comes free from 11 raw columns and a
  lookup table**, and everything this campaign has built -- 285 features, a 59-member blend, five
  rejected augmentation families -- buys the remaining **+0.028**. This is not a proven ceiling, but
  the reachable surface is small, and closing 7.8e-4 needs something structurally different.
- **The one concrete inference-policy defect found, now addressed.** No test-time model was trained on
  100% of the labels (Section 6e). v4 refits 32 LightGBM members on all 699,635 rows with an
  identical configuration and seed, and an iteration count taken as the **median of three independent
  5% early-stopping holdouts** (measured spread on the first config: [782, 1029, 973]) then corrected
  by only 1.039x. It is kept as a **complementary finalist** -- same 59 members, same equal weights,
  same OOF -- rather than replacing v3.
- **Next**:
  1. Finish the full-data refit, run `scripts/make_submission_v4.py`, pre-flight, and submit **once**.
  2. Re-check the final-submission selection rules on the competition Rules page near the deadline.
  3. Re-run `scripts/reproduce_finalist.py` from clean to confirm the submission is byte-reproducible.
  4. Do **not** re-tune the GBDT families, and do **not** revisit target encoding. Phase 8 tested the
     public solution's two TE mechanisms honestly — all-21 exact-value `TargetEncoder` and 18
     conditional cross keys through our own shrinkage machinery — and **both were rejected**, under both
     `extra_trees` and deterministic trees, and on ensemble complementarity as well as standalone. The
     measured reason is that our existing selective `te` block already captures the available signal
     and beats all-21 TE by **13.3e-5**; the value here is *interactions plus a shrinkage spectrum*,
     not column coverage.
  5. Section 20 is closed: **no independently collected external labelled dataset exists** for this
     schema, and the single anomaly carries a *different target definition* and is net harmful when
     appended (−78.2e-5 against a row-count-matched duplicate control). Stop dataset hunting.
  6. The one conclusion that survives every one of these negatives is **rho-independent and should
     govern what is tried next**: the top twenty occupy a 1.0e-4 band while sitting 7.8e-4 above us,
     and every gain actually measured is single-digit e-5. Closing that gap needs something
     structurally different, not more of the same.

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
