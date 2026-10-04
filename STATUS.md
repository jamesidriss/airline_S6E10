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