# S6E10 — Verified Competition Meta

_Live research log. Every claim is tagged VERIFIED / PROMISING HYPOTHESIS / UNVERIFIED / INVALID-LEAKY._
_Re-verified from official Kaggle sources on 2026-10-03 (`research/raw/kaggle_pages.json`)._

Phase15 refresh, 2026-10-08: **INVALID-LEAKY for our OOF admission contract** —
both inspected author-linked notebooks from discussions
[747358](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/747358)
and [747182](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/747182)
select boosting checkpoints using outer evaluation labels. Their reported OOF
scores are rejected as promotion evidence. The second notebook excludes its
separate pseudo-test population, which supports a limited transfer diagnostic,
not admission of its inner OOF. The original
[39-cross pseudocode](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745892)
also uses outer early stopping; our test keeps the inner10percent FIT-only split.
Exact source hashes and classifications are preserved in
`reports/sol_phase15_notebook_protocol_audit_20261008.json` and
`research/sol_phase15_refresh_20261008.md`. No competitor prediction CSV used.

---

## 1. Official facts (VERIFIED)

Source: `competitions.CompetitionService/GetCompetition` and
`competitions.PageService/ListPages` (competitionId `125224`).

| Field | Value |
|---|---|
| Title | Predicting Airline Satisfaction — Playground Series S6E10 |
| Competition id | 125224 · forumId 9538353 |
| Start | 2026-10-01 00:00 UTC |
| **Deadline** | **2026-10-31 23:59 UTC** |
| Metric | ROC-AUC (`Roc Auc Score`, `isMax=true`, 5 dp truncation) |
| Train / test | `(699635, 23)` / `(299844, 22)`; row id column = `id` |
| Target | `satisfaction`; positive rate `310339/699635 = 0.44361` |
| **Max daily submissions** | **10** |
| **Final submissions** | **up to 2** |
| Team size | 3, merges allowed |
| Public LB share | 20 % (`leaderboardPercentage`) |
| Prizes | Kaggle merchandise, 1st/2nd/3rd, once per person per series |
| Data licence | CC BY 4.0 |
| Teams / competitors / submissions | 434 / 442 / 2 121 (2026-10-03) |

### External data: **ALLOWED** (VERIFIED)
Rules §2.6 (Competition-Specific Rules): *"You may use data other than the Competition Data
('External Data') … you will ensure the External Data is either publicly available and equally
accessible to use by all Participants … at no cost …, or satisfies the Reasonableness criteria."*
§2.6(b) adds that *"the use of external data and models is acceptable unless specifically
prohibited by the Host"*. No S6E10-specific prohibition exists. §2.6(c) permits AMLT.

### Other verified rules that bind us
- **§2.8.b** — prize winners must deliver the final model's code and it must reproduce the submission.
- **§6.b** — public code sharing must be on the competition's forum/notebooks under an OSI licence.
- §2.9 — team merger requires the merged team's total submission count ≤ days × 10.

---

## 2. Data forensics (VERIFIED locally)

See `reports/forensics/forensics.json`, `reports/original_analysis.json`,
`reports/noise_ceiling.json`, `reports/duplicate_key_signal.json`.

| Finding | Value |
|---|---|
| Train/test marginal drift | **None.** All 21 KS tests p ≥ 0.084 (max KS 0.0027 on `Cleanliness`) |
| Adversarial validation (no `id`) | **AUC 0.50027** → train and test indistinguishable |
| Adversarial validation (with `id` blocks) | AUC 0.99999 — trivially separates because train ids are 0…699634 and test ids 699635…999478. Not usable. |
| `id` signal | **None.** Every digit / modulo / divisor probe gives AUC 0.5000–0.5009. 20 equal-width id bins all have pos-rate 0.440–0.450. |
| Nulls | `Arrival Delay in Minutes` only: 292 train / 130 test |
| Class vocabulary | `Business`, `Eco`, `Eco Plus` — **3 levels, no Premium Economy** |
| `Baggage handling` | **5 levels (1–5), never 0** — every other rating has 6 levels (0–5) |
| Duplicate structure | **48.96 %** of train rows share an (13 ratings + 4 categoricals) key; 63.66 % share a 13-rating key |

### The community's `Cleanliness` shift claim: **UNVERIFIED / most likely noise**
One discussion post reports a `Cleanliness` KS p = 0.0376. Across 21 tests at α = 0.05 the expected
number of false positives is ≈ 1.05, so p = 0.0376 is expected. Our own 21 KS tests give
p ≥ 0.084 and adversarial AUC 0.50027. **Do not act on it.**

---

## 3. Original (real) dataset — provenance (VERIFIED)

The competition Data page states the data was *"inspired by"* a dataset; we identified the exact
generator source by schema fingerprint:

**`arseniyshutko/binary-aviation-satisfaction-129k` → `data.csv`, shape `(129880, 22)`.**

| Property | original | S6E10 synthetic |
|---|---|---|
| rows × cols | 129 880 × 22 | 999 479 × 22/23 |
| `Class` levels | Business / Eco / Eco Plus | **identical 3** |
| `Baggage handling` levels | 5 (never 0) | **identical 5** |
| other ratings | 0–5 | **identical 6** |
| `Age` distinct | 75 | **75** |
| synthetic values present in original | — | 100 % (Age, both delays), 99.999 % (`Flight Distance`) |
| exact full-row matches | — | **21** of 999 479 (0.002 %) |
| positive rate | 0.4345 | 0.4436 |

Six other repackagings of the same 129 880 rows were downloaded and compared
(`teejmahal20`, `mysarahmadbhat`, `nilanjansamanta1210`, `raminhuseyn`, `yakhyojon`,
`binaryjoker`). None matches better.

Upstream attribution: the survey is commonly credited to TJ Klein; Maven Analytics mirrors it as
Public Domain. **No published checksum exists for either copy (UNVERIFIED).**
Canonical Kaggle copy: <https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction>
(its copy carries an extra `Inflight service` column, which S6E10 and `arseniyshutko` do not).

**Note on a same-shape decoy (VERIFIED):** `bismasajjad/airline-passenger-satisfaction-dataset`
is 5 columns × 3.3 kB with `Class ∈ {First Class, Economy, Business}` and three separate labels.
It is **not** this dataset.

### What the original data is actually worth here (all measured by us)
| Use of the original data | Measured effect |
|---|---|
| **Smoothed `P(y | exact value)` lookups per column** | **+0.001015** (raw 0.959029 → 0.960044) |
| **Conditional surfaces anchored on FD / Class / travel / online-boarding** | **+0.001333** alone; only +0.000009 on top of the other blocks (redundant) |
| Original-only teacher model prediction (teacher AUC 0.9550 on comp train) | +0.000035 on top of the `external` block |
| **Exact-row lookup** (27.1 % coverage, 9.1 % unique, 91.5 % label agreement) | **+0.000038** — rejected |
| Appending original rows | Reported −0.00038 by the community; not attempted again |

→ **Use the original dataset as knowledge, not as rows.**

---

## 4. The decisive negative results

### 4.1 Duplicate feature groups carry **no** recoverable label signal (VERIFIED)
48.96 % of rows share an (13 ratings + 4 categoricals) key, and the *in-sample* group label mean
reaches AUC 0.9967. But a **fold-safe** group target encoding on those exact keys gives only:

| key | train→test key coverage | fold-safe TE AUC |
|---|---|---|
| 13 ratings | 53.8 % | 0.8677 |
| 13 ratings + 4 categoricals | 38.1 % | 0.8178 |
| + Age | 9.4 % | 0.6303 |
| + Age + exact `Flight Distance` | 0.01 % | 0.5001 |

**Interpretation: the generator drew labels i.i.d. per row from p(x).** Duplicate rows share p(x)
but not their noise draw. Therefore:
- there is **no leak** through duplicates,
- target encoding cannot help (measured: −0.00005 on top of route-profile means),
- the achievable AUC is bounded by how well p(x) can be estimated, not by memorisation.

### 4.2 The original dataset's own CV AUC of 0.9948 is duplicate-inflated (VERIFIED)
Fitting a LightGBM to the original data with plain 5-fold CV gives 0.9948 — implausibly high for
this survey. It is inflated because validation rows have near-twins in the training fold. The
original-only teacher transferred to competition data only reaches 0.9550. Treat 0.9948 as an
upper bound on what a *duplicated* dataset can look like, not as a Bayes ceiling.

---

## 5. Live field state (2026-10-03, VERIFIED via `LeaderboardService` and Code page)

Public LB top: **0.96155**. Ranks 1–45 lie in 0.96125–0.96155; rank 100 = 0.96100;
rank 200 = 0.95898. Top-1 → top-45 spread is 3e-4.

**Calibration rule measured independently by four field authors: `public LB ≈ OOF − 0.00035`,
public LB noise ≈ ±0.0003.** Our own first submission confirms it: OOF 0.961273 → LB 0.960930
(gap −0.000343).

Highest *published* OOF anywhere in the field: **0.961647** (a 28-member level-2 stack).
Highest *published* notebook LB: 0.96132. So the podium is running ~2–3e-4 better than anything
public, i.e. **OOF ≈ 0.9619 is needed to lead.**

### Community ablation ladder to beat (from `arhancanli12/s6e10-generator-fingerprints-lgbm-cv-0-9610`, audited)
```
raw 21 columns                                    0.958898
+ value counts + original target means + digits   0.960250
+ GPT-2 BPE token keys                            0.960364
+ fold-safe target encoding                       0.960547
+ original-model prediction                       0.960769
+ slower/wider trees (lr 0.02, 127 leaves)        0.961029
equal-weight logit blend of lgb/xgb/cat/realmlp   0.961307
```
Their top feature importances: `orig_pred` (46 % of gain), `Online boarding_om`, `combo_om`.

### Community findings marked LEAKY or weak — and what we verified
| Claim | Our verdict |
|---|---|
| `yekenot/ps-s6-e10-realmlp-pytabkit` fits `factorize` + `value_counts` on the **full train set before CV** | **INVALID-LEAKY (mild, unsupervised).** It inflates that notebook's reported OOF; it does not inflate the LB. Its CV is not a clean baseline. Our pipeline fits nothing label-bearing outside the fold. |
| `giannhsstamelias` `Cleanliness` covariate shift p = 0.0376 | **UNVERIFIED → noise.** See §2. |
| GPT-2 BPE token keys of synthetic numerics | **PROMISING HYPOTHESIS for S6E10.** Measured +0.000025 for us (arhancanli +0.00011; S6E9 20th +0.00038). Retained: cheap and it did not hurt. |
| TE of rating **pairs** | Measured −0.00002 in the community. Not pursued. |
| Appending original rows | −0.00038. Not pursued. |
| Survey `0` as an N/A sentinel | **VERIFIED semantics**: `Online boarding == 0` → 62.0 % satisfied vs 10.5 % at 1; `Inflight wifi == 0` → 88.7 % satisfied. `Baggage handling` has exactly one zero in the original. Encoded as explicit `na_*` masks. Standalone effect not separately measurable (ratings are massively redundant: leave-one-out on `Online boarding` costs only 0.0006). |

### Transferable mechanisms from previous Playground seasons
| Mechanism | Evidence |
|---|---|
| Original data as **knowledge** (means / teacher), not rows | S6E9 3rd, S6E10 ×5 independent groups. **Adopted.** |
| Exact-value fold-safe target encoding | S6E7 29th (+0.0012), S6E9 20th (+0.0016). **Measured flat here (−0.00005)** because route-profile means already carry it. |
| Numeric + exact-categorical **twin** columns, backbone of RealMLP | S6E9 2nd/3rd, S6E1 2nd. **Adopted.** |
| **RealMLP** competitive-or-better; S6E8 1st place was a *single* RealMLP | **Adopted as a first-class family.** |
| Rank/logit-space blending, L2 logistic stacks, nested meta-CV | Every recent season's top solution. **Adopted.** |
| 5 → 10 folds | S6E9 20th +0.0001 per model. **In progress.** |
| GPT-2 BPE token fingerprints of synthetic numerics | S6E9 20th (+0.00038 LB), S6E9 2nd, S6E10 arhancanli. **Adopted.** |
| `base_margin` from a strong model's OOF logit | S6E9 (+0.00006…+0.00050). **Untested here — candidate.** |
| Residual/label-noise-exploiting tricks, TabPFN, greedy hill-climbing, ratio features | Measured flat or harmful in prior seasons. **Deprioritised.** |
| **Fitting blend weights to the public LB** | S6E9 2nd: public 0.94945 → private 0.94313 (Δprivate ≈ +16.7e-5 − 0.88·Δpublic, r = −0.97). **Hard prohibition.** |

---

## 6. Our own results (see `STATUS.md` and `experiments/ledger.jsonl`)

| View (LGBM, primary 5-fold) | OOF AUC |
|---|---|
| raw | 0.959029 |
| raw_ogsurf | 0.960362 |
| raw_trans | 0.960484 |
| raw_ext | 0.960044 |
| core3 | 0.960879 |
| **full** | **0.960904** |
| full, xgb / cat / realmlp | 0.960946 / 0.960909 / 0.960820 |
| 4-family equal-logit blend | 0.961273 → **public LB 0.960930** (gap −0.000343) |

Our early-stopping convention is *stricter* than the field's (an inner 10 % split carved from the
FIT rows, never the evaluation fold), so our OOF is not inflated by checkpoint selection.

---

## 7. New mechanism discovered here: randomised-split GBDTs (VERIFIED, ours)

**`extra_trees=True` in LightGBM is worth ≈ +2.3e-4 AUC over a fully grown deterministic
LightGBM on the identical feature view**, and it is the single strongest model we have built.

Measured on the `full` view, primary 5-fold, identical everything else:

| config | OOF AUC |
|---|---|
| lgbm 127 leaves, lr 0.02 (deterministic) | 0.960833 |
| lgbm 63 leaves, lr 0.03 | 0.960830 |
| lgbm 255 leaves, lr 0.02 | 0.960955 |
| lgbm max_bin 63 | 0.960901 |
| lgbm colsample 0.5 / subsample 0.6 | 0.960934 |
| **lgbm 127 leaves + `extra_trees=True`** | **0.961175** |
| lgbm 127 leaves + `extra_trees=True`, seed 1 | 0.961136 |
| lgbm 127 leaves + `extra_trees=True`, seed 2 | 0.961189 |

**Why this is the expected result and not a fluke.** Section 4.1 established that labels are
i.i.d. Bernoulli(p(x)). A deterministic GBDT greedily grows splits until the *training* labels
in each leaf are nearly pure, i.e. it fits the per-row noise draw as much as p(x). Extremely
randomised trees take random features and random thresholds, so each tree is a much weaker
learner whose errors are far less correlated; averaging many of them cancels more of the
independent noise while preserving the p(x) component. The same argument predicts that seed
bags and fold-count bags help, which we also measure.

**Nothing in the public S6E10 field uses this** — the field's feature views are shared almost
verbatim across notebooks, and `extra_trees` does not appear in any of the audited ladders.
It is cheap (one LightGBM flag) and it is the largest single-model gain we have measured.

### 7a. The flag is conditional on feature dimensionality — do not generalise it
Measured on the independent `shadow` folds (seed 777001), same code path:

| view | n_feat | deterministic | `extra_trees=True` | Δ | verdict |
|---|---|---|---|---|---|
| `full` | 285 | 0.960868 | **0.961143** | **+0.000275** | reproduced |
| `raw_ext` | 75 | 0.960067 | 0.958506 | −0.001561 | not reproduced |
| `raw` | 21 | 0.958959 | 0.953388 | −0.005571 | not reproduced (catastrophic) |

Randomly chosen features *and* thresholds need many candidate columns to land on a good split.
The flag is safe on our ensemble (every `extra_trees` member uses a rich view) and unsafe on a
minimal one.

---

## 8. Community notebook audit — the leaderboard-leading technique, tested honestly

`kozykappa/S6E10 | Local-Reliability Residual Blend` (`scriptVersionId=354946502`, public LB
**0.96152**, above ours) does **not train anything**. It reads a shared external OOF library
(`najiama/s6e10-oof`) plus four other notebooks, then post-processes them:

1. split the OOF into equal-frequency score regions,
2. measure each model's **local** ROC-AUC inside each region,
3. residualise the "aux carrier" against rank/logit summaries of the public ensemble,
4. apply the correction gated on local reliability and on disagreement.

**Audit verdict: not adoptable.** The correction is fitted and then evaluated on the same OOF
vector, so the reported number is optimistic by construction; and the inputs are another team's
predictions rather than our own, whose folds and preprocessing we cannot verify. This is exactly
the "never blend an unexplained prediction CSV just because it scores well" case.

**But the underlying question is legitimate, so we tested it ourselves**
(`scripts/local_reliability_gate.py`), with region weights for each fold computed only from the
*other* folds' rows:

| bin count | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| Δ vs equal weighting | −0.000000 | −0.000000 | −0.000000 | −0.000000 |

**No member beats the equal-weight blend in any score region**; per-bin local-AUC gains are
negative (two sampled members: −0.0137 and −0.0065 mean gain). An oracle control — a "member"
defined as `y` itself — is correctly detected at +0.434, so the machinery is not silently broken
and the zero is a real finding. **Region-local reliability gating is rejected with evidence.**

---

## 9. Original-data measurement that closes the mechanism story

Appending the 129,859 leak-audited original rows to training (`full` view, `extra_trees`):

| original-row sample weight | OOF AUC | Δ |
|---|---|---|
| 0.0 (excluded) | **0.961078** | — |
| 0.3 | 0.960493 | −0.000585 |
| 1.0 | 0.959975 | −0.001103 |

Monotone in weight. Since the original survey's p(x) is learnable to 0.9949 while the synthetic
labels only support 0.9612, the original rows teach a *different, sharper* function and bias the
model — yet their **conditional target statistics** are worth +0.001015, because those encode
feature-level structure that survives even when the overall dependence is much weaker.
**Use external data as knowledge, never as rows — measured, not assumed.**

## 10. SOL source audit, 7 October 2026

`goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160` contains a genuinely new
full-context TabPFN route representation. Its tree OOF cells early-stop on the
outer evaluation labels, and its meta folds differ from the base folds without
fully nested base training. Reject those tree/stack CV claims as comparable
evidence; do not import their prediction CSVs. Reimplement only the pretrained
inference mechanism with all context targets drawn from outer FIT, fixed primary
folds and no outer-fold model selection. The other inspected current leading
stacking notebooks import public prediction libraries of insufficient nesting
provenance. Detailed classifications and the ranked queue: `research/sol_queue.md`.

## 11. Neural validation and twin audit, 7 October 2026

All **12** neural members in the 59-member legacy portfolio supplied the outer
evaluation frame and labels as PyTabKit validation inputs. RealMLP's
`use_early_stopping=False` does not prevent the default `use_best_epoch=True`
checkpoint callback from restoring the best validation epoch; TabM also uses
validation for model selection. Reject those 12 legacy OOF vectors as honest
outer-fold evidence. Do not treat earlier neural ablation negatives as conclusive
when they share this contract. Banked artifacts remain preserved.

The numeric twins were independently factorized in each train/validation/test
frame. Different split vocabularies could assign the same numeric category code
to different distances. Twins now retain their literal numeric values, and
PyTabKit's FIT-only ordinal encoder handles shared and unseen values consistently.
The previous `twin=False` argument was ignored; it now removes twins. Neural
selection now receives only a stratified **10 percent** inner split of outer FIT,
seed **1**, with disjoint training and stopping rows. New run IDs carry
`sol_v2_inner10_literal_twins` to prevent accidental reuse of legacy results.
Regression tests flip outer labels and verify unchanged fold0 fit/ES inputs for
all three entry points, and test differing split vocabularies explicitly.

## 12. Generic test refit configuration audit, 7 October 2026

The old generic LightGBM full-prediction helper differed from its CV helper:
min_child_samples 20 versus 40, reg_lambda 0 versus 1, and omitted explicit
bagging_seed=seed+1 / feature_fraction_seed=seed+2. The generic CatBoost refit
default learning rate was 0.05 versus CV's 0.04. Explicit recipe overrides
could mask some differences; historical prediction vectors cannot prove their
parameters without a fitted-model manifest. Do not call these test refits exact
copies of the CV recipe. Banked predictions are preserved, not silently repaired.

The helpers now preserve estimator configuration. XGB and CatBoost selected
iteration indices are zero based and refit counts require +1; XGB index zero
must not be replaced by the maximum 6000-round budget. Parameter-identity and
first-tree regression checks pass. No leaderboard improvement is inferred from
these implementation changes; clean OOF/test replay remains necessary.

## Direct public-topic refresh, 7 October 23:58 UTC

[The39 categorical rating-context cross experiment](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/745892)
explicitly early-stops and restores checkpoints using the outer evaluation
targets. Its reported0.000336522/0.000284399 OOF gains are rejected as honest
admission evidence. Only its deterministic, label-free categorical mechanism
may be reimplemented with inner FIT selection and matched controls.
[The119-member public stack](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746148)
does not establish fully nested base-model training. Its reported0.96200 is
not comparable to our immutable, independently generated OOF. No shared
predictions have been adopted. See `sol_public_refresh_20261008.md`.


Phase 15 measured closure (2026-10-08): the raw/route portfolio reached full five-fold OOF 0.961788023, gaining 22.409 millionths versus v6 with five positive folds, but its mean/SE ratio of 2.403 fell below 2.5. Native39 standalone gains were 164.279 and 54.006 millionths; v6 slot gains were +9.512 and -2.574 millionths, so the portfolio benefit did not replicate. Route plus 13 fold-safe expected ratings gave a three-fold TabPFN portfolio mean gain of 6.306 millionths, below the 15-millionth continuation gate. The historical 20% classical diagnostic gained 19.079 millionths, below the 30-millionth clean-replay trigger. No new submission or qualified B emerged; v6 remains A. Details are in reports/sol_phase15_final_report_20261008.md and the append-only experiment ledger. No fourth branch was started.

Phase16 public-source rejection (2026-10-09): the noise notebook's imported135-member stack and Deotte's imported135-member TFM/public stack do not provide fully nested base-level validation; listed10fold members also disagree with the asserted5fold scheme. Meta-FIT OOF predictions can include meta-evaluation labels through their base training contexts. Reject their CV as local admission evidence. Naji's0.96208 notebook exposes only imported blend vectors, not its producer or selection process. The route notebook's ordinary GBDT early stopping uses outer evaluation labels; reject those reported gains under our inner-FIT stopping contract. No competitor vectors adopted. Source audit: research/phase16_independent_audit.md. The3.77% injected-flip interpolation is model/domain-equivalent corruption, not measured irreducible competition noise; see reports/phase16_noise_frontier.md.
