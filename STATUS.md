# STATUS — Kaggle Playground S6E10 (Airline Satisfaction)

_Last updated: 2026-10-03, after submission #1._

## Verified competition facts
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
| Public LB leader (day 3) | 0.96155 |

## Validation scheme
- **primary**: `StratifiedKFold(5, shuffle, seed=20261010)` — immutable, registered on disk.
- **shadow**: `StratifiedKFold(5, seed=777001)` — confirmation only, not yet used.
- **block10**: 10 folds for finalists.
- Early stopping uses an **inner 10 % split carved from the FIT rows**, never the evaluation
  fold. This makes our OOF ≈ 1e-4 *less* optimistic than the field's convention.
- Test predictions use a **second pass**: each fold model is refitted on all fold-fit rows at the
  median CV-selected iteration count. More data, no held-out label consulted.

## Current results

### Champion & submissions
| # | File | Members | OOF AUC | Public LB | LB − OOF |
|---|---|---|---|---|---|
| 1 | `v1_equal4` | 4 (xgb, cat, lgbm, realmlp) | 0.961273 | 0.960930 | −0.000343 |
| 2 | `v2_equal34` | 34, quality-weighted logit | **0.961438** | 0.960920 | −0.000518 |

### ⚠ The most important calibration result
Submission #2 gained **+0.000165 OOF** over #1 and moved the public LB by **−0.00001**.
Test-prediction Spearman between the two files is **0.9954** — and the *OOF* Spearman is also
0.9954, i.e. OOF and test orderings move together, so the CV is a faithful guide.

Consequence: **paired public-LB noise is ≈ ±0.0002.** To resolve a real improvement on the board
we need OOF gains ≳ +0.0003. Smaller OOF improvements are real but invisible on the public split
(59,969 rows ⇒ standalone AUC SE ≈ 0.0015; paired SE ≈ 0.0002 when ρ ≈ 0.995).

**Therefore: stop hill-climbing the public LB, and stop spending submissions on sub-0.0003 OOF
deltas.** Keep raising honest OOF; spend submissions only on ≥ +0.0003 steps and on the two final
selections.

### The two levers that actually moved single-model AUC
| Lever | Effect on a single model | Members |
|---|---|---|
| **`extra_trees=True` in LightGBM** | **+2.3e-4** (0.960833 → 0.961175) | 12 |
| **10-fold instead of 5-fold** | **+1.1e-4** (0.961136 → 0.961242) | 2 |
| leaves 255 / colsample 0.5 / max_bin 63 | +0.1e-4 … +1.2e-4 | 4 |
| RealMLP seed bag | ±0.9e-4 spread across seeds | 4 |
| CatBoost depth 6 vs 8 | +0.3e-4 | 1 |

`extra_trees` is not used anywhere in the public S6E10 field. It is predicted by our
i.i.d.-label-noise diagnosis: random splits decorrelate trees so averaging cancels more of the
per-row noise while keeping the p(x) component. The 10-fold gain is additive and additionally
decorrelates the member (its fold models see different 90 % subsets).

### Blend schemes (34 members, primary 5-fold)
| scheme | OOF AUC | note |
|---|---|---|
| equal over all | 0.961434 | zero selection freedom |
| quality-softmax (T = 2e-4) | 0.961438 | zero fitted parameters |
| family-balanced | 0.961407 | **worse** — upweights the weak cat/xgb members |
| **nested logit-LR stack** | **0.961466** | honest: coefficients fitted on 4 folds, scored on the 5th |

The nested-LR stack beats fixed weights by only 3e-5, which is inside the fold noise, so the
robust choice remains a fixed-weight blend.

### Feature-view ablation ladder (LGBM, primary folds)
| View | Blocks | n_feat | OOF AUC | Δ vs raw |
|---|---|---|---|---|
| raw | 21 raw columns | 21 | 0.959029 | — |
| raw_teacher | + original-only teacher (teacher AUC 0.9550) | 23 | 0.959064 | +0.000035 |
| raw_ext | + original-data smoothed target stats | 75 | 0.960044 | **+0.001015** |
| raw_trans | + counts / route profile / digits / N-A masks | 148 | 0.960484 | **+0.001455** |
| raw_ogsurf | + original-data conditional surfaces | 189 | 0.960362 | **+0.001333** |
| core3 | raw + cat twins + trans + external + teacher | 225 | 0.960879 | **+0.001850** |
| core3_ogsurf | + original surfaces | 393 | 0.960888 | +0.001859 |
| core3_te | + fold-safe target encoding | 273 | 0.960827 | +0.001798 |
| **full** | + GPT-2 BPE token keys | 285 | **0.960904** | **+0.001875** |

### Family comparison on the `full` view
| Family | OOF AUC | logit-corr to lgbm |
|---|---|---|
| xgb (depth 8) | 0.960946 | — |
| cat (depth 8) | 0.960909 | 0.9974 |
| lgbm (127 leaves, lr 0.02) | 0.960904 | — |
| realmlp (6 epochs, n_ens 8, twins) | 0.960820 / 0.960881 (seed 7) | 0.9959 |

### Ensemble geometry
| Blend | OOF AUC |
|---|---|
| equal-probability (4) | 0.961276 |
| **equal-logit (4)** | **0.961273** |
| equal-rank (4) | 0.961271 |
| greedy hill-climb (logit) | 0.961283 |
| nested logit-LR stack (C ∈ 0.03…3) | 0.961269 |

The three blend geometries are indistinguishable — as expected for a rank metric — so there is no
geometry to gain here. The gain came from **member breadth** (+0.00033).

## Key discoveries
1. **Source dataset identified (VERIFIED)**: `arseniyshutko/binary-aviation-satisfaction-129k`
   (129,880×22) — exact schema fingerprint (3 `Class` levels, 5 `Baggage handling` levels, 75
   `Age` levels, 100 % value-support coverage). Only **21** of 999,479 competition rows match an
   original row exactly → no row-level answer key.
2. **The original dataset is valuable as knowledge, not as rows.** Smoothed `P(y | exact value)`
   gives **+0.001015**; conditional surfaces give **+0.001333** alone; an original-only teacher
   reaches AUC 0.9550 on competition train but adds only +0.000035 on top of the target
   statistics (its information is subsumed). Kept only for ensemble diversity.
3. **Duplicate feature groups carry NO recoverable label signal (decisive).** 48.96 % of rows
   share an (13 ratings + 4 categoricals) key, but a fold-safe target encoding on those exact
   keys scores only AUC 0.8677 (vs 0.9609 for the full model). The generator drew labels **i.i.d.
   per row** from p(x): duplicate rows share p(x) but not their noise draw. Consequences —
   no leak through duplicates, target encoding cannot help (−0.00005 measured), and the ceiling
   is set by how well p(x) can be estimated. This is why we stopped chasing memorisation.
4. **The original dataset's own 0.9948 5-fold AUC is duplicate-inflated** — validation rows have
   near-twins in the training fold. Not a Bayes ceiling.
5. **`id` carries no signal** (all digit/modulo/divisor probes AUC 0.5000–0.5009; 20 id bins all
   0.440–0.450). Adversarial validation without `id`: **AUC 0.50027**. All 21 marginal KS tests
   p ≥ 0.084. **No covariate shift** — the community's `Cleanliness` p = 0.0376 claim is a
   false positive.
6. **`Flight Distance` behaves as a route ID** (3,474 levels, ~200 rows/value). Route-profile group
   statistics keyed on it are a large part of the `trans` block gain.
7. **Survey `0` is an N/A sentinel**: `Online boarding == 0` → 62.0 % satisfied vs 10.5 % at 1;
   `Inflight wifi == 0` → 88.7 %. `Baggage handling` has exactly one zero in the original.
   Encoded as explicit `na_*` masks; the standalone effect is not separable because the ratings
   are massively redundant (leave-one-out on `Online boarding` costs only 0.0006).

## Failed / rejected hypotheses — do not re-spend
| Idea | Measured | Verdict |
|---|---|---|
| Exact-row lookup into the original dataset | +0.000038 | rejected |
| Fold-safe TE on top of route-profile means | −0.00005 | rejected |
| Target encoding on duplicate feature keys | AUC 0.8677 standalone | rejected |
| Original-only teacher on top of `external` stats | +0.000035 | diversity only |
| Original-data conditional surfaces *on top of* `external` | +0.000009 | redundant |
| `id` digit / modulo / batch features | AUC ≈ 0.500 | rejected |
| GPT-2 BPE token keys | +0.000025 | kept (cheap, non-negative) |
| Blend-geometry search (prob / logit / rank / LR / greedy) | spread 1.4e-5 | no gain available |
| Appending original rows | reported −0.00038 by the field | not attempted |
| Pseudo-labelling | reported harmful by the field | not attempted |

## Kaggle submissions used today
**2 / 10** — `v1_equal4.csv` (0.960930), `v2_equal34.csv` (0.960920).

## Top active hypotheses (ranked)
1. **More 10-fold random-split LightGBM members.** 10-fold gave +1.1e-4 *and* decorrelates the
   member, so it is the only lever that improves a member twice over. Only 2 exist so far.
2. **RealMLP batch size.** Our RealMLP (0.96082) is 3.5e-4 below the field's best (0.96117);
   their recipe uses `batch_size=256` where we used 4096, a 16× difference in gradient steps per
   epoch. RealMLP members currently contribute +1.3…1.7e-5 each in the marginal analysis, so
   fixing this is the cheapest way to add decorrelated strength.
3. **CatBoost breadth** — only 2 members; it is the one genuinely different algorithm family
   left (XGBoost is near-duplicate with LightGBM at logit-corr 0.998).
4. **`base_margin` residual boosting** from a strong model's logit (prior season +6e-6…+5e-4;
   untested in S6E10).
5. **Auxiliary-task predicted class probabilities** for 4 key ratings (field: +0.00005).
6. **Shadow-fold confirmation** of the final stack before locking a finalist.

## Rejected — do not re-spend
| Idea | Measured |
|---|---|
| Exact-row lookup into the original dataset | +0.000038 |
| Fold-safe TE on duplicate feature keys | AUC 0.8677 standalone; −0.00005 in the model |
| Target encoding on top of route-profile means | −0.00005 |
| Original-only teacher on top of `external` stats | +0.000035 |
| Original conditional surfaces on top of `external` | +0.000009 |
| `id` digit / modulo / batch features | AUC ≈ 0.500 |
| Blend-geometry search (prob/logit/rank/LR/greedy) | spread ≤ 3e-5 |
| Family-balanced weighting | −0.00003 |
| Kaggle-like GOSS with bagging | invalid combination; dropped |
| Public-LB hill climbing | ±0.0002 paired noise; wasted submission |
| sklearn `ExtraTreesClassifier` (1000 trees) | too slow, marginal value — dropped for LightGBM `extra_trees` |

## Immediate next experiments
- [ ] Finish zoo 3 (CatBoost breadth, RealMLP batch sweep, 10-fold XT)
- [ ] If the stack clears **+0.0003 OOF** over 0.961438 → submission #3
- [ ] `base_margin` + auxiliary-task probes
- [ ] Shadow-fold confirmation of the finalist
- [ ] Reproduce the finalist from clean code, then choose the 2 final submissions by robustness

## Git
`main` — see `git log -1`. Repo: <https://github.com/jamesidriss/airline_S6E10>