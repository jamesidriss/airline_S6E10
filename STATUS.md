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
| What | OOF AUC (primary 5-fold) | Public LB |
|---|---|---|
| best single (`xgb/full`) | 0.960946 | — |
| **4-family equal-logit blend (`blend_v1_equal4`)** | **0.961273** | **0.960930** |
| CV → LB gap | | **−0.000343** |

The gap matches the field's independently measured calibration (`LB ≈ OOF − 0.00035`).
**Our CV is trustworthy; we do not need to recalibrate it.**

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
**1 / 10** — `v1_equal4.csv`, public **0.960930**, ref 56788249.

## Top active hypotheses (ranked)
1. **Member breadth.** The field's best published OOF (0.961647) is a 25–28-member stack. Our
   4-member blend gains +0.00033 over the best single; more members should add more. In progress:
   a 40-entry zoo (family × view × structure × seed × fold-count).
2. **10-fold members** for decorrelation from the 5-fold ones (prior season: +0.0001 per model).
3. **`base_margin` residual boosting** from a strong model's logit (prior season: +6e-6…+5e-4;
   untested in S6E10).
4. **Auxiliary-task predicted class probabilities** for 4 key ratings (field: +0.00005).
5. Equal-weight averaging of *all* admissible members vs gated greedy selection — the former has
   zero selection freedom and is therefore safer against OOF overfitting.

## Immediate next experiments
- [ ] Run the 40-member zoo (`scripts/run_zoo.py --save-test`)
- [ ] Gated stack + equal-weight-all comparison (`scripts/stack.py`)
- [ ] `base_margin` and auxiliary-task probes (`scripts/probe_new_signals.py`)
- [ ] Submission #2 only if the stack clears the gate by a real margin
- [ ] Shadow-fold confirmation before any finalist is locked

## Git
`main` @ see `git log -1`. Repo: <https://github.com/jamesidriss/airline_S6E10>