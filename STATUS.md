# STATUS — Kaggle Playground S6E10 (Airline Satisfaction)

_Last updated: 2026-10-03_

## Verified competition facts
| Item | Value |
|---|---|
| Metric | ROC-AUC (maximize) |
| Deadline | **2026-10-31 23:59 UTC** |
| Submissions | 10/day; **up to 2 final submissions** |
| Team size | 3 (merges allowed) |
| **External data** | **ALLOWED** (Rules §2.6) if public + free + equally accessible |
| Public LB share | 20% (test = 299,844 rows) |
| Prizes | Kaggle merchandise, top 3 |
| Data licence | CC BY 4.0 |
| Public LB leader (day 3) | 0.96155 |

## Validation scheme
- **primary**: `StratifiedKFold(5, shuffle, seed=20261010)` — immutable, all discovery CV.
- **shadow**: `StratifiedKFold(5, shuffle, seed=777001)` — confirmation only.
- `block10`: 10-fold for finalists.
- Early stopping uses an **inner 10% split carved from the FIT rows only** — never the eval fold.
  This makes our OOF ~1e-4 *less* optimistic than the field's convention.

## Current champion (LGBM on `full` view)
| Metric | Value |
|---|---|
| **OOF AUC (primary 5-fold)** | **0.960904** |
| Public LB | not submitted yet |
| Validation | primary |

## Feature-view ablation ladder (LGBM, primary folds, honest inner early stopping)
| View | Blocks | n_feat | OOF AUC | Δ vs raw |
|---|---|---|---|---|
| raw | 21 raw columns | 21 | 0.959029 | — |
| raw_teacher | + original-only teacher | 23 | 0.959064 | +0.000035 |
| raw_ext | + original-data smoothed target stats | 75 | 0.960044 | **+0.001015** |
| raw_trans | + counts / route-profile / digits / N-A | 148 | 0.960484 | **+0.001455** |
| core3 | raw + cat twins + trans + external + teacher | 225 | 0.960879 | **+0.001850** |
| core3_te | + fold-safe target encoding | 273 | 0.960827 | −0.000052 |
| **full** | + GPT-2 BPE token keys | **237** | **0.960904** | **+0.001875** |

## Key discoveries
1. **Source dataset identified (VERIFIED)**: `arseniyshutko/binary-aviation-satisfaction-129k`,
   129,880×22. Exact schema match; `Class` has 3 values (no Premium Economy); `Baggage handling`
   has 5 values (no 0); 100% of synthetic values occur in the original. Positive rate 0.4345 vs
   synthetic 0.4436. Only 17 train rows match an original row exactly → no row-level answer key.
2. **Original data is far more useful as *knowledge* than as *rows***. Appending original rows is
   reported to hurt (−0.00038). Smoothed `P(y | exact value)` lookups from original labels give
   **+0.001015**, the single biggest feature lever found so far.
3. **An original-only teacher model reaches AUC 0.9550 on competition train** but on top of the
   `external` block it adds almost nothing (+0.000035) — its information is largely subsumed by
   the per-value original target statistics. Teacher is kept for ensemble diversity only.
4. **Exact-row lookup into the original data is nearly worthless**: 27.1% coverage, 9.1% unique,
   91.5% label agreement, and adding it to raw LGBM gave **+0.000038**. The signal in the
   13 survey ratings is already exhausted by the model.
5. **`Flight Distance` behaves as a route ID** (3,474 levels, ~200 rows/value). Route-profile
   group statistics keyed on it are a major part of the `trans` block gain.
6. **`id` carries no signal.** Every digit/modulo/divisor probe gives AUC 0.5000–0.5009.
   Adversarial validation without `id`: **AUC 0.50027** → train and test are indistinguishable.
   All 21 marginal KS tests: p ≥ 0.084 → no covariate shift. (The community claim of a
   `Cleanliness` shift at p=0.0376 is consistent with ~0.6 expected false positives.)
7. **Fold-safe target encoding did not help** (−0.00005): the transductive route-profile means
   already carry that information.
8. **Survey `0` is an N/A sentinel** (`Online boarding == 0` → 62.0% satisfied vs 10.5% at 1).
   `N/A` masks added; measured inside the `trans` block.

## Failed / rejected hypotheses (do not re-spend)
- Exact-row original-data lookup as a feature: +0.000038. Rejected.
- Standalone `id` digit/modulo features: AUC ≈ 0.500. Rejected.
- Fold-safe TE on top of route-profile means: −0.00005. Rejected.
- Original-only teacher on top of `external` statistics: +0.000035. Kept only for diversity.

## Kaggle submissions used today
**0 / 10**

## Top active hypotheses
1. RealMLP (PyTabKit) on the `full` view with numeric+categorical twins — the field's best
   single model (0.96117) and architecturally orthogonal to GBDTs.
2. XGBoost / CatBoost on `full` for a decorrelated ensemble member.
3. Logit-space L2 logistic stack with nested cross-fitting (field: +0.00033).
4. 10-fold for finalists (field: +0.0001 per model).

## Immediate next experiments
- [ ] RealMLP `full` (6 epochs, n_ens=8, numeric twins) — primary folds
- [ ] XGB + CatBoost `full` — primary folds
- [ ] First submission (ensemble of the above) to establish the CV↔LB anchor
- [ ] Shadow-fold confirmation of `full` vs `core3`