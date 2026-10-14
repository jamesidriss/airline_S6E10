# H19-A -- fold-0 falsifier: difficulty-targeted reweighted LightGBM

- **verdict:** `INCONCLUSIVE`
- start SHA `b5b83a4a0533` -> run SHA `1eef4a0e381b`
- prereg `configs/phase19/h19a_prereg.json` sha256 `d02a8b74be49423a...` committed **before** any result
- eval fold 0 only; FIT folds 1-4; scheme `primary`; GPU jobs **0**; submissions **0**

## Anti-leakage

- Atlas diagnostics for folds 1-4 are **fold-0-contaminated** (set proof: classical(k) trains on folds != k which contains fold 0; numeric demo: fold-1 predictions pearson 0.998804 when fold 0 is in/out of training).
- NOT used as training weights: `per_row_burden.parquet`, `pred_a`, `pred_b`, stored route/aux.
- difficulty rebuilt **inside FIT only** (inner cross-fit, never fold 0).
- judge A/B pearson: 0.998605 (two fold-0-safe CPU LGBM judges).

> **blocking diagnostic (faithful target):** FAITHFUL TARGET BLOCKED: the preregistered difficulty target is the champion SHARED difficulty (route=TabPFN AND aux both fail). The champion route is TabPFN (GPU). Rebuilding it fold-0-safely for folds 1-4 requires 4 TabPFN trainings inside FIT = heavy GPU, explicitly out of scope for H19-A (CPU-only, no heavy GPU). The champion aux for folds 1-4 is itself fold-0-contaminated (trains on folds != k). Therefore the faithful shared difficulty CANNOT be recalculated strictly safely within the H19-A budget. This falsifier proceeds with an explicitly documented fold-0-safe CPU PROXY shared difficulty (two inner-cross-fitted LightGBM judges) -- see prereg documented_deviations DEV-1.

## Standalone (fold 0, exact AUC)

| member | AUC |
|---|---|
| control (uniform) | 0.9611731952 |
| treatment (difficulty) | 0.9611618970 |
| **delta** | **-1.130e-05** |
| champion route | 0.9614576551 |
| champion portfolio | 0.9618278143 |

## Shared-fail pair mechanism (frozen sampling)

- sampled 50,000,000 pairs, seed 20261019; shared rate 0.0324 +- 0.0000
- exact shared-fail pairs (fold 0): 156,922,098 (strict 156,921,389)
- on shared-fail pairs: treatment acc 0.0446+-0.0002 vs control acc 0.0406+-0.0002
- corrected 40,742 / degraded 34,344 / net rate 3.944e-03+-1.689e-04
- elsewhere (non-shared): net rate -1.355e-04+-9.536e-06 (new errors introduced)

## Correlations with champion halves

| member | pearson route | pearson aux |
|---|---|---|
| control | 0.997411 | 0.999624 |
| treatment | 0.997465 | 0.999618 |

## Frozen portfolio (equal-weight 11th aux member)

- base (aux10) 0.9618278143; +control 0.9618269891; +treatment 0.9618278564
- **delta (treat vs control): 8.672e-07**

## Decision (frozen rules)

- d = -1.130e-05 (bar 3e-05); s_rate = 3.944e-03 +- 1.689e-04 (sig=True); portfolio delta = 8.672e-07
- **INCONCLUSIVE** -- DEAD: d<3e-5 AND s_rate<=0 | PATH: d>=3e-5 AND s>=2SE AND port>=0 | else INCONCLUSIVE

## Recommendation

- no folds 1-4 replication recommended
- budget: 298s CPU, 0 GPU jobs, 0 submissions.

> fold 0 alone cannot validate the full promotion gates (>=1.5e-5 mean paired, >=2.5x paired SE, >=4/5 folds).
