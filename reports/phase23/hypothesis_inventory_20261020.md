# Phase 23 — hypothesis inventory (2026-10-19)

Every entry must name the evidence that would close it and the resource it needs.
Novelty is checked against `research/phase20_exhaustive/experiment_catalogue_96.json`
and `experiments/ledger.jsonl` before any run.

## Active

| id | hypothesis | resource | closing evidence |
|---|---|---|---|
| H23-01 | **Native raw8** (our folds/ckpt/geometry, `n_estimators=8`, full context) reproduces the external `tabpfn35` marginal against v7 (≈+3e-5 at w=0.10, 5/5 folds). | Kaggle T4 (~19 GPU-h), prereg `configs/phase23/raw8_kaggle_prereg.json` (sha `b9aec0ce…`) | `reports/phase23/raw8_comparison.json` gate (≥1.5e-5 mean, ≥2.5×SE, 5/5 folds, holm ≤.05, private80 p05≥0) |
| H23-02 | A **matched shadow-fold confirmation** of raw8 (seed 777001) would confirm any primary-passing candidate before submission. | Kaggle T4 (~19 GPU-h), separate prereg | shadow OOF + paired deltas vs v7; only run if H23-01 passes |
| H23-03 | **Logit-residual / v7-augmented stack specialist**: a CPU model trained with `logit(v7_oof)` as an input feature (cross-fitted) finds structure the linear logit blend cannot (e.g., regime-conditional corrections). | local CPU only | 5-fold OOF admission gate vs v7 + correlation ceiling vs existing members |

## Planned CPU Track-C candidates (to be novelty-checked before prereg)

1. H23-03 above.
2. Seed-varied **extratrees rich-view** members around the admitted family (cheap diversity; only if catalogue shows the exact seed/config pair absent).
3. **Calibration-conditional blend** (isotonic/Platt on the v7 logit in the near-0.5 band) — only if a pre-registered diagnostic first shows the band-specific AUC deficit is real.

## Closed (do not repeat)

| id | hypothesis | outcome |
|---|---|---|
| OBJ37–41 | five objective-augmentation candidates | NO_ADMISSION (phase 22) |
| — | row-13 TabPFN fixes | closed |
| FEAT48 | feature-augmentation block | closed |
| FLM12 / TabICL rich view | RESOURCE_BLOCKED (needs union of acceptance gates) |
| Lean84 / CatBoost CPU challengers | rejected at the gate (phase 22) |
| external diagnostics (24) | UNCERTIFIABLE (provenance); stay diagnostic-only |
| local GPU execution this phase | HARDWARE STOP + budget (15.678–24.043 h of 22 h) — do not run |

## Explicit non-goals

- No public-LB fitting anywhere (no submissions in this phase unless a separate committed decision authorizes one).
- No re-upload of the held 23 inventory candidates.
- No modification of submission guards or protected artefacts.
