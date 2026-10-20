# Phase 23 — hypothesis inventory (2026-10-19)

Every entry must name the evidence that would close it and the resource it needs.
Novelty is checked against `research/phase20_exhaustive/experiment_catalogue_96.json`
and `experiments/ledger.jsonl` before any run.

## Status after the probe (2026-10-20)

| id | status | evidence |
|---|---|---|
| H23-01 | **RESOURCE_BLOCKED — not executed** (hypothesis open, not refuted) | probe v2 OOM: full-context int8 KV ≈12.4 GiB + 2.14 GiB build activation > 14.56 GiB T4; local device stopped. `reports/phase23/raw8_probe_resource_report.md` |
| H23-02 | **NOT ACTIVATED** | no primary candidate; the shadow design is ready (prereg `05b7235e…`, ≈7.4 GPU-h) should a ≥16 GB accelerator appear |
| H23-04 | **CLOSED NEGATIVE** (see below) | ledger `phase23_20261020_id_generator_batch_structure` |

## Active

| id | hypothesis | resource | closing evidence |
|---|---|---|---|
| H23-01 | **Native raw8** (our folds/ckpt/geometry, `n_estimators=8`, full context) reproduces the external `tabpfn35` marginal against v7 (≈+3e-5 at w=0.10, 5/5 folds). | needs a **≥16 GB accelerator** (T4 14.56 GiB is 0.3–1 GiB short); prereg `configs/phase23/raw8_kaggle_prereg.json` (sha `b9aec0ce…`) | `reports/phase23/raw8_comparison.json` gate (≥1.5e-5 mean, ≥2.5×SE, 5/5 folds, holm ≤.05, private80 p05≥0) |
| H23-02 | A **matched shadow-fold confirmation** of raw8 (seed 777001, folds 2–3) would confirm any primary-passing candidate before submission. | Kaggle **≈7.4 GPU-h** (route + aux10 arms are the certified phase-17 shadow artifacts), prereg `configs/phase23/raw8_shadow_prereg.json` (sha `05b7235e…`) | `reports/phase23/raw8_shadow_comparison.json`: E2 ≥ 1e-5, ≥ 2.5×SE, both folds > 0 |

## Closed at the novelty check (no run)

| id | hypothesis | closure |
|---|---|---|
| H23-03 | Logit-residual / v7-augmented stack specialist (train on `logit(v7_oof)` + covariates, cross-fitted). | Closed without running: the phase-20 catalogue already covers residual structure and its surrounding analyses — #70/#72 (blend gating / hard-pair), #73/#74 (near-tie and residual-structure splits), #75/#79 (nested stacks), #51 (residual specialist) — all NO_ADMISSION. A new run would repeat a failed experiment with no new hypothesis. |
| H23-04 | **Id-order / generator-batch structure**: the synthetic generator left exploitable target structure along contiguous id ranges, reachable by a fold-safe block target encoding under generator-batch-aware splits. | **Closed negative (run, with numbers).** `AUC(id) = 0.500073`; positive-rate homogeneity across 10/50/100/400 contiguous id blocks: χ² p = 0.975/0.981/0.999/1.000; `sd(block rate)/binomial sd` = 0.70/1.05/1.05/1.00; fold-safe shrunk block TE alone scores AUC 0.5002–0.5006. Dispersion is pure binomial noise; nothing to exploit. Ledger `phase23_20261020_id_generator_batch_structure`. |

## Planned CPU Track-C candidates (to be novelty-checked before prereg)

1. Seed-varied **extratrees rich-view** members around the admitted family (cheap diversity; only if the catalogue shows the exact seed/config pair absent).
2. **Calibration-conditional blend** (isotonic/Platt on the v7 logit in the near-0.5 band) — only if a pre-registered diagnostic first shows the band-specific AUC deficit is real.

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
