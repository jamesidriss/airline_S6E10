# SOL campaign — independent audit and continuation

Checkpoint: 7 October 2026. Preserve all existing finalist CSVs and prediction vectors.

The independent audit reproduces v3 OOF **0.9615085784350211** and the v5
OOF candidate **0.9615234900843839**. Pooled gain **+0.00001491164936**;
mean paired fold gain **+0.00001535450060**, five positive folds, paired
t **4.084646**. The admission contract uses the fold mean, so the OOF
gate passes. The earlier gate failure/t=3.97 used pooled gain as numerator.

**The submitted v5 test file is not a verified counterpart of that OOF
candidate.** Its builder collapsed six intended extra-trees configurations
to seed 1/d127 defaults, pooled iteration records across controls/treatments,
and used five inner auxiliary folds against OOF's three. Fold-0 iteration
records are absent. No per-slot test vectors were saved. Its CSV, ordered
IDs and SHA-256 reproduce; its test-model methodology needs repair.

The target encoder's smoothing prior also included each fit row's own label.
The maximum direct prior contribution on primary fit rows is 1/559708
(about 1.79e-6 before shrinkage). No outer-validation label entered the
encoder, but the strict training-row self-exclusion contract failed.
The prior is now estimated separately inside each inner training fold;
a label-flip regression test catches this defect, including unseen keys.
Historical predictions are retained as legacy evidence, not silently relabeled
as clean reproductions.

The fold registry now binds array bytes and ordered IDs without regenerating
any assignment. Prediction reads check their stored hashes. The test runner
now preserves subprocess failures in its exit status. Phase 14C's correlation
no longer clips mean logits as though they were probabilities. v3 test
models are fold averages; v4 refits 32 members on all labels. The older claim
that every v3 test member used all labels is withdrawn.

The simulator ran **820** deterministic pseudo splits at a 299844-row
population: 200 random stratified, 200 fold-aware, and 420 segment-stressed.
Random splits give v3-minus-v5 private median **-0.00001514974** and
P05 **-0.00002399429**; public/private signs disagree **7.5%** of the time.
All 200 random private splits favor the v5 OOF vector. These diagnose fixed
OOF vectors and do not forecast competition ranks or repair v5 inference.
v4 is explicitly a v3 OOF proxy. Split definitions, row hashes, code hashes,
seeds and individual draws are saved; masks can be regenerated exactly.

The frozen 32-dimensional masked embedding has XGBoost standalone deltas
**+0.000180146, +0.000278289, -0.000093238** on primary folds 0/1/2;
mean **+0.000121732**, two positive folds. Actual legacy-v5 slot deltas
are **-0.000000608, -0.000004433, -0.000003414**. Thus the representation
has some standalone signal but no demonstrated ensemble admission. Adding
SSL to the auxiliary A0 baseline on fold0 loses **0.000028034** standalone
and **0.000001512** against the actual legacy-v5 slot. No blind full-five-fold
promotion. One frozen rich-view extra-trees control/treatment remains to test.
The canonical XGBoost control is bit-identical to the independent runner.
SSL reads only the 21 raw covariates; masking Age/Distance hides both twins.
12 epochs take about 1.6 seconds each on the measured GPU, and reconstruction
validation loss falls from 0.75618 to 0.59076. No satisfaction target is loaded.

Live CLI confirms v5 public **0.96103**, ref **56918343**; public lead
**0.96184** at this checkpoint. One submission used on 7 October, nine
remain under Kaggle's cap; two remain under the default daily campaign limit.
No new submission has been spent by this continuation.

The auxiliary-distribution ladder's independent canonical control is bit
identical to the repaired X2 A0, AUC **0.960944790** on fold0. Its independently
reconstructed expectations match the historical upstream cache within 1e-10.

| Treatment | Standalone AUC | Delta A0 | Actual legacy-v5 slot delta |
|---|---:|---:|---:|
| A1 residual + absolute residual | 0.961063943 | +0.000119153 | +0.000004925 |
| A2 + observed probability/surprisal | 0.961050966 | +0.000106176 | +0.000004473 |
| A3 + uncertainty/tails | 0.960828445 | -0.000116345 | -0.000000073 |
| A4 + aggregates | 0.960874771 | -0.000070019 | +0.000000657 |

A1 is frozen for fold1 confirmation; A3/A4 are not promoted. These are fold0
discovery numbers, not full OOF or submission evidence. The new evaluator
replaces the actual historical auxiliary slot. The original repair runner's
`operational_v5_slot_delta` compares A0 to itself and is withdrawn as an
operational metric; `record_sol_research.py` recomputes the real comparison.

All ten repair counterparts completed fold0. Their splice into legacy v3 has
AUC **0.9615119057**, +0.0000020641 vs legacy v5 on that fold. A structurally
fixed blend of only the ten clean counterparts scores **0.9614142040** on
fold0; it is a diagnostic, not an admitted finalist. Repaired slots use the
fixed strict TE prior. The other 49 legacy slots still need strict reproduction
before the whole v5 recipe can be called a clean finalist.

Independent research selected own TabPFN 3.5 full-context raw/route inference
as the highest-EV remaining mechanism. Public prediction stacks were rejected;
the source's tree/stack CV uses outer-eval stopping and insufficient nesting.
Rules/license/checkpoint provenance and a ranked five-entry queue are saved.
No imported competition predictions are used.

Windows torch lacks compiled FlashAttention. The first 1024-row prediction
probe stalled, and a forced-bfloat16 probe hit a library preprocessing TypeError;
both are INVALID infrastructure evidence, not model negatives. A supported
memory-efficient MQA backend passes an independent numerical reference (maximum
absolute gap **0.00048828125**). With it, 100,000 context rows fit in **28.219 s**,
and 1024 prediction rows take **0.531 s**, peak **9,324,301,312 bytes**. Full-context
auto memory mode spilled into shared RAM, so the current serious raw/route
comparison uses explicit memory saving. No TabPFN CV number exists yet.

Active work: one full-context TabPFN GPU job; one CPU-only auxiliary cache worker
for folds1/2. Ten-slot repair is checkpointed after fold0 for GPU serialization.
No additional submission has been spent, and no final slot is locked.

Current roles: BEST_A is v5 provisionally, pending repaired inference.
BEST_B_HEDGE is none. v4 is a fallback, with no distinct measured OOF.
The final two are not locked.

Evidence: `reports/sol_s0_audit.json`, `reports/sol_private_sim.json`,
`reports/sol_ssl12_training.json`, `reports/sol_b/`, `reports/sol_repair/`,
`experiments/private_finalists.json`. Reproduction begins with:

```powershell
.\.venv\Scripts\python.exe scripts\audit_sol_state.py
.\.venv\Scripts\python.exe scripts\private_lb_simulator.py
.\.venv\Scripts\python.exe scripts\run_sol_a.py --repair --folds 0,1,2,3,4 --tag sol_repair
.\.venv\Scripts\python.exe scripts\build_v5_aux_cross.py --dry-run
.\.venv\Scripts\python.exe scripts\train_sol_ssl.py --timing
.\.venv\Scripts\python.exe scripts\train_sol_ssl.py --epochs 12 --tag sol_ssl12
```
