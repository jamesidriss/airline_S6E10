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

The first 32-dimensional masked embedding increases an XGBoost fold-0
control from **0.960729290** to **0.960909436** (+0.000180146). Summary-only
gains +0.000030861; both gain +0.000099966. The embedding alone is frozen
for fold-1 confirmation. It still loses -0.000000608 when replacing the
existing auxiliary slot on v5, so no ensemble promotion is claimed.
The canonical XGBoost control is bit-identical to the independent runner.
SSL reads only the 21 raw covariates; masking Age/Distance hides both twins.
12 epochs take about 1.6 seconds each on the measured GPU, and reconstruction
validation loss falls from 0.75618 to 0.59076. No satisfaction target is loaded.

Live CLI confirms v5 public **0.96103**, ref **56918343**; public lead
**0.96184** at this checkpoint. One submission used on 7 October, nine
remain under Kaggle's cap; two remain under the default daily campaign limit.
No new submission has been spent by this continuation.

Active work: reconstruct the ten aux counterparts with corrected TE and
their own configurations; confirm the frozen SSL embedding; execute the
A0/A1/A2 distribution ladder once its upstream control reproduces.

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
