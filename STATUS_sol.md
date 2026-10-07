# SOL campaign — audit and continuation

Checkpoint: 7 October 2026, 20:54 UTC. **No clean finalist has yet been fully
reproduced. No new Kaggle submission was spent. Final slots remain unlocked.**
The campaign continues; this is not a final decision or research stopping point.

Banked v5: public **0.96103**, ref **56918343**. Legacy OOF reconstructs to
**0.9615234900843839** versus v3 **0.9615085784350211**. Mean paired fold gain
+0.00001535450060, SE0.0000037590797, t4.084646, 5/5 positive. That fixed-vector
gate passes, but the contracts below prevent an honest finalist claim.

## Audit findings

- All **12 neural members** (10 RealMLP, 2 TabM) selected checkpoints using
  outer evaluation labels. RealMLP disabled stopping but retained the library's
  default best-epoch restoration. Their CV claims are withdrawn. Corrected
  runners select on an inner 10 percent FIT partition, seed1.
- Numeric twins were factorized separately per split, potentially aliasing
  distances. Twins now preserve literal values, and `twin=False` works.
  New neural run IDs version the corrected protocol.
- Historical TE smoothing priors included each FIT row's own label. Maximal
  direct contribution was 1/559708=1.79e-6 before shrinkage. Both tables and
  priors now exclude own labels; historical vectors stay preserved.
- v5 test construction collapsed six XT configurations, mixed iteration
  records and used five auxiliary inner folds versus OOF's three. The corrected
  builder requires exact ten arms × five folds, seeds, tree counts and saved
  test vectors. Ten corrected counterparts exist for fold0. The remaining
  folds and 49 legacy slots still need reproduction.
- Fold arrays and ordered IDs are hash-bound without regeneration. Prediction
  reads verify hashes. The test failure counter and logit correlation were fixed.

Dropping the 12 neural vectors yields legacy classical47 AUC **0.9614916230444606**,
pooled loss **0.0000318670**. This is a diagnostic, not a clean alternative;
its historical TE priors remain affected. See `reports/sol_neural_audit.json`.

## Research evidence

| Branch | Frozen measurements | Decision |
|---|---|---|
| A1 residual/absolute residual | XGB deltas A0: f0 +0.000119153; f1 −0.000093787 | Failed replication; closed |
| A2 probability/surprisal | XGB deltas A0: f0 +0.000106176; f1 +0.000091126 | Frozen fold2 test pending; not admitted |
| A3/A4 uncertainty/aggregates | f0 −0.000116345/−0.000070019 standalone | No promotion |
| SSL32 embedding | XGB f0/1/2: +0.000180146/+0.000278289/−0.000093238; actual v5 slot negative3/3 | Closed standalone feature branch |
| SSL32 rich-view XT | B0 0.961174129; B1 0.961129753; delta −0.000044376; actual slot −0.000002372 | No promotion |
| SSL+aux A0 | f0 −0.000028034 standalone; slot −0.000001512 | No promotion |
| Surprise diagnostic | High-surprise pair error rate0.04723 vs0.03771 outside; modest within-margin enrichment | No specialist justified |
| Full-context TabPFN3.5 | No completed CV; raw FIT stopped at1530.8s on RAM guard | Resource-invalid, not negative |
| Query-chunk TabPFN probe |100k FIT33.844s;1024 predictions0.485s; peak3,062,673,920B | Probe passes; full run awaits reserve |

Attention reference gaps: FP16 **0.00048828125**, BF16 **0.0078125**, within
predeclared tolerances. Query chunks retain every key. GPU allocation is capped
at85 percent to avoid Windows shared-RAM spill. Further pre-fit RAM refusals
produced no scores and are resource evidence only.

Conditional multitask protocol: `research/sol_multitask_protocol.md` freezes
lambda0/0.05/0.2, matched architecture, SSL initialization and inner ES. Execute
only if A2 replicates on fold2. No multitask CV has been run.

Private simulation: **820** deterministic splits at299844-row population.
Random private v3−v5 median−0.00001514974, P05−0.00002399429; public/private
sign disagreement7.5%. This diagnoses legacy vectors; it does not repair their
contracts or forecast ranks. A clean candidate needs its own simulation.

## Resources and verification

Active: CPU-only fold2 auxiliary cache. It now reads only the21 raw covariates;
fold0/1 reproduce the exact original fingerprints in6 seconds. Larger jobs are
serialized. Authenticated Kaggle GPU quota is exhausted:61503s used versus
21600s allowed, resetting10 October. No remote GPU job was started.

Seven hash-verified duplicate static caches freed **6,488,621,126B**. Verification
began rebuilding them; that run was stopped and is not a pass. Views with identical
ordered static blocks now share canonical caches; dynamic features stay per-fold.
Array hashing streams identical bytes without a large temporary allocation;
compatibility tests preserve historical digests.

Last complete pre-fix test run: **70 passed, 0 failed**. New targeted neural,
generic GBDT, attention, normalization, alias and hash checks pass;30 critical
files have0 undefined names. Complete post-change verification remains required.

```powershell
.\.venv\Scripts\python.exe scripts\audit_sol_state.py
.\.venv\Scripts\python.exe scripts\audit_sol_neural.py
.\.venv\Scripts\python.exe scripts\cache_sol_aux.py --folds 2
.\.venv\Scripts\python.exe scripts\evaluate_sol_aux.py --folds 2 --variants A2
.\.venv\Scripts\python.exe scripts\run_sol_tabpfn.py --folds 0 --arms raw,route --batch-size 1024 --windows-mqa --memory-saving on --icl-bf16 --chunk-cells 262144 --col-chunk 1 --host-reserve-gib 4 --tag sol_tabpfn35_query_serial
.\.venv\Scripts\python.exe tests\run_tests.py
```

BEST_A is banked legacy v5 with explicit CV/test limitations. Verified clean
BEST_A and BEST_B_HEDGE are absent. v4 remains a legacy fallback with v3 OOF
proxy, not an independently measured hedge. A final decision report is premature
while promoted research and clean reproduction remain.
