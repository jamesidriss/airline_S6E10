# SOL campaign — audit and continuation

Checkpoint: 8 October 2026, 01:26 UTC. **No clean finalist has yet been fully
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
| A2 probability/surprisal | XGB deltas A0: f0 +0.000106176; f1 +0.000091126; f2 −0.000201215 | Failed three-fold promotion; closed |
| A3/A4 uncertainty/aggregates | f0 −0.000116345/−0.000070019 standalone | No promotion |
| SSL32 embedding | XGB f0/1/2: +0.000180146/+0.000278289/−0.000093238; actual v5 slot negative3/3 | Closed standalone feature branch |
| SSL32 rich-view XT | B0 0.961174129; B1 0.961129753; delta −0.000044376; actual slot −0.000002372 | No promotion |
| SSL+aux A0 | f0 −0.000028034 standalone; slot −0.000001512 | No promotion |
| Surprise diagnostic | High-surprise pair error rate0.04723 vs0.03771 outside; modest within-margin enrichment | No specialist justified |
| Full-context TabPFN3.5 raw | Honest f0 AUC 0.960972296; fixed 1/60 append gain +0.000021134; logit correlation 0.994532205; 1030.312 s | Positive discovery only; legacy comparison, no admission |
| Matched TabPFN3.5 route | Honest f0/f1/f2 AUC 0.961132446/0.961379166/0.961199565; f0 +0.000160150 versus raw; 1/60 legacy append gains +0.000020159/+0.000023193/+0.000023564 | Three route folds complete; fixed blend has only two complete auxiliary folds; no full-OOF admission |
| Frozen 50/50 route/strict auxiliary10 | f0/f1 AUC 0.961727180455/0.961935424840; paired gains +0.000312977866/+0.000329353076 versus clean auxiliary10; mean +0.000321165471, SE 0.000008187605, 2/2 positive | Replicated; complete primary and independent confirmation; no finalist claim |
| Corrected RealMLP recipe | f0 AUC 0.960758172; legacy slot diagnostic −0.000002531; 70.328 s; OOF/test vectors saved | Contract repair only; no improvement/admission claim |
| Query and activation buffer reuse | Complete-model 100k-context/1024-prediction probes are bit-identical to the reference; raw and route full-FIT now complete | Frozen for primary replication |
| Decoder linear-projection batching | Max probability gap 0.000226825 vs tolerance 0.000002 | Failed equivalence gate; reject implementation |

Attention reference gaps: FP16 **0.00048828125**, BF16 **0.0078125**, within
predeclared tolerances. Query chunks retain every key. GPU allocation is capped
at85 percent to avoid Windows shared-RAM spill. Further pre-fit RAM refusals
produced no scores and are resource evidence only.

**A2 is closed:** fold2 standalone delta −0.000201215; three-fold mean
−0.000001304, 2/3 positive. Actual legacy-slot mean +0.000002681, paired SE
0.000001624, 2/3 positive. No XT/Cat or multitask expansion is justified.

Conditional multitask protocol: `research/sol_multitask_protocol.md` freezes
lambda0/0.05/0.2, matched architecture, SSL initialization and inner ES. Execute
only if A2 replicates on fold2; that gate failed. No multitask CV has been run.

Private simulation: **820** deterministic splits at299844-row population.
Random private v3−v5 median−0.00001514974, P05−0.00002399429; public/private
sign disagreement7.5%. This diagnoses legacy vectors; it does not repair their
contracts or forecast ranks. A clean candidate needs its own simulation.

## Resources and verification

All twenty frozen auxiliary role/fold jobs are complete under
`sol_clean_aux10_isolated`. Every one of the ten fold0 controls has identical
predictions, FIT/validation matrix hashes and selected tree count versus strict
SOL-A A0. Their verified fixed portfolio is in `reports/sol_route_aux10_twofold.json`.
Active: `replay_sol_isolated.py`, tag `sol_clean_aux10_threefold`, folds0/1/2.
It verifies and reuses the twenty completed folds0/1 records, and fits the ten
remaining fold2 roles in separate processes. Five new XT fits are in progress
or complete; native CatBoost and three XGB roles remain. Interpret the fixed
three-fold portfolio only after all ten complete, before promotion to five folds.
One fresh process per fold avoids native allocator retention.

The all699635-row context resource probe failed before predictions in14.375s.
Fresh scaling-output reuse is bit-identical on10,247,168 BF16 values, leaves
the input query unchanged, and preserves CPU outputs and gradients. Both the
unmodified100k-context control and complete-model reuse treatment reproduce
the reference probability SHA exactly, max gap0, each27.812s total.
The full-context reuse retry also failed before predictions while explicitly
repeating attention values. Both attempts are resource-invalid, not negative
performance evidence. The predeclared full-test policy and probability-average
of five immutable primary-context fallback remain in
`research/sol_test_inference_contract.md`. No fallback has yet been executed.
Route fold1 completed in1027.359 s, with13675054592 B peak GPU allocation.
Its prediction SHA256 is83ca4330f955aef150dab5f45ebd2f1d432109659842e9fb371e88d6d1c0771a.
The first fold1 FIT succeeded in787.9 s but prediction was refused because the
Windows pagefile expansion left less than20 GiB free disk. No fold1 CV vector or
performance score was produced. This attempt remains preserved as resource-invalid.
Raw f0 is complete under `sol_tabpfn35_predict_guard`, with all 559708 FIT and
139927 evaluation rows. Peak GPU allocation was 13671814656 B (12.733 GiB).
The first completed full FIT was previously discarded by an overstrict prediction
RAM preflight. Prediction now requires 2 GiB before allocating its small batch;
the periodic abort threshold remains 1 GiB. Pre-fit reserve remains 4 GiB.
The route attempt in the raw process was refused before fitting; the completed
fresh-process run preserves that failed report. Both complete arms retain the
same seed/context/inference settings and differ only in route representation.
All three auxiliary probability caches are complete. Their builder reads only the21 raw covariates;
fold0/1 reproduce the exact original fingerprints in6 seconds. Larger jobs are
serialized. Authenticated Kaggle GPU quota is exhausted:61503s used versus
21600s allowed, resetting10 October. No remote GPU job was started.

Seven hash-verified duplicate static caches freed **6,488,621,126B**. Verification
began rebuilding them; that run was stopped and is not a pass. Views with identical
ordered static blocks now share canonical caches; dynamic features stay per-fold.
Array hashing streams identical bytes without a large temporary allocation;
compatibility tests preserve historical digests.

Additional exact aliases/unused, regenerable static caches freed 3,889,973,750 B
and 4,133,846,208 B, with streamed payload hashes and guarded workspace paths.
All used canonical views, raw/original data, folds, predictions, weights and
submissions are preserved. Cleanup manifests record regeneration commands.

Lossless filesystem compression now preserves every byte of five static caches.
NTFS compression saved874403714 B; stronger LZX compression subsequently saved
2986467556 B in allocated storage. Before/after file SHA256 values agree for
every processed file. Free disk recovered to about40.5 GiB. No file was deleted;
compression reports include exact reversal commands.

Additional lossless LZX compression of13 named dependency binaries recovered
1626813264 B. Every before/after file SHA256 agrees; package bytes, version and
model behavior are unchanged. The reports record exact reversal commands.
Current disk reserve is about20.4 GiB during the active route fold2.

The all-row failure included unnecessary MHA key/value copies even with16
matching heads. A separate opt-in backend now uses permutation/expansion views
without changing logical SDPA dimensions. CPU MHA/MQA, query alias protection
and key/value preservation checks pass. Whole-model GPU equivalence now passes;
this implementation has not produced scored CV or final test predictions.
The primary runner and original attention/activation helper bytes remain intact.

The complete-model100k head-view gate is bit-identical, max probability gap0;
peak GPU allocation falls2857153024→2663518720 B. Four FP16/BF16 MHA/MQA
GPU primitive comparisons are also bit-identical and preserve keys/values.
Full699635-row head-view inference failed in63.218s at a query-scaling GELU.
Reusing that fresh activation passes the complete-model100k gate again, gap0,
27.797s and2663518720 B peak, but all-row inference still failed in112.875s
at a full-size linear output. All failed probes produced no predictions or AUC.
One allocator-policy gate/probe remains prepared, not executed. If it cannot
make full-context inference feasible, the frozen five-context fallback runner
is ready. Its toy held-label flip check passes; no fallback model has been fitted.

The lower-priority native39 probe is implemented and protocol-frozen, not fitted.
It adds exact label-free rating/context categorical tuples to the unchanged
strict C1 control and changes only the existing0.05-weight slot. Literal tuple
identity, split-vocabulary consistency and target/ID independence checks pass.
Recent focused checks cover58 critical files with0 undefined names, full-context
default paths, confirmation/seed/fallback label exclusion and bootstrap ties.

New test certification and robustness scorecard runners are prepared. They
require complete primary admission, all ten median-capacity test refits, ordered
IDs, vector/source hashes, full shadow and block10 confirmation, paired bootstrap, private
simulation and one fixed seed1202 sensitivity replay. Seed1202 cannot replace
the frozen seed1201 portfolio. No test certificate, confirmation scorecard or
new submission has yet been produced.

Generic test refits also drifted from CV defaults: LightGBM changed leaf-size
and regularization defaults and omitted explicit bagging/feature seeds; CatBoost
changed the default learning rate. XGB/Cat indices needed +1 when passed as tree
counts. Helpers now preserve the CV recipe, including native CatBoost's first
selected tree. Regression checks cover parameter
identity and the first-tree edge case. Banked test predictions stay untouched.

The frozen classical replay runner and neural/portfolio certification scripts
are implemented. They require complete row-aligned, hash-bound predictions and
refuse partial or legacy members. They do not filter roles by new scores. The
full 47-classical/12-neural replay and ensemble scorecard remain outstanding.
The new frozen route/auxiliary10 combination now takes priority for honest
replication and finalist reproduction. A legacy candidate remains ineligible
until completely repaired; no clean 59-role claim is made.

Checkpoint4 full suite: **85 passed, 0 failed**. Checkpoint5: **89 passed, 1 failed**;
the sole failure was an unstaged new source file. After staging, the failed check
and two current-state/added checks passed (3/3). The failed full-suite log is
retained in `reports/sol_test_checkpoint5.json`. Secret scan found0 high/medium
findings. All nine actual-data view compositions and activation-reuse checks pass.
The complete-model GPU activation probe has max probability
gap **0**, retaining SHA a64dfea84287aad3f6c1675916f5f7e252c055d4eb45e22d79b265a930c3d543.

Three current focused checks pass: ambiguous completed foundation vectors are
rejected before scoring; flipping shadow evaluation labels leaves all FIT
inputs and predictions unchanged;43 critical files contain0 undefined names.
The separate frozen confirmation runner preserves the primary source bytes and
supports immutable shadow/block10 folds. No confirmation model has been fitted.
Direct public-topic refresh and rejected outer-ES/public-stack claims are in
`research/sol_public_refresh_20261008.md`. The39-cross mechanism replaces the
unsupported ModernNCA rerun in the five-entry queue; active weights stay fixed.

```powershell
.\.venv\Scripts\python.exe scripts\audit_sol_state.py
.\.venv\Scripts\python.exe scripts\audit_sol_neural.py
.\.venv\Scripts\python.exe scripts\replay_sol_neural.py --members z3_rm_bs256_e6 --folds 0 --save-test --tag sol_neural_clean_compact
.\.venv\Scripts\python.exe scripts\replay_sol_isolated.py --folds 0,1 --tag sol_clean_aux10_isolated
.\.venv\Scripts\python.exe scripts\evaluate_sol_foundation.py --folds 0,1 --classical-tag sol_clean_aux10_isolated --name sol_route_aux10_twofold
.\.venv\Scripts\python.exe scripts\run_sol_tabpfn_test.py --probe-only --tag sol_tabpfn35_full_context_probe
.\.venv\Scripts\python.exe tests\run_tests.py
```

BEST_A is banked legacy v5 with explicit CV/test limitations. Verified clean
BEST_A and BEST_B_HEDGE are absent. v4 remains a legacy fallback with v3 OOF
proxy, not an independently measured hedge. A final decision report is premature
while promoted research and clean reproduction remain.
