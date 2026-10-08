# SOL campaign — audit and continuation

Completed recovery checkpoint: 2026-10-08T11:51:37.015827+00:00. **Certified Champion A: v6_sol_tabpfn_route_aux10**, exact Kaggle ref56951906, COMPLETE, public AUC0.96144. Team rank234/1145 and9 submissions remaining at11:39 UTC. Full primary OOF0.9617656138245795; 5/5 paired gains positive, pooled gain0.00034690428011585617 vs clean auxiliary10; both predeclared shadow gains positive; seed1202 diagnostic complete. All299844 test rows certified from5 frozen route FIT contexts and10 exact auxiliary refits. One new submission used; no retuning or second upload. No credible clean Champion B found. Full shadow and block10 were not run; the small policy audit remains inconclusive. Current full suite114 passed,0 failed, with unchanged source since that run. Original banks, data and checkpoint hashes preserved. No active campaign model worker. Scientific campaign work is complete; closing Git sync is recorded in the final response. Detailed final evidence: reports/sol_final_recovery_report_20261008.md.

## Historical checkpoints

Recovery checkpoint: 2026-10-08T11:32:43.905686+00:00. Full test inference is certified for all299844 ordered test IDs. Five frozen559708-row primary FIT route contexts, seed1201, are equally averaged in probability space. Ten exact699635-row auxiliary refits use distinct frozen configurations and five-fold median tree counts. The final method blend remains50/50 logits. Candidate prediction array SHA b9ccd997acca2e5c05946e7581ec8d81ea21ab198eadfd99e535f86f651b7c7f. Private simulation and scientific decision are pending; no new submission has been spent.

Recovery checkpoint: 2026-10-08T09:07:56.234291+00:00. Independent confirmation is complete to the predeclared feasible scope: two positive shadow folds and the fixed primary fold0 seed1202 diagnostic. Seed1202 AUC0.9611372496955853 versus original1201 AUC0.9611324459037415 (difference0.0000048037918438). The finalist remains seed1201. Test queue started09:07 UTC, beginning with the bounded leakage-safe policy audit. Complete test certification and private diagnostics remain outstanding; no new submission has been spent.

Recovery checkpoint: 2026-10-08T08:54:10.191869+00:00. The frozen portfolio passes both predeclared shadow folds. Candidate AUCs 0.9612052540431654 and 0.9612181434319019; paired gains over strict auxiliary10 0.0002543102301415523 and 0.0003019852400100165. Mean gain 0.0002781477350757844, paired SE 0.00002383750493423209. All20 fresh auxiliary shadow fits and both route contexts completed. This is partial shadow confirmation only; block10 was not run. Seed1202 is in progress. Actual test inference, private simulation and submission are outstanding. No new submission has been spent.

Checkpoint: 8 October 2026, 07:17 UTC. **No clean finalist has yet been fully
reproduced. No new Kaggle submission was spent. Final slots remain unlocked.**
The campaign continues; this is not a final decision or research stopping point.

Recovery reused all33 verified vectors for primary folds0/1/2. Route fold3
completed under the unchanged frozen recipe: AUC0.9606084962415924,
559708 FIT rows,139927 validation rows and prediction SHA
09403f0cdedb6da75772407093b45c35221d992a9f34dbe8494dae6b04a4527c.
The earlier fold3 disk refusal remains resource-invalid evidence, without a
prediction or AUC. Route fold4 completed at AUC0.9618192023035632; all five
route vectors pass the authoritative source/data/checkpoint/ID/hash/AUC checks.
Auxiliary rating caches for folds3/4 are complete in735.641s and895.156s.
All ten frozen auxiliary roles now have complete verified five-fold OOF.
Thirty existing fold0/1/2 results were reused and twenty missing results fitted.
`reports/sol_route_aux10_primary_final.json` certifies all699635 ordered rows:
frozen candidate pooled AUC0.9617656138245795; clean auxiliary10
0.9614187095444636; route0.9612196007698864. All five paired gains are positive,
mean0.0003469980344849022, SE0.000025277324554686046. The frozen primary
admission gate passes. Independent confirmation, certified test inference and
private diagnostics remain required before submission.

The predeclared recovery scope is two shadow folds0/1 and one fixed primary
fold0 seed1202 replay; block10 is deferred. These checks cannot establish full
shadow OOF. Primary admission, complete test inference, robustness and an
evidence-bound decision remain required before submission.

The initial recovery suite returned110 passed,0 failed. A later source review
found that its static guard had linted pathname strings rather than file
contents, missing the scorecard's get_scheme import. The import and guard are
fixed. The corrected64-entry source check and missing-import fixture pass2/2;
earlier zero-undefined-name claims are superseded. The current full suite
completed at07:22 UTC: **114 passed,0 failed**, including the four new recovery
checks, actual-data view checks, leakage isolation, source tracking and six
mocked submission-I/O checks. No real submission was spent by the tests.

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
| Matched TabPFN3.5 route | Honest f0/f1/f2 AUC 0.961132446/0.961379166/0.961199565; f0 +0.000160150 versus raw; 1/60 legacy append gains +0.000020159/+0.000023193/+0.000023564 | Three route and auxiliary folds complete; no full-OOF admission |
| Frozen 50/50 route/strict auxiliary10 | f0/f1/f2 AUC 0.961727180455/0.961935424840/0.961787550190; paired gains +0.000312977866/+0.000329353076/+0.000345237952 versus clean auxiliary10; mean +0.000329189631, SE 0.000009313043, 3/3 positive | Passes declared promotion to complete primary five folds; independent confirmation and test still required; no finalist claim |
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

All thirty frozen auxiliary role/fold jobs are complete under
`sol_clean_aux10_threefold`. Every one of the ten fold0 controls has identical
predictions, FIT/validation matrix hashes and selected tree count versus strict
SOL-A A0. Their verified fixed portfolio is in `reports/sol_route_aux10_threefold.json`.
The interrupted route fold3 completed FIT but failed prediction's20 GiB disk
preflight at777.203 seconds; no OOF vector exists for that attempt. Recovery
found no surviving model worker, HEAD671826b matching the live GitHub branch,
43.4 GiB free disk and8.4 GiB available RAM. All33 completed vectors are
revalidated in `reports/sol_recovery_inventory_20261008.json`.
Active: fresh route retry3/4 under `sol_tabpfn35_route_recovery`, separate raw-only auxiliary caches3/4,
then isolated auxiliary completion under `sol_clean_aux10_primary` and complete
five-fold evaluation. Existing thirty completed role/fold records are verified
before reuse. Fold2 native C1 AUC0.961269750529,1812 trees,875.875 seconds.
One fresh process per fold avoids native allocator retention.

The critical recovery handoff supersedes the previous open-ended campaign and
revokes the supplied AGENTS.md agreement. The new finite scope is declared in
`research/sol_recovery_scope_20261008.json`: complete primary as
`sol_route_aux10_primary_final`, then preselected shadow folds0/1 and a fixed
seed1202 replay, five-context test inference, private diagnostics and submission
if qualified. Block10 is deferred as disproportionate extra compute and resource
risk; no block10 claim will be made. Partial shadow must remain explicitly
partial and show positive gains on both specified folds. No model or weight
has changed. No39-cross or clean59 replay is scheduled.

Recovery's full current suite: **110 passed,0 failed**, exit0. Separate new
partial-scope and pseudo-test partition checks also pass; current static guard
covers62 files with0 undefined names. A small100k-pool/20k-shadow-holdout policy
audit is prepared, not fitted, to compare single-context inference with five
80-percent contexts. It is not a full OOF estimate or a deployment test score.
The source-tracking check passes after recovery commit2a283d5. Secret scan has
0 high/medium findings. Live05:35 UTC Kaggle snapshot: rank336/1106, public
leader0.96185,10 submissions available; banked v5 remains0.96103.

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
The fresh allocator-policy100k-context gate also passes with max probability
gap0 and peak2663518720 bytes. Its all699635-row probe still exhausted GPU
memory at113.062 seconds, before any prediction. The declared five-context
probability-average test fallback is now the inference policy; its toy
held-label flip check passes, but no fallback test model has yet been fitted.

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

Submission I/O now reserves daily budget before uploading, preserves numeric
Kaggle references and polls only the exact recorded submission. Six isolated
mocked-network checks pass, including old completed submissions, uncertain
upload responses, local and remote caps, and filename/ref disagreement. These
checks spend no submission. Actual public transfer is still unmeasured for the
new portfolio.

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
