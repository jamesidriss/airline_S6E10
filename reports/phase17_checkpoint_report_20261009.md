# Phase 17 checkpoint — validated primary gain, qualification blocked by disk capacity

Checkpoint: 2026-10-09, 14:27 UTC. Evidence commit before this report: `a0f12722e00c13989f35b2bc8d3daec544d1dd3d`, verified equal to remote main. Historical baseline: `673a6c8783d83dfbbfd680bfbc7036de07568bc3`. The final report and closing proof index are committed afterward; their commit is reported in the delivery message.

The official two-estimator replacement achieved **0.9618529228157235 full primary OOF**, a **+0.0000873089911441** operational gain over immutable v6. All five paired fold gains are positive. This is a primary-validation result: matched independent TabPFN confirmation and certified test inference have not completed. **v6 remains certified Champion A; Champion B remains unresolved; Phase 17 used zero submissions.** The requested 0.9621 stretch remains unachieved, with a gap of 0.0002470771842764.

Full-context prediction correctly stopped when C: free space reached **18.495136 GiB**, below the unchanged **20 GiB** reserve. At the checkpoint, C: has **32.031040 GiB free**, host RAM has **9.207485 GiB free**, and no campaign model worker is active. A request to free at least 8 GiB more is pending. The source of the transient disk reservation has not been established. No user data was deleted, guards reduced, or system settings changed. The four-estimator resource branch is closed; the two-estimator candidate awaits capacity to finish qualification.

## Model table

The portfolio always uses 50% route logits and 50% unchanged auxiliary10 mean logits. “Full portfolio” below describes that recipe on the stated rows; a one-fold result is not full five-fold OOF. Test correlation means correlation with v6 on actual competition test predictions and is unavailable for both new candidates.

| Candidate | Internal estimators | Backend | Folds | Standalone route AUC | Full portfolio AUC | Δ vs v6 | Test correlation | Runtime | Verdict |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| Immutable v6 | 1 | Windows efficient SDPA, frozen precision | Primary 0–4; prior certified confirmation/test | 0.961219600770 | 0.961765613825 | 0 | 1, self-reference | Historical; not retimed | Certified Champion A |
| B0 exact replay | 1 | Same frozen backend | Primary 0 complete; new shadow 2 resource-invalid | 0.961132445904, primary 0 | 0.961727180455, primary 0 | 0, same primary fold | Original certified v6 | 1,010.688 s successful replay; 774.640 s failed shadow | Exact original prediction SHA; no new shadow result |
| B1 official internal2 | 2 | Windows efficient SDPA; one full cache at a time | All 5 primary folds complete | 0.961515978720 | **0.961852922816** | **+0.000087308991**, pooled | Unmeasured | **10,120.656 s**, primary fits/inference | Primary admission passes; shadow/test blocked |
| B2 official internal4 | 4 | Same verified sequential backend | Primary 0 complete; primary 1 partial/resource-invalid | 0.961511076454, primary 0 | 0.961831632639, primary 0 | +0.000104452183, same primary fold 0 | Unmeasured | 4,049.938 s successful fold; 4,587.812 s failed replication/recovery | Resource branch closed; no full OOF or model rejection |

## A. Technical resolution

The failed operator was PyTorch scaled-dot-product attention. The native memory-saving path sliced the query estimator batch to one while retaining a two-estimator cached key/value batch. Recorded tensors were Q `[1,16,1024,64]` and KV `[2,16,100000,64]`, contiguous bfloat16, with no mask, dropout, or causal attention. The selected efficient backend could not execute this batch combination. Flash attention was absent from the installed Windows build. The smaller 64-row query also failed. Broadcasting a different estimator's cache would change the model and was rejected.

The supported official setting `settings.tabpfn.max_batched_estimator_rows=0` makes native scheduling keep each estimator in its own cache group. It preserves `memory_saving_mode=True` and the frozen efficient-attention/precision policy. Native2 versus exact sequential2, and native4 versus exact sequential4, each had a maximum absolute probability difference of **1.1920928955078125e-7**, below the predeclared **2e-6** tolerance on the frozen 100,000-FIT/1,024-apply probe. Native4 repeated bit-exactly. Class unpermutation, estimator order, preprocessing, normalization, IDs, shapes, and finiteness are checked. The original full primary-fold0 B0 prediction reproduced its recorded SHA exactly.

This establishes equivalence for supported **unbatched native scheduling**. Default batched memory-saving inference remains invalid. The `memory_saving_mode='auto'` alternative differed by 0.0001661181449890 and was rejected before AUC evaluation. The official two-estimator configurations do not contain the original single-estimator configuration exactly; the result compares frozen operational recipes and is not a causal “old model plus one extra member” experiment.

The environment is Windows build 26200, Python 3.11.9, PyTorch 2.11.0+cu128, CUDA 12.8, TabPFN package 9.1.0, NVIDIA driver 591.86, RTX 5070 Ti, compute capability 12.0. The original TabPFN3.5 checkpoint SHA is `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`. Initial inspection found only a stopped docker-desktop WSL2 distribution and no running Linux Docker engine; working Linux CUDA was not established. Kaggle GPU quota was exhausted at inspection. The Windows solution made a Linux migration unnecessary. No environment or OS installation was performed.

Technical resolution took **16.45 minutes** of the 90-minute allowance. Native probe peaks were about **2.67 GiB** for two estimators and **2.69 GiB** for four; full-context single-cache peak allocation was **13,675,054,592 bytes**, about **12.736 GiB**. Full caches are released between members.

## B. Model performance

| Primary fold | B1 route AUC | B1 portfolio AUC | Paired portfolio gain vs v6 |
|---:|---:|---:|---:|
| 0 | 0.961457655131 | 0.961827814250 | +0.000100633795 |
| 1 | 0.961616293589 | 0.961988961145 | +0.000053536305 |
| 2 | 0.961529966227 | 0.961891298676 | +0.000103748487 |
| 3 | 0.960886425767 | 0.961099481811 | +0.000082155550 |
| 4 | 0.962135478128 | 0.962495639871 | +0.000099229860 |

The mean paired gain is **0.000087860799**, paired fold SE **0.000009368950**, or about **9.38 SE**. The 200-repeat paired-row bootstrap 95% interval is **[0.000059349370, 0.000113723725]**. These saved-prediction intervals do not include refitting uncertainty. Fold 1's own bootstrap interval crosses zero despite its positive point estimate.

All nine pooled passenger segments improve. The smallest pooled gain is Eco Plus, **+0.000017752492** over 30,019 rows. However, Eco Plus on fold 3 loses **0.000459898148** over 6,058 rows. This observed risk is retained and requires the matched confirmation step; pooled gains do not erase it.

B2's only complete primary fold gains 0.000104452183 over same-fold v6, only 0.000003818388 more than B1 on that fold. Its remaining OOF is unknown. A RAM guard prevented the original fold1 fourth-member fit; two bounded recovery fits then reached the disk reserve before prediction. All failed reports and the first three verified member contributions are retained. No AUC is assigned to those failures.

## C. Private leaderboard evidence

B1's full OOF correlation with v6 is **0.998808427901 Spearman**, **0.999824479231 logit Pearson**, and **0.999727902739 probability-residual Pearson**. Actual test correlations are unknown because B1 test predictions do not yet exist.

Exact positive-negative pair accounting across 120,813,731,344 pairs gives **148,033,941.5 rescued credits** and **137,485,816.5 damaged credits**. Their fractions are **0.001225307255** and **0.001137998264**, respectively; the net credit agrees with the measured pooled AUC gain. This is measured OOF ranking improvement, not knowledge of private test labels.

The fixed OOF-only robustness diagnostic completed **820 original partitions**: 200 random-stratified, 200 fold-aware, and 420 segment-stressed draws. Every partition-mask hash matched the existing bank; split-definition SHA is `7003ac1c9a0594d3e28f152cc9b556704ef9f1dc1545a1486dd76b164ed64472`. Relative to v6, B1's simulated private delta has **p05 +0.000055869511**, **median +0.000086067471**, and **p95 +0.000118479310**. All 820 private draws are positive; the minimum is **+0.000033055366**, and the worst 5% mean is **+0.000048228352**. Public/private signs disagree in **13/820** draws (1.5854%).

These draws condition on saved primary OOF predictions. They cannot certify unseen private labels, model refitting, unknown shifts, independent confirmation, deployment consistency, or test diversity. The diagnostic explicitly has `submission_eligible=false` and does not replace the certificate-dependent private simulator.

Fresh auxiliary10 preparation for shadow folds 2 and 3 is complete: all **20** satisfaction-model fits and both fold-safe rating-distribution caches passed report/source/ID/fingerprint checks. Their standalone mean-logit AUCs are **0.961794209507** and **0.961818928328**. These are not candidate-vs-control confirmation gains: the new B0 shadow2 prediction was resource-invalid, shadow3 was not started, and both B1 shadow predictions remain absent. No shadow result is imputed from primary or historical folds.

Certified Champion A remains **v6_sol_tabpfn_route_aux10**. No new candidate is certified for a private slot. Champion B is unresolved; an incomplete four-estimator result cannot fill the slot.

## D. Kaggle submissions and rules

**Zero Phase17 uploads** were made, and final-selection settings were not changed. The existing certified submission is **v6_sol_tabpfn_route_aux10.csv**, Kaggle reference **56951906**, status COMPLETE, public AUC **0.96144**, full primary OOF **0.9617656138245795**. Its certified test-array SHA is `b9ccd997acca2e5c05946e7581ec8d81ea21ab198eadfd99e535f86f651b7c7f`; uploaded CSV SHA is `828a962205c9529aebbdccac6217ad0dfce36ae34e3108aca45a6b0ff462fd7d`.

The authenticated checkpoint at **14:02:18 UTC** recorded team rank **270/1,296**, **10** daily submissions remaining, and public leader **0.96243**. This is the team's current rank, not a measured rank for an unsubmitted B1. Public metrics were not used to select weights or parameters.

The official [competition rules](https://www.kaggle.com/competitions/playground-series-s6e10/rules), refreshed through the authenticated Kaggle page API on October 9 and saved in `research/raw/phase17_rules_current_20261009.json`, allow ten daily submissions and up to two final judging selections. The deadline is **October 31, 2026, 23:59 UTC**. Recheck quota and selection rules before a later upload or final selection.

## E. Engineering and resources

The current full suite completed with **142 passed, 0 failed**. It includes the original leakage checks, Phase17 class/order/replay/equivalence checks, resource-guard preservation, verified resumability, unchanged partition definitions, and refusal of partial submission certificates. Critical source checks also passed. Target-dependent auxiliary features use inner out-of-sample FIT predictions and outer-FIT-only applied-row models; satisfaction and ID are excluded from the rating predictors. GBDT early stopping uses an inner FIT-only split. Registered primary/shadow fold hashes were not regenerated.

The checkpoint integrity audit at **14:26:16 UTC** passed on commit `a0f12722e00c13989f35b2bc8d3daec544d1dd3d`: **75 historical evidence files**, **8 baseline snapshots**, unchanged v6 bank, byte-exact historical ledger prefix, **19** unique new ledger entries, **1,266** tracked/untracked scan paths, and **154** changed committed blobs. Both secret-finding lists and committed/working byte mismatches were empty. A final integrity run after the report/status commit is indexed in `phase17_checkpoint_proof_index_20261009.json`; its actual results are recorded there. Bulk data, weights, and predictions stay ignored.

Unique model/probe wall time consumed is **6.5440275 hours**, including failed fits, the 20 new auxiliary fits, and both new auxiliary feature-cache builds. Verified reused contributions are counted once. This is not GPU-active time or the whole research elapsed time. The initial eight-hour allowance has not been exceeded. A finite **28-hour total ceiling** was declared after the replicated five-fold improvement; the extension is declared but not yet used. OOF diagnostics, tests, audits, and **125.109 seconds** of reversible cache compression are recorded separately.

Compression touched only 25 explicitly scoped generated cache files, about 9.515 GiB of logical bytes. All before/after logical SHA256 values match. Physical allocation fell by about **0.460 GiB**, which did not restore sufficient capacity. No dataset, checkpoint, v6 prediction, or unrelated user file was removed.

The last exact resource failure recorded **18.495136 GiB free** at prediction entry. Recent FIT telemetry had shown about 34 GiB, and the process exit restored about 34 GiB. The cause of the approximately 15.5 GiB transient change is unknown. The original one-estimator B0 shadow path also failed its disk guard despite its existing garbage-collection boundary. Therefore this is broader than the four-estimator implementation, and another unchanged-capacity retry is not justified. Read-only pagefile inspection recorded 29,998 MiB allocated, 2,445 MiB current use, and 14,564 MiB peak; these figures do not establish the cause.

At **14:27:14 UTC**, no campaign worker remained. GPU use was **1,407/16,303 MiB**, with 1% utilization from the desktop/runtime. Available host RAM was **9.207485 GiB** and C: free space **32.031040 GiB**. Full-context work awaits additional headroom; freeing **at least 8 GiB** is the requested intervention. It is a capacity prerequisite, not permission to alter the guard or delete user data.

## F. Explicit decisions

1. **Was the kernel problem solved?** Yes, using the supported official unbatched scheduler on Windows efficient SDPA. The default batched memory-saving path remains invalid.
2. **Is official two-estimator inference verified?** Yes. Native2/sequential2 maximum probability gap is 1.1920929e-7 ≤ 2e-6; original B0 full-fold prediction SHA matches exactly.
3. **Does internal ensembling improve v6?** Yes on full primary OOF: +0.000087308991 with the frozen unchanged-auxiliary portfolio. Test/private improvement remains unconfirmed.
4. **Is the improvement consistent?** All 5 primary fold gains are positive; mean +0.000087860799, SE 0.000009368950. The fold3 Eco Plus loss remains a limitation.
5. **Does it survive independent confirmation?** Unknown. Shared shadow auxiliary fits are ready; matched TabPFN control/candidate predictions are missing because of the disk failure.
6. **Does private robustness improve?** All 820 fixed OOF private draws improve, with median +0.000086067471 and p05 +0.000055869511. This is conditional simulation, not private-label evidence or test certification.
7. **Can 0.9621 be reached credibly?** It has not been reached. The measured score is 0.961852922816 and the remaining gap is 0.000247077184. No validated result establishes a route to the target; no impossibility or Bayes ceiling is claimed.
8. **Was a stronger Champion A produced?** A stronger primary candidate was produced. Certified Champion A remains v6 until the independent and test gates pass.
9. **Was a genuine Champion B found?** No qualified B has been established. The four-estimator candidate is incomplete and its resource branch is closed.
10. **Was anything submitted?** Zero new submissions. Existing v6 ref56951906 remains COMPLETE at public 0.96144.
11. **What is the next justified action?** Restore disk headroom, finish the frozen B1 matched shadow2/3 comparison, then—only if confirmation passes—perform all five exact full-context test runs, certify all 299,844 test rows, run the certificate-dependent robustness/decision gates, and upload only if eligible.
12. **Should this cycle stop?** Stop the bounded four-estimator resource branch and stop further full-context retries under unchanged capacity. The B1 qualification is resource-blocked, not completed or scientifically rejected. No new model search or automatic Phase18 is authorized by this checkpoint.

## Exact resume sequence

Do not rerun successful primary folds, change seeds/weights, regenerate fold IDs, or reuse incomplete predictions. Verify resources before every heavy job: one worker, host RAM ≥4 GiB before fitting, ≥2 GiB before prediction, unchanged GPU/disk guards, and enough C: headroom for the observed transient reservation. The additional 8 GiB request targets about 40 GiB pre-FIT free space and is not a guarantee against unrelated changes.

1. Preserve the failed `phase17_shadow_B0` report. Run `run_sol_tabpfn_confirm.py --reference reports/sol_tabpfn35_route/route_f0.json --scheme shadow --fold 2 --tag phase17_shadow_B0_capacity`, then fold3 only after success.
2. Run frozen `run_phase17_confirmation.py --mode shadow --primary-tag phase17_B1 --fold 2 --tag phase17_shadow_B1`, then fold3, one job at a time. Reuse the fully verified `phase17_shadow_aux10` caches without refitting them.
3. Run `evaluate_phase17_shadow.py --primary-tag phase17_B1 --candidate-tag phase17_shadow_B1 --control-tag phase17_shadow_B0_capacity --auxiliary-tag phase17_shadow_aux10`. Both paired fold gains must be nonnegative and their mean positive. Stop promotion if the frozen gate fails.
4. If confirmation passes, run `run_phase17_confirmation.py --mode test --primary-tag phase17_B1 --fold 0 --tag phase17_B1_test`, then folds1–4. Each uses the exact original primary outer-FIT population and all ordered test IDs. Official internal probability aggregation precedes equal-probability averaging across the five contexts; the final portfolio retains unchanged 50/50 route/auxiliary logits.
5. Assemble a uniquely named CSV with `assemble_phase17_test.py`, verify every raw contribution/configuration/source/ID/hash/CSV roundtrip, then run protected `simulate_phase17_private.py` and `decide_phase17_submission.py` with that actual certificate. Commit, audit and push the concrete eligible result before upload. Refresh rules/quota and obey the original daily cap; uncertain upload outcomes are not automatically retried.

Proof paths and SHA256 values are in `phase17_checkpoint_proof_index_20261009.json`. The checkpoint decision is machine-readable in `phase17_checkpoint_decision_20261009.json`. Failed experiments, partial contributions, old banked finalists, ledger history, and both resource evidence and numerical equivalence reports remain preserved.
