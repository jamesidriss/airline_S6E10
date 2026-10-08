# S6E10 — completed recovery campaign

Completed 8 October 2026. **Champion A: v6_sol_tabpfn_route_aux10.** Kaggle submission **56951906** is **COMPLETE**, public AUC **0.96144**. One new submission was used. No credible certified Champion B was found. Actual private ranking remains unknown.

## A. Recovery

Initial local HEAD and live GitHub main both verified as `671826b092d89ac73750a3c0baac6c21b3c1eb98`. The checkout contained pre-existing untracked logs; they were preserved. No surviving model worker was found. Available RAM was 8.4 GiB and disk 43.4 GiB.

The interrupted route fold 3 had completed fitting but failed the frozen 20 GiB prediction disk preflight after 777.203 seconds. It produced no validation vector or AUC. This remains resource-invalid evidence, not negative model evidence. Recovery fitted fresh route folds 3/4 under the unchanged recipe and completed the 20 missing auxiliary fits. All 33 valid primary fold 0/1/2 vectors were reused after source, data, fold, row-ID, parameter, prediction-hash and AUC verification. No original zoo, 39-cross branch or additional full-699635 GPU optimization was started.

The pre-upload checkpoint was `fc33684418d0e955a0e87222a06c545d7646be1f` and was pushed before upload. The final closing commit and live remote equality are reported in the closing chat response; this report records the source/evidence checkpoint rather than a self-referential commit hash.

## B. Complete primary five-fold result

All **699,635 ordered OOF rows** are present. The recipe remains seed 1201, route TabPFN 3.5 plus the ten corrected auxiliary roles, with fixed 50/50 method logits.

| Fold | Route AUC | Clean auxiliary10 AUC | Frozen candidate AUC | Paired gain vs auxiliary10 |
|---|---:|---:|---:|---:|
| 0 | 0.961132446 | 0.961414203 | 0.961727180 | +0.000312978 |
| 1 | 0.961379166 | 0.961606072 | 0.961935425 | +0.000329353 |
| 2 | 0.961199565 | 0.961442312 | 0.961787550 | +0.000345238 |
| 3 | 0.960608496 | 0.960573331 | 0.961017326 | +0.000443995 |
| 4 | 0.961819202 | 0.962092984 | 0.962396410 | +0.000303426 |

Pooled candidate AUC **0.9617656138245795**, clean auxiliary10 **0.9614187095444636**, route **0.9612196007698864**. Pooled candidate gain over auxiliary10 **+0.00034690428011585617**. Mean paired fold gain **+0.0003469980344849022**, SE **0.000025277324554686046**, **5/5 positive**. The frozen admission gate passes.

The 200-repeat paired-row bootstrap interval is **[+0.0002823551826645038, +0.00040949183996077666]**. It conditions on saved predictions and does not represent retraining or unknown distribution shift.

Historical comparisons are diagnostic: candidate minus v5 **+0.00024212374019561** and minus v3 **+0.0002570353895583466**. Legacy neural checkpoint selection, target-encoding priors and test refit defects remain documented. v4 uses v3 OOF as a proxy.

## C. Independent robustness and inference-policy audit

The scope was frozen before these jobs: shadow folds 0/1, primary fold 0 seed 1202, and the bounded policy audit. Full shadow OOF and block10 were not run.

| Shadow fold | Clean auxiliary10 | Frozen candidate | Paired gain |
|---|---:|---:|---:|
| 0 | 0.960950944 | 0.961205254 | +0.000254310 |
| 1 | 0.960916158 | 0.961218143 | +0.000301985 |

Both gains are positive. Mean **+0.0002781477350757844**, paired SE **0.00002383750493423209**. This passes the predeclared limited gate and remains **partial confirmation**, with no full shadow pooled AUC claimed. Block10 required disproportionate additional compute and larger contexts within the finite recovery budget.

The fixed seed replay returned route AUC **0.9611372496955853**, versus seed 1201 **0.9611324459037415** on the same fold. Difference **+0.000004803791843865923**; corresponding portfolio difference **+0.00002659845338104372**. Portfolio logit correlation is **0.9991642370**. Seed 1201 remains the finalist; this is one diagnostic fold, not five-fold seed validation.

The leakage-safe policy audit used one fixed 20,000-row shadow holdout and a 100,000-row training pool. Reference-context AUC **0.9602431262**; average of five approximately 80,000-row exclusion contexts **0.9602823011**. Difference **+0.0000391749**, bootstrap interval **[-0.0002856819, +0.0002721511]**. The audit is inconclusive about a true gain. It supplies no OOF score for the deployed test ensemble.

## D. Certified test inference

Five fresh route models each used its original **559,708-row primary FIT context**, seed 1201 and immutable feature, checkpoint, preprocessing and attention contracts. Each predicted all **299,844 test rows**. Their probabilities were equally averaged. This is the frozen CV-context inference fallback.

The six LightGBM extra-trees, three XGBoost and one native CatBoost auxiliary models each refitted on **699,635 labeled rows** at the median of its five valid selected tree counts. Their configurations remained distinct and matched CV. Auxiliary ratings used the frozen raw-only, three-fold, 250-round cache; satisfaction was excluded from its input.

The auxiliary component is the equal mean of ten logits. The final prediction is the logistic transform of half the route-component logit plus half the auxiliary-component logit. No calibration, seed selection or weight fitting was introduced.

Schema preflight passed: exact `id,satisfaction` columns, complete ordered IDs, finite probabilities in [0,1], correct row count and certified vector identity. Final float32 recomposition gap is **5.960464477539063e-08**.

Prediction array SHA256: `b9ccd997acca2e5c05946e7581ec8d81ea21ab198eadfd99e535f86f651b7c7f`.

Exact uploaded CSV SHA256: `828a962205c9529aebbdccac6217ad0dfce36ae34e3108aca45a6b0ff462fd7d`.

| Banked submission | Test Spearman vs v6 | Test logit correlation vs v6 |
|---|---:|---:|
| v3_final | 0.992677680 | 0.998783696 |
| v4_fulldata | 0.991947910 | 0.998569577 |
| v5_aux_cross | 0.993110672 | 0.998843156 |

## E. Private-first ranking and hedge decision

| Candidate | OOF AUC | Public AUC | Evidence / role |
|---|---:|---:|---|
| v6 frozen portfolio | 0.961765614 | 0.96144 | Full clean primary; two positive shadow folds; certified test; Champion A |
| Clean auxiliary10 | 0.961418710 | Unsubmitted | Trails A by 0.000346904; high test correlation; no credible hedge |
| Pure route | 0.961219601 | Unsubmitted | Trails A by 0.000546013; no credible hedge |
| Historical v5 | 0.961523490 diagnostic | 0.96103 | Banked, contracts defective; uncertified legacy fallback |
| Historical v4 | 0.961508578 v3 proxy | 0.96100 | Banked; no separate verified OOF |
| Historical v3 | 0.961508578 diagnostic | 0.96098 | Banked; historical contracts |

The simulator used seed 20261010, population 299,844, fixed 20/80 splits, 200 random, 200 fold-aware and 420 segment-stress draws. Candidate-independent confidence partitions were tied to fixed legacy v5. Old simulation artifacts were preserved.

For **v6 minus historical v5**, simulated private [5th, median, 95th] percentile gains were:

- Random: **[+0.000174222, +0.000246734, +0.000322389]**.
- Fold-aware: **[+0.000170031, +0.000240884, +0.000323622]**.
- Segment stress: **[+0.000156364, +0.000236552, +0.000315569]**.

The smallest fifth-percentile advantage among the 14 fixed segment scenarios was **+0.000131317**, for business travel. No simulated private ordering reversal occurred against clean components or legacy v3/v5 across 820 draws. Public/private sign disagreements versus v5 were 2.5%, 0.5% and 3.095% in the three modes. These are conditional resampling diagnostics; they cannot reveal private labels or estimate a probability of winning. Legacy comparisons remain diagnostic.

In two million sampled positive/negative pairs, v6 rescued **6,363** and damaged **5,812** pairs versus clean auxiliary10; the sampled net difference was +0.0002755. Against legacy v5 the diagnostic counts were **6,543 / 6,098**, net +0.0002225. These Monte Carlo values are not exact AUC arithmetic.

**Champion A: v6, ref 56951906. Champion B: no credible certified hedge.** Clean components have lower expected scores and lose every simulated private comparison; test logit correlations with A exceed 0.9989. Legacy v5 remains banked as an uncertified fallback. Final slots were reassessed; no final-selection UI change or artificial alternative blend was made.

The clean 59-member reconstruction is not strategically justified within this finite continuation after the large replicated clean gain. Repairing 47 classical and 12 neural members adds disproportionate compute. Its infrastructure, checkpoints and historical predictions are preserved. The campaign stops without a new feature, weight, seed or 39-cross search.

## F. Actual Kaggle result

Submission **56951906**, `v6_sol_tabpfn_route_aux10.csv`, is **COMPLETE**, public AUC **0.96144**. The exact newly recorded reference and filename were matched. The immutable post-submit snapshot at **2026-10-08 11:39:47 UTC** recorded team rank **234/1145**, **9 submissions remaining**, leader **0.96200**.

The public gain versus v5 is **+0.00041**. It supports transfer on the public portion; the actual 80% private ranking remains unknown. The scientific decision and exact CSV were committed before the score. One of at most two authorized new submissions was used; no second upload or public-score retuning followed.

## G. Repository, tests and durability

The current full suite passed **114 checks, 0 failed**. The initial 110-check run is preserved, but its static guard had linted path strings instead of file contents. The missing scorecard import and guard were repaired; the corrected guard checks all 64 critical entries and its missing-import regression passes. Source has not changed since the current full suite. Six submission-I/O tests used mocked network responses and consumed no real quota.

Authoritative certificates and current checks cover disjoint FIT/validation IDs, full OOF coverage, own-label exclusion, out-of-sample rating features, inner early stopping, category/feature identity, exact model parameters and tree counts, source tracking, finite probabilities and submission schema. Final saved AUCs recompute exactly. The original data, pretrained checkpoint and all three banked v3/v4/v5 CSV hashes match their recorded values. No campaign model worker remains active.

Pre-upload tree and decoded report-log scans found zero secrets. Reachable-history scans covered **126 commits, 1,041 standard text blobs and 170 historical log blobs**, including BOM-aware Windows log decoding, with zero HIGH/MEDIUM/LOW findings. The closing repository audit records the final tree and post-result additions. Historical untracked logs are preserved in the closing commit as historical evidence, with earlier static-guard claims explicitly superseded.

Important evidence is in `sol_route_aux10_primary_final.json`, `sol_route_aux10_primary_final_test_certificate.json`, `sol_route_aux10_recovery_scorecard.json`, `v6_sol_tabpfn_route_aux10_pre_result.json`, `v6_sol_tabpfn_route_aux10_result.json`, `sol_recovery_final_artifact_audit.json`, and the final proof index. Prediction arrays and model weights remain locally preserved under the established artifact ignore policy; certificates, reports, ledger and exact submission CSV are committed and pushed.

**No required campaign work remains blocked.** Remaining scientific uncertainty is the unknown private distribution, partial shadow scope, single seed replay and limited policy audit. Winning the eventual competition has not been established.
