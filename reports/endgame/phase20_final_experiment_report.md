# Phase 20 final experiment report — 96-item exhaustive campaign

**Generated:** 2026-10-19 UTC · **Git:** see `reports/endgame/gpu_budget_22h_final.json`
**Campaign state:** all 96 catalogue rows dispositioned · 550 append-only ledger entries
**Champion (immutable):** `v7_sol_tabpfn_internal2_aux10` — OOF **0.9618529228157235**, public
0.96146, Kaggle ref 57090508, test-pred SHA `2b744d0d…`, CSV SHA `fa4947fe…`

> This report distinguishes **disposition** from **execution**. A dispositioned row is
> *closed with a reason*; it is not a trained model. The fidelity column below is the
> authoritative record of how much compute each row actually received.

## 1. What the campaign was

96 catalogue items across 8 families (FLM 14, NN 16, OBJ 14, FEAT 14, CLASS 11, ENS 10,
GEN 8, DEP 9), executed under the repository's validation contract: immutable primary
folds (`StratifiedKFold(5, shuffle, seed 20261010)`, hash-checked registry), fold-safe
features only, preregistration before every predictive result, append-only ledger.

## 2. Execution accounting (honest)

| fidelity | meaning | rows |
|---|---|---|
| F5 | certified end-to-end (full OOF + test + submission-grade hashes) | 2 |
| F3 | full paired 5-fold CV against a frozen control | 2 |
| F2 | fold-0 screens with a frozen decision rule | 59 |
| F1 | audits / saved-array analyses (no new training) | 10 |
| F0 | catalogued, conceptual, license-, resource- or dependency-blocked — **no model was trained** | 23 |

F0 rows (never trained): 3, 7, 8, 12, 14, 33, 34, 43, 67, 70, 71, 73, 79, 81, 82, 83,
86, 87, 89, 90, 92, 93, 95.

Terminal-status distribution: **66 COMPLETE_NEGATIVE** · 8 DEPENDENCY_CLOSED ·
5 CONCEPTUAL_CLOSED · 3 LICENSE_BLOCKED · 2 RESOURCE_BLOCKED · 2 NOT_WARRANTED ·
2 HISTORICAL_EQUIVALENT · 6 PARTIAL_VALID · 1 LOW_VALUE_CLOSED · 1 COMPLETE_CERTIFIED ·
**0 NOT_STARTED · 0 PROMISING** (96 total).

Distinct model trainings actually executed in phase 20 exceed the F2 row count
because several screens reused trained members across rules; the ledger (550 entries)
is the itemized record.

## 3. The structural finding (the campaign's main scientific result)

Across ~29 independent candidate evaluations spanning in-context learners (TabPFN
family, TabICL, xRFM, TabFM, Mitra, Kumo), neural architectures (FT-Transformer,
RealMLP, TabM, MLP, ResNet, DCNv2 and ensembles), classical models (HistGB, ExtraTrees,
RandomForest, oblique, logistic), feature blocks (FEAT 14 items) and objective
variants (OBJ 37–41), every candidate lands on the same frontier:

* **Decorrelated but weak** — Spearman 0.87–0.99 with the champion at standalone
  0.948–0.958, i.e. 4–14e-3 below v7. Blending costs more than it adds.
* **Strong but redundant** — standalone 0.960–0.961 with Spearman ≥ 0.993 against the
  champion's boosted-tree aux10. No marginal value.

The champion sits at the Pareto corner: TabPFN-3.5 internal-2 (0.96146 fold-0 route
member) equal-logit blended with the 10-member aux10 boosted-tree ensemble
(6 LightGBM + 3 XGBoost + 1 CatBoost). The internal-2 upgrade over the single-context
route added **+87.3e-6** on primary and **+87.5e-6** on the independent shadow folds.

## 4. Objective variants (OBJ 37–41) — hypothesis falsified

Prereg `94561fce`; control parity exact (0.9576563038026185, recompute delta 0.0).
Frozen gate: port_delta ≥ +5e-5 **and** Spearman < 0.995 vs the route reference.

| arm | standalone | Δ vs control | blend Δ vs v7 | ρ(v7) | gate port_delta | verdict |
|---|---|---|---|---|---|---|
| bce_control | 0.957656 | — | −7.47e-4 | 0.936 | (parity reference) | control |
| supcon (37) | 0.957778 | **+1.22e-4** | −7.06e-4 | 0.938 | −6.61e-4 | FAIL |
| auc_surrogate (38) | 0.957515 | −1.42e-4 | −8.29e-4 | 0.934 | −7.72e-4 | FAIL |
| hard_mining (39) | 0.954052 | −3.60e-3 | −2.29e-4 | **0.891** | −2.19e-4 | FAIL |
| gce (40) | 0.955504 | −2.15e-3 | −2.17e-3 | 0.895 | −2.12e-3 | FAIL |
| ema (41) | 0.958221 | **+5.65e-4** | −4.81e-4 | 0.940 | −4.34e-4 | FAIL |

**0/5 pass.** The batch hypothesis — "the batch-1 neural screens were weak because of the
BCE objective" — is **falsified**: the objective lever is worth at most +5.65e-4 of
standalone AUC (EMA) against a family ceiling ≈3.6e-3 below v7. The two arms that buy
diversity (hard mining ρ=0.891, GCE ρ=0.895) pay 2–3.6e-3 of standalone accuracy for it.
No promotion to 5-fold OOF was justified for any arm.

> Transcription note: commit messages `86ab9ff` and `850ab88` state hard_mining/gce
> delta-vs-control as −3.60e-4 / −2.15e-4; the authoritative ledger entries and
> `reports/endgame/obj37_41_results.json` record **−3.60e-3 / −2.15e-3**. The artifacts
> are correct; the commit-message digits were a transcription slip (history is not rewritten).

## 5. Feature blocks (FEAT family) — one real signal, correctly bounded

* **FEAT48 hierarchical 13-rating structure** — the only rating-derived block to pass a
  full admission gate: on the 21-variable host, mean paired **+2.87e-4**, t = 4.92,
  **5/5 folds positive**, fold-0 parity exact. But the augmented host cannot enter the
  ensemble (equal-logit blend with v7: **−1.90e-3**, 0/5), and on the rich host the
  signal is absorbed (**+4.7e-4, t = 1.33**, gate fails). *Host-limited; closed.*
* **FEAT51 observed-minus-predicted residuals** — mean paired +7.05e-5, t = 1.54:
  misses the 2.5·SE limb. Real but inconsistent.
* **FEAT52/57 supervised factorization** — negative (−2.1e-3 to −2.9e-3). The
  unsupervised SVD variant helped only a weak base.
* The remaining FEAT items are negative or conceptually closed.

## 6. Resource- and license-blocked items (not negative results)

* **TabPFN-2.5 (row 12)** — three preregistrations, three distinct resource walls:
  eager KV-cache build OOM at the full 559k context (27.27 GiB), predict-time fp32
  attention materialization (50.42 GiB at 8192×50000 — v2.5 attention is not
  flash-eligible on this build), and a time-infeasible configuration (60 s/batch →
  4.3 h/arm). No predictive result exists; the architecture was never measured.
* **TabICL** — full-dose fine-tuning resource-blocked; a prior attempt produced
  `nvlddmkm` Event 153 GPU driver faults and a near-full system drive. A bounded
  100k-FIT fine-tune completed at 0.957404 standalone (portfolio delta −8.2e-4).
* **LimiX-2, LimiX-2M/16M, EXAONE** — license-blocked (non-commercial terms do not
  permit this competition's use).
* **FLM x rich view** — resource-blocked at 16 GB (TabICL), never measured.

## 7. Audits that shaped the endgame

* **H91 seed vs subset variance** — seed variance 1.8e-9…7.2e-9 vs subset variance
  6.0e-7…6.3e-7: the pipeline is seed-stable; fold-to-fold movement is data composition.
* **H94 champion stability (true champion reference)** — all 9 saved preprocessing/
  selection variants degrade an equal-logit blend with v7 on at least one fold
  (min Δ −1.7e-4…−7.0e-4). **Champion A must be locked unblended.**
* **GEN80 oracle AUC ≈ 0.964906** is label-dependent and *not* evidence that any
  executable blender reaches it.

## 8. Compute

Reconciled phase-19/20 GPU consumption before the endgame: **11.54 h documented**
(9.1 h at S7 close + 2.44 h of post-S7 ledger entries), carried at **~11.8 h**.
The endgame runs under a **user-authorized 22 h cap**
(`research/phase20_exhaustive/compute_budget_extension_22h.md`), enforced by
`scheduler.budget_check()` / `budget_charge()` with 3 regression tests.

## 9. Endgame execution (2026-10-19) — what the final run added

**Compute.** The user authorized raising the local GPU ceiling from 12 h to **22 h**
(`compute_budget_extension_22h.md`, prospective, with the scheduler enforcing the cap).
Reconciled consumption: 11.8 h before the endgame, **14.54 h after it** (7.46 h unspent,
deliberately — see §10).

**Priority 2 — the five prepared objectives.** Executed through the durable scheduler
(`J037`–`J041`, each with a DONE-manifest validator: report status, prediction SHA,
row count). The BCE control's parity was re-verified independently before any variant
was interpreted (recomputed AUC 0.9576563038026185, delta 0.0, prediction SHA
`5fc5e356…`). Result: **0/5 pass the frozen gate**; see §4. The batch hypothesis is
falsified: changing the training objective is worth at most +5.65e-4 of standalone AUC
(EMA) against a family ceiling ≈3.6e-3 below the champion. Both "buy diversity" arms
(hard mining ρ=0.891, GCE ρ=0.895) pay 2–3.6e-3 of standalone accuracy for it and still
lose the blend.

**Priority 4 — clean-code champion reproduction.** `JREP01` re-ran the certified
phase-17 pipeline for fold 0 under a prereg with a bitwise acceptance criterion and a
declared numerical-tolerance fallback. Outcome: **BITWISE_PARITY** — all four array
hashes and both member-config hashes match, max |Δ| = 0.0, AUC identical
(0.9614576551310231), 2021 s vs 2023 s. The fallback band was never invoked. The blend
geometry was independently recomputed from the ten aux10 members (max |Δ| 2.98e-8,
float32 artifact rounding).

**Priority 5 — Champion-B search (catalogue row 13, both checkpoint arms).**

| arm | context | standalone (fold 0) | blend Δ vs true v7 | 95 % CI | decision |
|---|---|---|---|---|---|
| TabPFN-2.6 | 100 k (full OOMed) | 0.956951 | −1.051e-3 | [−1.29e-3, −8.18e-4] | CLOSE |
| TabPFN-3 | 100 k | 0.956608 | −7.967e-4 | [−9.6e-4, −6.3e-4] | CLOSE |
| *control:* v3.5 (champion ckpt) | 100 k | 0.959544 | −5.281e-4 | [−7.0e-4, −3.5e-4] | interpretive |
| *control:* TabPFN-3 | **full 559,708** | 0.957746 | −6.243e-4 | [−7.9e-4, −4.5e-4] | CLOSE |

Two preregistered interpretation controls separate the context effect from the
checkpoint effect: the 100 k bound costs the champion's *own* checkpoint 1.91e-3,
while the checkpoint swap costs a further 2.94e-3 (v3) / 2.59e-3 (v2.6). The full-context
v3 run confirms it directly: 3.71e-3 below the champion member at identical context.
**Checkpoint quality dominates context size**, so both arms close on unconfounded
evidence and row 13 is terminal.

## 10. Submissions and final selection

* Endgame submissions: **0**. No candidate cleared PATH A (≥ +3e-4) or PATH B (within
  −2e-4 *and* structurally distinct). The only submission inside the PATH-B window is
  v6 (OOF 0.961766), which fails the distinctness clause (same TabPFN + aux10 family,
  logit corr ≈ 0.999+) and was already submitted (ref 56951906, public 0.96144 < v7's
  0.96146).
* **Finalist: Kaggle ref 57090508** (`v7_sol_tabpfn_internal2_aux10`), slot 2 empty.
  Final selection is a Kaggle UI action — the API exposes no selection method.
* Remaining budget unspent because every promotion path (TabPFN-3.5 internal-4,
  seed-averaged route, aux10 upgrades) costs 5.6–11.2 GPU-h for gains that no measured
  evidence supports, and H94 shows additive perturbations degrade the champion.

See `finalist_selection_report.md`, `private_robustness_report.md`,
`champion_reproducibility_report.md`, `kaggle_submission_status.json`,
`gpu_budget_22h_final.json` and `FINAL_HANDOFF.md`.