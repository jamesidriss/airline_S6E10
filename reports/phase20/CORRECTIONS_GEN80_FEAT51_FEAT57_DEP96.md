# Phase 20 Scientific Corrections (Append-Only)

**Date**: 2026-10-18  
**Authority**: User mandate to correct scientific interpretation issues without overwriting history.

---

## 1. GEN80 Oracle Ceiling — Interpretation Correction

**Original Report**: `reports/phase20/gen80_auc_ceiling/gen80_auc_ceiling.json`  
**Issue**: The oracle ceiling (0.964906) was presented as evidence that "stretch is THEORETICALLY reachable by a better blender."  
**Correction**: The oracle ceiling is a **label-dependent pairwise upper bound** (uses labels to cheat on every pos/neg pair). It is **not** evidence that an executable, label-free blender can achieve 0.962740. The oracle cheats by using ground-truth labels on every pair; a real blender must generalize from covariates only.  
**Correct Interpretation**: The oracle ceiling (+3.14e-3 headroom) shows the *mathematical maximum* if a blender could perfectly separate all pairs where at least one base model is correct. It does **not** imply the stretch target (0.96274) is achievable by any deployable blender. The actual equal-logit blend (0.961756) underperforms the route reference alone (0.961766).  
**Reference**: v7 champion OOF = 0.9618529228157235 (not the route reference 0.961766 used in the oracle). The stretch gap is +8.87e-4 from champion, not +9.74e-4 from route reference.

**Action**: Add this correction note to the GEN80 report. No re-measurement needed.

---

## 2. FEAT51 Residual Ratings — Feature Count & Paired Comparison Correction

**Original Report**: `reports/phase20/champion_residual_integration/champion_residual_integration.json`  
**Issue**: The report claims "full_residual (285 + 13 cross-fit residuals)" but the actual static feature count is 237 (the "full" view static features), not 285. The actual feature matrix was 237 static + 13 residuals = 250 features. The comparison to v7 champion (0.961853) used a single LGBM run, not a paired control/treatment with identical folds, feature views, and training policies.  
**Correction**: 
- The "full" static view has **237 features**, not 285. The 285 figure includes fold-safe TE columns that are computed per-fold, not static.
- The integration used 237 static + 13 residuals = 250 features.
- The comparison to v7 (a 59-member blend) via a single LGBM AUC is not a valid portfolio delta measurement. A proper test requires: (a) paired control (champion's full static view) and treatment (champion's full static view + 13 residuals) under identical 5-fold CV, identical LGBM recipe, identical folds, identical inner ES splits. Then measure the paired fold delta.

**Action**: 
1. Add correction note to the original report.
2. Run a proper paired control/treatment 5-fold CV: control = champion's static full view (237 features), treatment = same + 13 cross-fit residuals. Identical LGBM recipe, identical 5-fold splits, identical inner ES. Report paired fold deltas with bootstrap CI.

---

## 3. FEAT57 Low-Rank Interactions — Missing Mechanism

**Original Report**: `reports/phase20/feat57_lowrank_interactions/feat57_lowrank_interactions.json`  
**Issue**: The prereg requested "learned low-rank pairwise rating-interaction mechanism" (supervised, like a low-rank bilinear model trained with satisfaction signal). The implementation delivered **unsupervised SVD** (PCA on ratings), which is a distinct, weaker mechanism. The unsupervised SVD captures variance in ratings, not satisfaction-predictive interactions.  
**Correction**: The measured result (+3.19e-3 on 21raw, blend +3.81e-3) is valid for unsupervised SVD but does **not** test the preregistered hypothesis. The distinct missing mechanism is a **supervised low-rank pairwise interaction model** (e.g., factorized logistic on rating pairs, or a low-rank bilinear model trained with satisfaction labels via cross-fitting).  
**Action**: 
1. Add correction note distinguishing unsupervised SVD from the requested supervised mechanism.
2. Implement and test the supervised low-rank pairwise interaction mechanism (cross-fit, fold-safe, e.g., FactLog-style bilinear on rating pairs with satisfaction supervision).

---

## 4. DEP96 Block10 — Champion Distinction Clarification

**Original Report**: `reports/phase20/dep96_finalist_validation/dep96_finalist_validation.json`  
**Issue**: The report describes the experiment as "champion_rich_view: prod5_lgbm_full_primary recipe (xt_xt_f10) on 285-feature full view" and compares its block10 OOF (0.961050) to the v7 champion primary OOF (0.961853). But prod5_lgbm_full_primary is a **single LGBM model**, not the v7 champion (which is a 59-member blend including TabPFN, RealMLP, CatBoost, XGBoost, etc.). The block10 validation tested a standalone classical model, not the v7 champion blend.  
**Correction**: The block10 infrastructure is valid and ready. The experiment correctly measured a standalone LGBM on block10 folds. However, it did **not** validate the v7 champion blend on block10 folds. The v7 champion OOF (0.961853) was not re-measured on block10 folds.  
**Action**: 
1. Add correction note clarifying the distinction.
2. The block10 folds are registered and ready. When finalist B is selected, the full v7 blend (or the selected finalist blend) should be validated on block10 folds using the same per-fold assembly process.

---

## Summary of Required Actions

| Issue | Correction Type | Re-measurement Needed? |
|-------|----------------|------------------------|
| GEN80 | Interpretation note | No |
| FEAT51 | Correction note + paired 5-fold CV | Yes (paired control/treatment) |
| FEAT57 | Correction note + implement supervised mechanism | Yes (supervised low-rank) |
| DEP96 | Clarification note | No (infrastructure ready) |

All corrections are **append-only** — original reports and ledger entries are preserved. New correction documents and re-measurements are added.
## FEAT51 blend correction (2026-10-18, append-only, c)

- Corrected blend AUC: **0.961675** (FEAT51-treatment + route reference, equal logit).
- Champion A v7 OOF: **0.9618529228**. Blend minus champion: **-0.000178**.
- The reported +0.00059667 delta is vs the FEAT51 standalone model (0.961079), not vs v7. FEAT51 is NOT a champion improvement and is not promoted.
