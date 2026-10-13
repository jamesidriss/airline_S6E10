# Phase 18 — ranked research hypotheses

Generated 2026-10-14. Evidence citations refer to `reports/phase18/*.json` and the
Phase 17 certification chain. Ranking criterion: expected operational v7 portfolio
gain × probability of surviving the admission gate, per GPU-hour. The stretch gap
to 0.962100 is 247e-5 OOF; no single axis below is claimed to close it.

## Rank 1 — H18-A: TabPFN context coverage 80% -> 90% (Axis A/H)

- **Mechanism.** Each v7 route context trains on 80% of rows (559,708). S6E9
  measured that raising per-model training coverage (2 half-data bags -> 5 bags
  on 80%) gained +2.57e-5 (CatBoost+TabM) and +2.79e-5 (CatBoost), 5/5 folds,
  CIs excluding zero — with no feature/seed/hyperparameter change. The mechanism
  (more labels per final model) is model-family general.
- **Evidence for feasibility.** Peak GPU at 559,708 rows = 13.675 GB
  (`artifacts/phase17_B1/f0/member_0_stats.json`); total VRAM 16303 MiB = 17.09 GB
  decimal. Linear KV-cache projection at 629,671 rows = 15.39 GB (margin 1.7 GB).
  Members run sequentially, so estimator count does not raise peak.
- **Matched design.** Both arms evaluated on identical block10 fold-0 rows
  (69,964): control context = fixed-seed 559,708-row subsample of the other 9
  folds; treatment context = all 629,671. Same frozen params, n_estimators=2,
  same seeds. Only context size differs. Causality is established here; the
  promotion test (10-context route OOF vs champion on primary folds) follows only
  if discovery + replication pass.
- **Cost.** Memory probe ~15 min; matched discovery ~55 min; replication ~55 min;
  full 10-context OOF ~4.4 h; test contexts ~4.4 h. Worst case ~14 h of the 24 h
  GPU budget, gated at every stage.
- **Risk.** 1.7 GB projected VRAM margin (probe decides); protocol mismatch is
  controlled by the matched design; aux10 half of the portfolio is untouched.

## Rank 2 — H18-E: official TabPFN internal estimators 2 -> 4 (Axis E) — **CLOSED at wave 1 (2026-10-14)**

- **Verdict: NEGATIVE at the admission level.** Fold-0 route4 − route2 = +5.34e-5
  (standalone, real), but portfolio4 − v7 = **+3.82e-6** — a 7.1% conversion of
  route gain into portfolio gain. The 0.5 route + 0.5 aux10 logit blend dilutes
  route-only improvements almost entirely (route and aux10 share 0.9977
  correlation; the portfolio ranking is dominated by their shared component).
  Expected mean portfolio gain under perfect replication ≈ 0.4e-5, an order of
  magnitude below the 1.5e-5 admission gate. Folds 1-4 never run
  (~4.4h GPU saved); evidence: `reports/phase18_h18e_f0_evaluation.json`,
  closure record `phase18_h18e_close` in the ledger.
- **Methodological lesson (now a standing rule):** re-measure any route-level
  gain at the PORTFOLIO level on fold 0 BEFORE granting GPU replication
  budget. Conversion is change-dependent (~7% here, ~27% for v7's internal2
  change over v6).
- (Historical wave-1 evidence: route4 − route2 = +5.34e-5 on fold 0; member
  AUCs [0.96113, 0.96130, 0.96102, 0.96112], all below route4 0.961511;
  m0-m3 Spearman 0.9793; B2 folds 1-4 were RESOURCE_BLOCKED by disk, not
  model-rejected.)

## Rank 3 — H18-C: ratings-tuple-key exact-value TE (Axis C) — WEAKENED

- S6E8's exact-value TE was its largest transferable lever (+0.0032 there), and
  the FoldSafeTE machinery already exists (`src/features/s6e10.py`).
- **But the S6E10 evidence chain is negative:** single-column exact-value TE on
  all 21 raw columns was -13.3e-5 on the full view (`STATUS.md` L894); key-based
  (S13+M4) TE standalone AUC only 0.818-0.868 (`reports/duplicate_key_signal.json`);
  the error atlas shows unique-key rows (51% of train, where the key TE returns
  only the prior) carry the harder residual (AUC 0.9504 vs 0.9635 on dup rows).
- Kept alive only because a 17-column *combination* key differs from
  single-column TE; cheap CPU fold-0 test if higher ranks stall.

## Rank 4 — H18-B: EBM/GAM hierarchical additive members (Axis B)

- Expected: weak and redundant vs the existing GBDT/TabPFN portfolio; S6E7
  showed additive structures add nothing once interactions are modeled. Plan:
  one cheap CPU fold-0 standalone + marginal test, then close unless surprising.

## Rank 5 — H18-F (advanced ranking) / H18-G (latent factors) / H18-D (original-only)

- F: generic pairwise ranking is closed; only a materially different mechanism
  would qualify — none identified. G: prior SSL32 failed; factor models on the
  13 ratings are smoothed copies of existing columns. D: original-only teacher
  OOF is 0.959064 standalone and its information already enters via ogte_*/ogs_*/
  teach_* blocks; no marginal pathway identified. All three: no wave-1 planned.

## Champion B (hedge) status

Unchanged: UNRESOLVED. Test-side diversity structurally collapses (all measured
members >= 0.997 correlated with champion). None of the ranked axes produces a
structurally independent model. Do not invent a hedge.

## Promotion gates (frozen, unchanged)

Admission: mean paired portfolio gain >= 1.5e-5, >= 2.5x paired SE, positive in
>= 4/5 primary folds. Submission: only on OOF gain >= +3e-5 (private-hedge rule)
and only through the full pre-flight gate. Max 4 uploads this cycle.
