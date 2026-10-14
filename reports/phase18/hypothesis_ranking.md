# Phase 18 — ranked research hypotheses

Generated 2026-10-14. Evidence citations refer to `reports/phase18/*.json` and the
Phase 17 certification chain. Ranking criterion: expected operational v7 portfolio
gain × probability of surviving the admission gate, per GPU-hour. The stretch gap
to 0.962100 is 247e-5 OOF; no single axis below is claimed to close it.

## Rank 1 — H18-A: TabPFN context coverage 80% -> 90% (Axis A/H) — **CLOSED at wave 2 (2026-10-14)**

- **Verdict: mechanism REAL, operationally insufficient.** Matched design (both
  arms on identical block10 eval rows, only context size differs): route delta
  **+5.2976e-5** (block10 f0) and **+2.2209e-5** (block10 f1), 2/2 positive —
  the S6E9 deployment-coverage lesson transfers directionally to TabPFN.
  But mean +3.759e-5 failed the preregistered S3-launch rule (≥5.6e-5) by 1.5×,
  and the expected portfolio conversion (2.7e-6 – 1.0e-5 using the measured
  7.1–27% band) cannot reach the 1.5e-5 admission gate even optimistically.
  S3 (10 contexts, ~5.5h) and S4 test (~8h) never ran. GPU burned 2.6h, saved
  ~13.5h. Evidence: `reports/phase18/h18a_s1_f{0,1}_comparison.json`, closure
  record `phase18_h18a_close`.
- **S0 feasibility (kept for the record):** 90% context fits — peak 15.32 GB
  (< 16.6 GB abort, 17.09 GB hardware), fit 990s, full-test predict projects
  ~8 min/member.
- **Fold heterogeneity lesson:** f0 delta was 2.4× f1 — single-fold route
  deltas must not drive GPU commitments; the two-fold rule caught it.
- (Design history: control = fixed-seed 559,708-row subsample of the other 9
  folds; treatment = all 629,671; frozen v7 params, n_estimators=2, seed 1201;
  exact pair-credit identity verified in both comparisons.)

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

## Rank 3 — H18-C: ratings-tuple-key exact-value TE (Axis C) — **CLOSED at wave 1 (2026-10-14)**

- **Verdict: NEGATIVE / INSUFFICIENT at the admission level.** Wave 1 (all 10
  aux members, primary fold 0, prereg `configs/phase18/h18c_keyte_prereg.json`):
  portfolio fold-0 delta **+6.80e-6**, 4.4× below the predeclared +3.0e-5
  advance bar; aux10 +8.6e-6. All fidelity gates passed (certified feature
  hashes, library versions, EV gap ≤1e-10, bit-identical control replay).
- **Member deltas are unstable across families:** LGBM +16.5e-5 / +10.9e-5 /
  +8.6e-5 / −1.65e-5 / −0.3e-5 / −12.2e-5 (3 of 6 positive), XGB −5.3e-5 /
  −6.5e-5 / −21.1e-5 (0 of 3), Cat +1.5e-5. The mean-logit aggregation
  cancels the scatter — the feature helps some members and sabotages others.
- Consistent with the pre-existing negative evidence chain: all21 single-column
  TE −13.3e-5; key-TE standalone 0.818–0.868; unique-key rows (51%, prior-only
  fallback) carry the residual error. Evidence: `reports/phase18/h18c_f0.json`,
  closure record `phase18_h18c_close`.

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
