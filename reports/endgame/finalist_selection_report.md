# Finalist selection report — private-first endgame

**Decision date:** 2026-10-19 UTC · **Deadline:** 2026-10-31T23:59Z · **Quota used:** 0/10 today
**Policy:** `research/submission_policy_amendment_20261017.md` (PATH A / PATH B, standing authorization)

## 1. Decision

**Select exactly one submission for private judging: Kaggle ref `57090508`
(`v7_sol_tabpfn_internal2_aux10`). Slot 2 remains empty.**

No new candidate was uploaded in this endgame because none qualified under either
policy path. The second slot is deliberately **not** filled: the mandate's own
criterion for a second finalist is a *scientifically defensible* complement, and
every candidate measured this endgame failed its frozen gate before the slot
question could even arise.

## 2. Champion A — certified

| property | value |
|---|---|
| model | TabPFN-3.5 internal-2 route member ⟂ 10-member aux10 (6 LightGBM + 3 XGBoost + 1 CatBoost), frozen equal-logit blend |
| OOF (primary 5-fold) | **0.9618529228157235** |
| per-fold AUC | 0.961828 / 0.961989 / 0.961891 / 0.961099 / 0.962496 |
| public LB (observation) | 0.96146 (paired public noise floor ±2e-4) |
| Kaggle ref | 57090508 |
| test-pred SHA256 | `2b744d0df36a0b1143a92e1db18661ccd94fa881490784308624aaa0a0fc95af` |
| CSV SHA256 | `fa4947fe9fc381fe85b105c1c489375327f7bb2321e4ff7f0d9ffc7cd7c9c0bd` |
| clean-code reproduction | **BITWISE_PARITY** — all array hashes equal, max\|Δ\| = 0, ΔAUC = 0 |
| blend stability (H94) | no saved variant survives an equal-logit blend with it → champion locked unblended |

## 3. Why no PATH A candidate

PATH A requires ≥ +3e-4 OOF (or ≥ +1.5e-5 with 5/5 folds and t ≥ 2.5) against the
current submission. Every candidate measured this endgame was **screened at fold-0
under its own preregistered frozen gate first**, and none passed:

| candidate | gate metric | result |
|---|---|---|
| OBJ37 supcon | port_delta vs route_ref | −6.61e-4 FAIL |
| OBJ38 auc_surrogate | port_delta | −7.72e-4 FAIL |
| OBJ39 hard_mining | port_delta | −2.19e-4 FAIL (closest of the batch) |
| OBJ40 gce | port_delta | −2.12e-3 FAIL |
| OBJ41 ema | port_delta | −4.34e-4 FAIL (best standalone: +5.65e-4 vs its own control) |
| TabPFN-2.6 (row 13) | blend Δ vs true v7 | −1.051e-3, CI [−1.29e-3, −8.18e-4] |
| TabPFN-3 (row 13, full context) | blend Δ vs true v7 | −6.24e-4, CI [−7.87e-4, −4.47e-4] |

None reached the point where a 5-fold OOF would have been justified, so no
promotion was purchased and no submission quota was spent.

## 4. Why no PATH B candidate

PATH B needs OOF within −2e-4 of champion A **and** structural decorrelation
(logit corr < 0.99 or a distinct method family) **and** a written private-first case.

* Every candidate above is 3.6e-3 to 5.2e-3 below champion A — far outside the −2e-4 window.
* The only submission inside the window is **v6** (OOF 0.961766, −8.7e-5 vs v7). It fails
  the decorrelation clause: it is the same TabPFN + aux10 family with a single internal
  context, so its logit correlation with v7 is ≈0.999+. It was already submitted
  (ref 56951906) and scored **0.96144 < v7's 0.96146** — consistent with, not contrary to,
  the OOF ordering.
* A weak-but-decorrelated model is explicitly *not* an automatic hedge: a −4.9e-3 member
  dilutes a strong ensemble rather than insuring it. That is exactly what the hard_mining,
  gce, v2.6 and v3 measurements show (all decorrelated, all blend-negative).

## 5. Private-first robustness evidence for the selection

* **Covariate drift train↔test: negligible.** Max standardized mean difference **0.0033**,
  max categorical total-variation **0.0029** — train and test come from the same generator.
* **Score behaviour transfers.** OOF mean 0.444194 vs test mean 0.444115 (Δ 8e-5);
  fraction > 0.5: 0.433569 vs 0.433876.
* **Segment coverage matches** to within **0.0015** in share for every audited segment.
* **Member agreement on test** (route vs aux10 Spearman 0.977) is in the same range as
  OOF, so the equal-logit geometry behaves consistently off-distribution.
* **Seed stability** (H91): seed variance 1.8e-9…7.2e-9 ≪ subset variance 6.0e-7…6.3e-7 —
  the pipeline is not a seed lottery; fold movement is data composition.
* **Bootstrap context:** v7's own OOF AUC is 0.961854 ± 2.5e-4 (95% CI
  [0.961349, 0.962325]). The stretch target 0.962740 sits at the extreme upper tail of
  that distribution — consistent with "aspirational, not achievable by small gains".

Conclusion: because train and test are the same population, **OOF fidelity is the
correct selection criterion**, and champion A is the OOF-optimal candidate that also
reproduces bitwise and passes a strict stability audit.

## 6. Action required from the user (Kaggle UI)

The Kaggle python API exposes `competition_submit` (upload) and `competition_submissions`
(list) but **no final-selection method**; choosing which submission is judged privately is
a web-UI action:

1. Open the competition's **Submit** page.
2. Select **submission ref 57090508** (`v7_sol_tabpfn_internal2_aux10.csv`) for private judging.
3. Leave the second slot empty (or select any other submission only if a *new* qualified
   candidate appears — none has).
4. Confirm before **2026-10-31 23:59 UTC**.

No claim is made here about the private result; public feedback remains a measurement only.

## 7. What would change this decision

A future candidate would need: a frozen-screen pass, a full 5-fold OOF beating 0.961853 by
≥ +1.5e-5 with t ≥ 2.5 and 4/5 folds positive, an independently recomputed AUC from stored
arrays, a certified test vector with SHA, and — for slot 2 — structural decorrelation from
champion A that survives a private-first rationale. None of the 96 catalogue rows, the five
objective variants, or the two additional checkpoints meet that bar today.