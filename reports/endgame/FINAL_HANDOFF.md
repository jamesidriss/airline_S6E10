# FINAL_HANDOFF — Kaggle S6E10 (Playground Series Season 6 Episode 10)

**Date:** 2026-10-19 UTC · **Deadline:** 2026-10-31T23:59Z (≈12 days remaining)
**Repository:** `C:\Users\James\airline_S6E10` · **Remote:** `jamesidriss/airline_S6E10`
**Status:** ENDGAME COMPLETE — champion preserved, one finalist certified, no
unvalidated candidate uploaded.

## 1. Outcome statement (one of the four permitted forms)

> **(3) No new candidate passed admission; the existing champion v7 was preserved and the
> competition endgame was completed.** Champion A reproduces bitwise from clean code,
> is OOF-optimal among every candidate ever measured in this repository, and is already
> on the Kaggle board as ref **57090508**. No submission was spent this endgame because
> nothing qualified — not because nothing was tried.

No claim is made about the private result. The private leaderboard has not published.

## 2. The one action you must take (Kaggle UI)

1. Open the competition **Submit** page.
2. Select **ref 57090508** (`v7_sol_tabpfn_internal2_aux10.csv`) for private judging.
3. Leave the second slot empty — no qualified complement exists.
4. Confirm before 2026-10-31 23:59 UTC.

The Kaggle API has **no** final-selection method (`competition_submit` /
`competition_submissions` only), so this cannot be automated.

## 3. Champion (immutable, certified)

| property | value |
|---|---|
| experiment id | `v7_sol_tabpfn_internal2_aux10` |
| OOF (primary 5-fold) | 0.9618529228157235 |
| per-fold | 0.961828 / 0.961989 / 0.961891 / 0.961099 / 0.962496 |
| public (observation only) | 0.96146 · Kaggle ref 57090508 |
| test-pred SHA256 | `2b744d0df36a0b1143a92e1db18661ccd94fa881490784308624aaa0a0fc95af` |
| CSV SHA256 | `fa4947fe9fc381fe85b105c1c489375327f7bb2321e4ff7f0d9ffc7cd7c9c0bd` |
| reproduction | BITWISE_PARITY (all hashes equal, ΔAUC = 0) |
| in prediction store | yes (OOF + test, `artifacts/predictions/v7_*`) |

## 4. What this endgame executed (all preregistered before results)

| work | outcome |
|---|---|
| OBJ37 supcon | CLOSED_NEGATIVE — +1.22e-4 vs BCE control; gate −6.61e-4 |
| OBJ38 auc_surrogate | CLOSED_NEGATIVE — −1.42e-4 vs control; gate −7.72e-4 |
| OBJ39 hard_mining | CLOSED_NEGATIVE — most decorrelated (ρ=0.891), gate −2.19e-4 |
| OBJ40 gce | CLOSED_NEGATIVE — gate −2.12e-3 |
| OBJ41 ema | CLOSED_NEGATIVE — best standalone of the batch (+5.65e-4 vs control), gate −4.34e-4 |
| OBJ37–41 batch verdict | **hypothesis falsified** — the neural family's weakness is the representation ceiling, not the BCE objective |
| v7 clean-code reproduction | **BITWISE_PARITY** (finalist requirement satisfied) |
| TabPFN-2.6 screen (row 13) | CLOSE — blend −1.051e-3, CI entirely < 0 |
| TabPFN-3 screen (row 13) | CLOSE — blend −7.967e-4 (100k ctx) and −6.243e-4 (**full** ctx), CIs entirely < 0 |
| interpretation controls | context effect −1.91e-3 vs checkpoint effect −2.94e-3 ⇒ checkpoint dominates; row 13 closed unconfounded |

## 5. Catalogue and process state

* 96 rows dispositioned, **0 NOT_STARTED, 0 PROMISING** (ledger: 556 append-only entries).
* Fidelity accounting (honest): F5 = 2, F3 = 2, F2 = 59, F1 = 10, **F0 = 23 (never trained a model)**.
* Test suite: **234 passed, 0 failed** (231 baseline + 3 new budget-guard regression tests).
* GPU: **14.54 h consumed of the 22 h authorized**; 7.46 h deliberately unspent — every
  promotion path (internal-4, seed-averaged route, aux10 upgrades) costs 5.6–11.2 h for a
  candidate that no measured evidence supports.
* Kaggle: 7 submissions on record (all reconciled with the remote); **0 uploaded this
  endgame**; quota 0/10 used; no duplicate or unknown submissions.

## 6. Deliverables in `reports/endgame/`

| file | content |
|---|---|
| `phase20_final_experiment_report.md` | full campaign report (execution accounting, structural findings, all families) |
| `obj37_41_results.json` | verified per-arm results incl. recomputed AUCs and SHA checks |
| `champion_reproducibility_report.md` / `.json` | bitwise reproduction evidence |
| `candidate_comparison.json` | private-first comparator (paired gate arithmetic, bootstraps, segments, drift) |
| `private_robustness_report.md` / `.json` | transfer and stability evidence |
| `finalist_selection_report.md` | the PATH A/B decision and the UI action |
| `kaggle_submission_status.json` | remote reconciliation, quota, finalist state |
| `gpu_budget_22h_final.json` | budget amendment, per-run GPU accounting, stop conditions |
| `FINAL_HANDOFF.md` | this file |

## 7. Honest limitations

* The private result is unknown; every claim above is a validation-based estimate.
* Rows closed at F0 (23) or resource/license-blocked were **never measured** — they are
  not negative results.
* Two commit messages (`86ab9ff`, `850ab88`) transcribe a hard_mining/gce delta with the
  wrong magnitude; the ledger entries and result artifacts are authoritative and correct.
* TabPFN-3's internal multi-checkpoint ensemble was prereg'd as *not run* unless the
  single-context arm was competitive; it was not, so that idea remains unbudgeted.

## 8. If you want more compute spent

A defensible use of the remaining 7.46 h would be a preregistered **internal-4**
TabPFN-3.5 screen (≈1.1 h) to measure whether the champion's own geometry has headroom;
promotion would additionally need ~5.6 h OOF + ~5.6 h test inference, which the current
balance cannot cover. That path should only start if the budget is extended again.