# Compute budget extension — 12 h → 22 h (prospective)

**Status:** ACTIVE from the timestamp below. This document records a user-authorized
increase of the local GPU research ceiling. It does **not** retroactively justify any
past decision, does **not** modify any safety guard, and does **not** authorize paid
cloud compute.

## 1. Authority

Direct user authorization in the active session (2026-10-19 UTC), verbatim intent:

> Increase the local GPU research budget from 12 hours to 22 hours in total.
> This is explicit authorization. DO NOT ASK AGAIN BEFORE SPENDING WITHIN THIS NEW
> AUTHORIZED LIMIT.

The authorization is a **total**, not an increment to be re-applied: the ceiling is
**22 project GPU-hours**, of which ~11.8 h were already consumed.

## 2. Numbers

| field | value |
|---|---|
| Previous cap (phase19_authorized) | **12 h** |
| New cap | **22 h** |
| Increment | **+10 h** |
| Documented consumption at reconciliation | **11.54 h** |
| Estimated total consumption carried forward | **~11.8 h** (11.54 documented + ~0.26 h unrecorded early-phase-20 runs, carried at the conservative estimate) |
| Remaining balance | **~10.2 h** |
| Effective | 2026-10-19 UTC |
| Scope | local research GPU only (RTX 5070 Ti, 16 GB) |

## 3. Consumption reconciliation (audit trail)

The budget window opened with the phase-20 program (`campaign_budget.json` →
`phase19_authorized.spent_at_phase20_start.gpu_heavy_h = 0.0`); phase-17 artifacts
predate it and are **not** charged to this cap.

| source | hours | evidence |
|---|---|---|
| S7 close documented tally | 9.1 | `research/phase20_exhaustive/campaign_state.json` → `gpu_hours_note` (itemized: FTT-batch1024 killed 1.0, FTT bounded 0.69, INET 0.01, dcnv2/factlog <0.01; rffsvm CPU) |
| `p20_obj34_finetune_tabpfn35_blocked` | 0.87 | ledger line 537 (`gpu_hours`) |
| `p20_obj37_41_parity_gate` | 0.70 | ledger line 541 (`gpu_hours`, runtime 2513.1 s) |
| `p20_flm12_tabpfn25_blocked` | 0.87 | ledger line 542 (`gpu_hours`) |
| **documented subtotal** | **11.54** | |
| unrecorded early-phase-20 GPU runs (e.g. FLM9 screen) | ~0.26 (estimate) | no per-run timing recorded; carried conservatively at the previous session's recount |
| **total carried** | **~11.8** | |

Reconciliation method: sum of `gpu_hours` fields in `experiments/ledger.jsonl` for
entries dated on/after 2026-10-17, plus the documented S7 tally, cross-checked against
`total_seconds`/`fit_predict_seconds` in `reports/**` for runs inside the window.

## 4. Allocation of the remaining ~10.2 h

| item | est. hours | note |
|---|---|---|
| OBJ37–41 objective arms (5 × ~0.70 h measured) | 3.50 | measured, not estimated: the `bce_control` parity arm took 2513 s |
| Clean-code v7 fold-0 reproduction | 0.57 | measured from the original B1 f0 run (2023 s) |
| reserve: finalist validation + full-test inference + submission | 3.00 | a full 5-fold test-inference path costs ~0.7 h/fold for TabPFN-class members |
| reserve: contingency / re-runs / end-of-run analysis | ~3.13 | held unspent until a qualified candidate exists |

**Promotion stop condition.** If an OBJ arm passes its frozen fold-0 screen, promotion
to a 5-fold OOF run costs ~3.5 h. It is permitted **only if** the promotion plus a
full test-inference path for any qualified candidate still fits inside the balance at
that moment. Otherwise the fold-0 result is recorded as a promising screen that was not
promoted for budget reasons — an honest, recorded outcome, never a silent drop.

## 5. What this does NOT authorize

1. Paid cloud GPU rental or paid API tokens.
2. Switching the OpenCode provider away from the configured Space Bunny route.
3. Exceeding **22 total** project GPU-hours (hard ceiling; exceeding it requires a new
   explicit extension request, documented, never a quiet overrun).
4. Disabling hardware safety: Windows GPU timeout protection stays on; memory-placement
   guards stay on unless explicitly disclosed; no disk offloading; no unbounded
   memory-mapped caches; no retry of known resource-infeasible configurations unchanged.
5. Deleting historical artifacts, rewriting the ledger, force-pushing, or resetting to
   an older commit.
6. Bypassing Kaggle quota guards (`src/submission/kaggle_io.py` cap is never edited).

## 6. Stop conditions

Execution halts and reports when any of these fires:

- remaining balance ≤ 0 (hard cap reached),
- repeated GPU driver faults (historical reference: `nvlddmkm` Event 153 on TabICL),
- a required credential is missing,
- a competition rule conflict appears,
- an official Kaggle action requires user interaction,
- the deadline (2026-10-31T23:59Z) requires the endgame calendar to take precedence.

## 7. Enforcement

`scripts/phase20/scheduler.py` carries `budget_check()`, called by `run` before any GPU
job: it reads `campaign_budget.json`, refuses to launch when the remaining balance is
≤ 0, and records the decision in the runtime ledger. A missing budget file does not
block execution (documented fail-open for sandboxed test runs); an exhausted or
over-cap balance always blocks.

---

*Recorded 2026-10-19 (UTC), prospective, append-only. History preserved: the 12 h cap
and its S7 accounting remain untouched in `campaign_budget.json`.*