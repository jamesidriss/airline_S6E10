# Phase 23 — RESUME STATE

Updated 2026-10-19 ~23:00 UTC (working session 2026-10-19/20).

## What this phase is doing

Certify or kill the **native raw-context 8-estimator TabPFN-3.5 lead** — the strongest
unresolved Phase-22 signal — using **our own folds and our own checkpoint**, executed on
**free Kaggle GPU quota only** (the local RTX 5070 Ti is under a hardware stop and its 22 h
budget cannot cover the run).

- Champion stays protected: `v7_sol_tabpfn_internal2_aux10`, OOF `0.9618529228157235`,
  ref `57090508`. Nothing here modifies it.
- Frozen prereg: `configs/phase23/raw8_kaggle_prereg.json`
  sha256 `b9aec0ce67aa0cef33f6f8e9a546d85af8f5a9aa7aebc6667e038a65a87f8fe0`.
- Frozen evaluator: `scripts/phase23/compare_raw8.py` (same statistics/partitions as
  `scripts/phase22/compare.py`; refuses to overwrite; zero submissions).
- Kernel: `research/phase23/kaggle/raw8/main.py` + `kernel-metadata.json`
  (script kernel, GPU, internet, competition source `playground-series-s6e10`).

## The hypothesis in one paragraph

The external `tabpfn35` vector (cdeotte, CC0) is RAW 21 columns, full labelled context,
8 internal estimators, mixed precision; on our labels it gains **+2.99e-5 pooled** over v7 at
w=0.10 (5/5 folds, paired SE 4.97e-6; w=0.25: +4.22e-5). That vector is uncertifiable
(producer folds seed 42, unaudited pipeline). If the effect is a property of the *recipe*,
our own raw8 member (folds 20261010, pinned checkpoint `Prior-Labs/tabpfn_3_5@06bf2ba3`,
sha `ece4d67e…`, `random_state=1201`, `n_estimators=8`) should reproduce a mean paired fold
delta of the same order at w=0.10.

## Current state (as committed)

1. `raw8_expected_hashes.json` — data/fold reference hashes the kernel asserts
   (ids `b8941ff7…`, y `5ab86061…`, folds `b41f1d7c…`, X-round6 `b2a49d79…`, XT `a7004d08…`;
   139 927 rows/fold; pos-rate 0.44357272).
2. Kaggle kernel **probe** version pushed (PROBE=True):
   checks T4×2, internet/pip `tabpfn==9.1.0`, checkpoint sha, raw data hashes, then
   **stock vs ported vs lean-patched equivalence ≤ 2e-6**, then full-context (559 708 rows)
   member-0 fit + 8 192-row predict to measure per-unit timing/memory.
3. If probe passes → flip `PROBE=False` → production push: 40 units (5 folds × 8 members),
   2 GPU workers, deadline-aware, durable per-unit val/test logits, then in-kernel assembly
   of `raw8_oof.npy` / `raw8_test.npy` + fold AUCs.

## Budgets

| resource | state |
|---|---|
| local GPU | **blocked** (driver 610.88 vs phase-17 stable 591.86; 6 `nvlddmkm` 153 faults Oct 17–19). Budget 15.678–24.043 h of 22 h cap → do not use. |
| Kaggle GPU | free 30 h/week; raw8 ≈ 19 GPU-h over one ≤ 12 h session (internal stop 10.6 h). |
| submissions | 10/10 used 2026-10-19 (inventory batch, final for that batch). Nothing in this phase submits. |
| local CPU | Track-C experiments while Kaggle runs. |

## Resume checklist

```powershell
# 0) confirm nothing drifted
git log --oneline -3
.\.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'.');from src.common import file_sha256;from pathlib import Path;print(file_sha256(Path('configs/phase23/raw8_kaggle_prereg.json')))"

# 1) kernel status / logs
.\.venv\Scripts\python.exe -m kaggle kernels status jamesripoll/s6e10-phase23-raw8-tabpfn35

# 2) when the run is COMPLETE: download outputs
.\.venv\Scripts\python.exe -m kaggle kernels output jamesripoll/s6e10-phase23-raw8-tabpfn35 -p "$env:TEMP\kagpull\raw8"
#    expect: probe_summary.json (probe) or units/* + raw8_oof.npy + raw8_test.npy + raw8_result.json (production)

# 3) ingest into artifacts/phase23/raw8/ (raw8_oof.npy, raw8_test.npy, units/*) then evaluate
.\.venv\Scripts\python.exe scripts\phase23\compare_raw8.py
#    -> reports/phase23/raw8_comparison.json  (gate: >=1.5e-5 mean, >=2.5*SE, 5/5 folds, holm<=.05, private80 p05>=0)

# 4) record in experiments/ledger.jsonl, commit, push. No submission without a separate committed decision.
```

## Decision tree

- **Gate PASS** → record `NOMINATE_MATCHED_SHADOW`; next step is a matched shadow (seed 777001)
  Kaggle run with its own prereg (≈2nd session) — still zero submissions.
- **Gate FAIL** → `NO_ADMISSION` in the ledger; the raw-context-8 lead is closed at certified
  level; external diagnostic stays ineligible. Champion untouched.
- **Partial run** (session died / members missing) → `OUTCOME_PARTIAL`; finish the missing units
  in a second Kaggle session (outputs of session 1 re-uploaded as a private dataset if needed).
- **GPU compute request** (only if raw8 is borderline and a richer local variant becomes
  decisive): driver rollback to 591.86 + budget clarification to be requested from the user with
  the incident report; do not touch the machine now.

## Reference numbers

| item | value |
|---|---|
| v7 OOF | 0.9618529228157235 (sha `1981745c…`) |
| external tabpfn35 standalone OOF (our labels) | 0.9611450967635963 |
| external tabpfn35 w0.10 delta | +2.99448370622013e-05 (5/5 folds) |
| external tabpfn35 w0.25 delta | +4.223376716561145e-05 (5/5 folds) |
| certified route2 portfolio (fold 0, 1→2→4) | 0.9611324 → 0.9614577 → 0.9615111 |
| our internal4 portfolio 2→4 gain (fold 0) | +3.818e-6 |
| admission gate (AGENTS.md) | ≥1.5e-5 mean, ≥2.5×paired SE, ≥4/5 folds |
| public-LB noise | ±0.0002 → submit only on OOF gain ≳ +0.0003 |
