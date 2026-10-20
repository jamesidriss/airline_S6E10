# Phase 23 — raw8 probe result and resource verdict

Updated 2026-10-20. Zero submissions. Champion untouched.

## What was run

Probe v2 of `research/phase23/kaggle/raw8/main.py` on the free Kaggle kernel
`jamesripoll/s6e10-phase23-raw8-tabpfn35` (2 × Tesla T4, 14.56 GiB usable per device,
tabpfn==9.1.0 from pip, internet on, competition source mounted). The probe is the frozen
gate of `configs/phase23/raw8_kaggle_prereg.json` (sha `b9aec0ce…`).

## Result A — the port is numerically sound on T4 (PASS)

| check | value | gate |
|---|---|---|
| stock `predict_proba` vs ported `prepare`+engine | max abs Δ = **1.1920928955078125e-07** | ≤ 2e-6 |
| ported vs ported + lean KV-build patch | max abs Δ = **0.0** | ≤ 2e-6 |
| checkpoint sha256 (materialised real copy) | `ece4d67e…` OK | must match |
| data + fold hashes | OK | must match |
| GPU | 2 × Tesla T4 14.6 GiB, fp16 autocast (no bf16) | T4×2 |

This is a durable asset: the ported, lean-patched TabPFN-3.5 path reproduces the stock
path exactly on the target hardware, and the checkpoint-materialisation fix from probe v1
works.

## Result B — the frozen member does not fit a T4 (hard blocker)

```
[raw8] B prepare on 559708 rows: 6s, members=8
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.14 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 1.82 GiB is free.
12.74 GiB is in use (12.41 GiB allocated by PyTorch).
   at build_member_engine -> _build_cache -> model(...) -> mlp_classification(train_emb)
```

Memory arithmetic for **one** full-context member (559,708 rows, int8 KV, hidden ≈ 2048):

| component | GiB |
|---|---|
| int8 KV cache for the full context | ≈ 12.4 |
| full-context build activation (`train_emb`, fp16) | 2.14 |
| weights (1.6 GB checkpoint, fp16) | ≈ 1.5 |
| **peak** | **≈ 14.9 – 16** |
| Tesla T4 usable | **14.56** |
| RTX 5070 Ti usable (certified local runs) | 15.9 |

The member is **~0.3–1 GiB short on a T4** and fits a 16 GB local card by ~1 GiB — which is
exactly consistent with the certified phase-17 route members having run locally.

### Why the obvious escapes do not work

| escape | verdict |
|---|---|
| `keep_cache_on_device=False` | **no.** tabpfn moves caches to CPU *after* building them (`stage_caches_on_cpu` / `cache.to(device)`), so the build peak is unchanged. |
| `device=['cuda:0','cuda:1']` | **no.** tabpfn's multi-device path parallelises **ensemble members** across devices ("Ensemble members are dispatched across available GPUs"); a single member's KV cache is never sharded. |
| `kv_cache_precision='fp8'` | **no.** 8-bit, same size as int8. |
| `memory_saving_mode=True` | already on; it bounds member batching, not the row dimension of the context pass. |
| row-chunked cache build | the context pass is not decomposable across row blocks (rows attend to each other), so this needs a deep custom patch with its own equivalence proof. |
| P100 (16 GB) | Kaggle script kernels expose T4×2 only; the accelerator cannot be selected. |
| local RTX 5070 Ti | under the hardware stop (driver 610.88, `nvlddmkm` Event 153). Not bypassed. |

## Result C — cost arithmetic had it run

Measured on the certified local route member (full 559,708-row context, same checkpoint):
`member_0_fit = 788.3 s`, `member_0_predict (139,927 val rows) = 225.2 s`, fold total
`2,035.9 s` for two members. Test prediction (299,844 rows) ≈ 2.14 × the val pass ≈ 482 s.

- per unit (fit + val + test) ≈ **1,495 s ≈ 25 min** at RTX-5070-Ti speed;
- 40 units ⇒ **≥ 16.6 GPU-h** at that speed, **≥ 25 GPU-h** at any plausible T4 factor (1.5–2.8×);
- quota available: **28.57 GPU-h** (refresh 2026-10-24, +30 h afterwards).

So even with the memory problem solved, the frozen 40-unit run would consume essentially
the whole remaining weekly quota (or require two sessions plus a private-dataset transfer of
the partial units between them).

## Result D — value versus the admission rule

The external `tabpfn35` diagnostic (8 estimators, raw geometry, foreign folds) adds
**+2.994e-5** to v7 at w=0.10 (5/5 folds, paired SE 4.97e-6). Our native member was expected
to reproduce that order of magnitude. AGENTS.md requires an OOF gain **≳ +3e-4** before a
submission is even arguable, i.e. **10× more than the best case of this campaign**.

Therefore, even a full PASS could not change the final submission. The campaign's remaining
value was scientific (closing the external-vector question), and its cost is ≥ 25 GPU-h plus
two long Kaggle sessions — behind a memory wall that the available hardware cannot cross.

## Verdict

- `raw8` primary protocol: **RESOURCE_BLOCKED_HARDWARE** — the frozen member (full context,
  int8 KV, `keep_cache_on_device=True`) exceeds a T4's usable memory by ~0.3–1 GiB; the only
  device that can run it is under a hardware stop.
- `H23-01` (native reproduction): **NOT EXECUTED — resource blocked**. Not a scientific
  refutation: the hypothesis remains open.
- `H23-02` (matched shadow confirmation): **NOT ACTIVATED** (no primary candidate).
- Champion `v7_sol_tabpfn_internal2_aux10` (ref 57090508) remains the private-LB answer;
  Finalist B analysis is in `reports/phase23/finalist_B_analysis.md`.
- Zero submissions spent. Quota used by the whole phase: **1.43 GPU-h**.

## What would unblock it (for the record, not planned)

1. A ≥ 16 GB accelerator on Kaggle (cannot be selected) or restoration of the local GPU with
   a cleared hardware stop.
2. Or a context-subsampled member (e.g. 300k rows, ≈ 6.6 GiB cache) — but that is a
   *different, weaker* hypothesis and would not reproduce the external full-context result.
3. Or a row-chunked cache build with its own equivalence proof — deep patch, high risk.

None of these is justified by a sub-`+3e-4` expected gain.
