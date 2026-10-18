# Champion v7 clean-code reproducibility report

**Prereg:** `P20-ENDGAME-V7-REPRODUCTION` (`configs/phase20/endgame_v7_reproduction_prereg.json`,
sha `0091f8381b126695b6c05323bafe060335d8436cd7824fd8981b18d3de901e8b`), committed before the run.
**Run:** `scripts/run_phase17_tabpfn.py --estimators 2 --fold 0 --tag phase17_B1_replication_20261019`
executed through the scheduler (job `JREP01`, fingerprint `48d71f1b5909171c`) under the 22 GPU-h cap.
**Result:** **BITWISE_PARITY** — verdict recorded in `reports/endgame/champion_reproducibility.json`.

## 1. What was reproduced

The certified v7 champion's TabPFN-3.5 component on primary fold 0: two internal
context estimators fitted on all 559,708 FIT rows (folds 1–4) and inference over the
139,927 fold-0 rows, using the committed phase-17 pipeline, the current `tabpfn`
library bytes, and the immutable fold registry.

## 2. Acceptance rule (frozen before the run)

* **Primary — bitwise**: every listed `arr_sha256` matches exactly.
* **Fallback (not needed)**: max|Δ| ≤ 1e-6 **and** Spearman ≥ 0.9999 **and** |ΔAUC| ≤ 1e-6.
* Anything outside: `REPRODUCTION_FAILED`, documented, champion untouched.

## 3. Result — every target matched bitwise

| array | certified SHA256 | reproduced | match |
|---|---|---|---|
| `prediction.npy` (route, fold 0) | `eda2772041b59987…` | `eda2772041b59987…` | ✅ |
| `member_0_raw_logits.npy` | `455aaf4b15c27927…` | `455aaf4b15c27927…` | ✅ |
| `member_1_raw_logits.npy` | `d93428a576ce77fa…` | `d93428a576ce77fa…` | ✅ |
| `apply_ids.npy` | `b8b6784bc1d8d4b9…` | `b8b6784bc1d8d4b9…` | ✅ |
| member config hashes (2) | `de5090d7…`, `6781d617…` | identical | ✅ |

* max |Δ| on route probabilities: **0.000e+00**
* AUC: certified **0.9614576551310231** → reproduced **0.9614576551310231** (Δ = 0)
* Runtime: 2021.1 s vs the original 2022.7 s (0.1 % difference)

**No numerical tolerance was required.** The fallback band was never invoked, and no
claim of "approximately equal" is made anywhere: this is exact bitwise parity.

## 4. Blend-geometry re-verification (secondary check)

`portfolio.npy` is an assembly artifact (equal-logit blend of the route member with the
10-member aux10 mean), not a runner output. Recomputing it in float64 from the ten
certified aux10 members plus the route vector reproduces the certified portfolio to
**max |Δ| = 2.98e-08** — float32 rounding of the stored artifact, well inside the
prereg's 1e-6 band. The blend geometry (equal-logit, equal weights, one TabPFN member
against the 10-member aux10 mean) is therefore independently confirmed.

## 5. Integrity of the immutable references

Verified unchanged before and after the run:

* champion OOF AUC **0.9618529228157235**, OOF array SHA `1981745c…`
* champion test prediction SHA **2b744d0df36a0b1143a92e1db18661ccd94fa881490784308624aaa0a0fc95af**
* submission CSV SHA **fa4947fe9fc381fe85b105c1c489375327f7bb2321e4ff7f0d9ffc7cd7c9c0bd**

The reproduction wrote only to new paths (`artifacts/phase17_B1_replication_20261019/`,
`reports/phase17_B1_replication_20261019_f0.json`). The certified artifacts were never
overwritten (the runner refuses to write to an existing tag path).

## 6. Conclusion

The AGENTS.md finalist requirement — *"reproduce each finalist from clean code before
selecting"* — is satisfied for champion v7 at the strictest possible standard: the
model is bitwise reproducible from committed source, identical dependency bytes, and
immutable data on this hardware. Champion A is certified as a finalist.