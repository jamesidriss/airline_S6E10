# PHASE 19 — OpenCode handoff

**Date:** 2026-10-14 · **Champion A:** `v7_sol_tabpfn_internal2_aux10` = 0.961852923 OOF
(hash-verified byte-exact this session) · **Champion B:** UNRESOLVED ·
**Submissions:** 0 of 2 new used · **GPU:** idle (RTX 5070 Ti), 0 of 12h new spent.

Phase 18 closed all four of its hypotheses by frozen rules. Phase 19 opens with the
**shared-error Atlas** (H19-ATLAS) — an exact decomposition of *where* the champion's
residual lives — plus two audits that closed two tempting but dead branches
(neural/classical gap; zero-rating features).

## 1. The Atlas (Axis 19-A) — the campaign's new map

`scripts/phase19/build_shared_error_atlas.py` → `reports/phase19/shared_error_atlas.{json,md}`,
`shared_error_segments.parquet`, `per_row_burden.parquet`. Every integer identity, the
composition identity, and the AUC-mass decomposition are asserted **exact** (max error
< 1.1e-16). Recorded in `experiments/ledger.jsonl` as `p19_shared_error_atlas`.

**Composition (bit-exact):** the champion portfolio is
`sigmoid((logit(route) + classical()) / 2).astype(float32)` where `classical(k, 'sol_clean_aux10_primary', …)`
returns the float64 mean of the 10 aux members' logits for fold-k validation rows.
Verified max gap 0.0. The float32 aux10 bank round-trip fails `np.array_equal` (rejected
as the authority); an earlier 0.188 diff was an operator-precedence bug in a temp script.

**Counting convention (verified exact):** `AUC = 1 − (strict_wrong + 0.5·ties)/pairs`.
Cross-class ties exist (route ~1500/fold, port ~130, aux ~110), so `assert ties == 0` is
wrong. The monotone theorem (portfolio = sigmoid(mean-of-logits) is monotone in both
halves ⇒ shared-fail set ⊆ portfolio-fail set) is proven and empirically holds on all
5 folds; shared errors were located by exact Fenwick 2-D dominance sweeps, not max-rank.

**Where the residual lives (pooled over 1.208e11 cross-fold pairs):**

| region | pairs | AUC mass | share of fail |
|---|---|---|---|
| **shared** (both halves fail the same pair) | 3.932e9 | 0.032547 | **85.3 %** |
| route-only fail | 6.99e8 | 0.005937 | 15.6 % |
| aux-only fail | 7.04e8 | 0.006034 | 15.8 % |
| aggregation loss (≥1 half right, average wrong) | 6.55e8 | 0.005600 | 14.6 % |

Per-fold shared share is stable (85.1–85.6 %). Reading: **85 % of the residual is
shared** — unreachable by reweighting or averaging the two halves (both closed axes).
The only reachable levers are (a) a new inductive bias that is *right where both halves
are wrong*, and (b) the 15 % aggregation-loss pool (needs a strong AND decorrelated
third member — champion B, unresolved).

**The strongest new number — difficulty is covariate-locatable:** a cross-fitted 5-fold
LightGBM (inner 10 % ES from FIT rows only; outer-fold labels never used) predicts
"top-quintile shared-burden row" from the **285-feature rich view only** (no labels, no
model scores) at **AUC 0.9318** (folds 0.9310–0.9323, SE 0.0002; Spearman vs burden
0.825). The *satisfaction* context AUC of the same model is only **0.534** — it locates
*difficulty*, not the label. A trivial min-margin baseline is 0.074. Hard rows are
covariate-locatable without labels.

**But locating ≠ fixing:** the original-data teacher is *confidently wrong* on **27.3 %**
of top-burden rows vs **5.47 %** overall (**5.0× enrichment**). Hard rows deviate from
original-distribution structure — they are at least partly label noise / generator
artifacts, i.e. possibly irreducible. This is the tension H19-A must resolve: the
specialist must be *right* on shared-fail pairs, not merely trained on them.

**Segments:** `unique_rating_key` 254,260 rows (36 %) AUC 0.95032; Eco 0.90654;
Personal Travel 0.83431 (202k rows); Eco Plus 0.93357. Burden rates fairly uniform
(~0.032–0.035), so difficulty is row-level, not segment-level.

## 2. Two audits that closed two branches

**Neural/classical gap (§37 step 9):** the hypothesis "neural lacked the rich view" is
**DISPROVEN**. Corrected-protocol RealMLP on the full 285-feature view (numeric twins,
inner ES): 4 seeds 0.960789–0.960881; TabM 0.960693; corrected RealMLP f0 0.960758.
Historical RealMLP/TabM members (`z5_rm_ens32` 0.961017, `z5_tabm_e25` 0.960693) already
used the rich view. All neural is **~1e-3 below champion** (0.961853). H18-B: a weaker
member hurts the portfolio. Neural = reserve with a strong negative prior (H19-E).

**Zero-rating semantics (§37 step 10):** rating 0 means "service not offered", not
"worst" — `P(satisfied | wifi=0) = 0.887` vs `0.391` at rating 1; the 0-bucket breaks
1..5 monotonicity on 6/13 columns; zero shares match train/test. **But this is already
implemented** in `src/features/s6e10.py`: `na_*` zero indicators, `na_count`/`na_any`,
zero×segment crosses, zero-aware rating aggregates (`rate_cnt`, `rate_mean_pos`, …).
Not a new feature block. The public notebook's insight is real but absorbed.

## 3. Ranked hypotheses (registry: `research/phase19_hypothesis_registry.jsonl`)

| rank | id | hypothesis | status |
|---|---|---|---|
| 1 | **H19-A** | difficulty-targeted reweighted member; right on shared-fail pairs | EXECUTE (fold 0 first) |
| 2 | **H19-B** | aux10 deployment policy: bag-of-fold vs full-fit, pseudo-test | EXECUTE (CPU, pseudo-test standard) |
| 3 | **H19-C** | smoothed nearest-key on unique-key rows | EXECUTE (fold-0 screen only) |
| 4 | H19-D | aggregation-loss tiebreaker / champion B | reserve, no candidate |
| 5 | H19-E | neural reserve | reserve, do not spend GPU |

Full rationale: `research/phase19_research_queue.md`. Each of the top 3 is preregistered
(hash + commit before results) with a fold-0 falsifying experiment and a frozen decision
rule. **H19-A's leakage control:** the Atlas outer `pred_a` must NOT be reused as
training weights (its training folds include the eval fold → leak); recompute with an
inner cross-fit inside the FIT fold, mirroring TE.

## 4. Compute & submission budget

`research/phase19_compute_budget.json`. Ceilings: ≤12h new heavy GPU, ~30h CPU,
≤1 heavy GPU job at a time. Phase 19 top-3 are **CPU-only** (0h GPU planned; the 12h is
held in reserve). Submissions: at most 2 new, 0 used; pre-result manifest + hashes +
commit before upload; `python -m kaggle` only; only on OOF gain ≳ +3e-5 with a written
reason (H19-B is pseudo-test standard and cannot satisfy this — finalist-selection
evidence only).

## 5. Continuation conventions

- **Never leak.** Fold-safe features; inner ES from FIT rows only; immutable fold
  registries (hash-checked). A public method that leaks → document in
  `research/current_meta.md` and reject.
- **Never fit to the public leaderboard.** Paired public-LB noise ≈ ±2e-4; public LB is
  a measurement, not an objective.
- **Preregister** every serious experiment (hash + commit before results). Promotion
  gates frozen: mean paired ≥ 1.5e-5, ≥ 2.5× paired SE, ≥ 4/5 folds; prefer pooled ≥ 3e-5.
- **Closed branches stay closed** (list in the research queue). Zero-rating,
  conditional-density, residual-group-bias closed this cycle.
- Preserve Git history; append-only ledgers; secret-scan before push; no force-push.
- New namespace: `configs|scripts|src|reports|artifacts|tests/phase19/`.
- Test suite: `.\.venv\Scripts\python.exe tests\run_tests.py` (147 tests). Untracked `.py`
  files at repo root fail `test_no_source_files_missing_from_git` — clean up temp files.

## 6. File map (this session)

- `scripts/phase19/build_shared_error_atlas.py` — Atlas builder (Fenwick dominance,
  exact identities, difficulty predictor). **Completed, all asserts pass.**
- `reports/phase19/shared_error_atlas.{json,md}` — authoritative Atlas numbers.
- `reports/phase19/{shared_error_segments,per_row_burden}.parquet` — diagnostics.
- `research/phase19_hypothesis_registry.jsonl` — 5 preregistered hypotheses.
- `research/phase19_research_queue.md` — ranked queue + evidence citations.
- `research/phase19_compute_budget.json` — GPU/CPU/submission budgets.
- `experiments/ledger.jsonl` — appended `p19_shared_error_atlas` + segment-sha correction.
- `reports/phase19/experiment_scoreboard.{json,md}` — live scoreboard (below).

## 7. Scoreboard (initial)

See `reports/phase19/experiment_scoreboard.md`. Atlas complete; H19-A/B/C preregistered
and pending fold-0 execution.

## 8. Session update — H19-B closed (NEGATIVE) + foundation-model screen (all NEGATIVE)

**H19-B deployment policy — verdict `NEGATIVE` (frozen rules).** Question: does replacing
the certified aux10 inference policy (one full-FIT refit per role) with a **bag-of-five
inner-fold members** (equal mean-logit) improve the complete v7 ranking on pseudo-test
fold 4? Answer: **no — it uniformly hurts.** 10 roles × 5 bags = 50 member fits on 4/5-FIT
subsets; the certified full-FIT control reproduced **bit-exact** (identity gap 0.0).

- aux: control 0.9620929850 → bag5 0.9620533686 (**Δ −3.96e-5**)
- portfolio: control 0.9624956399 → bag5 0.9622173311 (**Δ −2.783e-4**, paired SE 1.747e-4,
  −1.59σ); **all 22 predeclared segments negative** (worst Personal Travel −1.13e-3);
  logit Pearson 0.999, mean |Δprob| 0.104.
- **Mechanism:** each bag member trains on only ~80 % of FIT (~447k rows), so averaging five
  under-trained members loses to one model fit on all 559k rows. **The certified
  one-full-FIT-refit-per-role policy is confirmed correct; no change to v7.** No promotion,
  no submission (pseudo-test standard cannot pass the OOF gate). Ledger
  `p19_h19b_bag5_deployment`; report `reports/phase19/h19b_bag5_f4.json`.
- **Eval-note:** the exact-pair identity assertion was relaxed from absolute `<1e-9` to
  relative `<1e-9`; at 4.83e9 pairs the `roc_auc` rank-sum and exact-credit paths differ by
  ~3e-13 relative (abs ~4.5e-7) — an internal sanity check, not a decision threshold.

**Foundation-model screen (fold 0, `H19-FM-SCREEN`, prereg `c467f9a`).** Installable
tabular foundation models given a shot as an 11th member on the champion raw-21+coarse10
view, 20k stratified fold-0-safe context, 50k fold-0 eval rows, equal-logit blend vs the
champion portfolio. Promising bar = portfolio Δ ≥ +5e-5 AND logit-corr < 0.995.

| model | AUC | port Δ | corr vs port | note |
|---|---|---|---|---|
| Causilo | 0.956684 | −1.01e-3 | 0.986 | causal-discovery + TabPFN |
| TabICL-v2 | 0.956343 | −1.40e-3 | 0.984 | `tabicl-classifier-v2` ckpt |
| xRFM | 0.930634 | −9.29e-3 | 0.824 | decorrelated but far too weak |
| TabDPT | (rerun `use_flash=False`) | — | — | flash-attn kernel unavailable on 5070 Ti |
| TabFM | (deferred→run) | — | — | needs jax backend (installed jax CPU-only) |

**Conclusion:** all tested models sit ~1e-3 to 9e-3 **below** the champion when blended —
none is promising. The champion route is TabPFN-v3.5 **full-context** (559k rows); these
candidates were given only 20k context, so this is a screen, not a full-context head-to-head
— but the gap is large and consistent with the strong prior that the champion's full-context
TabPFN + aux10 portfolio is already near the achievable ceiling. **Champion A unchanged;
Champion B still UNRESOLVED; 0 submissions.**

**Environment (reversible):** installed `tabicl`, `tabdpt`, `causilo`, `tabfm`, `omegaconf`,
`absl-py`, `jaxtyping<0.3`, `typeguard<3`, `jax 0.7` (CPU-only) via `--no-deps` /
version-pinned installs. Certified stack verified unchanged (numpy 2.4.6, torch 2.11+cu128,
sklearn 1.9.1, scipy 1.17.1, lightgbm 4.7.0). Heavy-GPU use still 0 h.
