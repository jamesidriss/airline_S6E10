# STATUS_p12.md — Phase 12 closed

## Verdict

**NO ADMISSION. No submission.** Native LightGBM categorical splits produce a small, real
standalone effect that is **worthless at the ensemble level** — exactly the pattern Phase 11R found
for native CatBoost counterparts, but now measured in the strongest family rather than the weakest.

The checkpoint question has a direct answer: **categorical handling changes random-split behaviour
without adding information the blend lacks.**

## The champion configuration (Q1)

Recovered from its authoritative runner, **not** from `src/models/gbdt.py`:

| field | value | source |
|---|---|---|
| member | `xt_xt_d127_s1` | ledger, OOF 0.961136 |
| view / scheme / seed | `full` / `primary` / **1 fixed on every fold** | `run_xt_zoo.py:36` |
| params | `_fit_lgbm_es` defaults verbatim: lr 0.02, num_leaves 127, min_child_samples 40, colsample 0.8, subsample 0.8, subsample_freq 1, reg_lambda 1.0, max_bin 255, n_estimators 6000, ES patience 300 | `run_views.py` |
| **plus** | `extra_trees=True` | `run_xt_zoo.py:36` |
| protocol | inner 10% carve of FIT rows for ES, train on the other 90%, predict the OUTER fold at `best_iteration` | `run_zoo.py::run_gbdt` |
| **also** | `inner_seed = k` (the fold index) | `run_xt_zoo.py:111` |

Feature matrix: **285 columns, 48 of them `te_`**. The raw categoricals are already global ordinal
integers, so declaring them categorical adds no column and duplicates nothing.

## Was it already tested? (Q2)

**No.** No `lgbm` ledger entry anywhere declares `categorical_feature`; neither
`src/features/s6e10.py` nor `src/features/view.py` emits such a flag. The only prior native-
categorical work was CatBoost (Phase 10B/11R) — a different library, different mechanism. Genuinely
new.

## Results (Q3, Q4, Q5, Q6)

**L0 reproduces the champion on all five folds: +0.0001, +0.0000, +0.0000, +0.0000, −0.0001e-5.**
Worst |Δ| = 0.0001e-5. The harness *is* the champion.

| arm | standalone Δ vs champion, per fold | mean | SE | t | pos | **marginal on v3** | mean | t | pos | corr w/ champ |
|---|---|---|---|---|---|---|---|---|---|---|
| **L1** META4 | +6.0, −1.1, +5.6, +4.5, **−25.4** | **−2.10e-5** | 5.96 | −0.35 | 3/5 | +0.01, −0.07, +0.10, +0.02, +0.02 | **+0.02e-5** | +0.67 | 4/5 | 0.9982–0.9993 |
| **L2** META4+irregular | +5.5, +5.6, +6.5, −0.7, −7.5 | **+1.89e-5** | 2.67 | +0.71 | 3/5 | −0.03, −0.02, +0.15, +0.12, −0.04 | **+0.04e-5** | +0.88 | 2/5 | 0.9989–0.9992 |

- **L3 was NOT run** — it was gated on L1/L2 looking promising and neither was.
- **L2 categorical splits per fold: 7,312–11,716.** L1: 3,513–9,472. The treatment is emphatically
  not inert.

**L2's marginal of +0.04e-5 is 37× short of the +1.5e-5 admission gate.**

## Q7: new information, or just different random splits?

**Different random splits.** Both arms create thousands of categorical splits yet produce predictions
**0.998–0.999 logit-correlated** with the member they replace, and swapping them into the exact v3
slot moves the blend by +0.02 to +0.04e-5 — inside the fold noise.

Mechanistic reading: `cat_smooth=10` / `cat_l2=10` make a LightGBM categorical split a *shrunk target
statistic*, i.e. the same smoothing family the 48 `te_` columns already encode from the same columns.
The `extra_trees` family already randomises its split search aggressively, so there is little for an
ordered categorical search to add that survives averaging over 59 members.

One asymmetry worth recording: **META4's 2–3 levels sit at or below `max_cat_to_onehot=4`**, so those
columns get *one-hot*, not subset, splits. L1 is therefore not a strict subset of L2's test.

## Three bugs, all mine, all with evidence

1. **A split detector that reported a working treatment as inert.** Counting `"||"` returned 0 for
   every configuration; LightGBM's `dump_model()` uses `decision_type "=="` with an integer
   threshold, and only the saved model text uses pipe lists. This would have closed Phase 12 before
   it started and been indistinguishable from a real negative.
2. **Two fold-index coincidences, both invisible on fold 0.** The champion uses a *fixed* seed and
   `inner_seed=k`. I used `seed+k` and `inner_seed=0` — both coincide on fold 0 (`1+0==1`,
   `k=0`) and diverge after. L0 showed +0.0001e-5 on fold 0 but +4.50e-5 / −6.57e-5 on folds 1–2.
   Fixing the seed alone still left the mismatch; that residue is what exposed `inner_seed`.
   **A fold-0-only screen cannot detect either.** The reproduction check now runs on every fold.
3. **A `te_` reference compared across folds.** `te_` columns are refitted per fold, so the check
   aborted with "te_ columns changed vs control" on a correct run. Now keyed by fold — which makes it
   assert the property that actually matters: every arm of a fold got identical features.

## Methodological finding worth carrying forward

**A three-fold screen overstated the effect by 3.1×.** Folds 0–2 gave L2 +5.87e-5; five folds gave
+1.89e-5. L1 went the other way (+3.47e-5 → −2.10e-5). Fold-level spread here is several times the
mean, so **five folds is the minimum for a standalone claim**. This is the second instance of the
same failure mode as Phase 11R's gate B: a criterion satisfied for a reason that carries no weight.

## Runtime and compute (Q8)

~35–77 s per arm per fold on 8 threads; 5 folds × 3 arms ≈ 12 min total. The entire hypothesis was
affordable — the reason it failed is mechanism, not cost.

## Git, tests (Q9, Q10)

- HEAD pushed; see `git log`.
- `tests/test_phase12_lgb_cat.py`: **224/224**, pinning the champion config against `run_views.py`
  verbatim, asserting no prior lgbm run used categoricals, handling-only invariance, code safety,
  the ES carve, the detector against ground truth, `extra_trees`/categorical coexistence, and the
  fixed-seed + `inner_seed=k` conventions.
- Ledger at **168** append-only lines including both withdrawn claims.

## Submission (Q11)

**No.** Nothing cleared the gate. `v3_final` (OOF 0.961509, public 0.960980) and `v4_fulldata`
(public 0.961000) remain the immutable finalists. 4 submissions used; none in 2026-10-07.

## Strongest remaining direction

**Auxiliary-task expected-value features** — two independent public sources, absent from our view:

- `goodpjw2008` trains 16 auxiliary models, each predicting **one rating from the other 20 columns**
  with no satisfaction label, and uses `Σ_k k·P(k)` per rating as 13 features.
  **+13e-5** one XGBoost, **+9e-5** one LightGBM, **+5.9e-5** on the stack with 5/5 folds up.
- `sachith7` independently uses "aux expected ratings", computed once over train+test.
- We have 48 `te_` + 54 `ogte_*` + 2 `teach_*`, but our teacher predicts *satisfaction* from the
  original dataset — a different mechanism. **No per-rating auxiliary target exists in our view.**

It is label-free (so fold-safe with no new infrastructure), reported an order of magnitude above
anything Phase 11 or 12 produced, and `goodpjw2008` narrows the target precisely: the *surprise*
features from those same models do nothing, only the plain expected value helps.