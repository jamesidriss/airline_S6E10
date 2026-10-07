# AGENTS.md — working agreement for this repository

## Mission
Win (or finish as high as possible) Kaggle Playground Series S6E10 by **robust private-LB**
performance. Compute cost is secondary; scientific soundness is not.

## Non-negotiables

1. **Never leak.** Every target-dependent transform is fitted inside the training fold.
   Automated checks live in `tests/test_pipeline.py`:
   `test_fold_safe_te_never_sees_apply_rows`, `test_crossfit_row_never_sees_own_label`,
   `test_external_features_use_no_competition_label`, `test_original_overlap_audit_is_small`.
   If a public method leaks, document it in `research/current_meta.md` and reject it.
2. **Never fit to the public leaderboard.** S6E9's runner-up measured
   Δprivate ≈ +16.7e-5 − 0.88·Δpublic (r = −0.97) after tilting to the public board.
   The public LB is a *measurement*, not an objective.
   **Measured noise floor:** submission #2 gained +0.000165 OOF over #1 and moved the board by
   −0.00001. Paired public-LB noise is ≈ **±0.0002** (public split = 59,969 rows ⇒ standalone
   AUC SE ≈ 0.0015; paired SE ≈ 0.0002 at ρ ≈ 0.995). **Only submit on an OOF gain ≳ +0.0003.**
3. **Never spend a submission without a reason.** 10/day is a budget, not a target. Default
   0–3/day. `src/submission/kaggle_io.py` tracks usage and hard-refuses past the cap.
4. **Never commit secrets or bulk data.** `.gitignore` excludes `kaggle.json`, `.env`,
   `data/raw/`, `data/original/`, `artifacts/`, predictions. `data/MANIFEST.md` records
   checksums and re-download commands instead.
5. **Every claim needs a number.** Record experiments in `experiments/ledger.jsonl` with the
   paired fold deltas, correlation with the champion, and a verdict.

## Loop
`RESEARCH → HYPOTHESIS → IMPLEMENT → TEST → OOF → COMPARE → RECORD → DECIDE → COMMIT → PUSH`

## Validation contract
- Discovery CV: `primary` = StratifiedKFold(5, shuffle, seed=20261010), immutable, registered
  on disk in `data/cache/folds/` and hash-checked. **Never regenerate.**
- Confirmation: `shadow` = StratifiedKFold(5, seed=777001). Use sparingly, never to tune.
- Finalists: `block10` = 10 folds, seed 20261010.
- `blocked_id_scheme()` keeps contiguous id ranges inside a fold; use it to test whether any
  id-derived feature survives a generator-batch-aware split.
- Early stopping uses an **inner 10 % split carved from the FIT rows**, never the eval fold.
  Keep it that way so OOF stays comparable with itself.

## Layout
```
src/common.py            paths, hashing, seeding, data loading
src/validation/folds.py  immutable fold registry
src/validation/compare.py paired deltas, bootstrap, correlations
src/features/s6e10.py    feature blocks (static / transductive / external / fold)
src/features/view.py     named views = bundles of blocks; fold-safe assembly
src/models/gbdt.py       lgbm / xgb / catboost / extratrees / histgb
src/models/realmlp.py    RealMLP + TabM with numeric->categorical twins
src/ensemble/lab.py      blend geometry, nested stacks, admission gate
src/submission/          pre-flight gate, store, kaggle I/O + daily cap
experiments_ledger.py    append-only experiment log + champion pointer
scripts/                 runners (run_views, run_models, run_batch, make_submission, …)
tests/run_tests.py       15 tests, dependency-free runner
```

## Danger zone: `extra_trees` is an interaction, not a free win
`extra_trees=True` is our largest single-model lever (+2.3e-4) **but only on a rich feature
view**, verified on the independent `shadow` folds:

| view | n_feat | deterministic | extra_trees | Δ |
|---|---|---|---|---|
| `full` | 285 | 0.960868 | **0.961143** | **+0.000275** |
| `raw_ext` | 75 | 0.960067 | 0.958506 | −0.001561 |
| `raw` | 21 | 0.958959 | 0.953388 | −0.005571 |

Randomly chosen features *and* thresholds need many candidate columns to land on a good split.
**Never apply the flag to a minimal view.** Every `extra_trees` member in the ensemble is on a
rich view; a future refactor that trims features must re-check this.

## Feature-block provenance discipline
| prefix | meaning | fold-safe? |
|---|---|---|
| *(raw)* | the 21 raw columns | yes |
| `*__cat` | exact-value categorical twin | yes (label-free) |
| `cnt_*`, `fdm_*`, `fdb_*`, `*_m*`, `*_d*`, `*_len`, `na_*`, `rate_*`, `ix_*` | transductive, label-free, fitted on train+test | yes — no label involved |
| `tok*` | GPT-2 BPE token keys | yes (deterministic) |
| `ogte_*`, `ogs_*`, `teach_*` | built from **original-dataset labels only** | yes — no competition label involved |
| `te_*` | fold-safe target encoding | **must** be recomputed per fold with inner cross-fitting |
| `aux_*` | covariate-rating predictive distributions and signatures; no satisfaction input | **must** use inner OOS probabilities for FIT and outer-FIT-only models for applied rows |
| `ssl_*` | masked reconstruction of the 21 covariates on train+test | yes — no satisfaction label or id accepted by the encoder |

Any new feature must state which row of this table it belongs to.

## Admission gate
A new ensemble member is admitted only if, against the current stack:
mean paired fold gain ≥ **1.5e-5** AUC, ≥ **2.5 × paired fold SE**, and positive in **≥ 4 of 5**
folds. (The rule the S6E9 runner-up used; S6E10's rank-1 → rank-45 spread is only 3e-4 wide.)

## Commands
```powershell
.\.venv\Scripts\python.exe tests\run_tests.py                 # tests
.\.venv\Scripts\python.exe scripts\run_views.py --views full --models lgbm,xgb,cat --folds primary --tag prod5 --save-test
.\.venv\Scripts\python.exe scripts\run_models.py --view full --model realmlp --folds primary --tag prod5 --epochs 6 --ens 8 --save-test
.\.venv\Scripts\python.exe scripts\run_batch.py --plan core   # batch of diversity jobs
.\.venv\Scripts\python.exe scripts\make_submission.py --members a,b,c --name v2 --kind equal_logit
.\.venv\Scripts\python.exe -c "import sys;sys.path.insert(0,'.');from src.submission.kaggle_io import submit;from pathlib import Path;submit(Path('submissions/v2.csv'),'desc')"
```

## Final-submission strategy
- Up to **2** final submissions may be chosen for private judging.
- Pick by robustness and complementary methodology, not by the two highest public scores.
- Reproduce each finalist from clean code before selecting.
- Near the deadline, re-check the current selection rules on the competition Rules page.
