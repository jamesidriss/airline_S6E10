# Phase 19 — experiment scoreboard

Live. Updated as each hypothesis resolves. Champion A: `v7_sol_tabpfn_internal2_aux10`
= 0.961852923 OOF (hash-verified). Stretch target 0.962100000 (+2.47e-4; exact 0.0002470771842765). Submissions
0 of 2 used. GPU 0 of 12h new spent.

| id | hypothesis | experiment | result | verdict |
|---|---|---|---|---|
| **H19-ATLAS** | where does the residual live? | exact Fenwick 2-D dominance over 1.208e11 pairs + cross-fitted difficulty predictor | shared fail **85.3 %**; difficulty AUC **0.9318** (SE 0.0002); teacher 5.0× enrichment on hard rows; unique-key 36 % @ 0.95032 | **COMPLETE** (diagnostic; ledger `p19_shared_error_atlas`) |
| **H19-A** | difficulty-targeted reweighted member is right on shared-fail pairs | fold 0: inner-cross-fit difficulty → reweighted LGBM vs baseline; count specialist-correct on shared-fail pairs | *pending* | PREREGISTERED |
| **H19-B** | aux10 bag-of-fold deployment beats full-fit on a pseudo-test | carve fold 4 holdout; 5 fold-fit vs 1 full-fit on folds 0-3; predict holdout | *pending* | PREREGISTERED (pseudo-test standard) |
| **H19-C** | smoothed nearest-key helps unique-key rows | fold 0: +nearest-key TE feature; AUC overall + unique-key subset | *pending* | PREREGISTERED (fold-0 screen) |
| **H19-D** | aggregation-loss tiebreaker / champion B | — | no candidate (all members ≥0.995 corr, weaker) | RESERVE, UNRESOLVED |
| **H19-E** | neural reserve | — | RealMLP/TabM ~1e-3 below champion; rich view already used | RESERVE, NEGATIVE PRIOR |

## Closed this cycle (do not reopen)

| branch | evidence | verdict |
|---|---|---|
| zero-rating / not-offered features | already in `src/features/s6e10.py` (`na_*`, `rate_*`, zero×segment) | ABSORBED, not new |
| conditional-density / generative ratio | Phase 7 `p7_class_conditional_generative`: standalone 0.931, all blend weights negative, logit-corr 0.87 | CLOSED |
| residual group-bias offsets | Phase 10: +2.00e-5 did not transfer to v3 | CLOSED |
| neural rich-view gap | audit: historical + corrected neural already on full view, all ~1e-3 below champion | DISPROVEN |

## Frozen gates (unchanged)
- Admission: mean paired fold gain ≥ 1.5e-5 AUC, ≥ 2.5× paired SE, ≥ 4/5 folds, vs
  current stack; prefer pooled ≥ 3e-5.
- Submission: ≤2 new, pre-result manifest + hashes + commit, `python -m kaggle` only,
  only on OOF gain ≳ +3e-5 with a written reason.
