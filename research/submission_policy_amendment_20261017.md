# Submission policy amendment — prospective, dated 2026-10-17

**Status:** ACTIVE, prospective. This document records a user-authorized policy change
that takes effect from its timestamp onward. It does **not** retroactively justify any
past submission decision, and it does **not** modify any code guard. Official Kaggle
limits always win over anything written here.

## 1. What changed and why

Historical policy (S6E9 lessons, AGENTS.md rule 2–3): submit only on an OOF gain
≳ **+0.0003** against the current submission; default 0–3 submissions/day; every upload
was to be confirmed with the user first.

On **2026-10-17** the user issued a standing authorization that supersedes the
"confirm each upload" step:

> "Si une soumission vaut le coup d'être envoyée ou qu'on pense gagner sur le private,
> alors soumets sur Kaggle ! Je veux absolument gagner."

Recorded in full, verbatim, as the authority for this amendment.

## 2. The amended rule

Autonomous upload is authorized **without asking the user each time** when the candidate
falls under **PATH A** or **PATH B** below and the full certification checklist (§4) has
been completed with a written `SUBMISSION_DECISION` record (§5) committed *before* the
upload.

| | PATH A — stronger champion | PATH B — credible private hedge |
|---|---|---|
| Intent | Replace/augment the current champion on robustness | Hold complementary structure that could win the *private* 80% split |
| Evidence required | Full 5-fold primary OOF (or confirmation-fold replication) beating champion A OOF by ≥ **+3e-4**, or ≥ **+1.5e-5 with 5/5 folds positive and t ≥ 2.5** against the matched control | OOF within **−2e-4** of champion A *and* a documented decorrelation argument (logit corr with champion A **< 0.99**, or a structurally distinct method family) *and* a written PATH-B rationale for why the private split could rank it differently |
| What we claim | Nothing about public LB | Nothing about public LB |
| Max per day | Shared 10/day official cap; self-imposed ≤ 5/day for PATH B | same |

Notes:

- The historical **+0.0003 OOF** bar remains the default for *routine* candidates. PATH B
  is the only documented exception and it exists because the private split is 80% of the
  test rows and our two finalist slots reward *complementary* methods, not just the two
  highest public scores (AGENTS.md final-submission strategy).
- The public LB is a **measurement**, never an objective (rule 2). No candidate is
  admitted, weighted, or rejected because of a public score. ±0.0002 paired noise floor
  stands.
- The **10/day cap is Kaggle's**, enforced by `src/submission/kaggle_io.py`, and is never
  raised, bypassed, or reconfigured by this amendment. `remaining()` is read before every
  upload.
- 2 finalist slots ≠ lifetime upload cap. Uploading for information is still bounded by
  §2 and by quota.

## 3. What this amendment explicitly does NOT change

1. **Never leak.** Every target-dependent transform stays fold-fitted. Unchanged.
2. **Never fit to the public LB.** Unchanged. This amendment does not license public-LB
   tuning, weight selection on public score, or "one more try" after an unfavorable
   public result.
3. **Champion A immutability.** `v7_sol_tabpfn_internal2_aux10` artifacts (OOF
   0.9618529228157235, public 0.96146, ref 57090508, test-pred SHA
   `2b744d0d…`, CSV SHA `fa4947fe…`) are never overwritten by a new candidate. New
   candidates get new names and new artifact paths.
4. **Preregistration.** A candidate cannot be submitted unless its construction was
   preregistered and hashed before results existed (AGENTS.md loop).
5. **No quota-guard edits.** If a submission is ever blocked by the daily cap, we stop.
   We do not edit the guard, the config, or the clock.
6. **No force-push, no ledger rewrite.** Every decision, including a submission that
   later looks wrong on the private board, is append-only history.

## 4. Certification checklist (must all pass before any upload)

A candidate is uploadable only if every line is satisfied and each is *evidenced* by a
file path or hash, not by memory:

1. Prereg JSON exists, was hashed, and was committed before any result file it governs.
2. Fold scheme is the immutable primary registry (`data/cache/folds/`, hash-checked);
   no regenerated folds.
3. Full 5-fold OOF predictions exist for the candidate, stored with SHA-256.
4. OOF AUC recomputed from stored arrays matches the reported value to ≤ 1e-9.
5. No overlap between any fit-fold and its eval-fold rows (test suite green).
6. Early stopping used an inner 10% split carved from FIT rows only.
7. Fold-safe TE / aux / ssl blocks: `tests/test_pipeline.py` green (4 leak tests).
8. External features use no competition label (`test_external_features_use_no_competition_label`).
9. Test-set predictions produced by certified code path with recorded git commit.
10. Test-pred SHA-256 recorded; for a champion replacement, also CSV SHA-256 after build.
11. `src/submission/make.py` pre-flight gate passes (schema, ids, finiteness, range).
12. Candidate compared against the *current* submission on paired folds with mean Δ, SE, t, folds-positive.
13. Admission-gate arithmetic recomputed independently (not copied from a report).
14. Public LB *not* consulted during any of the above; if a public number is quoted in the
    decision record, it is quoted as an observation with its noise floor.
15. PATH selected (A or B) and its evidence line cited concretely.
16. Daily quota read (`kaggle_io.remaining()`) and sufficient.
17. Description string prepared with candidate name, OOF, Δ, and fold evidence — no
    marketing language.
18. Full test suite green (`tests/run_tests.py`) at the candidate's git commit.
19. Secret scan green at the candidate's git commit.
20. `SUBMISSION_DECISION` record written and committed **before** upload (§5).

## 5. The SUBMISSION_DECISION record

Appended to `experiments/ledger.jsonl` with `"kind": "SUBMISSION_DECISION"` and
committed before calling `kaggle_io.submit`:

```json
{
  "kind": "SUBMISSION_DECISION",
  "ts": "<utc iso>",
  "candidate": "<name>",
  "path": "A | B",
  "oof": <float>,
  "oof_delta_vs_champion": <float>,
  "folds": {"per_fold": [], "positive": "n/5", "mean": [], "se": [], "t": []},
  "logit_corr_vs_champion": <float>,
  "evidence": {"prereg_sha256": "", "oof_pred_sha256": "", "test_pred_sha256": "",
               "git": "", "tests": "227 passed", "secret_scan": "0 findings"},
  "quota_remaining_before": <int>,
  "public_lb_consulted": false,
  "rationale": "<one paragraph, PATH-specific>",
  "policy_ref": "research/submission_policy_amendment_20261017.md"
}
```

After upload: append the Kaggle ref + public score as a separate `SUBMISSION` entry
(observation only), same as historical practice.

## 6. Finalists

The two finalist selections for private judging remain governed by AGENTS.md:
chosen by robustness and complementary methodology, reproduced from clean code, not by
the two highest public scores. This amendment does not change finalist selection; it
only removes the per-upload conversation for PATH A/B candidates.

## 7. Sunset

This amendment expires when the competition deadline passes (2026-10-31T23:59Z) or when
the user revokes it, whichever comes first. Revocation is a new dated document; this one
is never edited in place.

---

*Recorded 2026-10-17 (UTC), prospective from that timestamp, in the spirit of the
append-only discipline this repository already follows. Authority: user directive quoted
verbatim in §1. Enforcement: human process + existing quota guard; no code guard was
modified to enable this.*
