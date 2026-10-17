# Phase 20 — S7 recovery and fidelity-repair close-out

**Date:** 2026-10-18. **Scope:** five original experiment items (15, 17, 30, 64, 65), all valid primary fold-0 screens. All S7 experiments were preregistered under `configs/phase20/s7_fidelity_repair_prereg.json` (SHA256 `c10cf3fec2efa1e35eb561009afcf72f717dbc0274490df46fb049145e79e5ed`) before treatment results. The original OOM reports were retained.

| Item | Experiment | Fold-0 standalone AUC | 50/50 blend delta vs v7 | Corr. to v7 | Result |
|---|---|---:|---:|---:|---|
| 15 | FT-Transformer, 285-feature rich view, bounded GPU batches | 0.958638712 | -0.000616678 | 0.992069 | Negative for this bounded recipe |
| 30 | InteractionNet, rich view, chunked inference | 0.958710300 | -0.000938788 | 0.983815 | Negative |
| 17 | Full-matrix DCNv2, rich view | 0.959040786 | -0.000977878 | 0.979865 | Negative |
| 65 | Low-rank factorized logistic (rank 16), rich view | 0.959788072 | -0.000681542 | 0.986785 | Negative |
| 64 | RFF approximate RBF SVM, rich view | 0.929997214 | -0.002174932 | 0.867091 | Negative |

All comparisons use the same certified v7 primary fold-0 target population (139,927 held-out rows; portfolio reference AUC 0.9618278142504795), not a complete five-fold OOF. The preregistered +0.00005 discovery bar was not met. **No model promotion or Kaggle submission.**

## Engineering and recovery

- Original FT-Transformer and InteractionNet rich-view attempts failed on enormous unchunked internal-validation prediction tensors, not on an independently established inability to train the architecture.
- The bounded runs now provide real measured values. InteractionNet kept its training batch at 4096 and chunked inner validation/inference at 4096; FT-Transformer used a lower train batch (512) plus chunked inference (2048), as permitted by the resource-fallback preregistration.
- **Scientific nuance:** reducing the FT-Transformer training batch changes stochastic optimization, even with the same architecture and loss. Its failure is evidence against the *bounded training recipe*; the original 4096-batch training recipe did not complete. Do not claim strict training equivalence or a universal negative architecture verdict.
- The scalar-cross DCNv2 proxy did not establish the full-matrix DCNv2 result; full-matrix DCNv2 has now been measured directly and is negative in the frozen screen.
- Additive logistic regression did not establish the result of a factorized bilinear logistic model; rank-16 factorized logistic is now measured directly.
- Kernel Ridge was not an evaluation of RFF SVM; the RFF SVM candidate is now measured directly.
- Historical H19-C, H19-B and foundation-model negatives remain unchanged.

## Persistence and integrity

- The experiment ledger now has **500 entries**, including precisely one `p20_s7_ftt_full_bounded` and one `p20_s7_inet_full_bounded`; these were already appended before recovery and were not appended twice.
- The five S7 jobs are marked CLOSED_NEGATIVE in `research/phase20_exhaustive/scheduler.json` with correct executable commands, source/config fingerprints, and validated report/prediction SHA256 manifests.
- `scripts/phase20/s7_scheduler_finalize.py` is idempotent and verifies the five outputs; it never retrains.
- The scheduler has a corrected no-timeout default, a live PID-bearing GPU lock, and a source-code-aware fingerprint even when invoked through python.exe.
- The Phase 20 catalogue rows and campaign budget were updated. Approximate new Phase 20 GPU consumption: **9.1/12 h**, leaving about **2.9 h** under the existing cap; recalculate from the definitive compute ledger before any extension.
- V7 is not modified. The submission authorization/policy committed previously remains separate from this S7 repair.

## Limitations and next step

A negative fixed 50/50 blend is a screening result, not a mathematical proof that every nested ensemble policy loses. These experimental recipes can be closed without closing every adjacent research idea.

Proceed through the remaining CPU-first, truly distinct, eligible items of the 96-item catalogue. Observe the existing GPU budget, hardware guard, private-first 20% public / 80% private rule, and full OOF+independent validation requirements before any new submission.

## Final validation before commit

- **231 software tests passed, 0 failed** in the complete repository suite (2026-10-18); four new scheduler integration tests included.
- **Secret scan:** 0 findings.
- **Certified v7:** test prediction `2b744d0df36a0b1143a92e1db18661ccd94fa881490784308624aaa0a0fc95af` MATCH; submission CSV `fa4947fe9fc381fe85b105c1c489375327f7bb2321e4ff7f0d9ffc7cd7c9c0bd` MATCH.
- **Scheduler:** 96 recorded jobs: 56 PENDING, 22 CLOSED_NEGATIVE, 9 DEPENDENCY_BLOCKED, 5 DONE, 3 LICENSE_BLOCKED and 1 PROMOTED. All five S7 outputs hash-verified and retrospectively indexed.
- **Submissions:** none. No active ML training was duplicated during recovery.
