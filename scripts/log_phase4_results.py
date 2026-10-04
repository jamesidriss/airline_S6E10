"""Log the Phase 4 conclusions (TabR stop, kNN-TE negative) and the learning-curve diagnostic.

The learning curve is the most strategically important entry here: it changes the campaign's
self-assessment from "representation-limited plateau" to "still data-limited", which redirects what
is worth trying next.

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.jsonl"

ENTRIES = [
    {
        "exp_id": "tabr_full_ctx48_e10_fold0",
        "oof_auc": 0.960329,
        "fold_aucs": [0.960329],
        "fold": 0,
        "scheme": "primary",
        "verdict": "REJECT for the ensemble -- confound resolved, still no blend gain",
        "config": {"view": "full", "d_main": 128, "context_size": 48, "batch_size": 4096,
                   "memory_efficient": True, "n_epochs": 10, "patience": 4},
        "cost": {"train_seconds": 1354.7, "inference_seconds": 27.8,
                 "peak_vram_allocated_gb": 10.15, "peak_host_rss_gb": 3.249,
                 "retrieval_queries": 5045564, "actual_k": 48, "nan": 0, "oom_events": 0},
        "context_scaling": {
            "ctx32": {"auc": 0.960282, "train_seconds": 549.6},
            "ctx48": {"auc": 0.960329, "train_seconds": 1354.7},
            "note": ("50% more neighbours bought +4.7e-5 for 2.46x the compute, and did NOT "
                     "increase decorrelation: logit corr vs finalist 0.99502 at ctx48 versus "
                     "0.99501 at ctx32."),
        },
        "fixed_weight_diagnostic_on_finalist": {
            "finalist_alone": 0.961501,
            "w0.02_logit": -0.000001, "w0.05_logit": -0.000003, "w0.10_logit": -0.000008,
            "w0.02_prob": -0.000003, "w0.05_prob": -0.000009, "w0.10_prob": -0.000022,
        },
        "verdict_detail": (
            "TabR is stopped. It achieved what it was chosen for -- it is the most decorrelated "
            "model in the pool (logit corr 0.99502 vs a 0.99577 floor among existing pairs, and "
            "Spearman 0.97727, the lowest ever measured here) -- but at two independent context "
            "sizes, every predeclared fixed blend weight was negative. Its 1.2e-3 standalone deficit "
            "is simply larger than its decorrelation is worth. Further tuning would be spending "
            "hours per fold on a family whose marginal value is measurably zero."
        ),
    },
    {
        "exp_id": "knnte_fold0",
        "oof_auc": 0.960953,
        "fold_aucs": [0.960953],
        "fold": 0,
        "scheme": "primary",
        "verdict": "REJECT -- harmful and redundant",
        "baseline_auc": 0.961103,
        "delta": -0.000150,
        "block": {"n_columns": 12, "build_seconds": 29,
                  "columns": ["te_16", "te_64", "te_256", "te_unw_16", "te_unw_64", "te_unw_256",
                              "dist_16", "dist_64", "dist_256", "disp_16", "disp_64", "disp_256"],
                  "metric_space": "21 raw survey variables -> 16 numeric dims "
                                  "(standardised numerics + one-hot categoricals)",
                  "standalone_auc_of_te": {"te_16": 0.943745, "te_64": 0.946756, "te_256": 0.946725}},
        "cost": {"model_seconds": 70.4, "vs_tabr_train_seconds": 1354.7},
        "correlation": {"vs_finalist_logit": 0.99831, "vs_finalist_spearman": 0.989,
                        "vs_xt_logit": 0.99761, "vs_xt_spearman": 0.9834},
        "protocol": ("retrieval database contains only outer-FIT rows; every fit row is encoded by "
                     "a 5-fold inner cross-fit that excludes it; inner-ES/val/test rows are encoded "
                     "against the full outer-FIT-minus-ES database; retrieval exact fp32 on GPU, "
                     "validated against a float64 reference."),
        "note": (
            "A pure k-NN with no model at all reaches 0.9468 on this fold, so the neighbourhood "
            "metric is genuinely informative and the fold discipline is sound -- the block is simply "
            "redundant with what the 285-feature view already extracts, and it displaces better "
            "features under colsample_bytree. It is MORE correlated with the champion than TabR was "
            "(Spearman 0.989 vs 0.977), so it is the opposite of what this campaign needed."
        ),
    },
    {
        "exp_id": "lcurve_fold0",
        "oof_auc": None,
        "fold_aucs": [],
        "verdict": "STRATEGIC -- we are still DATA-limited, not representation-limited",
        "model": "lgbm extra_trees on the full view (champion single-model configuration)",
        "seeds_per_fraction": 3,
        "curve": [
            {"frac": 0.125, "n": 69963, "auc": 0.959318, "std": 0.000139},
            {"frac": 0.25, "n": 139927, "auc": 0.960020, "std": 0.000060},
            {"frac": 0.5, "n": 279854, "auc": 0.960696, "std": 0.000204},
            {"frac": 0.75, "n": 419781, "auc": 0.960956, "std": 0.000133},
            {"frac": 1.0, "n": 559708, "auc": 0.961212, "std": 0.000108},
        ],
        "per_doubling_gains_e5": {"0.125->0.25": 7.0, "0.25->0.5": 6.8, "0.5->1.0": 5.2},
        "seed_std_e5": 1.1,
        "conclusion": (
            "The curve is close to linear in log(n) and barely decelerates: each doubling of training "
            "data still buys 5-7e-5, and the final doubling's 5.2e-5 is about 5x the seed-to-seed "
            "standard deviation, so it is real signal and not sampling noise. Extrapolating, "
            "another doubling (~1.1M rows) would be worth roughly +5e-5, which is a sixth of the "
            "entire rank-1-to-rank-45 spread. This campaign has been treating itself as "
            "representation-limited; the evidence says it is still DATA-limited. The actionable "
            "consequences are (a) in-distribution data augmentation, which is untested here and "
            "directly manufactures the resource that is scarce, and (b) continued variance "
            "reduction by averaging. This is a directional diagnostic, not a Bayes ceiling."
        ),
        "bug_found_first": (
            "The first run returned AUCs of 0.47-0.54 (anti-correlated) for every fraction below 1.0 "
            "and a correct 0.961 only at 1.0. Cause: the assembled matrix is indexed by position "
            "within the outer-FIT rows, but positions were being mapped through the subsample, so "
            "the wrong rows were read. It looked correct only at frac=1.0 where subsample == fit. "
            "Anti-correlated AUCs are what exposed it -- a plausible-looking smooth curve would not "
            "have been caught by inspection."
        ),
    },
]


def main() -> int:
    existing = set()
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line).get("exp_id"))
                except Exception:  # noqa: BLE001
                    pass
    added = 0
    with LEDGER.open("a", encoding="utf-8") as f:
        for e in ENTRIES:
            if e["exp_id"] in existing:
                print(f"  skip: {e['exp_id']}")
                continue
            e.setdefault("date", "2026-10-04")
            f.write(json.dumps(e) + "\n")
            added += 1
            print(f"  appended: {e['exp_id']} [{e['verdict']}]")
    print(f"\n{added} entries appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
