"""Log the first VALID TabR fold-0 result, its blend test, and the measured context-cost curve.

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
        "exp_id": "tabr_full_ctx32_e10_fold0",
        "oof_auc": 0.960282,
        "fold_aucs": [0.960282],
        "fold": 0,
        "scheme": "primary",
        "verdict": "WEAK BUT MOST DECORRELATED -- no fixed blend weight improves the finalist",
        "config": {"view": "full", "assembled_features": 285, "model_frame_features": 289,
                   "d_main": 128, "context_size": 32, "batch_size": 4096,
                   "memory_efficient": True, "candidate_encoding_batch_size": None,
                   "n_epochs": 10, "patience": 4, "lr": 0.0003121273641315169},
        "cost": {"train_seconds": 549.6, "inference_seconds": 11.4,
                 "peak_vram_allocated_gb": 10.15, "peak_host_rss_gb": 6.454,
                 "retrieval_queries": 5605272, "actual_k": 32, "retrieval_dim": 128,
                 "nan": 0, "oom_events": 0, "seconds_per_epoch": 54.1},
        "protocol": ("Early stopping / checkpoint selection used only the inner 10% split carved "
                     "from the outer-FIT rows; the outer fold was scored exactly once afterwards. "
                     "n_refit=0 verified from sklearn_base.py:443-454, so pytabkit does not refit "
                     "on train+val. The outer AUC is therefore a valid fold-0 measurement."),
        "gate": {
            "band": "weak_but_potentially_orthogonal [0.959301, 0.9603)",
            "stop_line": 0.959301,
            "clears_stop_line": True,
        },
        "correlations_fold0": {
            "vs_finalist": {"pearson_prob": 0.99843, "pearson_logit": 0.99501, "spearman": 0.97647},
            "vs_xt_lightgbm": {"pearson_prob": 0.99791, "pearson_logit": 0.99377, "spearman": 0.96998},
            "vs_realmlp": {"pearson_prob": 0.99808, "pearson_logit": 0.99434, "spearman": 0.9715},
            "vs_tabm": {"pearson_prob": 0.99774, "pearson_logit": 0.99403, "spearman": 0.975},
            "existing_family_logit_corr_range": [0.99577, 0.99908],
        },
        "fixed_weight_diagnostic_on_finalist": {
            "finalist_alone": 0.961501,
            "w0.02_logit": -0.000002, "w0.02_prob": -0.000004,
            "w0.05_logit": -0.000004, "w0.05_prob": -0.000010,
            "w0.10_logit": -0.000009, "w0.10_prob": -0.000024,
            "note": ("Every predeclared fixed weight HURTS, monotonically in the weight. Measured "
                     "on the same fold it is chosen on, so it is indicative only -- but with all "
                     "six cells negative there is no signal to justify 5-fold compute at ctx 32."),
        },
        "interpretation": (
            "TabR at context 32 is 1.2e-3 below the finalist standalone, and although it IS the "
            "most decorrelated model in the pool (logit corr 0.99501 vs a 0.99577 floor among the "
            "existing pairs), that margin is far too small to pay for the accuracy deficit. Its "
            "Spearman of 0.97647 is the lowest we have measured, which is the one encouraging "
            "signal, but correlation alone did not translate into blend gain."
        ),
        "confound_being_resolved": (
            "context 32 was forced by a VRAM wall (TabR's default 96 costs 30 s/step and thrashes "
            "at 15.7 of 16.3 GB). TabR's context IS its neighbourhood size, so a proper-sized "
            "context might be both stronger and still decorrelated. Measured cost curve: "
            "ctx 32 -> 1.03 s/step, ctx 48 -> 0.72 s/step, ctx 64 -> 20.4 s/step, ctx 96 -> 30.0 "
            "s/step. The cliff sits between 48 and 64, so ctx 48 is nearly free and is being "
            "tested before TabR is judged."
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
