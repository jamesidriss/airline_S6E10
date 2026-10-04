"""Log the TabR context-32 fold-0 screening measurements to experiments/ledger.jsonl.

Separate from log_tabr_perf.py because these are the first MEASURED TabR training curves at the
viable operating point, plus the fold-0 gate decision that follows from them.

Idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "experiments" / "ledger.jsonl"


def build_entries():
    return [
        {
            "exp_id": "tabr_ctx32_fold0_screening_epochs0_7",
            "oof_auc": None,
            "fold_aucs": [],
            "verdict": "INCOMPLETE (run reaped at epoch 7, no prediction produced)",
            "note": (
                "First run at the viable operating point (context 32, d_main 128, batch 4096, "
                "memory_efficient=True, single-pass candidate encoding). 8 epochs completed at a "
                "steady ~150 s/epoch, cumulative 1157 s. Peak VRAM 10.15 GB allocated / 14.23 GB "
                "reserved versus 15.7 of 16.3 GB at context 96, and host RSS settled at 1.35 GB. "
                "The run was then killed without any Python traceback -- but memory was NOT the "
                "cause this time, since both VRAM and RSS were well inside limits, so the "
                "silent-death pattern is most likely external reaping rather than exhaustion."
            ),
            "epoch_seconds": [106.8, 152.1, 158.1, 150.5, 148.1, 149.1, 152.6, 138.1],
            "peak_vram_allocated_gb": 10.15,
            "peak_host_rss_gb": 1.35,
            "durability_note": (
                "Every epoch record was flushed to disk as it was produced, so this run still "
                "yielded its timing evidence despite producing no prediction. That is the sink fix "
                "working as intended."
            ),
            "protocol_note": (
                "The inner-validation ROC-AUC column is null for this run: the recorder bugs (labels "
                "under batch['Y'] rather than batch[1]; val_accuracy being a torchmetrics Metric "
                "object rather than a float) were found by reading these records and fixed "
                "afterwards, so later runs do carry the curve."
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
        for e in build_entries():
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
