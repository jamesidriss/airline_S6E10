"""FEAT56 + FEAT58 ledger/catalogue updates (batched helper)."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "experiments" / "ledger.jsonl"
CAT = ROOT / "research" / "phase20_exhaustive" / "experiment_catalogue_96.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(entry: dict) -> None:
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def set_cat(num: int, finding: str, fidelity: str, status: str) -> None:
    cat = json.loads(CAT.read_text(encoding="utf-8"))
    for row in cat["experiments"]:
        if row.get("experiment_number") == num:
            row["phase20_finding"] = finding
            row["fidelity_reached"] = fidelity
            row["status"] = status
    CAT.write_text(json.dumps(cat, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    which = sys.argv[1]
    if which == "56":
        rep = json.loads((ROOT / "reports/phase20/feat56_local_transforms/feat56_local_transforms.json").read_text(encoding="utf-8"))
        m = rep["metrics"]
        append({
            "exp_id": "p20_feat56_local_transforms",
            "hypothesis_id": "H56-LOCAL-TRANSFORMS",
            "date": "2026-10-18", "ts": utcnow(), "family": "FEAT", "experiment_number": 56,
            "fold_scheme": "primary fold-0 (F2 screen, strict frozen rule)",
            "prereg": {"path": "configs/phase20/feat56_local_transforms_prereg.json",
                       "sha256": hashlib.sha256((ROOT / "configs/phase20/feat56_local_transforms_prereg.json").read_bytes()).hexdigest()},
            "arch": "within-(Customer Type x Class) ECDF transform of 4 continuous covariates, "
                    "outer-FIT-only (label-free), added to 21raw. LGBM xt_xt_f10.",
            "results": m,
            "finding": (
                "Fold-0: control 0.953423, treatment 0.953637 (+2.15e-4 descriptive). "
                "FROZEN RULE FAILS: port_delta -1.875e-3 (blend of candidate with route "
                "reference 0.9599 vs portfolio 0.9617) and corr 0.9120 (< 0.995 satisfied, "
                "port limb fails). The candidate is far too weak standalone to add portfolio "
                "value; context-sensitive transforms do not change this. NEGATIVE."),
            "verdict": "COMPLETE_NEGATIVE", "closes_rows": [56],
        })
        set_cat(56,
                "Within-segment ECDF transforms (4 numerics, outer-FIT-only): aug delta "
                "+2.15e-4 descriptive but port_delta -1.875e-3 (need >= +5e-5), corr 0.9120. "
                "Frozen rule NEGATIVE.",
                "F2", "COMPLETE_NEGATIVE")
        print("ledger + catalogue row 56 updated")
    elif which == "58":
        rep = json.loads((ROOT / "reports/phase20/feat58_rare_combination/feat58_rare_combination.json").read_text(encoding="utf-8"))
        m = rep["metrics"]
        finding = (
            f"Fold-0: control {m['control_fold0_auc']}, treatment {m['treatment_fold0_auc']} "
            f"({m['augmentation_delta_descriptive_only']:+.2e} descriptive). "
            f"FROZEN RULE: port_delta {m['port_delta']:+.3e} (need >= +5e-5), corr "
            f"{m['candidate_spearman_vs_portfolio']} (need < 0.995) -> "
            f"{'PROMISING' if m['promising_frozen_rule'] else 'NEGATIVE'}.")
        append({
            "exp_id": "p20_feat58_rare_combination",
            "hypothesis_id": "H58-RARE-COMBINATION",
            "date": "2026-10-18", "ts": utcnow(), "family": "FEAT", "experiment_number": 58,
            "fold_scheme": "primary fold-0 (F2 screen, strict frozen rule)",
            "prereg": {"path": "configs/phase20/feat58_rare_combination_prereg.json",
                       "sha256": hashlib.sha256((ROOT / "configs/phase20/feat58_rare_combination_prereg.json").read_bytes()).hexdigest()},
            "arch": "adaptive (count-based) shrinkage target encoding of 4-way categorical "
                    "combinations (Gender x Customer Type x Type of Travel x Class, <=48 "
                    "cells), strict outer-FIT-only EVAL stats + inner cross-fit within "
                    "outer-FIT, ES labels excluded from all sources. LGBM xt_xt_f10.",
            "results": m,
            "finding": finding,
            "verdict": "COMPLETE_NEGATIVE" if not m["promising_frozen_rule"] else "PROMISING",
            "closes_rows": [] if m["promising_frozen_rule"] else [58],
        })
        set_cat(58, finding,
                "F2", "PROMISING" if m["promising_frozen_rule"] else "COMPLETE_NEGATIVE")
        print("ledger + catalogue row 58 updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
