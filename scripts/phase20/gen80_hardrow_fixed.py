"""GEN80 hard-row analysis recalculation: use absolute logits for confidence ordering.

Recalculates the hard-row AUC-by-decile using |logit| as confidence measure,
not signed logits. Preserves original report with appended correction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "gen80_auc_ceiling"

ROUTE = "sol_route_aux10_primary_final"
RICH = "blend_v3_final"
STRETCH = 0.96274
CHAMP_OOF = 0.9618529228157235


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def auc_by_abs_confidence(y: np.ndarray, prob: np.ndarray, abs_logit: np.ndarray) -> list[dict]:
    """AUC on the easiest X% of rows ranked by |logit| confidence (absolute distance from 0.5)."""
    order = np.argsort(-abs_logit)  # most confident first (highest |logit|)
    n = len(y)
    out = []
    for frac in (0.5, 0.8, 0.9, 0.95, 1.0):
        k = int(n * frac)
        idx = order[:k]
        out.append({
            "top_frac": frac,
            "n_rows": k,
            "auc": round(float(roc_auc_score(y[idx], prob[idx])), 6),
        })
    # hardest decile (least confident 10%)
    idx = order[int(n * 0.9):]
    out.append({
        "top_frac": "hardest_10pct",
        "n_rows": len(idx),
        "auc": round(float(roc_auc_score(y[idx], prob[idx])), 6),
    })
    return out


def main() -> int:
    ds = load_raw()
    y = ds.train[TARGET].to_numpy()

    route = load_oof("sol_route_aux10_primary_final")
    rich = load_oof("blend_v3_final")

    route_auc = float(roc_auc_score(y, route))
    rich_auc = float(roc_auc_score(y, rich))
    blend = (logit(route) + logit(rich)) / 2.0
    blend_auc = float(roc_auc_score(y, blend))

    # Hard-row analysis with ABSOLUTE logits (confidence = |logit|)
    route_logit = logit(route)
    route_abs_logit = np.abs(route_logit)
    rich_logit = logit(load_oof("blend_v3_final"))
    rich_abs_logit = np.abs(rich_logit)

    route_deciles = auc_by_abs_confidence(y, route, route_abs_logit)
    rich_deciles = auc_by_abs_confidence(y, load_oof("blend_v3_final"), rich_abs_logit)

    report = {
        "prereg": "configs/phase20/gen80_auc_ceiling_prereg.json",
        "prereg_sha256": "82af441ed577d5cdf2dade569516568de529108a29be6c63b16e688ac39fcc24",
        "correction_20261018_v2": "HARD-ROW RECALCULATION 2026-10-18: Confidence now measured by ABSOLUTE logit (|logit| = distance from decision boundary), not signed logit. This correctly identifies hard rows as those near the decision boundary (|logit| small), regardless of which class they favor.",
        "h80_oracle": {
            "route_reference": {"id": "sol_route_aux10_primary_final", "oof_auc": round(route_auc, 6)},
            "rich_blend": {"id": "blend_v3_final", "oof_auc": round(rich_auc, 6), "n_members": 59},
            "equal_logit_blend_auc": round(float(roc_auc_score(load_raw().train["satisfaction"].to_numpy(), (logit(load_oof("sol_route_aux10_primary_final")) + logit(load_oof("blend_v3_final"))) / 2.0)), 6),
            "oracle_ceiling": 0.964906,
            "headroom_oracle_minus_route": 0.003140,
            "stretch_target": 0.96274,
            "stretch_gap_vs_champion": 0.000887,
            "stretch_reachable_by_blend": "false_oracle_is_not_achievable_by_deployable_blender",
            "note": "Oracle is label-dependent upper bound; real blenders must generalize from covariates."
        },
        "h80_hard_rows_abs_logit": {
            "confidence_metric": "absolute_logit (|logit| = distance from decision boundary)",
            "route_reference": route_deciles,
            "rich_blend": rich_deciles,
        },
        "h80_hard_rows_signed_logit_original": {
            "note": "Original analysis used signed logit (incorrectly treating high positive logits as 'easy' regardless of class). This is preserved for reference.",
        }
    }
    path = OUT / "gen80_auc_ceiling_v2.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print("Route reference (|logit| confidence):")
    for d in route_deciles:
        print(f"  {d['top_frac']}: n={d['n_rows']} auc={d['auc']}")
    print("Rich blend (|logit| confidence):")
    for d in rich_deciles:
        print(f"  {d['top_frac']}: n={d['n_rows']} auc={d['auc']}")
    print(f"Report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())