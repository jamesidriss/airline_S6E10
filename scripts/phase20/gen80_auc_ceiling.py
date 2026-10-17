"""P20-GEN80-AUC-CEILING (item 80) -- prereg 82af441e, frozen before results.

H80_oracle: oracle (label-using) upper bound on blending sol_route_aux10 (route
reference, OOF 0.961766) with blend_v3_final (59-member rich blend, 0.961509).
H80_hard:  champion AUC by confidence decile -- where do errors concentrate?

Read-only on saved OOF arrays. The oracle is an analysis ceiling, NOT a model.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/gen80_auc_ceiling.py
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

ROUTE = "sol_route_aux10_primary_final"   # best route-based reference in store
RICH = "blend_v3_final"                    # best rich-view blend (59 members)
STRETCH = 0.96274
CHAMP_OOF = 0.961766


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def oracle_auc(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Pair-based oracle AUC: fraction of (pos,neg) pairs where at least one of
    a/b ranks the positive above the negative.

    This upper-bounds the AUC of ANY single blender over a and b: on any pair
    where at least one model is correct, a (label-cheating) blender could get
    the pair right. Computed exactly in O(n log n) via a Fenwick sweep:
    process rows in ascending a; for each negative j, count already-seen
    positives i with a[i] < a[j] AND b[i] < b[j] (both-wrong pairs).
    """
    n = len(y)
    order_a = np.argsort(a, kind="stable")
    b_rank = np.empty(n, dtype=np.int64)
    b_rank[np.argsort(b, kind="stable")] = np.arange(1, n + 1)  # 1..n

    bit = np.zeros(n + 2, dtype=np.int64)

    def bit_add(i: int, v: int = 1) -> None:
        while i <= n:
            bit[i] += v
            i += i & -i

    def bit_sum(i: int) -> int:
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    both_wrong = 0
    for idx in order_a:
        br = b_rank[idx]
        if y[idx] == 0:
            both_wrong += bit_sum(br - 1)
        else:
            bit_add(br, 1)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return 1.0 - both_wrong / (n_pos * n_neg)


def auc_by_decile(y: np.ndarray, conf: np.ndarray) -> list[dict]:
    """AUC on the easiest X% of rows (ranked by |confidence|)."""
    order = np.argsort(-conf)  # most confident first
    n = len(y)
    out = []
    for frac in (0.5, 0.8, 0.9, 0.95, 1.0):
        k = int(n * frac)
        idx = order[:k]
        out.append({
            "top_frac": frac,
            "n_rows": k,
            "auc": round(float(roc_auc_score(y[idx], conf[idx])), 6),
        })
    # hardest decile
    idx = order[int(n * 0.9):]
    out.append({
        "top_frac": "hardest_10pct",
        "n_rows": len(idx),
        "auc": round(float(roc_auc_score(y[idx], conf[idx])), 6),
    })
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    y = ds.train[TARGET].to_numpy()

    route = load_oof(ROUTE)
    rich = load_oof(RICH)
    route_auc = float(roc_auc_score(y, route))
    rich_auc = float(roc_auc_score(y, rich))

    # equal-logit blend (the measured, non-oracle blend)
    blend = (logit(route) + logit(rich)) / 2.0
    blend_auc = float(roc_auc_score(y, blend))

    oracle = oracle_auc(y, route, rich)
    headroom = oracle - route_auc
    gap = STRETCH - CHAMP_OOF

    # hard-row analysis on the route reference (champion proxy)
    conf = np.log(np.clip(route, 1e-7, 1 - 1e-7) / np.clip(1 - route, 1e-7, 1))
    deciles = auc_by_decile(y, conf)

    report = {
        "prereg": "configs/phase20/gen80_auc_ceiling_prereg.json",
        "prereg_sha256": "82af441ed577d5cdf2dade569516568de529108a29be6c63b16e688ac39fcc24",
        "h80_oracle": {
            "route_reference": {"id": ROUTE, "oof_auc": round(route_auc, 6)},
            "rich_blend": {"id": RICH, "oof_auc": round(rich_auc, 6), "n_members": 59},
            "equal_logit_blend_auc": round(blend_auc, 6),
            "oracle_ceiling": round(oracle, 6),
            "headroom_oracle_minus_route": round(headroom, 6),
            "stretch_target": STRETCH,
            "stretch_gap_vs_champion": round(gap, 6),
            "stretch_reachable_by_blend": bool(headroom >= gap),
            "interpretation": (
                f"Oracle ceiling {oracle:.6f} vs champion {CHAMP_OOF}: headroom "
                f"{headroom:+.6f}. Stretch gap {gap:+.6f}. "
                + ("Stretch THEORETICALLY reachable by a better blender over these two."
                   if headroom >= gap else
                   "NO blend of these two reaches the stretch -> need a new signal (FEAT block).")),
        },
        "h80_hard_rows": {
            "confidence": "route reference logit",
            "auc_by_decile": deciles,
        },
    }
    path = OUT / "gen80_auc_ceiling.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    o = report["h80_oracle"]
    print(f"route={o['route_reference']['oof_auc']} rich={o['rich_blend']['oof_auc']} "
          f"blend={o['equal_logit_blend_auc']} oracle={o['oracle_ceiling']}")
    print(f"headroom={o['headroom_oracle_minus_route']:+.6f} "
          f"gap={o['stretch_gap_vs_champion']:+.6f} "
          f"reachable={o['stretch_reachable_by_blend']}")
    for d in deciles:
        print(f"  {d['top_frac']}: n={d['n_rows']} auc={d['auc']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
