"""Phase 24 -- inventory of stored OOF vectors vs the certified champion.

Read-only diagnostic. Reports, for every stored *_oof.npy in artifacts/predictions:
  - pooled OOF AUC (recomputed from the array)
  - paired delta vs v7 with per-fold paired SE
  - Pearson correlation of logits with v7 on the pooled OOF
  - the best single-weight logit blend with v7 on the pooled rows
and writes reports/phase24/oof_inventory.json. No labels are written out, no
model is fitted, nothing is submitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import DATA, file_sha256  # noqa: E402

PRED = ROOT / "artifacts" / "predictions"
OUT = ROOT / "reports" / "phase24"
CHAMP = "v7_sol_tabpfn_internal2_aux10"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Rank AUC with half credit for ties (identical convention to the project)."""
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order].astype(np.float64)
    if w is not None:
        w_sorted = w[order].astype(np.float64)
    else:
        w_sorted = np.ones_like(y_sorted)
    pos_w = w_sorted * y_sorted
    neg_w = w_sorted * (1.0 - y_sorted)
    # group ties
    tie_start = np.empty(len(s_sorted), dtype=np.int64)
    tie_size = np.empty(len(s_sorted), dtype=np.float64)
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        tie_start[i:j] = i
        tie_size[i:j] = j - i
        i = j
    cum_neg = np.cumsum(neg_w)
    neg_before = np.concatenate([[0.0], cum_neg[:-1]])
    neg_up_to_tie_end = cum_neg[tie_start + tie_size.astype(np.int64) - 1]
    # strict wins: positives beat all negatives with strictly lower score
    contrib = pos_w * neg_before
    # ties: half credit within the tie group
    same = (tie_size > 1)
    if same.any():
        idx = np.where(same)[0]
        # neg weight inside the group excluding itself
        grp_neg_total = neg_up_to_tie_end[idx] - neg_before[idx]
        own = np.where(y_sorted[idx] > 0, neg_w[idx], 0.0)
        contrib[idx] += 0.5 * pos_w[idx] * (grp_neg_total - own)
    denom = pos_w.sum() * neg_w.sum()
    return float(contrib.sum() / denom)


def fold_ids() -> np.ndarray:
    from src.common import load_cached_parquet
    from src.validation.folds import get_scheme

    tr, _ = load_cached_parquet()
    y = tr["satisfaction"].values.astype("int8")
    return get_scheme("primary", y, tr["id"]).folds


def main() -> None:
    from src.common import load_cached_parquet

    tr, _ = load_cached_parquet()
    y = tr["satisfaction"].values.astype("int8").astype(np.float64)
    folds = fold_ids()
    champ = np.load(PRED / f"{CHAMP}_oof.npy")
    base_auc = auc(y, champ)
    print(f"champion {CHAMP}: pooled OOF AUC {base_auc:.16f}  sha {file_sha256(PRED / f'{CHAMP}_oof.npy')}")
    ch_logit = logit(champ)

    per_fold_base = [auc(y[folds == k], champ[folds == k]) for k in range(5)]
    print("per fold:", ["%.9f" % v for v in per_fold_base])

    rows = []
    for p in sorted(PRED.glob("*_oof.npy")):
        name = p.name[: -len("_oof.npy")]
        a = np.load(p)
        if a.shape != champ.shape or not np.isfinite(a).all():
            rows.append({"name": name, "skipped": "shape/nonfinite", "shape": list(a.shape)})
            continue
        # normalise to probabilities if the vector looks like logits
        if a.min() < -1e-6 or a.max() > 1 + 1e-6:
            pv = 1.0 / (1.0 + np.exp(-np.clip(a, -60, 60)))
        else:
            pv = a
        try:
            d = [auc(y[folds == k], pv[folds == k]) - per_fold_base[k] for k in range(5)]
        except Exception:
            rows.append({"name": name, "skipped": "auc error"})
            continue
        d = np.asarray(d)
        se = float(d.std(ddof=1) / np.sqrt(5)) if np.isfinite(d).all() else float("nan")
        lg = logit(pv)
        corr = float(np.corrcoef(lg, ch_logit)[0, 1])
        best = None
        for w in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            b = (1 - w) * ch_logit + w * lg
            g = auc(y, b) - base_auc
            if best is None or g > best[1]:
                best = [w, g]
        rows.append(
            {
                "name": name,
                "pooled_auc": auc(y, pv),
                "mean_paired_delta": float(d.mean()),
                "paired_se": se,
                "folds_positive": int((d > 0).sum()),
                "corr_logit": corr,
                "best_blend_w": best[0],
                "best_blend_delta": best[1],
                "sha256": file_sha256(p),
            }
        )

    rows = [r for r in rows if "pooled_auc" in r]
    rows.sort(key=lambda r: -r["best_blend_delta"])
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "champion": CHAMP,
        "champion_auc": base_auc,
        "champion_sha256": file_sha256(PRED / f"{CHAMP}_oof.npy"),
        "per_fold_champion": per_fold_base,
        "n_vectors": len(rows),
        "rows": rows,
    }
    (OUT / "oof_inventory.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n{'name':58} {'pooled':>10} {'delta':>11} {'se':>9} {'+f':>3} {'corr':>8} {'bestw':>6} {'blend':>10}")
    for r in rows[:35]:
        print(
            f"{r['name'][:58]:58} {r['pooled_auc']:.9f} {r['mean_paired_delta']:+.3e} "
            f"{r['paired_se']:.2e} {r['folds_positive']:3d} {r['corr_logit']:8.4f} "
            f"{r['best_blend_w']:6.2f} {r['best_blend_delta']:+.3e}"
        )


if __name__ == "__main__":
    main()
