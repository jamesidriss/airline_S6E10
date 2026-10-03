"""Compare every stored experiment in one place.

Produces reports/all_models.md: per-experiment metrics, logit/Spearman correlation matrix,
marginal-gain-when-added-to-stack analysis, and the best blends found by several geometries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--min-auc", type=float, default=0.0)
    ap.add_argument("--require-test", action="store_true")
    ap.add_argument("--out", default="all_models.md")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds
    idx = store._load_index()

    rows, P = [], {}
    for k, v in idx.items():
        m = v.get("meta", {})
        if not m.get("auc"):
            continue
        if m.get("family") in ("blend", "stack"):
            continue  # derived predictions must never become members of their own stack
        if m["auc"] < args.min_auc:
            continue
        if args.require_test and "test" not in v:
            continue
        o = store.load_oof(k).astype("float64")
        if len(o) != len(y):
            continue
        P[k] = o
        rows.append({"exp_id": k, "auc": m["auc"], "family": m.get("family", ""),
                     "view": m.get("featureset", ""), "scheme": v.get("fold_scheme", "primary"),
                     "has_test": "test" in v, "seed": m.get("seed", "")})
    rows.sort(key=lambda r: -r["auc"])
    df = pd.DataFrame(rows)

    L = []
    L.append("# All experiments (primary 5-fold OOF)\n")
    L.append(f"n = {len(df)}   (min_auc={args.min_auc}, require_test={args.require_test})\n")
    L.append(df.to_markdown(index=False))
    L.append("\n## Logit-space Pearson correlation\n")
    C = lab.corr_matrix(P, "logit")
    L.append(C.round(4).to_markdown())
    L.append("\n## Spearman correlation\n")
    S = lab.spearman_matrix(P)
    L.append(S.round(4).to_markdown())

    # marginal gain when each member is added to the equal-weight blend of the rest
    L.append("\n## Marginal value of each member\n")
    L.append("`gain_vs_best_single` = AUC of equal-logit blend of all members, minus the best "
             "single member. `marginal` = AUC(all) − AUC(all minus this member); positive means "
             "the member earns its place.\n")
    names = list(P)
    Mall = np.column_stack([lab.tform(P[n], "logit") for n in names])
    a_all = lab.auc(y, Mall.mean(axis=1))
    best_single = df.iloc[0]
    marg = []
    for i, n in enumerate(names):
        keep = [j for j in range(len(names)) if j != i]
        a = lab.auc(y, Mall[:, keep].mean(axis=1))
        marg.append({"exp_id": n, "auc": float(df[df.exp_id == n].iloc[0]["auc"]),
                     "marginal_in_blend": round(a_all - a, 6)})
    L.append(pd.DataFrame(marg).sort_values("marginal_in_blend", ascending=False).to_markdown(index=False))

    L.append("\n## Blend geometry over all members\n")
    for kind in ("prob", "logit", "rank"):
        b = lab.equal_blend(P, y, folds, kind=kind)
        L.append(f"- equal-{kind}: **{b['auc']:.6f}**  folds={[round(x,6) for x in b['fold_aucs']]}")
    g = lab.greedy_hill_climb(P, y, folds, kind="logit", rounds=80)
    L.append(f"- greedy-logit: **{g['auc']:.6f}**")
    L.append(f"\n- best single: **{best_single['auc']:.6f}** (`{best_single['exp_id']}`)")
    L.append(f"- equal-logit all: **{a_all:.6f}**  (gain {a_all - best_single['auc']:+.6f})")

    out = REPORTS / args.out
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:8]))
    print("...")
    print(f"equal-logit all = {a_all:.6f}   best single = {best_single['auc']:.6f} "
          f"({best_single['exp_id']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()