"""Build the two finalist candidates and characterise the risk each one carries.

The competition lets us select **up to two** submissions for private judging. Two candidates
with nearly identical OOF are only worth banking if they fail differently. This script builds
them and reports how different they actually are:

  A. `equal_all`      -- equal weight per member. Maximum variance reduction, but the LightGBM
                         family is 32 of 59 members, so a systematic LightGBM-only error would
                         dominate.
  B. `family_balanced`-- equal weight per *family*, split evenly inside the family. Costs ~7e-6
                         of OOF, which is inside the fold noise, but removes the single-family
                         concentration risk.

If the two are rank-identical, banking both is pointless; the script says so explicitly rather
than assuming diversity.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.submission.kaggle_io import remaining, used_today  # noqa: E402
from src.submission.make import build  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def family_of(meta: dict) -> str:
    f = meta.get("family", "?")
    if f == "lgbm" and (meta.get("params") or {}).get("extra_trees"):
        return "lgbm_xt"
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-auc", type=float, default=0.9605)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    idx = store._load_index()

    P, meta = {}, {}
    for k, v in idx.items():
        m = v.get("meta", {})
        if not m.get("auc") or m.get("family") in ("blend", "stack"):
            continue
        if m["auc"] < args.min_auc or "test" not in v:
            continue
        o = store.load_oof(k)
        if len(o) == len(y):
            P[k] = o.astype("float64")
            meta[k] = m
    names = list(P)
    M = np.column_stack([lab.tform(P[n], "logit") for n in names])
    T = np.column_stack([lab.tform(store.load_test(n).astype("float64"), "logit")
                         for n in names])

    fams = defaultdict(list)
    for i, n in enumerate(names):
        fams[family_of(meta[n])].append(i)
    w_fam = np.zeros(len(names))
    for f, ii in fams.items():
        for i in ii:
            w_fam[i] = 1.0 / (len(fams) * len(ii))

    cands = {"equal_all": np.ones(len(names)) / len(names), "family_balanced": w_fam}
    out = {}
    vecs = {}
    for nm, w in cands.items():
        oof = expit(M @ w)
        test = expit(T @ w)
        auc = float(roc_auc_score(y, oof))
        fa = lab.fold_aucs(y, oof, folds)
        vecs[nm] = (oof, test)
        conc = max(float(np.max(np.abs(w[fams[f]]))) for f in fams)
        out[nm] = {"oof_auc": round(auc, 6), "fold_aucs": [round(x, 6) for x in fa],
                   "n_members": len(names), "max_family_concentration": round(conc, 5),
                   "families": {f: len(ii) for f, ii in fams.items()}}
        store.save(f"cand_{nm}", oof, test, fold_scheme="primary",
                   meta={"family": "blend", "featureset": nm, "auc": round(auc, 6),
                         "members": names})
        print(f"{nm:<18} OOF={auc:.6f}  folds={[round(x,6) for x in fa]}  "
              f"max single-family weight={conc:.3f}")
    a, b = "equal_all", "family_balanced"
    sp = float(spearmanr(vecs[a][1], vecs[b][1]).statistic)
    out["test_spearman_between_candidates"] = sp
    print(f"\ntest-prediction Spearman between the two candidates: {sp:.6f}")
    if sp > 0.9995:
        print("  -> rank-identical: banking both adds no diversity. Submit only the higher-OOF one.")
    elif sp > 0.998:
        print("  -> weakly differentiated. Worth banking only if a scheme failure is a real risk.")
    else:
        print("  -> meaningfully differentiated: a genuine hedge.")

    if not args.dry_run:
        for nm in (a, b):
            build(vecs[nm][1], name=f"v4_{nm}", notes=f"finalist candidate {nm}", oof_auc=out[nm]["oof_auc"])
    out["submissions_used_today"] = used_today()
    out["submissions_remaining"] = remaining()
    save_json(out, REPORTS / "finalists.json")
    print("\nwrote", REPORTS / "finalists.json")


if __name__ == "__main__":
    main()