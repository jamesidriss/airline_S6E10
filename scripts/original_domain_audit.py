"""How different are the ORIGINAL public survey rows from the COMPETITION rows?

Why this matters before testing any augmentation with original data
-------------------------------------------------------------------
Appending the original 129,880 rows with their own labels was already measured as harmful. The
usual explanation ("different labels") is not obviously the right one, because this generator was
built FROM the original survey, so the feature distributions should be close. If the feature
distributions ARE close, then the damage must come from the conditional p(y|x) rather than from
p(x) -- which would mean original rows are useful as X and useless as labelled examples, and that is
exactly the Family C hypothesis.

So this measures the two separately:

  p(x) agreement
    * a domain classifier (original vs competition train) scored by AUC; 0.5 means indistinguishable
    * nearest-neighbour distance from each original row to the competition set, versus a
      competition-to-competition baseline of the same size
    * marginal agreement per raw column: numeric overlap of quantiles, categorical level overlap
    * category frequency differences, which are the most likely generator fingerprint

  y|x agreement
    * the ORIGINAL-DATA TEACHER's AUC on competition labels (already measured at ~0.9550) versus
      the in-domain model at ~0.9612. A domain classifier that cannot separate the domains while the
      teacher is 6e-3 worse in AUC localises the mismatch in the conditional.

A domain AUC near 0.5 with a materially worse teacher would be a strong, concrete argument that
original rows supply usable X but not usable labels -- and therefore that Family C is worth testing,
while re-appending original labels (already measured harmful) is not.

Usage: python scripts/original_domain_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features import s6e10 as S6  # noqa: E402


def numeric_matrix(df, cols):
    return df[cols].to_numpy(dtype="float64")


def main() -> None:
    tr, te = load_cached_parquet()
    orig = S6.load_original()
    # NOTE: the cached parquet stores categoricals as pandas str (pyarrow-backed), NOT object,
    # so an is_object_dtype test silently classifies all four of them as numeric and then fails to
    # cast them. This is detected explicitly instead.
    def _is_cat(dtype):
        return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) \
            or str(dtype) in ("str", "string", "category")
    cat_cols = [c for c in tr.columns
                if c not in (TARGET, ID_COL) and _is_cat(tr[c].dtype)]
    num_cols = [c for c in tr.columns
                if c not in (TARGET, ID_COL) and c not in cat_cols]
    print(f"competition train {tr.shape} | original {orig.shape}")
    print(f"  numeric cols {len(num_cols)} | categorical cols {len(cat_cols)}")

    out = {"n_comp": int(len(tr)), "n_orig": int(len(orig)),
           "numeric_cols": num_cols, "categorical_cols": cat_cols}

    # ---------------------------------------------------------------- 1. domain classifier
    Xc = numeric_matrix(tr, num_cols)
    Xo = numeric_matrix(orig, num_cols)
    # `Arrival Delay in Minutes` is the one genuinely missing column, so impute before any
    # distance-based or model-based comparison. The median is taken over the POOLED rows, which
    # involves no labels, and it is applied identically to both domains so the comparison is not
    # distorted by differing fill values.
    med = np.nanmedian(np.vstack([Xc, Xo]), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    n_missing_c = int(np.isnan(Xc).sum())
    n_missing_o = int(np.isnan(Xo).sum())
    out["missing_values"] = {"competition": n_missing_c, "original": n_missing_o,
                             "fraction_competition": n_missing_c / max(1, Xc.size),
                             "fraction_original": n_missing_o / max(1, Xo.size)}
    Xc = np.where(np.isnan(Xc), med, Xc)
    Xo = np.where(np.isnan(Xo), med, Xo)
    print(f"\n  missing values: competition {n_missing_c} "
          f"({n_missing_c/max(1,Xc.size):.5f} of numeric cells), "
          f"original {n_missing_o} ({n_missing_o/max(1,Xo.size):.5f})")
    sc = StandardScaler().fit(np.vstack([Xc, Xo]))
    Zc, Zo = sc.transform(Xc), sc.transform(Xo)
    # balance the two domains so AUC is not inflated by class imbalance
    n = min(len(Zc), len(Zo))
    idx_c = np.random.default_rng(0).choice(len(Zc), n, replace=False)
    idx_o = np.random.default_rng(1).choice(len(Zo), n, replace=False)
    Xd = np.vstack([Zc[idx_c], Zo[idx_o]])
    yd = np.r_[np.zeros(n), np.ones(n)]
    aucs = []
    for trn, tst in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        lr = LogisticRegression(max_iter=2000, C=0.3)
        lr.fit(Xd[trn], yd[trn])
        aucs.append(float(roc_auc_score(yd[tst], lr.predict_proba(Xd[tst])[:, 1])))
    dom_auc = float(np.mean(aucs))
    out["domain_classifier_auc"] = dom_auc
    print(f"\n  domain classifier AUC (0.5 = indistinguishable) = {dom_auc:.4f} "
          f"(5-fold, balanced, {n} per side)")

    # ---------------------------------------------------------------- 2. NN distances
    import torch

    from src.models.tabr_retrieval import TorchExactL2Index, verify_matches_reference

    ok = verify_matches_reference(device="cuda", n_db=20_000, n_q=512, dim=16, n=8)["passes"]
    if ok:
        idx = TorchExactL2Index(d_main=len(num_cols), device=torch.device("cuda"),
                                query_chunk=256)
        # The index must be built on competition rows DISJOINT from those used as queries. An
        # earlier version indexed the whole competition set and then queried a sample of that same
        # set, so every query found itself at distance 0 and the density ratio came out as 4.4e6 --
        # an artefact of self-matching, not a property of the domains.
        g = np.random.default_rng(2)
        perm = g.permutation(len(Zc))
        n_half = len(perm) // 2
        idx_rows, qry_rows = perm[:n_half], perm[n_half:n_half + 20_000]
        idx.add(torch.as_tensor(Zc[idx_rows].astype("float32"), device="cuda"))
        n_orig_used = min(20_000, len(Zo))
        d_o, _i = idx.search(torch.as_tensor(Zo[:n_orig_used].astype("float32"),
                                        device="cuda"), 1)
        d_c, _i2 = idx.search(torch.as_tensor(Zc[qry_rows].astype("float32"), device="cuda"), 1)
        out["nn_dist_original_to_comp"] = float(d_o.mean())
        out["nn_dist_comp_to_comp"] = float(d_c.mean())
        out["nn_index_rows"] = int(len(idx_rows))
        out["nn_query_rows_comp"] = int(len(qry_rows))
        out["nn_query_rows_orig"] = int(n_orig_used)
        ratio = float(d_o.mean() / max(d_c.mean(), 1e-9))
        out["nn_dist_ratio"] = ratio
        print(f"  mean NN distance  original -> competition   : {d_o.mean():.4f} (n={n_orig_used})")
        print(f"  mean NN distance  competition -> competition : {d_c.mean():.4f} "
              f"(n={len(qry_rows)}, disjoint from the index)")
        print(f"  ratio = {ratio:.3f}  (1.0 = equally dense; >1 = original rows sit in sparser "
              f"regions of the same support)")
        del idx
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 3. marginal agreement
    marg = {}
    worst = []
    for c in num_cols:
        a, b = tr[c].to_numpy("float64"), orig[c].to_numpy("float64")
        qs = np.linspace(0.05, 0.95, 19)
        qa, qb = np.quantile(a, qs), np.quantile(b, qs)
        scale = max(np.std(a), 1e-9)
        gap = float(np.mean(np.abs(qa - qb)) / scale)
        # fraction of original values outside the competition 1st-99th percentile range
        lo, hi = np.quantile(a, 0.01), np.quantile(a, 0.99)
        outside = float(np.mean((b < lo) | (b > hi)))
        marg[c] = {"mean_quantile_gap_in_sd": round(gap, 4),
                   "frac_orig_outside_comp_p01_p99": round(outside, 4)}
        worst.append((gap, c, outside))
    worst.sort(reverse=True)
    out["numeric_marginals"] = marg
    print("\n  numeric columns, worst 5 by mean quantile gap (in SD units):")
    for gap, c, outside in worst[:5]:
        print(f"    {c:<34} gap={gap:.4f} SD   outside_p01_p99={outside:.4f}")

    cat_cmp = {}
    for c in cat_cols:
        pa = tr[c].astype(str).value_counts(normalize=True)
        pb = orig[c].astype(str).value_counts(normalize=True)
        levels = sorted(set(pa.index) | set(pb.index))
        tvd = float(0.5 * np.abs(pa.reindex(levels).fillna(0)
                                 - pb.reindex(levels).fillna(0)).sum())
        cat_cmp[c] = {"total_variation": round(tvd, 4),
                      "levels_comp": {k: round(float(pa.get(k, 0)), 4) for k in levels},
                      "levels_orig": {k: round(float(pb.get(k, 0)), 4) for k in levels}}
    out["categorical_marginals"] = cat_cmp
    print("\n  categorical columns, total-variation distance (0 = identical frequencies):")
    for c, v in sorted(cat_cmp.items(), key=lambda kv: -kv[1]["total_variation"]):
        print(f"    {c:<34} TV={v['total_variation']:.4f}")

    # ---------------------------------------------------------------- 4. verdict
    max_tv = max(v["total_variation"] for v in cat_cmp.values())
    out["max_categorical_tv"] = max_tv
    verdict = _verdict(out)
    out["interpretation"] = verdict
    print("\n=== interpretation ===")
    for line in verdict:
        print(f"  {line}")

    save_json(out, REPORTS / "original_domain_audit.json")
    print("\nwrote", REPORTS / "original_domain_audit.json")


def _verdict(out):
    lines = []
    a = out.get("domain_classifier_auc")
    if a is not None:
        if a < 0.60:
            lines.append(f"p(x) is close: a logistic domain classifier reaches only AUC {a:.3f}, so "
                         "the original rows are NOT a different marginal distribution.")
        elif a < 0.75:
            lines.append(f"p(x) differs moderately (domain AUC {a:.3f}) -- a shift exists but the "
                         "domains overlap substantially.")
        else:
            lines.append(f"p(x) differs strongly (domain AUC {a:.3f}).")
    r = out.get("nn_dist_ratio")
    if r is not None:
        lines.append(f"Original rows sit {r:.2f}x further from the competition set than competition "
                     f"rows sit from each other; >1 means they occupy sparser regions of the same "
                     f"support.")
    tv = out.get("max_categorical_tv")
    if tv is not None:
        lines.append(f"Largest categorical frequency shift (total variation) = {tv:.3f}.")
    lines.append("Known from the campaign: the original-data TEACHER reaches ~0.9550 on competition "
                 "labels against ~0.9612 for an in-domain model.")
    lines.append("If p(x) is close while the conditional is materially worse, then original rows "
                 "are usable as X and unusable as labelled examples -- which is precisely the "
                 "Family C hypothesis, and a reason re-appending original labels could not help.")
    return lines


if __name__ == "__main__":
    main()
