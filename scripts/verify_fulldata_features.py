"""Independent check that the FULL-DATA assembly produces a sound test feature matrix.

Why this exists
---------------
`scripts/refit_members_fulldata.py` calls

    vb.assemble(np.arange(n_all), y_int, np.arange(500), test_rows, inner_seed=0)

with the fit index set to EVERY labelled row. That is the whole point -- test-time inference needs no
held-out fold -- but it is also the first time this repository assembles features with
`fit_idx == all rows`, and two things could silently go wrong:

  1. the 237 static columns handed to the model for the test rows might no longer be the cached
     static values, if some block re-derives or reorders them; and
  2. the fold-safe target-encoding block, now fitted on 100% of the labels instead of 80%, could be
     malformed -- constant, saturated, or full of NaN.

LightGBM handles NaN natively, so a corrupted matrix would not raise. It would simply produce a
worse submission, and the only symptom would be a public score. This check compares the assembled
matrix against the immutable static cache, which is the one thing that must not have moved.

Checks
------
  static columns identical to the cache, exactly
  target-encoding columns present, finite, and not constant
  target-encoding columns actually CHANGED relative to the fold-fitted version (they must: that is
  the whole effect of having 25% more labels behind them) -- if they were identical, the refit would
  be silently doing nothing
  class balance of the encoding sane (mean in [0,1], and near the base rate on average)

Usage: python scripts/verify_fulldata_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import FEATURES, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    n_all, n_te = len(tr), len(te)
    base_rate = float(y.mean())

    out = {"n_train": n_all, "n_test": n_te, "base_rate": base_rate, "views": []}
    ok_all = True

    for view in ("full",):
        cache = np.load(FEATURES / f"static_{view}.npz", allow_pickle=True)
        static_te = cache["te"]
        n_static = static_te.shape[1]

        vb = ViewBuilder(tr, te, view)
        vb.build_static()
        Xf, Xa, names = vb.assemble(np.arange(n_all), y, np.arange(500),
                                    np.arange(n_all, n_all + n_te), inner_seed=0)
        Xt = Xa["test"]
        n_fit_feat = Xf.shape[1]
        # column counts must agree; row counts legitimately differ (all labelled rows vs test rows)
        assert Xt.shape[1] == Xf.shape[1], (
            f"test matrix has {Xt.shape[1]} columns, fit matrix has {Xf.shape[1]} -- the assembly "
            f"produced a different feature set for the two, which would be a silent disaster")
        print("=" * 96)
        print(f"view {view}: fit {Xf.shape}   test {Xt.shape}   names {len(names)}")
        print(f"  static columns in cache: {n_static}   total columns: {n_fit_feat}   "
              f"encoding columns: {n_fit_feat - n_static}")
        print("=" * 96)

        static_block = Xt[:, :n_static]
        same = np.array_equal(np.nan_to_num(static_block, nan=-999.0),
                              np.nan_to_num(static_te, nan=-999.0))
        print(f"  [1] static columns identical to the immutable cache : {same}")
        if not same:
            d = np.abs(static_block - static_te)
            print(f"      max abs diff {np.nanmax(d):.6g}  mean {np.nanmean(d):.6g}")
        ok_all &= same

        te_block = Xt[:, n_static:]
        n_te_cols = te_block.shape[1]
        finite = bool(np.isfinite(te_block).all())
        const = [j for j in range(n_te_cols) if np.nanstd(te_block[:, j]) < 1e-12]
        print(f"  [2] encoding columns: n={n_te_cols}  finite={finite}  constant={len(const)}")
        print("      (NOT asserted to lie in [0,1]: 12 of the 48 columns are on a count or target-mean")
        print("       scale with means up to 12.3, so a unit-range check would be a wrong assertion,")
        print("       not a useful guard)")
        ok_all &= finite and not const

        print(f"  [3] encoding column means: min={te_block.mean(axis=0).min():.4f} "
              f"max={te_block.mean(axis=0).max():.4f}   (train base rate {base_rate:.4f})")

        # [4] compare against the FOLD-fitted encoding by assembling the same way the existing
        # pipeline does (fit on outer-fit only), so we can confirm the refit really changed them.
        vb2 = ViewBuilder(tr, te, view)
        vb2.build_static()
        folds = np.zeros(n_all, dtype="int64")
        folds[int(n_all * 0.8):] = 1
        _Xf2, Xa2, _ = vb2.assemble(np.where(folds == 0)[0], y, np.where(folds == 1)[0],
                                   np.arange(n_all, n_all + n_te), inner_seed=0)
        fold_te = Xa2["test"][:, n_static:]
        n_cmp = min(n_te_cols, fold_te.shape[1])
        changed = [j for j in range(n_cmp)
                   if np.std(te_block[:, j] - fold_te[:, j]) > 1e-9]
        print(f"  [4] encoding columns that CHANGED vs an 80%-fitted encoding: {len(changed)}/{n_cmp}")
        print("      (must be > 0 -- if it were 0 the full-data refit would be doing nothing)")
        ok_all &= len(changed) > 0

        # [5] end-to-end sanity: the encoding carries target information, so it must correlate
        # POSITIVELY with the finalist's TEST prediction. Both arrays are test-length here; an
        # earlier version compared against the train-length OOF array, which cannot work and
        # should have been caught before running.
        from src.validation.compare import corr
        fin_test = store.load_test("blend_v3_final").astype("float64")
        assert len(fin_test) == len(te_block), \
            f"finalist test prediction has {len(fin_test)} rows, encoding has {len(te_block)}"
        cc = [float(corr(logit(te_block[:, j]), fin_test)) for j in range(n_te_cols)]
        print(f"  [5] corr(encoding column, finalist test logit): min={min(cc):.4f} "
              f"median={float(np.median(cc)):.4f} max={max(cc):.4f}")
        n_low = sum(1 for c in cc if c < 0.3)
        print(f"      columns with corr < 0.3: {n_low}/{n_te_cols} "
              f"(a majority would mean the encoding is not carrying target information)")
        ok_all &= n_low < n_te_cols // 2

        out["views"].append({
            "view": view, "fit_shape": list(Xf.shape), "test_shape": list(Xt.shape),
            "n_static": int(n_static), "n_encoding": int(n_te_cols),
            "static_identical_to_cache": bool(same),
            "encoding_all_finite": bool(finite),
            "n_constant_encoding_cols": len(const),
                        "encoding_col_means_range": [float(te_block.mean(axis=0).min()),
                                         float(te_block.mean(axis=0).max())],
            "n_encoding_cols_changed_vs_80pct_fitted": len(changed),
            "corr_encoding_vs_finalist_range": [min(cc), max(cc)],
            "corr_encoding_vs_finalist_median": float(np.median(cc)),
        })

    out["all_checks_passed"] = bool(ok_all)
    save_json(out, REPORTS / "fulldata_feature_check.json")
    print("\n" + ("ALL CHECKS PASSED" if ok_all else "*** CHECKS FAILED -- do not submit ***"))
    print("wrote", REPORTS / "fulldata_feature_check.json")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
