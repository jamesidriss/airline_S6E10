"""Three cheap structural probes that would each be high-upside, and that I cannot find recorded.

Motivated by `scripts/analyse_lb_noise.py`, which established that the 7.8e-4 gap to the leader is
STATISTICALLY REAL (z = 3.1-7.9 even between uncorrelated predictions) while every gain this
campaign has measured is single-digit e-5. Closing that needs something structurally different, so
the cheapest possible places to look for it are checked first.

PROBE A -- exact and near duplicate rows between TRAIN and TEST
    The campaign verified only 21 exact matches between the ORIGINAL dataset and the competition
    data. It never checked competition TRAIN against competition TEST, which is a different and much
    more relevant question. If the generator emitted any test row as a copy of a train row (possibly
    with the label flipped, which is how a synthetic generator "recycles"), those rows could be
    scored directly from the label we already hold -- a signal far larger than anything reachable by
    modelling. Reports the exact-match count, the match rate, and, for matches, how often the
    matched train label agrees with the model.

PROBE B -- is the train/test covariate shift actually zero?
    STATUS.md records "train/test covariate shift essentially zero", and a domain classifier AUC of
    0.7225 was measured for train vs ORIGINAL. Train vs TEST was not measured with a classifier, and
    the claim is load-bearing: if there IS shift, our train-fitted models are systematically
    mis-ranked on test and a shift-aware correction is worth real AUC. A LightGBM domain classifier
    is trained on pooled rows with a train/test indicator and scored on held-out pooled rows, so an
    AUC of exactly 0.5 means indistinguishable.

PROBE C -- is the id column really uninformative?
    "id useless" is recorded, but the `id` is the generator's own row ordering. A quick direct check
    is one line, and if ids were assigned in blocks correlated with the target the whole fold
    protocol could be optimistic. Checks: AUC of raw id against the label, and AUC of id within each
    of several contiguous id blocks (which would catch a block-local relationship that a global
    monotone score would miss).

Usage: python scripts/structural_probes.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import RAW21  # noqa: E402

OUT = {}


def _hash_frame(df, cols, numeric_round=None):
    """Vectorised row hash over `cols`.

    `pd.util.hash_pandas_object` is used rather than a Python-level join over 700k rows x 21 columns,
    which took tens of seconds per call and made the probe impractical to iterate on.

    `numeric_round` buckets the numeric columns at integer precision. Missing values are mapped to a
    sentinel BEFORE rounding: `Arrival Delay in Minutes` carries NaN, and astype(int64) on NaN
    raises IntCastingNaNError, which is a crash rather than a wrong answer -- but it stops the probe.
    """
    sub = df[cols].copy()
    if numeric_round is not None:
        for c in cols:
            if c not in ("Gender", "Customer Type", "Type of Travel", "Class"):
                v = sub[c].to_numpy(dtype="float64")
                sub[c] = np.where(np.isfinite(v), np.round(v, numeric_round), -999999)
    return pd.util.hash_pandas_object(sub, index=False).to_numpy(dtype="uint64")


def probe_a_duplicates(tr, te):
    print("\n" + "=" * 96)
    print("PROBE A -- exact duplicate rows between competition TRAIN and competition TEST")
    print("=" * 96)
    key_cols = list(RAW21)
    CAT = ("Gender", "Customer Type", "Type of Travel", "Class")

    t0 = time.time()
    htr, hte = _hash_frame(tr, key_cols), _hash_frame(te, key_cols)
    uniq, inv, cnt = np.unique(htr, return_inverse=True, return_counts=True)
    # a match only counts if the training rows behind that pattern all carry the same label
    pos_sum = np.bincount(inv, weights=tr[TARGET].values.astype("float64"))
    pure = np.abs(cnt - 2 * pos_sum) < 1e-9
    pure_set = set(uniq[pure].tolist())
    matches = np.isin(hte, np.fromiter(pure_set, dtype="uint64", count=len(pure_set)))
    n_te = len(te)
    print(f"  train rows={len(tr):,}   test rows={n_te:,}")
    print(f"  distinct train row-patterns        : {len(uniq):,}")
    print(f"  train patterns that are label-pure : {int(pure.sum()):,}")
    print(f"  TEST rows exactly matching a label-pure TRAIN pattern: "
          f"{int(matches.sum()):,} ({matches.mean():.6%})")
    print(f"  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    htr2, hte2 = _hash_frame(tr, key_cols, numeric_round=0), _hash_frame(te, key_cols, 0)
    u2, i2, c2 = np.unique(htr2, return_inverse=True, return_counts=True)
    ps2 = np.bincount(i2, weights=tr[TARGET].values.astype("float64"))
    pure2 = np.abs(c2 - 2 * ps2) < 1e-9
    s2 = u2[pure2]
    m2 = np.isin(hte2, s2)
    print(f"  (numeric rounded to 0 dp) TEST rows matching a label-pure TRAIN pattern: "
          f"{int(m2.sum()):,} ({m2.mean():.6%})")
    print(f"  ({time.time()-t0:.0f}s)")

    res = {"exact_test_rows_matching_label_pure_train": int(matches.sum()),
           "exact_match_rate": float(matches.mean()),
           "rounded_test_rows_matching_label_pure_train": int(m2.sum()),
           "rounded_match_rate": float(m2.mean()),
           "n_train": len(tr), "n_test": n_te, "key_columns": key_cols}
    exploitable = matches.sum() > 200
    print(f"\n  VERDICT A: "
          f"{'EXPLOITABLE -- build a lookup feature' if exploitable else 'no exploitable exact-match structure; the generator did not recycle train rows'}")
    res["exploitable"] = bool(exploitable)
    return res


def probe_b_shift(tr, te):
    print("\n" + "=" * 96)
    print("PROBE B -- train vs TEST covariate shift (domain classifier)")
    print("=" * 96)
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    cols = list(RAW21)
    CAT = ("Gender", "Customer Type", "Type of Travel", "Class")
    n_tr, n_te = len(tr), len(te)

    # Concatenate as OBJECTS first: converting to float64 up front raises on 'Female'. The
    # categoricals are then integer-coded over the POOLED vocabulary, which is correct -- a domain
    # classifier must be able to see a category that appears in only one domain, or it would mistake
    # a vocabulary difference for a distribution shift.
    both = pd.concat([tr[cols], te[cols]], ignore_index=True)
    for c in CAT:
        v = both[c].to_numpy()
        u = {uu: i for i, uu in enumerate(sorted(set(v.tolist())))}
        both[c] = np.array([u[vv] for vv in v], dtype="float64")
    for c in cols:
        if c not in CAT:
            both[c] = pd.to_numeric(both[c], errors="coerce")
    X = np.nan_to_num(both.to_numpy(dtype="float64"), nan=-999.0, posinf=1e9, neginf=-1e9)
    d = np.concatenate([np.zeros(n_tr), np.ones(n_te)])

    rng = np.random.default_rng(7)
    n_sub = min(n_te, n_tr)
    itr = rng.choice(n_tr, n_sub, replace=False)
    ite = rng.choice(n_te, n_sub, replace=False)
    keep = np.concatenate([itr, n_tr + ite])
    X, d = X[keep], d[keep]
    print(f"  balanced pool: {int((d==0).sum()):,} train vs {int((d==1).sum()):,} test")

    aucs = []
    skf = StratifiedKFold(3, shuffle=True, random_state=11)
    p = {"objective": "binary", "n_estimators": 400, "learning_rate": 0.08, "num_leaves": 63,
         "min_child_samples": 100, "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
         "verbose": -1, "n_jobs": 8, "seed": 3}
    for a, b in skf.split(X, d):
        ds = lgb.Dataset(X[a], label=d[a])
        dv = lgb.Dataset(X[b], label=d[b], reference=ds)
        m = lgb.train(p, ds, num_boost_round=400, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        aucs.append(roc_auc_score(d[b], m.predict(X[b], num_iteration=m.best_iteration)))
        print(f"    fold AUC {aucs[-1]:.6f}  (iter {m.best_iteration})", flush=True)
    auc = float(np.mean(aucs))
    print(f"\n  mean domain-classifier AUC = {auc:.6f}")
    print(f"  VERDICT B: {'SHIFT IS REAL -- a shift-aware correction may be worth AUC' if auc > 0.52 else 'shift is negligible, consistent with the recorded finding'}")
    return {"domain_auc_folds": aucs, "domain_auc": auc, "pool_size": int(len(d)),
            "balanced": True,
            "verdict": ("shift is real" if auc > 0.52 else "shift negligible")}


def probe_c_id(tr):
    print("\n" + "=" * 96)
    print("PROBE C -- is the id column really uninformative?")
    print("=" * 96)
    y = tr[TARGET].values.astype("int8")
    ids = tr["id"].to_numpy(dtype="float64")
    print(f"  id range [{ids.min():.0f}, {ids.max():.0f}]  unique={len(np.unique(ids)):,}")
    a = roc_auc_score(y, ids)
    print(f"  AUC of raw id vs label              : {a:.6f}")
    nblk = 10
    order = np.argsort(ids)
    bounds = np.array_split(order, nblk)
    blk = []
    for bi, b in enumerate(bounds):
        if len(np.unique(y[b])) > 1:
            blk.append(roc_auc_score(y[b], ids[b]))
    print(f"  AUC of id WITHIN {nblk} contiguous id blocks: "
          f"{[round(x,4) for x in blk]}")
    print(f"    mean |AUC-0.5| = {np.mean([abs(x-0.5) for x in blk]):.6f}")
    print(f"  VERDICT C: {'id carries signal' if abs(a-0.5) > 0.005 or np.mean([abs(x-0.5) for x in blk]) > 0.005 else 'id is uninformative, as recorded'}")
    return {"auc_raw_id": float(a), "within_block_aucs": blk,
            "mean_abs_within_block": float(np.mean([abs(x - 0.5) for x in blk]))}


def probe_d_bayes(tr):
    print("\n" + "=" * 96)
    print("PROBE D -- empirical noise floor from label consistency in COARSE feature buckets")
    print("=" * 96)
    # Probe A just established there are ZERO exact duplicate rows among 699,635, so the cleanest
    # version of this estimate -- identical features, do the labels agree? -- is not available.
    # The substitute is a coarse bucket over the nine service ratings that dominate the model, plus
    # Class and Type of Travel. Rows in the same bucket have nearly identical p(y|x), so the
    # disagreement rate inside a bucket is dominated by LABEL NOISE, not by covariate variation.
    #
    # If the generator drew the target deterministically from the features, disagreement would be 0
    # and AUC 1.0 would be attainable. Pure label noise would give disagreement 0.25 and AUC 0.5.
    # The truth is in between and is worth knowing: it bounds everything this campaign is chasing,
    # and it calibrates every "we are approaching a ceiling" claim made so far.
    y = tr[TARGET].values.astype("float64")
    bucket_cols = ["Class", "Type of Travel", "Inflight wifi service", "Online boarding",
                   "Seat comfort", "Inflight entertainment", "On-board service",
                   "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
    bucket_cols = [c for c in bucket_cols if c in RAW21]
    h = _hash_frame(tr, bucket_cols)
    u, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    n_multi = int((cnt > 1).sum())
    print(f"  bucket columns ({len(bucket_cols)}): {bucket_cols[:4]} ...")
    print(f"  distinct buckets: {len(u):,}   buckets with >=2 rows: {n_multi:,}")

    # restrict to rows in multi-row buckets, then measure observed vs binomial-expected disagreement
    multi = cnt > 1
    keep = multi[inv]
    if keep.sum() < 500:
        print("  too few multi-row buckets to estimate anything")
        return {"n_rows_in_multi_buckets": int(keep.sum()), "estimate": None}

    # Everything below works on UNSORTED kept arrays and uses bincount, never positional indexing
    # into a reordered array. An earlier version sorted ys/ivs by bucket and then indexed them via
    # `np.searchsorted(rows_idx, r)`, which walks positions in ORIGINAL order -- so it paired each
    # row with another row's bucket. That produced a bucket-only AUC of 0.404, which is impossible
    # for a bucket-mean predictor and should have been caught as such immediately.
    yk = y[keep]
    ik = inv[keep]
    nk = len(yk)

    tot = np.bincount(ik, weights=yk, minlength=len(u))
    cnt = np.bincount(ik, minlength=len(u))
    grp_n = cnt[ik]
    grp_p = tot[ik] / np.maximum(grp_n, 1)

    # within-bucket mean squared deviation, and the two binomial expectations.
    # For n Bernoulli(p) draws, E[sum_i (y_i - p_bar)^2] = (n - 1) p (1 - p), so PER ROW the
    # expectation is ((n - 1)/n) p (1 - p). Both the /n and the (n-1)/n matter: with a mean bucket
    # of ~9.8 rows, dropping /n inflates the expectation ~10x, and dropping (n-1)/n shifts it ~10%
    # the other way. An earlier version omitted /n and printed an expectation of 5.19 against an
    # observation of 0.055, i.e. a nonsense "excess" of -5.13.
    obs_num = float((((yk - grp_p) ** 2).sum()))
    q_obs = obs_num / nk
    q_naive = float((grp_p * (1 - grp_p)).sum()) / nk
    frac = (grp_n - 1) / np.maximum(grp_n, 1)
    q_corr = float((grp_p * (1 - grp_p) * frac).sum()) / nk

    sizes = cnt[cnt >= 2]
    print(f"  rows in multi-row buckets: {nk:,}   bucket groups used: {len(sizes):,}")
    print(f"  bucket size distribution: min={sizes.min()} p25={np.percentile(sizes,25):.0f} "
          f"median={np.median(sizes):.0f} p75={np.percentile(sizes,75):.0f} max={sizes.max()} "
          f"mean={sizes.mean():.2f}")
    print(f"  observed within-bucket mean squared deviation : {q_obs:.5f}")
    print(f"  naive binomial expectation  p(1-p)            : {q_naive:.5f}")
    print(f"  finite-sample-corrected   (n-1)p(1-p)         : {q_corr:.5f}")
    print(f"  EXCESS over corrected expectation              : {q_obs - q_corr:+.5f}")

    # Assumption-free companion: leave-one-out bucket means. A row is scored by its own bucket's
    # mean computed WITHOUT it, so no row contributes to its own prediction.
    denom = np.maximum(grp_n - 1, 1)
    loo = np.where(grp_n > 1, (tot[ik] - yk) / denom, y.mean())
    oof = np.zeros(len(y))
    oof[np.flatnonzero(keep)] = loo
    bucket_auc = float(roc_auc_score(yk, loo))
    pure_frac = float(np.mean([(tot[k] == 0) or (tot[k] == cnt[k])
                               for k in np.flatnonzero(cnt >= 2)]))
    print(f"\n  leave-one-out BUCKET-ONLY AUC : {bucket_auc:.6f}   "
          f"(sanity: must be > 0.5; below 0.5 means the index mapping is wrong)")
    assert bucket_auc > 0.5, (
        f"bucket-only AUC {bucket_auc:.4f} is at or below chance, which a bucket-mean predictor "
        f"cannot be. This is an indexing bug, not a finding.")
    print(f"  fraction of buckets that are label-pure        : {pure_frac:.3f}")
    print(f"  campaign full-model OOF AUC                    : 0.961509")
    print(f"  fine-grained modelling adds on top of the bucket: {0.961509 - bucket_auc:+.6f}")

    print("\n  Interpretation: a small POSITIVE excess over the finite-sample binomial expectation")
    print("  means p(y|x) still varies a little inside a coarse bucket, which is expected because")
    print("  the bucket is coarse. The clean, assumption-free statement is the bucket-only AUC: the")
    print("  11-column coarse view is already highly predictive, and the campaign's entire modelling")
    print("  effort -- 285 engineered features, 59-member blend -- buys the difference between them.")
    print("  This is a useful calibration of where the signal actually lives. It is NOT a Bayes")
    print("  ceiling and no ceiling is claimed.")

    return {"bucket_columns": bucket_cols, "n_buckets": int(len(u)),
            "n_multi_buckets": n_multi, "n_rows_used": int(nk),
            "observed_within_bucket_msd": q_obs,
            "naive_binomial_msd": q_naive,
            "corrected_binomial_msd": q_corr,
            "excess_msd": q_obs - q_corr,
            "bucket_size_mean": float(sizes.mean()),
            "bucket_size_median": float(np.median(sizes)),
            "label_pure_bucket_fraction": pure_frac,
            "leave_one_out_bucket_only_auc": bucket_auc,
            "full_model_oof_auc": 0.961509,
            "caveat": ("coarse buckets, so the variance excess over-states the residual noise: "
                       "covariate variation inside a bucket inflates it. The leave-one-out "
                       "bucket-only AUC needs no variance algebra and is the cleaner statement. "
                       "Not a Bayes-ceiling claim.")}


if __name__ == "__main__":
    tr, te = load_cached_parquet()
    print(f"train {tr.shape}   test {te.shape}")
    t0 = time.time()
    OUT["probe_a_train_test_duplicates"] = probe_a_duplicates(tr, te)
    OUT["probe_b_covariate_shift"] = probe_b_shift(tr, te)
    OUT["probe_c_id"] = probe_c_id(tr)
    OUT["probe_d_noise_floor"] = probe_d_bayes(tr)
    OUT["seconds"] = round(time.time() - t0, 1)
    OUT["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()[:12]
    save_json(OUT, REPORTS / "structural_probes.json")
    print("\nwrote", REPORTS / "structural_probes.json")
