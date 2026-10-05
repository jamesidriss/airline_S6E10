"""Pseudo-labelled TEST-DISTRIBUTION rows -- the one thing the rejected augmentation work did not cover.

Why this is not "augmentation again"
------------------------------------
Phase 5 rejected synthetic augmentation with a clean mechanism: interpolating between neighbours or
copying rows manufactures no new information, and the damage (off-manifold geometry, ~24e-5) swamped
any benefit. Pseudo-labelling is a different object and deserves its own test rather than being
swept up in that verdict:

  * the added rows are REAL covariate rows drawn from the actual evaluation distribution, not
    synthetic points interpolated in feature space;
  * they are therefore ON-manifold by construction, so the specific mechanism that sank Phase 5
    does not apply;
  * their labels come from a model, so they are noisy, and the risk is confirmation bias rather
    than geometry.

What makes it honestly measurable
---------------------------------
The trick is to treat a held-out FOLD as if it were the unlabelled test set:

  1. fit M0 on outer-fit rows only, early stopping on a 10% carve of outer-fit -> it0
  2. M0 predicts fold k's covariates. Those predictions never saw a fold-k label.
  3. build an augmented training set and refit with the SAME early-stopping protocol
  4. score fold k with its TRUE labels, which were never used for anything

This is completely honest: fold k plays the role of the unlabelled test set, its labels are used
only for the single final scoring step, and the pseudo-labels are no better informed than a real
submission-time pseudo-labelling would be.

THE MATCHED CONTROL IS THE POINT
--------------------------------
Three arms, all with the same row count, the same protocol and the same seed:

  ctl     outer-fit minus inner-ES. No extra rows.                    (the existing protocol)
  dup     ctl + N DUPLICATED outer-fit rows carrying their TRUE labels. This is Phase 5's
          duplicate control: it buys extra row count with zero new information. If an augmented arm
          beats ctl but not dup, the gain is row count, not pseudo-label information.
  pseudo  ctl + N fold-k rows carrying M0's HARD pseudo-labels, selected as the most confident
          fraction by |p - 0.5|.

Arms are identical in size, so `pseudo - dup` isolates exactly the thing under test: are
on-manifold, model-labelled rows from the evaluation distribution worth more than duplicated rows?

Soft targets were rejected in Phase 1-2, so the pseudo-label is HARD. The selection fraction is
predeclared, not tuned on the evaluation fold.

Usage: python scripts/run_pseudo_label.py --folds 0,1 --fracs 0.25,0.5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

BASE = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 127,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True,
        "verbose": -1, "n_jobs": 8}
ROUNDS = 6000


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(p):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(p, dtype="float64"), -35, 35)))


def train(X, yy, esX, esY, seed, rounds=ROUNDS):
    import lightgbm as lgb
    p = dict(BASE)
    p.update(n_estimators=rounds, random_state=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2)
    ds = lgb.Dataset(X, label=yy)
    dv = lgb.Dataset(esX, label=esY, reference=ds)
    m = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m, int(m.best_iteration or rounds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--fracs", default="0.25,0.5")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds
    fracs = [float(f) for f in args.fracs.split(",")]
    klist = [int(x) for x in args.folds.split(",")]

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    fin = store.load_oof("blend_v3_final").astype("float64")
    results = []

    for k in klist:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit, y_int, val, None, inner_seed=k)
        Xv = Xa["val"]
        tr_g, es_g = _inner_es_split(fit, y_int, args.seed + k)
        pos = {int(v): i for i, v in enumerate(fit)}
        tr_l = np.array([pos[int(v)] for v in tr_g])
        es_l = np.array([pos[int(v)] for v in es_g])
        assert not (set(tr_g.tolist()) & set(val.tolist()))
        print(f"\n{'='*94}\nfold {k}   outer-fit={len(fit):,}   train={len(tr_l):,} "
              f"({len(tr_l)/len(y):.1%} of labels)   eval={len(val):,}   feats={len(names)}"
              f"\n{'='*94}", flush=True)

        set_seed(args.seed + k)
        t0 = time.time()
        m0, it0 = train(Xf[tr_l], y[tr_g], Xf[es_l], y[es_g], args.seed + k)
        p_ctl = m0.predict(Xv, num_iteration=it0)
        auc_ctl = float(roc_auc_score(y[val], p_ctl))
        print(f"  ctl                       iter={it0:>5}  AUC={auc_ctl:.6f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

        # M0's predictions on the held-out fold -- the pseudo-labels. No fold-k label was used.
        p_pl = m0.predict(Xv, num_iteration=it0)
        conf = np.abs(p_pl - 0.5)

        for f in fracs:
            n_add = int(len(val) * f)
            sel = np.argsort(-conf)[:n_add]                 # most confident fold-k rows
            hard = (p_pl[sel] > 0.5).astype("float64")
            bal = float(hard.mean())

            # --- arm: pseudo. extra rows are fold-k covariates with M0's hard labels ---
            set_seed(args.seed + k)
            t0 = time.time()
            Xaug = np.vstack([Xf[tr_l], Xv[sel]])
            yaug = np.concatenate([y[tr_g], hard])
            m1, it1 = train(Xaug, yaug, Xf[es_l], y[es_g], args.seed + k)
            p_ps = m1.predict(Xv, num_iteration=it1)
            auc_ps = float(roc_auc_score(y[val], p_ps))
            t_ps = time.time() - t0

            # --- arm: dup. SAME row count, duplicated outer-fit rows with their TRUE labels ---
            rng = np.random.default_rng(args.seed + k + 31)
            dup_l = rng.choice(tr_l, size=n_add, replace=True)
            set_seed(args.seed + k)
            t0 = time.time()
            Xdup = np.vstack([Xf[tr_l], Xf[dup_l]])
            ydup = np.concatenate([y[tr_g], y[fit][dup_l]])
            m2, it2 = train(Xdup, ydup, Xf[es_l], y[es_g], args.seed + k)
            p_dp = m2.predict(Xv, num_iteration=it2)
            auc_dp = float(roc_auc_score(y[val], p_dp))
            t_dp = time.time() - t0

            print(f"  pseudo frac={f:<5} n_add={n_add:>7,} (pseudo-positive rate {bal:.3f})  "
                  f"iter={it1:>5}  AUC={auc_ps:.6f}  vs ctl {(auc_ps-auc_ctl)*1e5:+6.1f}e-5  "
                  f"vs dup {(auc_ps-auc_dp)*1e5:+6.1f}e-5  ({t_ps:.0f}s)", flush=True)
            print(f"  dup    frac={f:<5} n_add={n_add:>7,}                          "
                  f"iter={it2:>5}  AUC={auc_dp:.6f}  vs ctl {(auc_dp-auc_ctl)*1e5:+6.1f}e-5  "
                  f"({t_dp:.0f}s)", flush=True)

            bg = {}
            for w in (0.05, 0.10, 0.20, 0.30, 0.50):
                mix = w * logit(p_ps) + (1 - w) * logit(fin[val])
                bg[str(w)] = float(roc_auc_score(y[val], mix)) - float(
                    roc_auc_score(y[val], fin[val]))
            print(f"         corr(pseudo,ctl) logit={corr(logit(p_ps), logit(p_ctl)):.5f} "
                  f"spearman={spearman(p_ps, p_ctl):.5f} | "
                  f"finalist blend gains: "
                  f"{', '.join(f'w={a}:{b*1e5:+.1f}' for a, b in bg.items())}")
            results.append({
                "fold": k, "frac": f, "n_add": n_add, "pseudo_positive_rate": bal,
                "ctl": {"auc": auc_ctl, "iter": it0},
                "pseudo": {"auc": auc_ps, "iter": it1, "seconds": round(t_ps, 1),
                           "delta_vs_ctl": auc_ps - auc_ctl, "delta_vs_dup": auc_ps - auc_dp,
                           "finalist_blend_gains": bg,
                           "logit_corr_vs_ctl": corr(logit(p_ps), logit(p_ctl))},
                "dup": {"auc": auc_dp, "iter": it2, "seconds": round(t_dp, 1),
                        "delta_vs_ctl": auc_dp - auc_ctl},
            })

    print("\n" + "=" * 94)
    print(f"{'frac':>7}{'fold':>6}{'ctl AUC':>12}{'pseudo':>12}{'dup':>12}"
          f"{'ps-ctl':>10}{'ps-dup':>10}")
    for r in results:
        print(f"{r['frac']:>7.2f}{r['fold']:>6}{r['ctl']['auc']:>12.6f}{r['pseudo']['auc']:>12.6f}"
              f"{r['dup']['auc']:>12.6f}{r['pseudo']['delta_vs_ctl']*1e5:>+9.1f}e"
              f"{r['pseudo']['delta_vs_dup']*1e5:>+9.1f}e")
    print("\n=== the comparison that matters: pseudo - dup ===")
    summ = {}
    for f in fracs:
        pd_ = [r["pseudo"]["delta_vs_dup"] for r in results if r["frac"] == f]
        cc = [r["pseudo"]["delta_vs_ctl"] for r in results if r["frac"] == f]
        summ[str(f)] = {"mean_pseudo_minus_dup": float(np.mean(pd_)),
                        "mean_pseudo_minus_ctl": float(np.mean(cc)),
                        "folds_positive_vs_dup": int(sum(1 for d in pd_ if d > 0)),
                        "n_folds": len(pd_)}
        print(f"  frac={f:<5} pseudo-dup mean {np.mean(pd_)*1e5:+6.1f}e-5  "
              f"(positive {sum(1 for d in pd_ if d > 0)}/{len(pd_)})   "
              f"pseudo-ctl mean {np.mean(cc)*1e5:+6.1f}e-5")

    best = max(summ, key=lambda k2: summ[k2]["mean_pseudo_minus_dup"])
    s = summ[best]
    print(f"\n  best frac={best}: pseudo-dup {s['mean_pseudo_minus_dup']*1e5:+.1f}e-5")
    if s["mean_pseudo_minus_dup"] >= 5e-5 and s["folds_positive_vs_dup"] == s["n_folds"]:
        print("  VERDICT: pseudo-labelled evaluation-distribution rows beat duplicated rows -- the "
              "information is real.\n           PROMOTE to all folds.")
    elif s["mean_pseudo_minus_dup"] > 0:
        print("  VERDICT: positive but below +5e-5. A hint that on-manifold pseudo-labels carry "
              "some information\n           beyond row count, not enough to act on. Hold.")
    else:
        print("  VERDICT: REJECT. Pseudo-labelled evaluation rows do not beat duplicated rows, so "
              "adding test\n           covariates with model-derived labels adds no usable signal. "
              "Confirmation bias cancels any\n           boundary sharpening. Do not spend more "
              "here -- and note this is a DIFFERENT\n           verdict from Phase 5's synthetic "
              "augmentation, with a different mechanism.")

    save_json({"results": results, "summary": summ, "view": args.view, "scheme": args.scheme,
               "folds": klist, "fracs": fracs, "seed": args.seed,
               "mechanism_under_test": ("are on-manifold rows from the evaluation distribution, "
                                        "carrying model-derived hard labels, worth more than the "
                                        "same number of duplicated rows"),
               "honesty": ("fold k plays the role of the unlabelled test set; its labels are used "
                           "only for the single final scoring step, and the pseudo-labels come "
                           "from a model that never saw them"),
               "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip()[:12]},
              REPORTS / "pseudo_label.json")
    print("\nwrote", REPORTS / "pseudo_label.json")


if __name__ == "__main__":
    main()
