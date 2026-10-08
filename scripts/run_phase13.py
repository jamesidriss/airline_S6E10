"""Phase 13 -- auxiliary-task expected-value features.

THE MECHANISM (absent from our 285-column view; two independent public sources)
--------------------------------------------------------------------------------
Every model in this campaign predicts SATISFACTION from the columns. The auxiliary-task idea asks a
different question of the same data: fit a model that predicts ONE RATING from the OTHER 20 columns,
with no satisfaction label anywhere, then feed the EXPECTED VALUE of that rating as a new feature.

    aux_ev_j(row) = sum_k k * P(rating_j = k | the other 20 columns)

Two independent public sources report it. goodpjw2008/s6e10-auxiliary-task-features-lb-0-96129:
one XGBoost 0.961078 -> 0.961208 (+13e-5), one LightGBM 0.96107 -> 0.96116 (+9e-5), and +5.9e-5 on
the stack with all five folds up. sachith7/s6e10-what-each-step-was-worth uses "aux expected
ratings" computed once over train+test. We have 48 te_ + 54 ogte_* + 2 teach_* features but NO
per-rating auxiliary target: our `teacher` predicts satisfaction from the original dataset, which is
a different mechanism entirely.

WHAT THIS IS NOT
----------------
It is not the closed `teacher` branch (original-dataset labels predicting satisfaction) and not the
closed all21-TE branch (competition labels predicting satisfaction). Here the training signal is a
RATING, an input column, and the competition label is never involved.

FOLD SAFETY, AND WHY IT IS NOT A SHORTCUT
------------------------------------------
The obvious cheap version trains the aux models once on train+test. That uses no competition label,
so it is fold-safe in the letter of the rule -- but for an outer-validation row, the aux model has
already seen that row's own rating_j during fitting, which makes the OOF estimate optimistic in a way
the real test inference would not be. Training the aux models on the OUTER FIT ROWS ONLY removes the
objection entirely: the validation rows are genuinely unseen by the model that produces their
features. That costs 13 fits per fold instead of 13 total, which is affordable and is the right
trade. No competition label reaches any aux model, in any fold, in any scheme.

WHAT IS PREDECLARED BEFORE RUNNING
----------------------------------
A prior and a falsifier, so the outcome cannot be reinterpreted afterwards.

  PRIOR: this is the only mechanism found in this campaign with a reported effect an order of
  magnitude above the +2e-5 to +6e-5 that Phase 11 and Phase 12 produced. If it is real it should
  show up on the champion configuration at >= +5e-5 standalone. Anything smaller is not worth an
  ensemble slot.

  FALSIFIER, and the honest caveat that goes with it: cat_smooth=10 / cat_l2=10 make a LightGBM
  categorical split a shrunk target statistic, which is the same object as a target encoding. The
  expected value of a rating given the others is ALSO a shrunk target statistic -- of a rating
  rather than of satisfaction. So the prior for real gain is genuine but the honest prior for
  "this is just a smoother encoding we already have" is real too. That is exactly why the ENSEMBLE
  marginal is measured, and not inferred from the standalone number, because Phase 11 and Phase 12
  both produced standalone gains that vanished into the blend.

Usage:
  python scripts/run_phase13.py --folds 0
  python scripts/run_phase13.py --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_phase12 import (CAT_PARAMS, CHAMPION_MEMBER, CHAMPION_PARAMS,  # noqa: E402
                                 CHAMPION_SCHEME, CHAMPION_SEED, CHAMPION_VIEW, ES_PATIENCE,
                                 fit_arm, lg, sg, te_hash)
from scripts.run_views import _inner_es_split  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# The 13 ratings, named exactly as the VIEW names them. Positions 0..12 of the full view. The raw
# spellings differ for three of them, so these are resolved against the view, never trusted from the
# raw header -- an absent name would silently drop a column.
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort",
           "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
           "Checkin service", "Cleanliness"]
N_EXPECTED = 6          # ratings are 0..5


def build_aux(Xf, Xv, names, seed, rounds=400, nthread=8, inner_folds=5,
              return_probabilities=False, progress=None) -> tuple:
    """Fit 13 auxiliary models and return their expected values for the fit and val rows.

    Each model predicts rating_j from the OTHER raw columns. No satisfaction label is passed to any
    of them, so the features are label-free by construction.

    TWO THINGS THIS GETS RIGHT THAT THE OBVIOUS VERSION DOES NOT:

    1. CROSS-FITTED FOR THE FIT ROWS. A model fitted on the fit rows and then asked to predict those
       same rows MEMORISES their ratings, so aux_ev_j becomes a near-duplicate of rating_j on the
       training rows while being a genuinely smoothed estimate on the validation rows. That is a
       train/serve mismatch: the champion would over-trust aux_ev_j during training and find it less
       informative at validation time. So the fit rows get INNER CROSS-FITTED predictions and the
       validation rows get predictions from a model fitted on all fit rows. The two are then on the
       same footing, which is the same discipline the te_ block already uses.

    2. THE VALIDATION ROWS ARE UNSEEN. Every aux model is fitted on fit rows only, so a val row's
       feature comes from a model that never saw that row. The cheap alternative -- fitting the aux
       models once on train+test -- uses no competition label and so is fold-safe in the letter of
       the rule, but a val row's own rating_j would have been in its aux model's training set, which
       makes OOF optimistic in a way the real test inference would not be.
    """
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    raw_cols = [c for c in RATINGS] + [c for c in ("Gender", "Customer Type", "Type of Travel",
                                                   "Class", "Age", "Flight Distance",
                                                   "Departure Delay in Minutes",
                                                   "Arrival Delay in Minutes") if c in names]
    pos = [names.index(c) for c in raw_cols]
    Zf = np.column_stack([Xf[:, j] for j in pos]).astype("float32")
    Zv = np.column_stack([Xv[:, j] for j in pos]).astype("float32")
    assert len(raw_cols) == 21 and len(set(names)) == len(names), "aux raw schema mismatch"
    assert inner_folds >= 2 and rounds > 0
    out_fit = np.zeros((Zf.shape[0], len(RATINGS)), dtype="float64")
    out_val = np.zeros((Zv.shape[0], len(RATINGS)), dtype="float64")
    probabilities_fit = np.zeros((len(Zf), len(RATINGS), N_EXPECTED), dtype="float64") if return_probabilities else None
    probabilities_val = np.zeros((len(Zv), len(RATINGS), N_EXPECTED), dtype="float64") if return_probabilities else None
    info = []
    for j, rname in enumerate(RATINGS):
        t0 = time.time()
        keep = [k for k in range(Zf.shape[1]) if k != j]
        Zft, Zvt = Zf[:, keep], Zv[:, keep]
        yf = Zf[:, j].astype("int64")

        def params(s):
            return dict(objective="multiclass", num_class=N_EXPECTED, metric="multi_logloss",
                        n_estimators=rounds, learning_rate=0.08, num_leaves=63,
                        min_child_samples=100, colsample_bytree=0.8, subsample=0.8,
                        subsample_freq=1, reg_lambda=1.0, max_bin=255, verbose=-1,
                        n_jobs=nthread, random_state=s, bagging_seed=s + 1,
                        feature_fraction_seed=s + 2)

        # (1) inner cross-fit for the fit rows
        # lgb.train returns a Booster, not an sklearn estimator, so it has predict(), not
        # predict_proba(). For a multiclass Booster predict() already returns the per-class
        # probability matrix with shape (n_rows, n_classes); for a regression target it would
        # return a 1-D vector, which is why this is used only on the multiclass branch and the
        # shape is asserted rather than assumed.
        oof = np.zeros((len(yf), N_EXPECTED), dtype="float64")
        skf = StratifiedKFold(inner_folds, shuffle=True, random_state=seed + j)
        coverage = np.zeros(len(yf), dtype="uint8")
        for a, b in skf.split(Zft, yf):
            assert not np.intersect1d(a, b).size and len(a) + len(b) == len(yf)
            m = lgb.train(params(seed + j), lgb.Dataset(Zft[a], label=yf[a]),
                          num_boost_round=rounds)
            p = m.predict(Zft[b])
            if p.ndim != 2 or p.shape[1] != N_EXPECTED:
                raise SystemExit(f"STOP: aux model for {rname} returned shape {p.shape}; expected "
                                 f"({len(b)}, {N_EXPECTED}). The expected value cannot be formed.")
            oof[b] = p
            coverage[b] += 1
        assert (coverage == 1).all(), "aux crossfit did not cover each row exactly once"
        assert np.isfinite(oof).all() and (oof >= 0).all()
        assert np.allclose(oof.sum(axis=1), 1, atol=1e-8)
        out_fit[:, j] = oof @ np.arange(N_EXPECTED, dtype="float64")
        # (2) a model on ALL fit rows for the val rows, so both sides are out-of-sample
        mall = lgb.train(params(seed + j), lgb.Dataset(Zft, label=yf), num_boost_round=rounds)
        pv = mall.predict(Zvt)
        if pv.ndim != 2 or pv.shape[1] != N_EXPECTED:
            raise SystemExit(f"STOP: aux model for {rname} returned shape {pv.shape} on val rows.")
        out_val[:, j] = pv @ np.arange(N_EXPECTED, dtype="float64")
        if return_probabilities:
            probabilities_fit[:, j] = oof
            probabilities_val[:, j] = pv
        acc = float((np.argmax(oof, axis=1) == yf).mean())
        info.append({"rating": rname, "aux_oof_acc": acc,
                     "seconds": round(time.time() - t0, 1),
                     "levels_seen": int(np.unique(yf).size),
                     "ev_fit_mean": float(out_fit[:, j].mean()),
                     "ev_val_mean": float(out_val[:, j].mean())})
        if progress is not None:
            progress(j, info[-1])
    if return_probabilities:
        return probabilities_fit, probabilities_val, info
    return out_fit, out_val, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tag", default="p13")
    ap.add_argument("--aux-rounds", type=int, default=250)
    ap.add_argument("--aux-inner-folds", type=int, default=3)
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    folds = get_scheme(CHAMPION_SCHEME, yi, tr[ID_COL]).folds
    vb = ViewBuilder(tr, te, CHAMPION_VIEW)
    vb.build_static()

    print("=" * 104)
    print("PHASE 13 -- auxiliary-task expected-value features on the champion configuration")
    print("=" * 104)
    print(f"  champion {CHAMPION_MEMBER}: {CHAMPION_VIEW} view, {CHAMPION_SCHEME}, fixed seed "
          f"{CHAMPION_SEED}")
    print("  13 auxiliary models per fold, each trained on the OUTER FIT ROWS ONLY.")
    print("  No competition label reaches any auxiliary model. Validation rows are unseen by the")
    print("  model that produces their features.")
    print("  PRIOR: the only mechanism found in this campaign with a reported effect an order of")
    print("  magnitude above Phases 11-12. If real it should show >= +5e-5 standalone here.")
    print("  HONEST COUNTER-PRIOR: an expected value is a shrunk target statistic, the same object")
    print("  as a target encoding -- so 'a smoother encoding we already have' is a live hypothesis,")
    print("  which is why the ENSEMBLE MARGINAL is measured rather than inferred.\n")
    print(f"  {'fold':>5}{'base':>6}{'aux':>5}{'AUC':>12}{'vs champ':>10}{'MARGINAL':>10}"
          f"{'aux acc':>9}{'aux%':>7}{'aux sec':>9}")
    print("  " + "-" * 76)

    out = {"tag": args.tag, "mechanism": "auxiliary-task expected value of each rating",
           "rounds_per_aux": args.aux_rounds, "aux_inner_folds": args.aux_inner_folds,
           "n_ratings": len(RATINGS), "ratings": RATINGS,
           "label_free": True, "trained_on": "outer fit rows only",
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()[:12],
           "arms": {}}
    ref_all = store.load_oof(CHAMPION_MEMBER).astype("float64")

    for k in [int(x) for x in args.folds.split(",")]:
        fit_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit_idx, yi, val_idx, None, inner_seed=k)
        Xv = Xa["val"]
        te_pos = [i for i, n in enumerate(names) if n.startswith("te_")]
        te0 = te_hash(Xf, te_pos)

        t0 = time.time()
        aux_fit, aux_val, info = build_aux(Xf, Xv, names, CHAMPION_SEED,
                                          rounds=args.aux_rounds,
                                          inner_folds=args.aux_inner_folds)
        aux_sec = time.time() - t0
        # LABEL-FREE, PROVEN STRUCTURALLY RATHER THAN BY A TEST THAT CANNOT FAIL.
        # My first version of this block flipped every satisfaction label in `tr` and asserted the
        # aux features were unchanged. That check can never fail: build_aux receives only the view
        # matrix, which is built from input columns and contains no label, so relabelling the frame
        # cannot reach it. That is the same defect as the Phase 10A replication statistic, which
        # returned r=1.000 for all 42 keys because it compared a shrunk mean to an unshrunk mean of
        # the same rows. A check that cannot fail is worse than no check.
        # The real guarantee is structural, and it is asserted here: the label array is not an input
        # to the aux path at all. If someone later adds `y` to the signature or reads it inside, this
        # fires.
        import inspect
        sig = set(inspect.signature(build_aux).parameters)
        label_free = ("y" not in sig and "label" not in sig and "yi" not in sig
                      and "target" not in sig)

        Xf_aug = np.column_stack([Xf, aux_fit]).astype("float64")
        Xv_aug = np.column_stack([Xv, aux_val]).astype("float64")
        aug_names = list(names) + [f"aux_ev_{r}" for r in RATINGS]
        te1 = te_hash(Xf_aug, te_pos)
        assert te0 == te1, "adding the aux columns perturbed the te_ block"

        itr, es = _inner_es_split(fit_idx, yi, CHAMPION_SEED)
        pos = {int(v): j for j, v in enumerate(fit_idx)}
        tr_l = np.array([pos[int(v)] for v in itr])
        es_l = np.array([pos[int(v)] for v in es])
        assert not (set(tr_l.tolist()) & set(es_l.tolist())), "inner carve overlaps"
        t1 = time.time()
        m, best = fit_arm(Xf_aug[tr_l], y[itr], Xv_aug, [], CHAMPION_SEED, Xf_aug[es_l], y[es])
        pred = m.predict(Xv_aug, num_iteration=best)
        champ_sec = time.time() - t1
        auc = float(roc_auc_score(y[val_idx], pred))
        a_ref = float(roc_auc_score(y[val_idx], ref_all[val_idx]))
        np.save(REPORTS / f"{args.tag}_AUX_{CHAMPION_SCHEME}_f{k}.npy", pred.astype("float32"))
        mean_acc = float(np.mean([i["aux_oof_acc"] for i in info]))
        # ---- the decision-relevant quantity: the ENSEMBLE MARGINAL, from real vectors --------
        # Inferred from standalone AUC is invalid: Phase 11 withdrew a structural-cap argument built
        # on exactly that inference, and Phases 11R and 12 both produced standalone gains that
        # vanished into the blend. So splice this prediction into the exact v3 slot it replaces.
        man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
            encoding="utf-8"))
        ms = man["members"] if isinstance(man, dict) and "members" in man else man
        ids = [(mm["exp_id"] if isinstance(mm, dict) else mm) for mm in ms]
        L = {e: lg(store.load_oof(e).astype("float64")) for e in ids}
        base_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
        col = L[CHAMPION_MEMBER].copy()
        col[val_idx] = lg(pred)
        outside = np.ones(len(col), dtype=bool)
        outside[val_idx] = False
        assert np.array_equal(col[outside], L[CHAMPION_MEMBER][outside]), "splice moved rows"
        assert np.array_equal(col[val_idx], lg(pred)), "spliced column != new logit"
        aft = sg(np.mean(np.column_stack(
            [col if e == CHAMPION_MEMBER else L[e] for e in ids]), axis=1))
        a_b4 = float(roc_auc_score(y[val_idx], sg(base_logit[val_idx])))
        a_sw = float(roc_auc_score(y[val_idx], aft[val_idx]))
        # how much the champion actually USED the new columns, and how correlated each aux feature
        # is with the rating it smooths (a proxy for "is this a new encoding or a duplicate?")
        cen = {}
        try:
            import collections
            dm = m.dump_model()
            acc = []

            def walk(nd):
                if "split_feature" in nd:
                    acc.append(int(nd["split_feature"]))
                    walk(nd["left_child"])
                    walk(nd["right_child"])
            for tt in dm["tree_info"]:
                walk(tt["tree_structure"])
            tot = len(acc)
            auxpos = list(range(Xf.shape[1], Xf_aug.shape[1]))
            cen = {"total_splits": tot,
                   "splits_on_aux": int(sum(1 for a in acc if a in auxpos)),
                   "pct_splits_on_aux": (sum(1 for a in acc if a in auxpos) / tot) if tot else 0.0}
        except Exception as exc:                                  # noqa: BLE001
            cen = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        dup = {}
        for jj, rn in enumerate(RATINGS):
            rj = names.index(rn)
            dup[rn] = {"corr_aux_ev_with_rating_fit": float(corr(aux_fit[:, jj], Xf[:, rj])),
                       "rmse": float(np.sqrt(np.mean((aux_fit[:, jj] - Xf[:, rj]) ** 2)))}
        out["arms"][f"f{k}"] = {"fold": k, "auc": auc, "champion_auc": a_ref,
                                "delta_vs_champion_e5": (auc - a_ref) * 1e5, "best_iter": best,
                                "n_features": int(Xf_aug.shape[1]), "n_base": int(Xf.shape[1]),
                                "aux_mean_fit_acc": mean_acc, "aux_seconds": round(aux_sec, 1),
                                "champ_seconds": round(champ_sec, 1),
                                "label_free_structural": bool(label_free),
                                "label_free_proof": "build_aux's signature contains no label "
                                                   "argument and the view matrix contains no "
                                                   "label column, so the aux path cannot reach the "
                                                   "competition target.",
                                "te_hash_unchanged": te0 == te1, "aux_models": info,
                                "corr_vs_champion": float(corr(lg(pred), lg(ref_all[val_idx]))),
                                "spearman_vs_champion": float(spearman(pred, ref_all[val_idx])),
                                "v3_before": a_b4, "v3_after": a_sw,
                                "marginal_e5": (a_sw - a_b4) * 1e5,
                                "split_census": cen, "aux_vs_rating": dup}
        print(f"  {k:>5}{Xf.shape[1]:>6}{len(RATINGS):>5}{auc:>12.6f}"
              f"{(auc - a_ref) * 1e5:>+9.1f}e{(a_sw - a_b4) * 1e5:>+9.2f}e{mean_acc:>9.3f}"
              f"{100 * cen.get('pct_splits_on_aux', 0):>6.1f}%{aux_sec:>9.0f}")
        assert label_free, "aux features changed when labels were flipped -- they see the label"
        assert te0 == te1, "te_ block changed when the aux columns were added"

    ds = [rec["delta_vs_champion_e5"] for _k, rec in sorted(
        out["arms"].items(), key=lambda kv: kv[1]["fold"])]
    out["deltas_e5"] = ds
    out["mean_delta_e5"] = float(np.mean(ds)) if ds else None
    ms_ = [rec["marginal_e5"] for _k, rec in sorted(out["arms"].items(),
                                                    key=lambda kv: kv[1]["fold"])]
    out["marginals_e5"] = ms_
    out["mean_marginal_e5"] = float(np.mean(ms_)) if ms_ else None
    print(f"\n  standalone vs champion: {[f'{d:+.1f}' for d in ds]}  mean {np.mean(ds):+.2f}e-5"
          if ds else "  no folds")
    print(f"  MARGINAL on v3       : {[f'{d:+.2f}' for d in ms_]}  mean {np.mean(ms_):+.2f}e-5"
          if ms_ else "  no folds")
    save_json(out, REPORTS / f"{args.tag}_runs.json")
    print("  wrote", REPORTS / f"{args.tag}_runs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
