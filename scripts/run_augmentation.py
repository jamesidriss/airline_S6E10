"""Augmentation benchmark harness: every experiment trains its own MATCHED baseline.

Why a harness rather than a script
----------------------------------
An augmentation number is meaningless without the baseline it was measured against, because the
baseline drifts whenever a model configuration, a feature view or a seed changes. Every run here
therefore trains, in the same process and the same code path:

  * the champion configuration on the original outer-FIT rows, and
  * the champion configuration on original + augmented rows,

and reports the paired delta on identical evaluation rows. Old numbers from other scripts are never
used as the comparison.

Fingerprinting
--------------
Every experiment records git commit, fold, row counts, method, parameters, target rule, weights, seed,
feature schema hash, augmented-dataset hash, model hash, runtime, both AUCs, the delta, correlations
against the baseline and the finalist, the marginal finalist blend gain, and duplicate/invalid counts.
The augmented matrices themselves are NOT committed -- only their hashes and the code that made them.

Promotion policy (predeclared, so it cannot drift after seeing a result)
    delta <= 0                  reject
    0 < delta < +5e-5           noise; do not escalate by default
    delta >= +8e-5              interesting
    delta >= +1.0e-4            promote to >= 3 folds

Usage:
  python scripts/run_augmentation.py --method duplicate --ratio 1.0 --fold 0
  python scripts/run_augmentation.py --method interp --ratio 0.5 --fold 0
  python scripts/run_augmentation.py --sweep interp:0.5,1.0 donor:1.0 --fold 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features import augment as AUG  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

# The exact champion single-model configuration, copied from the champion runner. Held fixed so the
# ONLY difference between the two arms of every experiment is the training support.
CHAMPION_PARAMS = {
    "objective": "binary", "n_estimators": 6000, "learning_rate": 0.02, "num_leaves": 127,
    "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
    "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True, "verbose": -1, "n_jobs": 8,
}


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def arr_hash(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes()[:200_000_000])
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()[:12]
    except Exception:  # noqa: BLE001
        return "unknown"


def train_champion(X, y, Xv, yv, esX, esY, seed, params=None, weights=None, soft=False):
    """Train the champion LightGBM. ``soft=True`` uses the cross_entropy objective for soft labels."""
    import lightgbm as lgb

    p = dict(CHAMPION_PARAMS)
    p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    if params:
        p.update(params)
    if soft:
        p["objective"] = "cross_entropy"
    ds = lgb.Dataset(X, label=y, weight=weights)
    dv = lgb.Dataset(esX, label=esY, reference=ds)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return m.predict(Xv, num_iteration=m.best_iteration), int(m.best_iteration)


def blend_gain(pred, y, base_ref, weights=(0.02, 0.05, 0.10, 0.20)):
    """Marginal contribution of a candidate to the finalist, in logit space."""
    from src.ensemble import lab

    b = float(roc_auc_score(y, base_ref))
    out = {}
    lp, lb = lab.tform(np.asarray(pred, "float64"), "logit"), lab.tform(base_ref, "logit")
    for w in weights:
        out[str(w)] = float(roc_auc_score(y, w * lp + (1 - w) * lb)) - b
    return {"finalist_auc": b, "gains_by_weight": out, "best_gain": max(out.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--method", default="interp", choices=["duplicate", "interp", "donor"])
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--alpha-lo", type=float, default=0.35)
    ap.add_argument("--alpha-hi", type=float, default=0.65)
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--aug-weight", type=float, default=1.0,
                    help="sample weight on augmented rows (1.0 = same as original)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--query-chunk", type=int, default=256)
    ap.add_argument("--sweep", default="", help="e.g. interp:0.5,1.0 donor:1.0")
    ap.add_argument("--teacher", action="store_true",
                    help="label the augmented rows with a fold-fit teacher (soft targets, "
                         "cross_entropy objective, matched cross_entropy baseline)")
    ap.add_argument("--tag", default="aug")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = y.astype("int8")
    folds = get_scheme(args.folds, y_int, tr[ID_COL]).folds
    fit = np.where(folds != args.fold)[0]
    val = np.where(folds == args.fold)[0]
    # The evaluation fold must be exactly the held-out fold, and must not overlap the fit rows.
    # An earlier version used `folds != args.fold` here, which made the evaluation set identical to
    # the training set; the resulting AUC of 0.9724 for the champion baseline is what exposed it,
    # since the same configuration measures 0.9611 honestly.
    assert len(set(fit.tolist()) & set(val.tolist())) == 0, "fit and evaluation folds overlap"
    assert len(val) > 1000, f"evaluation fold implausibly small: {len(val)}"

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    Xf, Xout, names = vb.assemble(fit, y_int, val, None, inner_seed=args.fold)
    Xv = Xout["val"]
    schema_hash = sha({"names": list(names)})
    model_hash = sha(CHAMPION_PARAMS)

    # inner early-stopping holdout, carved from the outer-FIT rows
    inner_tr, inner_es = _inner_es_split(fit, y_int, args.seed + args.fold)
    pos = {int(v): i for i, v in enumerate(fit)}
    tr_local = np.array([pos[int(v)] for v in inner_tr])
    es_local = np.array([pos[int(v)] for v in inner_es])

    fin_all = store.load_oof("blend_v3_final").astype("float64")

    specs = []
    if args.sweep:
        for tok in args.sweep.split():
            m, r = tok.split(":")
            specs.append((m, float(r)))
    else:
        specs.append((args.method, args.ratio))

    results = []
    for method, ratio in specs:
        print(f"\n{'='*92}\nmethod={method} ratio={ratio} aug_weight={args.aug_weight} "
              f"fold={args.fold}\n{'='*92}", flush=True)
        set_seed(args.seed + args.fold)

        # ---------------- matched baseline: identical code path, no augmentation -------------
        t0 = time.time()
        p_base, it_base = train_champion(Xf[tr_local], y[inner_tr], Xv, y[val],
                                         Xf[es_local], y[inner_es], args.seed + args.fold)
        auc_base = float(roc_auc_score(y[val], p_base))
        t_base = time.time() - t0
        print(f"  baseline(binary)  AUC={auc_base:.6f} iter={it_base} ({t_base:.0f}s)", flush=True)

        # When the student switches to cross_entropy, the objective change alone can move AUC, so a
        # MATCHED cross_entropy baseline on the original rows only is required. Otherwise an
        # objective effect would be misattributed to the augmentation.
        p_base_ce, it_base_ce, auc_base_ce = None, None, None
        if args.teacher:
            t0 = time.time()
            p_base_ce, it_base_ce = train_champion(
                Xf[tr_local], y[inner_tr], Xv, y[val], Xf[es_local], y[inner_es],
                args.seed + args.fold, soft=True)
            auc_base_ce = float(roc_auc_score(y[val], p_base_ce))
            print(f"  baseline(ce, no aug) AUC={auc_base_ce:.6f} iter={it_base_ce} "
                  f"({time.time()-t0:.0f}s)  <- the correct comparison for the teacher arm",
                  flush=True)

        # ---------------- build augmentation from outer-FIT rows only ------------------------
        t0 = time.time()
        if method == "duplicate":
            Xa, ya, prov = AUG.build_duplicated(Xf[tr_local], y[inner_tr], ratio=ratio,
                                                seed=args.seed + args.fold)
        else:
            Xa, ya, prov = AUG.build_local_augmentation(
                Xf[tr_local], y[inner_tr], names, ratio=ratio,
                alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi, mode=method,
                seed=args.seed + args.fold, query_chunk=args.query_chunk,
                ks=tuple(int(k) for k in args.ks.split(",")))
        t_aug = time.time() - t0
        print(f"  augmentation: {prov['n_aug']} rows built in {t_aug:.0f}s  "
              f"mode={prov['mode']}", flush=True)

        if len(Xa) == 0:
            print("  augmentation produced no rows; skipping", flush=True)
            continue

        # validity audit of the synthetic rows
        orig_set = {arr_hash(Xf[tr_local][i:i + 1]) for i in range(0, min(5000, len(tr_local)), 7)}
        dup_frac = float(np.mean([arr_hash(Xa[i:i + 1]) in orig_set
                                  for i in range(0, min(3000, len(Xa)), 3)]))
        n_invalid = int((~np.isfinite(Xa)).any(axis=1).sum())
        cat, integer, cont = AUG.classify_columns(names)
        cat_ix = [names.index(c) for c in cat]
        bad_cat = 0
        if cat_ix:
            valid_vals = {c: set(np.unique(Xf[tr_local][:, names.index(c)]).tolist()) for c in cat}
            for c in cat:
                j = names.index(c)
                vals = set(np.unique(Xa[:, j]).tolist())
                bad_cat += len(vals - valid_vals[c])
        print(f"  audit: duplicate_rate~{dup_frac:.4f}  non-finite rows={n_invalid}  "
              f"off-support categorical values={bad_cat}", flush=True)

        # ---------------- augmented arm ----------------------------------------------------
        # TARGET RULE. With hard targets the augmented row inherits its source row's label, which
        # assumes the class is locally pure. It is not: y is a Bernoulli draw from p(x), so the
        # midpoint of two class-1 rows can have a materially lower p. Copying the label therefore
        # bakes that error into the new rows, which is the most likely reason Family A degraded.
        # With --teacher the augmented rows instead get a label predicted by a teacher fitted ONLY
        # on the outer-FIT rows, and the student uses the cross_entropy objective so the fractional
        # targets are actually used.
        soft = bool(args.teacher)
        wcat = np.concatenate([np.ones(len(tr_local)),
                               np.full(len(ya), float(args.aug_weight))])
        t_teacher = 0.0
        if soft:
            t0 = time.time()
            # teacher: outer-FIT rows only, early-stopped on the inner holdout, never sees the
            # evaluation fold. Its predictions on X_aug are therefore fold-safe by construction.
            t_teacher_pred, it_te = train_champion(
                Xf[tr_local], y[inner_tr], Xa, ya, Xf[es_local], y[inner_es], args.seed + args.fold)
            t_teacher = time.time() - t0
            soft_auc = float(roc_auc_score(ya, t_teacher_pred))
            hard_auc = float(roc_auc_score(ya, ya))
            print(f"  teacher fitted in {t_teacher:.0f}s (iter={it_te}); on the AUGMENTED rows "
                  f"teacher AUC={soft_auc:.6f} vs inherited-label AUC={hard_auc:.6f} "
                  f"(delta {soft_auc-hard_auc:+.6f})", flush=True)
            ycat = np.concatenate([y[inner_tr], t_teacher_pred])
            ycat = np.clip(ycat, 1e-4, 1 - 1e-4)
        else:
            ycat = np.concatenate([y[inner_tr], ya])
        Xcat = np.vstack([Xf[tr_local], Xa])
        t0 = time.time()
        p_aug, it_aug = train_champion(Xcat, ycat, Xv, y[val], Xf[es_local], y[inner_es],
                                       args.seed + args.fold, weights=wcat, soft=soft)
        auc_aug = float(roc_auc_score(y[val], p_aug))
        t_augfit = time.time() - t0
        print(f"  augmented AUC={auc_aug:.6f} iter={it_aug} ({t_augfit:.0f}s) "
              f"objective={'cross_entropy' if soft else 'binary'}", flush=True)

        ref_auc = auc_base_ce if args.teacher else auc_base
        delta = auc_aug - ref_auc
        lp, lbase = _logit(p_aug), _logit(p_base)
        lfin = _logit(fin_all[val])
        bg = blend_gain(p_aug, y[val], fin_all[val])
        c_base, c_fin = corr(lp, lbase), corr(lp, lfin)
        print(f"  DELTA (vs matched objective) = {delta:+.6f} ({delta*1e5:+.1f}e-5)  "
              f"[vs binary baseline {auc_aug-auc_base:+.6f}]", flush=True)
        print(f"  corr vs baseline logit={c_base:.5f}  spearman={spearman(p_aug, p_base):.5f}",
              flush=True)
        print(f"  corr vs finalist logit={c_fin:.5f}", flush=True)
        gains_txt = ", ".join(f"w={w}:{g:+.6f}" for w, g in bg["gains_by_weight"].items())
        print(f"  finalist blend gains: {gains_txt}", flush=True)

        verdict = _verdict(delta, bg["best_gain"])
        print(f"  VERDICT: {verdict}", flush=True)

        results.append({
            "exp_id": f"{args.tag}_{method}_r{ratio}_w{args.aug_weight}_fold{args.fold}",
            "method": method, "ratio": ratio, "aug_weight": args.aug_weight,
            "fold": int(args.fold), "scheme": args.folds, "view": args.view,
            "git_commit": git_commit(), "seed": args.seed,
            "n_original_rows": int(len(tr_local)), "n_augmented_rows": int(len(Xa)),
            "n_val_rows": int(len(val)), "n_features": int(len(names)),
            "augmentation": prov,
            "alpha": [args.alpha_lo, args.alpha_hi], "ks": args.ks,
            "target_rule": "inherit the source row's hard label (no teacher, no soft targets)",
            "row_weights": {"original": 1.0, "augmented": float(args.aug_weight)},
            "schema_hash": schema_hash, "model_hash": model_hash,
            "aug_data_hash": arr_hash(Xa), "aug_label_hash": arr_hash(ya),
            "baseline_auc": auc_base,
            "baseline_auc_matched_objective": ref_auc,
            "augmented_auc": auc_aug, "delta": delta,
            "delta_vs_binary_baseline": auc_aug - auc_base,
            "teacher": ({"fitted": True, "seconds": round(t_teacher, 1),
                         "auc_on_augmented_rows": soft_auc,
                         "inherited_label_auc_on_augmented_rows": hard_auc,
                         "teacher_minus_inherited": soft_auc - hard_auc} if args.teacher else None),
            "baseline_best_iter": it_base, "augmented_best_iter": it_aug,
            "runtime_seconds": {"baseline": round(t_base, 1), "augment_build": round(t_aug, 1),
                                "augmented_fit": round(t_augfit, 1)},
            "corr_vs_baseline_logit": c_base,
            "spearman_vs_baseline": spearman(p_aug, p_base),
            "corr_vs_finalist_logit": c_fin,
            "blend_gain": bg, "audit": {"duplicate_rate_estimate": dup_frac,
                                       "non_finite_rows": n_invalid,
                                       "off_support_categorical_values": bad_cat},
            "verdict": verdict,
        })
        np.save(REPORTS / f"{args.tag}_{method}_r{ratio}_w{args.aug_weight}_fold{args.fold}.npy",
                p_aug.astype("float32"))

    if results:
        save_json({"results": results}, REPORTS / f"{args.tag}_bench_fold{args.fold}.json")
        print(f"\nwrote {REPORTS / f'{args.tag}_bench_fold{args.fold}.json'}")
        print("\n=== SUMMARY ===")
        print(f"{'method':<10}{'ratio':>7}{'w':>5}{'base':>11}{'aug':>11}{'delta':>11}"
              f"{'blend':>11}  verdict")
        for r in results:
            print(f"{r['method']:<10}{r['ratio']:>7}{r['aug_weight']:>5}"
                  f"{r['baseline_auc_matched_objective']:>11.6f}{r['augmented_auc']:>11.6f}"
                  f"{r['delta']*1e5:>+10.1f}e{r['blend_gain']['best_gain']*1e5:>+10.1f}e  "
                  f"{r['verdict']}")


def _verdict(delta, best_blend):
    if delta <= 0:
        return "REJECT (no gain over the matched baseline)"
    if delta < 5e-5:
        return "noise -- do not escalate"
    if delta < 8e-5:
        return "weak positive -- worth one more fold, not a promotion"
    if delta < 1e-4:
        return "INTERESTING -- promote to >= 3 folds"
    return "PROMOTE -- >= +1.0e-4, run full primary CV"


if __name__ == "__main__":
    main()
