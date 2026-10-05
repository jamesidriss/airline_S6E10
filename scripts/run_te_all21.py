"""Phase 8A: isolate `te_all21` against the champion view. Matched arms, same code path.

The experiment
--------------
Arms differ in EXACTLY two things -- the feature view and `extra_trees` -- and share the inner
early-stopping split, the seed, every hyper-parameter, and the row set. The control arm reproduces the
known fold-0 baseline (0.961299 at 797 rounds) as a live check that the harness is matched; if it
does not, nothing else in the table is interpretable.

  full            285 feat   champion `full` view, existing `te` block        <- BASE
  full_te21       306 feat   full + te_all21 STACKED (old `te` kept)
  full_all21te    258 feat   full with old `te` REMOVED, te_all21 in its place

The swap view is not optional. Two redundant TE systems can inflate variance and, under `extra_trees`
with `colsample_bytree=0.8`, compete for the same random 80% of split candidates -- so a genuinely
useful block can look harmful purely from being stacked. Separating "more TE columns" from "TE columns
competing" requires both arms.

Deterministic control (section 8)
---------------------------------
This campaign has already seen a feature block help random splits while hurting deterministic ones
(the `enrich` experiment), because `extra_trees` can use a column as a random split candidate that a
greedy deterministic tree has no use for. Running both tells us whether `te_all21` carries predictive
information or merely supplies extra random candidates:

  both improve                  -> strong genuine signal
  det improves, xt flat        -> real but redundant with the xt pool; may still make diversity
  xt improves, det hurts       -> random-feature artefact, like `enrich`
  neither                      -> reject

Index discipline (the bug class that produced three fake results in this campaign)
-----------------------------------------------------------------------------------
`assemble()` returns matrices indexed by POSITION WITHIN the fit set; `y` is indexed by GLOBAL row.
Features use the local index and labels use the global one, always through the explicitly named
`fit_rows_global` / `fit_local` pair, and disjointness is asserted.

Usage: python scripts/run_te_all21.py --folds 0 --arms base,stack,replace,det_base,det_stack,det_replace
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

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

BASE = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 127,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8}
ROUNDS = 6000

# (label, view, extra_trees). "base" must reproduce the recorded fold baseline.
ARMS = [
    ("base", "full", True),
    ("stack", "full_te21", True),
    ("replace", "full_all21te", True),
    ("det_base", "full", False),
    ("det_stack", "full_te21", False),
    ("det_replace", "full_all21te", False),
]


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view-prefix", default="")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--arms", default="base,stack,replace,det_base,det_stack,det_replace")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--smooth", default="auto")
    ap.add_argument("--tag", default="te_all21")
    ap.add_argument("--arm-spec", default="",
                    help="comma list of name:view:xt, overriding the built-in arm table; used so "
                         "the same matched harness serves Phase 8A and 8B and later blocks")
    args = ap.parse_args()

    import lightgbm as lgb

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.arm_spec:
        # An explicit spec REPLACES the built-in table outright. Filtering it through --arms was a
        # bug: the default --arms list does not contain Phase 8B's names, so three of the six arms
        # were silently dropped and the run reported a partial table as if it were complete.
        arms = []
        for item in args.arm_spec.split(","):
            nm, view, xt = item.split(":")
            arms.append((nm.strip(), view.strip(), xt.strip().lower() in ("1", "true", "yes")))
        base_for = {"stack": "base", "replace": "base", "tec_stack": "base", "tec_swap": "base",
                    "n21_stack": "base", "n21_swap": "base",
                    "det_stack": "det_base", "det_swap": "det_base", "det_replace": "det_base"}
    else:
        arms = [(n, v, x) for n, v, x in ARMS if n in wanted]
        base_for = {"stack": "base", "replace": "base",
                    "det_stack": "det_base", "det_replace": "det_base"}
    xt_of = {n: x for n, _, x in arms}
    fin = store.load_oof("blend_v3_final").astype("float64")

    vcache = {}
    out = {"tag": args.tag, "scheme": args.scheme, "seed": args.seed,
           "smooth": args.smooth, "folds": {}, "arm_pairing": {}, "arms": {n: {"view": v, "extra_trees": x}
                                                        for n, v, x in arms},
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}

    for k in [int(x) for x in args.folds.split(",")]:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        # global indices for labels, local positions for the assembled matrices -- see the docstring
        fit_rows_g, es_rows_g = _inner_es_split(fit, y_int, args.seed + k)
        pos = {int(v): i for i, v in enumerate(fit)}
        fit_l = np.array([pos[int(v)] for v in fit_rows_g])
        es_l = np.array([pos[int(v)] for v in es_rows_g])
        assert len(fit_l) + len(es_l) == len(fit) and not (set(fit_l) & set(es_l))
        assert not (set(fit.tolist()) & set(val.tolist())), "fit/eval overlap"

        print(f"\n{'='*100}\nfold {k}   outer-fit={len(fit):,}   train={len(fit_l):,} "
              f"({len(fit_l)/len(y):.1%} of labels)   eval={len(val):,}\n{'='*100}", flush=True)
        fold_rec = {}

        for label, view, xt in arms:
            if view not in vcache:
                vcache[view] = ViewBuilder(tr, te, view)
                vcache[view].build_static()
            vb = vcache[view]
            Xf, Xa, names = vb.assemble(fit, y_int, val, None, inner_seed=k,
                                        te_all21_smooth=args.smooth)
            Xv = Xa["val"]
            p = dict(BASE)
            if xt:
                p["extra_trees"] = True
            p.update(n_estimators=ROUNDS, random_state=args.seed + k,
                     bagging_seed=args.seed + k + 1, feature_fraction_seed=args.seed + k + 2)
            t0 = time.time()
            ds = lgb.Dataset(Xf[fit_l], label=y[fit_rows_g])
            dv = lgb.Dataset(Xf[es_l], label=y[es_rows_g], reference=ds)
            m = lgb.train(p, ds, num_boost_round=ROUNDS, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            it = int(m.best_iteration or ROUNDS)
            pred = m.predict(Xv, num_iteration=it)
            auc = float(roc_auc_score(y[val], pred))
            dt = time.time() - t0
            bg = {}
            for w in (0.05, 0.10, 0.20, 0.30, 0.50):
                bg[str(w)] = float(roc_auc_score(
                    y[val], w * logit(pred) + (1 - w) * logit(fin[val]))) - float(
                    roc_auc_score(y[val], fin[val]))
            fold_rec[label] = {"auc": auc, "iter": it, "n_features": int(Xf.shape[1]),
                               "seconds": round(dt, 1), "blend_gain": bg,
                               "pred": pred}
            print(f"  {label:<12} {view:<15} xt={str(xt):<5} nfeat={Xf.shape[1]:>4} "
                  f"iter={it:>5}  AUC={auc:.6f}  ({dt:.0f}s)", flush=True)
            np.save(REPORTS / f"{args.tag}_{label}_fold{k}.npy", pred.astype("float32"))

        # ---- paired deltas against the matched control for the same extra_trees setting ----
        print(f"\n  {'arm':<14}{'n_feat':>7}{'iter':>7}{'AUC':>12}{'delta vs ctrl':>15}"
              f"{'logit corr':>13}{'spearman':>11}{'blend@0.2':>11}")
        print(f"  {'-'*96}")
        for label in [n for n, _, _ in arms]:
            r = fold_rec[label]
            ctrl_name = base_for.get(label, label)
            ctrl = fold_rec.get(ctrl_name)
            if ctrl is None or label == ctrl_name:
                print(f"  {label:<14}{r['n_features']:>7}{r['iter']:>7}{r['auc']:>12.6f}"
                      f"{'(control)':>15}{'':>13}{'':>11}{r['blend_gain']['0.2']*1e5:>+10.1f}e")
                continue
            dd = r["auc"] - ctrl["auc"]
            print(f"  {label:<14}{r['n_features']:>7}{r['iter']:>7}{r['auc']:>12.6f}"
                  f"{dd*1e5:>+14.1f}e{corr(logit(r['pred']), logit(ctrl['pred'])):>13.5f}"
                  f"{spearman(r['pred'], ctrl['pred']):>11.5f}"
                  f"{r['blend_gain']['0.2']*1e5:>+10.1f}e")
        out["folds"][str(k)] = {kk: {a: b for a, b in vv.items() if a != "pred"}
                                for kk, vv in fold_rec.items()}
        out["folds"][str(k)]["_pairing"] = base_for
        out["arm_pairing"] = base_for

    # ---- fold-0 interpretation (section 8 decision table), derived rather than asserted ----
    if "0" in out["folds"]:
        f0 = out["folds"]["0"]

        def d(lbl):
            c = base_for.get(lbl, lbl)
            return None if lbl not in f0 or c not in f0 else f0[lbl]["auc"] - f0[c]["auc"]

        tested = [n for n, _, _ in arms if n in f0 and base_for.get(n, n) in f0
                  and n != base_for.get(n, n)]
        deltas = {n: d(n) for n in tested}
        xt_d = [v for n, v in deltas.items() if xt_of[n]]
        det_d = [v for n, v in deltas.items() if not xt_of[n]]

        print("\n" + "=" * 100)
        print("FOLD 0 INTERPRETATION (the section 8 decision table)")
        print("=" * 100)
        for n, v in deltas.items():
            print(f"  {n:<14} {'extra_trees' if xt_of[n] else 'deterministic':<14} "
                  f"delta vs {base_for[n]:<9} = {v*1e5:+.1f}e-5")
        allpos = all(v > 0 for v in deltas.values()) and len(deltas) >= 2
        anypos = any(v > 0 for v in deltas.values())
        if allpos:
            v_txt = "STRONG GENUINE SIGNAL -- every arm improves against its matched control."
        elif not anypos:
            v_txt = "REJECT -- no arm improves against its matched control."
        elif det_d and all(x > 0 for x in det_d) and (not xt_d or all(x <= 0 for x in xt_d)):
            v_txt = ("real but redundant with the extra_trees pool: deterministic improves while the "
                     "random-split model is flat. May still create ensemble diversity.")
        else:
            v_txt = ("MIXED -- the gains do not hold across both model types, which is the shape a "
                     "random-feature artefact takes (the `enrich` block behaved this way).")
        print(f"  VERDICT: {v_txt}")
        out["fold0_interpretation"] = {"deltas": deltas, "verdict": v_txt}

    save_json(out, REPORTS / f"{args.tag}.json")
    print("\nwrote", REPORTS / f"{args.tag}.json")


if __name__ == "__main__":
    main()
