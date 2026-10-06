"""Phase 10B -- native CatBoost categorical learning, matched 2x2 (C0/C1/C2/C3).

The gap this fills
------------------
Audit: zero `cat_features` and zero CatBoost `boosting_type` anywhere in src/ or scripts/. Every
CatBoost model this campaign ever fitted received Gender, Customer Type, Type of Travel, Class and
all 13 service ratings as ORDINAL FLOATS. CatBoost's defining mechanism -- ordered target statistics
for categorical features -- was never exercised, and the best CatBoost ever recorded (0.9610555) is
below the deterministic LightGBM. "CatBoost is worse" was never actually tested.

The 2x2, with NOTHING else varying
----------------------------------
  C0  full numeric view, Plain            -- the current control; must reproduce the established
                                              CatBoost fold result or the harness is wrong
  C1  full numeric view, Ordered          -- isolates ordered boosting
  C2  full numeric view + native cats, Plain   -- isolates native categorical CTRs
  C3  full numeric view + native cats, Ordered -- the full CatBoost-specific mechanism

C0 vs C1 differ ONLY in boosting_type. C0 vs C2 differ ONLY by appended twin columns, and C0's
numeric block is byte-identical to what the existing pipeline feeds CatBoost. That is asserted in
tests/test_native_cat.py, so the 2x2 is interpretable.

Round selection follows the Phase 9 fixed-round protocol, NOT the existing CatBoost path
--------------------------------------------------------------------------
The existing `_fit_cat_es` early-stops on a 10% carve and then throws those rows away, so its model
trains on 90% of outer-fit. Here the iteration count is chosen by inner early stopping and the model
is then REFITTED on 100% of outer-fit at that fixed count, and the outer fold is scored ONCE. Doing
the same for all four arms is what makes the comparison fair. Refitting on 100% was worth +5.3e-5 in
Phase 9, which is larger than most effects being chased here, so leaving it out would have buried it.

Usage:
  python scripts/run_catboost_native.py --arms C0,C1,C2,C3 --folds 0
  python scripts/run_catboost_native.py --arms C3 --timing-probe 60     # real-scale rate
  python scripts/run_catboost_native.py --report-only --tag p10b --folds 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.native_cat import (attach, cat_frame, cat_indices, default_cat_cols,  # noqa: E402
                                verify_no_target)
from scripts.run_views import _inner_es_split  # noqa: E402

# The established CatBoost config, from scripts/run_views.py::_fit_cat_es. lr/depth/l2 are its
# defaults; only the mechanism under test varies between arms.
BASE = dict(iterations=6000, learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
            thread_count=8, verbose=0, allow_writing_files=False, eval_metric="AUC")

ARMS = {
    "C0": {"boosting_type": "Plain",   "cats": False},
    "C1": {"boosting_type": "Ordered", "cats": False},
    "C2": {"boosting_type": "Plain",   "cats": True},
    "C3": {"boosting_type": "Ordered", "cats": True},
    # one higher CTR interaction setting, tested only AFTER a native-cat arm shows signal (§13)
    "C2c2": {"boosting_type": "Plain", "cats": True, "max_ctr_complexity": 2},
    "C3c2": {"boosting_type": "Ordered", "cats": True, "max_ctr_complexity": 2},
    # high-cardinality twins, tested separately and only after C2/C3 shows signal (§15)
    "C4": {"boosting_type": "Plain", "cats": True, "extra_src": ["Age"]},
    "C5": {"boosting_type": "Plain", "cats": True, "extra_src": ["Flight Distance"]},
}
TASK = "CPU"
ES_ROUNDS = 6000
ES_PATIENCE = 300


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build(vb, tr, fit, val, test, cats_on, extra_src):
    """Numeric frame from the champion view, optionally with native categorical twins appended."""
    Xf, Xa, names = vb.assemble(fit, None, val, test, inner_seed=0)
    if not cats_on:
        f = pd.DataFrame(np.asarray(Xf, dtype=np.float32), columns=list(names))
        a = {k: pd.DataFrame(np.asarray(v, dtype=np.float32), columns=list(names))
             for k, v in Xa.items()}
        return f, a, []
    src = default_cat_cols(True) + list(extra_src or [])
    cf = cat_frame(tr, src, fit)
    cv = cat_frame(tr, src, val)
    # the test frame is built from `te`, so its own source block is required
    raise_if_target_tr(tr, src)
    f, cf_names = attach(Xf, names, cf)
    a = {"val": attach(Xa["val"], names, cv)[0]}
    if "test" in Xa:
        ct = cat_frame(tr, src, test)
        a["test"] = attach(Xa["test"], names, ct)[0]
    return f, a, cf_names


def raise_if_target_tr(tr, src):
    verify_no_target([f"ncat__{c}" for c in src])
    if any(c not in tr.columns for c in src):
        missing = [c for c in src if c not in tr.columns]
        raise KeyError(f"categorical source columns absent from the frame: {missing}")


def fit_cat(frame, y, params, cat_names, seed, n_rounds):
    from catboost import CatBoostClassifier
    p = dict(BASE)
    p.update(params)
    p["random_seed"] = seed
    p["iterations"] = int(n_rounds)
    if "iterations" not in params:
        p["iterations"] = int(n_rounds)
    p.pop("eval_metric", None)
    p["task_type"] = TASK
    m = CatBoostClassifier(**p)
    kw = {"cat_features": cat_indices(frame, cat_names)} if cat_names else {}
    m.fit(frame, y, **kw)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="C0,C1,C2,C3")
    ap.add_argument("--view", default="full")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--tag", default="p10b")
    ap.add_argument("--task", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--timing-probe", type=int, default=0,
                    help="if >0, fit each arm ONCE at this round count on the full outer-fit rows "
                         "and report s/round. Required before committing to a multi-hour arm.")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    global TASK
    TASK = args.task

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; known {sorted(ARMS)}")

    v3 = store.load_oof("blend_v3_final").astype("float64")

    if args.report_only:
        return report_only(args, y, folds, v3)

    out = {"tag": args.tag, "view": args.view, "scheme": args.scheme, "task_type": TASK,
           "arms": arms, "folds": {}, "base_params": {k: v for k, v in BASE.items()},
           "protocol": "inner ES on a 10% carve of outer-fit selects the iteration count; the model "
                       "is then refitted on 100% of outer-fit at that fixed count; the outer fold is "
                       "scored once",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}

    for k in [int(x) for x in args.folds.split(",")]:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        assert not (set(fit.tolist()) & set(val.tolist()))
        vb = ViewBuilder(tr, te, args.view)
        vb.build_static()

        # ---- timing probe at REAL scale, before committing ----
        if args.timing_probe:
            print(f"\n{'='*100}\nTIMING PROBE fold {k}: {args.timing_probe} rounds on all "
                  f"{len(fit):,} outer-fit rows, {args.task}\n{'='*100}", flush=True)
            proj = {}
            for a in arms:
                spec = ARMS[a]
                f, _, cn = build(vb, tr, fit, val, None, spec["cats"], spec.get("extra_src"))
                params = {"boosting_type": spec["boosting_type"], "task_type": TASK,
                          "random_seed": args.seed + k}
                params.update({kk: vv for kk, vv in spec.items()
                               if kk not in ("boosting_type", "cats", "extra_src")})
                t0 = time.time()
                fit_cat(f, y[fit], params, cn, args.seed + k, args.timing_probe)
                dt = time.time() - t0
                per = dt / args.timing_probe
                proj[a] = {"seconds": round(dt, 1), "sec_per_round": round(per, 4),
                           "n_features": int(f.shape[1]), "n_cat": len(cn),
                           "projected_900_rounds_h": round(per * 900 / 3600, 2)}
                print(f"  {a:<5} {f.shape[1]:>4} feat ({len(cn)} cat)  {args.timing_probe} rounds "
                      f"in {dt:>7.1f}s = {per:.4f}s/round -> 900 rounds ~ "
                      f"{per*900/3600:.2f} h", flush=True)
            out.setdefault("timing", {})[str(k)] = proj
            save_json(out, REPORTS / f"{args.tag}_timing.json")
            print("\nwrote", REPORTS / f"{args.tag}_timing.json")
            return

        print(f"\n{'='*100}\nfold {k}   outer-fit={len(fit):,}  eval={len(val):,}\n{'='*100}",
              flush=True)
        rec = {}
        for a in arms:
            spec = ARMS[a]
            f, A, cn = build(vb, tr, fit, val, None, spec["cats"], spec.get("extra_src"))
            seed = args.seed + k
            params = {"boosting_type": spec["boosting_type"], "task_type": TASK,
                      "random_seed": seed}
            params.update({kk: vv for kk, vv in spec.items()
                           if kk not in ("boosting_type", "cats", "extra_src")})

            # ---- inner ES on a 10% carve of outer-fit, to pick the iteration count ----
            itr_g, es_g = _inner_es_split(fit, y_int, seed)
            pos = {int(v): i for i, v in enumerate(fit)}
            itr_l = np.array([pos[int(v)] for v in itr_g])
            es_l = np.array([pos[int(v)] for v in es_g])
            from catboost import CatBoostClassifier
            p_es = dict(BASE)
            p_es.update(params)
            p_es["iterations"] = ES_ROUNDS
            p_es["eval_metric"] = "AUC"
            m_es = CatBoostClassifier(**p_es)
            kw = {"cat_features": cat_indices(f.iloc[itr_l], cn)} if cn else {}
            t0 = time.time()
            m_es.fit(f.iloc[itr_l], y[itr_g], eval_set=(f.iloc[es_l], y[es_g]),
                     early_stopping_rounds=ES_PATIENCE, verbose=0, **kw)
            n_iter = int(m_es.get_best_iteration() or ES_ROUNDS)
            t_es = time.time() - t0
            del m_es

            # ---- refit on 100% of outer-fit at that fixed count ----
            t0 = time.time()
            m = fit_cat(f, y[fit], params, cn, seed, n_iter)
            pred = m.predict_proba(A["val"])[:, 1]
            t_refit = time.time() - t0
            auc = float(roc_auc_score(y[val], pred))
            np.save(REPORTS / f"{args.tag}_{a}_fold{k}.npy", pred.astype("float32"))
            bg = {}
            for w in (0.01, 0.02, 0.03, 0.05, 0.10):
                bg[str(w)] = (float(roc_auc_score(
                    y[val], float(w) * logit(pred) + (1 - float(w)) * logit(v3[val])))
                    - float(roc_auc_score(y[val], v3[val]))) * 1e5
            rec[a] = {"auc": auc, "inner_best_iter": n_iter, "n_fit_rows_es": int(len(itr_l)),
                      "n_fit_rows_refit": int(len(fit)), "n_features": int(f.shape[1]),
                      "n_cat": len(cn), "seconds_es": round(t_es, 1),
                      "seconds_refit": round(t_refit, 1), "blend_gains_e5": bg,
                      "params": params, "cat_cols": cn}
            print(f"  {a:<5} boosting={spec['boosting_type']:<8} cats={len(cn):<3} "
                  f"iters={n_iter:<5} refit_rows={len(fit):,}  AUC={auc:.6f}  "
                  f"(ES {t_es:.0f}s + refit {t_refit:.0f}s)  "
                  f"blend@2% {bg['0.02']:+.2f}e-5", flush=True)

        ctrl = rec.get("C0")
        print(f"\n  {'arm':<6}{'AUC':>12}{'delta vs C0':>14}{'logit corr vs v3':>18}"
              f"{'spearman':>11}{'blend@2%':>11}")
        print(f"  {'-'*72}")
        for a in arms:
            r = rec[a]
            if a == "C0":
                print(f"  {a:<6}{r['auc']:>12.6f}{'(control)':>14}{'':>18}{'':>11}"
                      f"{r['blend_gains_e5']['0.02']:>+11.2f}")
                continue
            pv = np.load(REPORTS / f"{args.tag}_{a}_fold{k}.npy").astype("float64")
            d = r["auc"] - ctrl["auc"]
            print(f"  {a:<6}{r['auc']:>12.6f}{d*1e5:>+13.1f}e"
                  f"{corr(logit(pv), logit(v3[val])):>18.5f}"
                  f"{spearman(pv, v3[val]):>11.5f}{r['blend_gains_e5']['0.02']:>+11.2f}")

        verdicts = []
        for a in arms:
            if a == "C0":
                continue
            r = rec[a]
            d = (r["auc"] - ctrl["auc"]) * 1e5
            b2 = r["blend_gains_e5"]["0.02"]
            if d >= 8.0:
                verdicts.append(f"{a}: standalone {d:+.1f}e-5 >= +8e-5 -> PROMOTE to fold 1")
            elif d >= 3.0:
                verdicts.append(f"{a}: standalone {d:+.1f}e-5 (+3..+8e-5) -> fold 1 if diversity "
                                f"also useful (blend@2% {b2:+.2f}e-5)")
            elif b2 >= 1.5:
                verdicts.append(f"{a}: standalone flat ({d:+.1f}e-5) but blend@2% {b2:+.2f}e-5 "
                                f">= +1.5e-5 -> fold 1")
            else:
                verdicts.append(f"{a}: standalone {d:+.1f}e-5, blend@2% {b2:+.2f}e-5 -> REJECT")
        print("\n  " + "\n  ".join(verdicts))
        out["folds"][str(k)] = {"control": "C0", "control_auc": ctrl["auc"] if ctrl else None,
                                "arms": {a: {kk: vv for kk, vv in r.items() if kk != "cat_cols"}
                                         for a, r in rec.items()},
                                "verdicts": verdicts}

    save_json(out, REPORTS / f"{args.tag}.json")
    print("\nwrote", REPORTS / f"{args.tag}.json")


def report_only(args, y, folds, v3) -> None:
    from src.validation.compare import corr, spearman
    out = {"tag": args.tag, "report_only": True, "folds": {}}
    for k in [int(x) for x in args.folds.split(",")]:
        val = np.where(folds == k)[0]
        preds, aucs = {}, {}
        for a in ARMS:
            f = REPORTS / f"{args.tag}_{a}_fold{k}.npy"
            if f.exists():
                p = np.load(f).astype("float64")
                preds[a], aucs[a] = p, float(roc_auc_score(y[val], p))
        if not preds:
            continue
        ctrl = aucs.get("C0")
        print(f"\n{'='*104}\nfold {k}  rebuilt from saved predictions "
              f"({len(preds)} arms)\n{'='*104}")
        print(f"  {'arm':<6}{'AUC':>12}{'delta vs C0':>14}{'logit corr vs v3':>18}"
              f"{'spearman':>11}" + "".join(f"{'blend@'+w:>11}"
                                             for w in ("0.01", "0.02", "0.05", "0.10")))
        print(f"  {'-'*104}")
        rec = {}
        for a in sorted(preds, key=lambda z: -aucs[z]):
            p = preds[a]
            v0 = float(roc_auc_score(y[val], v3[val]))
            bg = {w: (float(roc_auc_score(y[val], float(w) * logit(p)
                                          + (1 - float(w)) * logit(v3[val]))) - v0) * 1e5
                  for w in (0.01, 0.02, 0.05, 0.10)}
            d = (aucs[a] - ctrl) * 1e5 if ctrl else None
            rec[a] = {"auc": aucs[a], "delta_vs_C0_e5": d,
                      "logit_corr_vs_v3": corr(logit(p), logit(v3[val])),
                      "spearman_vs_v3": spearman(p, v3[val]),
                      "blend_gains_e5": bg}
            dtxt = "(control)" if a == "C0" else f"{d:>+13.1f}e"
            print(f"  {a:<6}{aucs[a]:>12.6f}{dtxt:>14}{rec[a]['logit_corr_vs_v3']:>18.5f}"
                  f"{rec[a]['spearman_vs_v3']:>11.5f}"
                  + "".join(f"{bg[w]:>+11.2f}" for w in (0.01, 0.02, 0.05, 0.10)))
        out["folds"][str(k)] = {"control_auc": ctrl, "arms": rec}
    save_json(out, REPORTS / f"{args.tag}_report.json")
    print("\nwrote", REPORTS / f"{args.tag}_report.json")


if __name__ == "__main__":
    main()
