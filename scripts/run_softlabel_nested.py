"""Phase 2b: soft targets from a STRONG teacher, generated with full nesting.

Why this is a different experiment from `run_softlabel.py`
----------------------------------------------------------
The external teacher reaches AUC 0.9550 on competition labels. Shrinking the observed label
toward it is monotonically harmful (measured: -4.2e-4 at lambda=0.20), because that teacher is
informative but biased relative to the synthetic p(x). Shrinkage only pays when the target of
shrinkage is both informative and roughly unbiased, so the right test needs a *strong* teacher.

The leak we must avoid
----------------------
Our global 59-member OOF cannot be reused naively. For an outer-fold TRAIN row r, that OOF comes
from models trained on every fold except r's own -- which includes the outer VALIDATION rows. Using
it as a training target therefore leaks outer-validation labels into the student's targets. The
student then predicts those same outer-validation rows. That is real leakage, however subtle.

This script generates the teacher honestly, with full nesting:
  for each outer fold k:
      inner 5-fold split of the OUTER-TRAIN rows only
      train the teacher on inner-train, predict inner-val   ->  p_inner for every outer-train row
      soft targets for outer-train: q = (1-lambda)*y + lambda*p_inner
      train the student on outer-train with q, score on outer-val

No outer-validation label ever influences any teacher prediction used as a target.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.softlabel import soft_targets  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_softlabel import run_lgbm  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

TEACHER_PARAMS = dict(learning_rate=0.03, num_leaves=127, extra_trees=True)


def nested_teacher(vb, y, folds, k, teacher_mode, seed, n_inner=5):
    """Inner-cross-fitted teacher predictions for every outer-fold TRAIN row.

    Returns (p_inner_over_all_rows, stats). ``p_inner`` is only meaningful on the fit rows; the
    validation rows are filled with the outer-fold mean so they can never influence anything.
    """
    fit = np.where(folds != k)[0]
    val = np.where(folds == k)[0]
    p = np.zeros(len(y))

    if teacher_mode == "hard":   # control: teacher == the student's own features, no info added
        p[fit] = y[fit]
        return p, {"mode": "hard-control", "teacher_auc_fit": None}

    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 500 + k)
    Xall = vb.static_tr
    for i, (a_rel, b_rel) in enumerate(skf.split(Xall[fit], y[fit])):
        a = fit[a_rel]          # inner-train absolute rows
        b = fit[b_rel]          # inner-val   absolute rows
        set_seed(seed + i)
        inner_tr, inner_es = _inner_es_split(a, y.astype("int8"), seed + 1000 + i)
        pos = {v: j for j, v in enumerate(a)}
        tr_local = np.array([pos[v] for v in inner_tr])
        es_local = np.array([pos[v] for v in inner_es])
        Xa, Xout, _ = vb.assemble(a, y.astype("int8"), b, None, inner_seed=1000 + i)
        # Xout is a dict; Xout["val"] is the inner-val block. The ES holdout is carved out of the
        # inner-TRAIN block.
        o, it = run_lgbm(Xa[tr_local], y[inner_tr].astype("float64"), Xout["val"],
                         TEACHER_PARAMS, seed + i, Xa[es_local], y[inner_es].astype("float64"))
        p[b] = o
    stats = {"mode": teacher_mode, "teacher_auc_fit": float(roc_auc_score(y[fit], p[fit]))}
    return p, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--lams", default="0,0.05,0.10,0.20,0.35,0.50")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="softnest")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    ntr = len(tr)
    folds = get_scheme(args.folds, y.astype("int8"), tr[ID_COL]).folds

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    lams = [float(x) for x in args.lams.split(",")]
    oofs = {lam: np.zeros(ntr) for lam in lams}
    iters = {lam: [] for lam in lams}
    stats = []

    for k in sorted(set(folds.tolist())):
        t0 = time.time()
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        p_inner, st = nested_teacher(vb, y, folds, k, "strong", args.seed)
        st["fold"] = int(k)
        stats.append(st)
        print(f"  fold{k}: nested teacher AUC(train)={st['teacher_auc_fit']:.6f}", flush=True)

        set_seed(args.seed + k)
        Xf, Xa, _ = vb.assemble(fit, y.astype("int8"), val, None, inner_seed=k)
        inner_tr, inner_es = _inner_es_split(fit, y.astype("int8"), args.seed + k)
        pos = {v: j for j, v in enumerate(fit)}
        tr_local = np.array([pos[v] for v in inner_tr])
        es_local = np.array([pos[v] for v in inner_es])
        esX, esY = Xf[es_local], y[inner_es].astype("float64")
        for lam in lams:
            q_all = soft_targets(y[fit], p_inner[fit], lam)
            o, it = run_lgbm(Xf[tr_local], q_all[tr_local], Xa["val"], TEACHER_PARAMS,
                             args.seed + k, esX, esY)
            oofs[lam][val] = o
            iters[lam].append(it)
        print(f"  fold{k} done ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 96)
    print(f"NESTED SOFT-TARGET RESULTS  view={args.view} scheme={args.folds} "
          f"teacher=strong lgbm+extra_trees (5x inner CV)")
    print("=" * 96)
    rows = []
    base = None
    for lam in lams:
        auc = float(roc_auc_score(y, oofs[lam]))
        fa = [float(roc_auc_score(y[folds == kk], oofs[lam][folds == kk]))
              for kk in sorted(set(folds.tolist()))]
        if lam == 0.0:
            base = auc
        rows.append({"lam": lam, "oof_auc": round(auc, 6),
                     "delta_vs_lambda0": None if base is None or lam == 0.0
                     else round(auc - base, 6),
                     "fold_aucs": [round(x, 6) for x in fa],
                     "median_iter": int(np.median(iters[lam])) if iters[lam] else None})
        d = rows[-1]["delta_vs_lambda0"]
        print(f"  lambda={lam:<5} OOF AUC={auc:.6f}  delta={'' if d is None else f'{d:+.6f}'}  "
              f"folds={[round(x, 6) for x in fa]}")

    for lam in lams:
        store.save(f"{args.tag}_{args.view}_l{lam}_{args.folds}", oofs[lam], None,
                   fold_scheme=args.folds,
                   meta={"family": "lgbm_soft_nested", "featureset": args.view,
                         "auc": round(float(roc_auc_score(y, oofs[lam])), 6), "lam": lam})
    best = max(rows, key=lambda r: r["oof_auc"])
    save_json({"rows": rows, "teacher_stats": stats, "best_lambda": best["lam"],
               "best_auc": best["oof_auc"], "view": args.view, "scheme": args.folds},
              REPORTS / f"{args.tag}_{args.view}.json")
    print(f"\nbest lambda={best['lam']} auc={best['oof_auc']:.6f}")
    print("wrote", REPORTS / f"{args.tag}_{args.view}.json")


if __name__ == "__main__":
    main()