"""Phase 14 -- do the VALIDATED auxiliary rating features help XGBoost and CatBoost too?

THE QUESTION
------------
Phase 13 established that auxiliary expected-rating features are real: +12.25e-5 standalone on an
extra_trees LightGBM, 5/5 folds, and the first consistently positive ensemble marginal in the
campaign (+0.745e-5 over six slots, t +3.99). But the gain could not be harvested, and the reason is
specific and measured: each new prediction correlates 0.998-0.999 with the member it replaces, so
six slots bought 2.8x rather than 6x.

That points at the obvious next question. The extra_trees LightGBM family is 24 of 59 slots and its
members are near-clones of each other, so a mechanism that mostly re-ranks rows *within* that family
has little room. XGBoost (8 slots) and CatBoost (7 slots) are structurally different families with
different inductive biases, so the SAME features might produce predictions that are both better and
more independent. That is a testable claim, and this phase tests it.

WHAT IS FIXED AND NOT REDESIGNED
--------------------------------
The auxiliary features are reused EXACTLY as validated in Phase 13 -- the cached per-fold arrays
reports/p13b_aux{fit,val}_f{k}.npy, produced by scripts/run_phase13.py::build_aux with inner
cross-fitting, expected-value construction over 13 ratings, and outer-fit-only fitting. Nothing about
the mechanism is changed here. That is deliberate: changing both the features and the model family at
once would confound the interaction, and the only question worth answering is the INTERACTION.

So the single changed variable per arm is: 13 extra columns appended, everything else identical.

THE ARMS, AND WHY THESE THREE XGB CONFIGS
-----------------------------------------
Read from the finalist manifest and the ledger, not inferred from names.
  X0  prod5_xgb_full_primary   strongest XGB member in v3, OOF 0.960946, seed 1
  X1  xt_xgb_lossguide         grow_policy=lossguide, max_leaves=127 -- a different TREE-GROWTH
                               algorithm, not just different hyper-parameters, so the strongest
                               structural contrast available
  X2  zoo_xgb_d6               shallow depth 6 with heavier regularisation -- the opposite end of the
                               capacity range

Both xt_xgb_* come from the extra_trees zoo pass but with XGBoost's own randomised-growth knobs
(colsample_bynode, grow_policy), so they are the closest existing analogue of the extra_trees
mechanism inside the XGBoost family.

THE CATBOOST DECOMPOSITION, WHICH IS THE POINT OF 14B
-----------------------------------------------------
Phase 10B established that native categorical CTRs are worth +6.6e-5 at model level (4/5 folds).
So CatBoost gives a clean 2x2:

    numeric base          numeric + aux
    native-cat base       native-cat + aux

If the two mechanisms compound, native+aux will exceed BOTH single-mechanism cells. If it does not,
no synergy is claimed -- the instruction is explicit that invented synergy is worse than none, and
this campaign has two withdrawn claims on record from inferring a mechanism's blend value instead of
measuring it.

Only Plain boosting. Ordered remains closed. No Age-cat, no Flight-Distance-cat, no ctr2.

FOLD DISCIPLINE, CARRIED OVER FROM THE PHASE 12 LESSON
------------------------------------------------------
run_zoo.py:209 passes inner_seed=k, and run_gbdt passes the entry's FIXED seed without adding the
fold index. Both are fold-indexed conventions that coincide with the naive default on fold 0 and
diverge after. The control arm for every configuration therefore reproduces the STORED member before
any treatment is trusted, and the check runs on every fold. A treatment measured against a harness
that is not the member is not evidence.

Usage:
  python scripts/run_phase14.py --folds 0
  python scripts/run_phase14.py --folds 0,1,2,3,4
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_phase12 import CAT_PARAMS, CHAMPION_SCHEME, CHAMPION_VIEW, lg, sg, te_hash  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# ---- XGBoost: defaults copied verbatim from scripts/run_views.py::_fit_xgb_es ----------------
XGB_BASE = dict(objective="binary:logistic", eval_metric="auc", n_estimators=6000,
                learning_rate=0.03, max_depth=8, min_child_weight=8, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=2.0, max_bin=256, tree_method="hist",
                device="cuda", n_jobs=8, early_stopping_rounds=250)
XGB_ARMS = {
    "X0": ("prod5_xgb_full_primary", 1, {}),                      # strongest v3 XGB member
    "X1": ("xt_xgb_lossguide", 12, dict(learning_rate=0.03, max_depth=0, grow_policy="lossguide",
                                        max_leaves=127, min_child_weight=8,
                                        colsample_bytree=0.8)),
    "X2": ("zoo_xgb_d6", 8, dict(learning_rate=0.04, max_depth=6, min_child_weight=5,
                                 colsample_bytree=0.85, subsample=0.9, reg_lambda=5.0,
                                 reg_alpha=0.05)),
}
# ---- CatBoost: defaults copied verbatim from scripts/run_views.py::_fit_cat_es ----------------
CAT_BASE = dict(iterations=6000, learning_rate=0.04, depth=8, l2_leaf_reg=3.0, thread_count=8,
                verbose=0, allow_writing_files=False, eval_metric="AUC")
CAT_ARMS = {
    "C0": ("z3_cat_d8_s2", 3, dict(learning_rate=0.04, depth=8), False),   # numeric base
    "C1": ("z3_cat_d8_s2", 3, dict(learning_rate=0.04, depth=8), True),    # native-cat C2 repr
}
AUX_TAG = "p13b"     # the cached, validated Phase 13 auxiliary features
N_CAT = 17           # Phase 13's validated native-category twin set (META4 + SERVICE13)


def fit_xgb(Xf, yf, Xv, seed, params, es_X, es_y):
    import xgboost as xgb
    p = dict(XGB_BASE)
    p.update(params)
    p.update(random_state=seed)
    m = xgb.XGBClassifier(**p)
    m.fit(Xf, yf, eval_set=[(es_X, es_y)], verbose=False)
    it = int(m.best_iteration)  # zero is a valid selected iteration
    return m.predict_proba(Xv)[:, 1], it


def fit_cat(frame_fit, y_fit, frame_val, seed, params, cat_names, es_frame, es_y):
    from catboost import CatBoostClassifier
    from scripts.native_cat import cat_indices
    p = dict(CAT_BASE)
    p.update(params)
    p.update(random_seed=seed, boosting_type="Plain")
    ci = cat_indices(frame_fit, cat_names) if cat_names else []
    kw = {"cat_features": ci} if ci else {}
    m = CatBoostClassifier(**p)
    m.fit(frame_fit, y_fit, eval_set=(es_frame, es_y), early_stopping_rounds=300, verbose=0, **kw)
    return m.predict_proba(frame_val)[:, 1], int(m.get_best_iteration() or p["iterations"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tag", default="p14")
    ap.add_argument("--arms", default="X0,X1,X2,C0,C1")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    folds = get_scheme(CHAMPION_SCHEME, yi, tr[ID_COL]).folds
    vb = ViewBuilder(tr, te, CHAMPION_VIEW)
    vb.build_static()

    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    L = {e: lg(store.load_oof(e).astype("float64")) for e in ids}
    base_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    assert float(np.abs(sg(base_logit).astype(np.float32).astype("float64") - v3).max()) == 0.0
    from scripts.native_cat import cat_cardinality, default_cat_cols

    print("=" * 112)
    print("PHASE 14 -- the SAME validated auxiliary features, now on XGBoost and CatBoost")
    print("=" * 112)
    print("  auxiliary features: reused EXACTLY from Phase 13, cached per fold, not redesigned")
    print(f"  arms: {args.arms}   folds: {args.folds}")
    print("  single changed variable per arm: 13 extra columns appended. Nothing else moves.\n")
    print(f"  {'arm':<4}{'fold':>5}{'variant':>9}{'nfeat':>7}{'AUC':>12}{'d(ctrl)':>9}"
          f"{'O/A corr':>10}{'spear':>9}{'v3 swap':>10}{'iters':>7}{'sec':>7}")
    print("  " + "-" * 98)
    print("  d(ctrl) = aux minus the SAME harness's control, which is the honest treatment delta.")
    print("  The bracketed figure is aux minus the STORED member; where those differ, the stored")
    print("  member came from a different runner and its reproduction gap must not be charged to")
    print("  the treatment.")

    out = {"tag": args.tag, "aux_source": AUX_TAG, "arms": {}, "folds": {},
           "xgb_base": XGB_BASE, "cat_base": CAT_BASE,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()[:12]}

    for k in [int(x) for x in args.folds.split(",")]:
        fit_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit_idx, yi, val_idx, None, inner_seed=k)
        Xv = Xa["val"]
        te_pos = [i for i, n in enumerate(names) if n.startswith("te_")]
        te0 = te_hash(Xf, te_pos)
        af = REPORTS / f"{AUX_TAG}_auxfit_f{k}.npy"
        av = REPORTS / f"{AUX_TAG}_auxval_f{k}.npy"
        if not (af.exists() and av.exists()):
            raise SystemExit(f"STOP: no cached auxiliary features for fold {k}. Run run_phase13b.py "
                             f"first so the validated features are reused rather than rebuilt.")
        aux_fit, aux_val = np.load(af), np.load(av)
        assert aux_fit.shape[1] == 13, f"expected 13 aux columns, got {aux_fit.shape[1]}"
        Xf_aug = np.column_stack([Xf, aux_fit]).astype("float64")
        Xv_aug = np.column_stack([Xv, aux_val]).astype("float64")
        assert te_hash(Xf_aug, te_pos) == te0, "appending aux columns perturbed the te_ block"
        itr, es = _inner_es_split(fit_idx, yi, 1)
        pos = {int(v): j for j, v in enumerate(fit_idx)}
        tr_l = np.array([pos[int(v)] for v in itr])
        es_l = np.array([pos[int(v)] for v in es])
        assert not (set(tr_l.tolist()) & set(es_l.tolist())), "inner carve overlaps"
        assert len(tr_l) + len(es_l) == len(fit_idx), "inner carve does not partition"
        out["folds"][f"f{k}"] = {"n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
                                 "te_hash": te0, "aux_shape": list(aux_fit.shape)}

        for aname in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
            for variant in ("ctrl", "aux"):
                if aname.startswith("X"):
                    member, seed, params = XGB_ARMS[aname]
                    A = Xf if variant == "ctrl" else Xf_aug
                    Av = Xv if variant == "ctrl" else Xv_aug
                    t0 = time.time()
                    pred, it = fit_xgb(A[tr_l], y[itr], Av, seed, params, A[es_l], y[es])
                    secs = time.time() - t0
                    catn = 0
                else:
                    member, seed, params, native = CAT_ARMS[aname]
                    catcols = default_cat_cols(True) if native else []
                    catn = len(catcols)
                    A = Xf if variant == "ctrl" else Xf_aug
                    Av = Xv if variant == "ctrl" else Xv_aug
                    # Column labels must follow the MATRIX, not the original view: the aux variant has
                    # 298 columns, and reusing the 285-name list produced a shape mismatch.
                    cols = ([n for n in names]
                            + ([f"aux_ev_{i}" for i in range(aux_fit.shape[1])]
                               if variant == "aux" else []))
                    t0 = time.time()
                    if native:
                        from scripts.native_cat import attach, cat_frame, verify_no_target
                        catcols = [c for c in catcols if c in names]
                        verify_no_target([f"ncat__{c}" for c in catcols])
                        F, cn = attach(A.astype("float64"), cols,
                                       cat_frame(tr, catcols, fit_idx))
                        V, _ = attach(Av.astype("float64"), cols,
                                      cat_frame(tr, catcols, val_idx))
                        FE, _ = attach(A[tr_l].astype("float64"), cols,
                                       cat_frame(tr, catcols, fit_idx[tr_l]))
                        EE, _ = attach(A[es_l].astype("float64"), cols,
                                       cat_frame(tr, catcols, fit_idx[es_l]))
                    else:
                        cn = []
                        # F = the 90% inner-TRAIN rows, which is what the ES model fits on, and
                        # EE = the disjoint 10% carve it early-stops on. Passing the carve as the
                        # fit frame is a label/data length mismatch AND would make early stopping
                        # evaluate on rows the model trained on -- the same defect that contaminated
                        # Phase 11R's first run.
                        F = pd.DataFrame(A[tr_l], columns=cols)
                        V = pd.DataFrame(Av, columns=cols)
                        FE = F
                        EE = pd.DataFrame(A[es_l], columns=cols)
                    assert len(FE) == len(itr), f"CatBoost fit rows {len(FE)} != inner-train labels {len(itr)}"
                    assert len(EE) == len(es), f"CatBoost ES rows {len(EE)} != carve labels {len(es)}"
                    assert not (set(tr_l.tolist()) & set(es_l.tolist())), "carve overlaps"
                    pred, it = fit_cat(FE, y[itr], V, seed, params, cn, EE, y[es])
                    secs = time.time() - t0
                auc = float(roc_auc_score(y[val_idx], pred))
                ref = store.load_oof(member).astype("float64")
                a_ref = float(roc_auc_score(y[val_idx], ref[val_idx]))
                np.save(REPORTS / f"{args.tag}_{aname}_{variant}_f{k}.npy",
                        pred.astype("float32"))
                rec = {"arm": aname, "variant": variant, "fold": k, "member": member,
                       "seed": seed, "params": params, "n_features": int(Av.shape[1]),
                       "n_native_cat": catn, "auc": auc, "stored_member_auc": a_ref,
                       "delta_vs_stored_e5": (auc - a_ref) * 1e5, "iters": it,
                       "seconds": round(secs, 1),
                       "corr_vs_stored": float(corr(lg(pred), lg(ref[val_idx]))),
                       "spearman_vs_stored": float(spearman(pred, ref[val_idx]))}
                if variant == "aux":
                    c = L[member].copy()
                    c[val_idx] = lg(pred)
                    outside = np.ones(len(c), dtype=bool)
                    outside[val_idx] = False
                    assert np.array_equal(c[outside], L[member][outside]), "splice moved rows"
                    aft = sg(np.mean(np.column_stack(
                        [c if e == member else L[e] for e in ids]), axis=1))
                    a_b4 = float(roc_auc_score(y[val_idx], sg(base_logit[val_idx])))
                    a_sw = float(roc_auc_score(y[val_idx], aft[val_idx]))
                    rec["v3_swap_e5"] = (a_sw - a_b4) * 1e5
                    rec["v3_before"] = a_b4
                    rec["v3_after"] = a_sw
                out["arms"].setdefault(aname, {}).setdefault(variant, {})[f"f{k}"] = rec
                # The HONEST delta is aux-minus-CONTROL within THIS harness, not aux-minus-stored.
                # For X1 and X2 the stored member was produced by a different runner whose
                # reproduction gap (+1.1e-5 and -5.0e-5) would otherwise be folded into the
                # treatment effect and could masquerade as either a gain or a loss.
                if variant == "aux" and f"f{k}" in out["arms"].get(aname, {}).get("ctrl", {}):
                    rec["delta_vs_own_ctrl_e5"] = (
                        rec["auc"] - out["arms"][aname]["ctrl"][f"f{k}"]["auc"]) * 1e5
                d = (f"{rec['delta_vs_own_ctrl_e5']:+7.1f}e" if "delta_vs_own_ctrl_e5" in rec
                     else f"{'-':>8}")
                dstore = f"{rec['delta_vs_stored_e5']:+6.1f}e"
                ca = (f"{rec['corr_vs_stored']:>10.5f}" if variant == "aux" else f"{'-':>10}")
                sw = (f"{rec['v3_swap_e5']:>+9.2f}e" if variant == "aux" else f"{'-':>10}")
                print(f"  {aname:<4}{k:>5}{variant:>9}{Av.shape[1]:>7}{auc:>12.6f}{d}"
                      f"{ca}{rec['spearman_vs_stored']:>9.5f}{sw}{it:>7}{secs:>7.0f}"
                      f"   (vs stored {dstore})")

    save_json(out, REPORTS / f"{args.tag}_runs.json")
    print("\n  wrote", REPORTS / f"{args.tag}_runs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
