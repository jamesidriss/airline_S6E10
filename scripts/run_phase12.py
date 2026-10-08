"""Phase 12 -- native categorical splits in the winning LightGBM extra_trees family.

THE HYPOTHESIS
--------------
Our strongest single family is LightGBM with extra_trees=True. The 285-column full view contains
raw categorical variables and their numeric copies, but LightGBM sees them as ordinary numbers, so
its ordered target-based categorical split algorithm has never been exercised inside the winning
configuration. CatBoost's native CTRs were worth +6.64e-5 at model level (Phase 10B), and
LightGBM's mechanism is DIFFERENT (an ordered, shrunk target statistic with cat_smooth/cat_l2 rather
than a CTR with priors). This asks whether native categorical splits improve the extra_trees family
WITHOUT removing the existing 48-column competition TE block.

This is NOT the public notebook's failed experiment. That one substituted numerics-as-categories
INSTEAD OF target encoding and confounded learning rate and seed. Here the te_ block stays
byte-identical and only the split algorithm changes, which is why L0 -> L1 is a clean single-variable
comparison.

WHAT THE AUDIT ALREADY ESTABLISHED (scripts/audit_lgb_native_cat.py, must be run first)
----------------------------------------------------------------------------------------
  * categorical_feature works on this CPU build, by index and by name.
  * extra_trees=True does NOT disturb the categorical search: a forced probe gives 125 categorical
    splits with extra_trees True and 125 with it False, the same count and the same code set.
    extra_trees randomises the NUMERICAL threshold search only. The two mechanisms coexist.
  * On the real 285-column view, declaring META4 produces 58 categorical splits that the control has
    zero of, so the treatment is OBSERVABLE and not inert.
  * LightGBM routes NaN in a declared categorical as MISSING (the -1 internal marker appears in
    feature_infos), so a missingness sentinel cannot silently become a real category code.
  * The audited columns are static ordinal encodings, so train/val/test codes agree structurally and
    no category map is fitted anywhere.

A MECHANISTIC PRIOR THAT IS RECORDED UP FRONT, NOT AFTER THE FACT
-----------------------------------------------------------------
cat_smooth=10 and cat_l2=10 make a LightGBM categorical split a SHRUNK target statistic with an
explicit prior. That is the same smoothing family our 48-column te_ block already supplies in the
same matrix. So the prior expectation is a SMALL effect, not new information. This is stated before
the run so a small result is a prediction met rather than a surprise reinterpreted afterwards.

ARMS (predeclared, no post-hoc additions)
  L0  control           no categorical declaration at all
  L1  META4             Gender, Customer Type, Type of Travel, Class
  L2  META4 + irregular META4 plus Online boarding, Inflight wifi service, Gate location
  L3  all 17 discrete   META4 plus all 13 survey ratings  -- only if L1 or L2 looks promising
      Age, Flight Distance and both delay columns stay NUMERIC in every arm. High-cardinality
      categoricals are explicitly NOT tested here; Phase 11R already found Flight Distance-as-
      category negligible for CatBoost and that result is not re-run under a different library.

PROTOCOL (must match the champion exactly, or the comparison is void)
  Champion member: xt_xt_d127_s1 -- view=full, scheme=primary, seed=1,
  params = scripts/run_views.py::_fit_lgbm_es defaults + extra_trees=True.
  Training follows run_zoo.py::run_gbdt: an inner 10% carve of the FIT rows is used for early
  stopping, the model is trained on the other 90%, and the OUTER fold is predicted at
  best_iteration. No outer-validation label ever reaches early stopping.
  THE SEED IS FIXED AT 1 FOR EVERY FOLD. scripts/run_xt_zoo.py:112 calls
  run_gbdt(..., seed) with the zoo entry's single seed and never adds the fold index. My first
  version used CHAMPION_SEED + k, which coincidentally matches on fold 0 (1+0 == 1) and diverges
  everywhere else: L0 then scored +2.7e-5 and +2.8e-5 against the stored champion on folds 1 and 2
  instead of +0.0e-5. A fold-0-only screen would never have caught it, because fold 0 is exactly
  where the two conventions coincide. The L0 reproduction check is therefore run on EVERY fold, not
  just the first, and it is the check that exposed this.
  L0 must reproduce the stored xt_xt_d127_s1 fold score before any treatment is trusted.

Usage:
  python scripts/run_phase12.py --arms L0,L1,L2 --folds 0
  python scripts/run_phase12.py --report
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
from scripts.run_views import _inner_es_split  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# ---- the champion configuration, recovered from its authoritative runner -----------------
# scripts/run_xt_zoo.py:36 registers ("xt_d127_s1", LGBM, "full", "primary",
#     dict(learning_rate=0.02, num_leaves=127, extra_trees=True), seed=1)
# scripts/run_zoo.py::run_gbdt dispatches to scripts/run_views.py::_fit_lgbm_es for LGBM.
# The defaults below are copied from _fit_lgbm_es verbatim. They are NOT src/models/gbdt.py
# defaults, which the brief explicitly warned against using.
CHAMPION_PARAMS = dict(
    objective="binary", metric="auc", n_estimators=6000, learning_rate=0.02, num_leaves=127,
    min_child_samples=40, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    reg_lambda=1.0, max_bin=255, verbose=-1, n_jobs=8,
    extra_trees=True,                     # <- the lever this campaign is built on
)
CHAMPION_MEMBER = "xt_xt_d127_s1"
CHAMPION_SEED = 1
CHAMPION_VIEW = "full"
CHAMPION_SCHEME = "primary"
ES_PATIENCE = 300

# Pinned categorical hyperparameters -- LightGBM 4.7 documented defaults, written out so the run
# record contains the values that governed the result. Left implicit, m.params returns None for
# all of them and the arm silently depends on the installed library's defaults.
CAT_PARAMS = {"cat_smooth": 10, "cat_l2": 10, "max_cat_threshold": 32,
              "max_cat_to_onehot": 4, "min_data_per_group": 100}

META4 = ["Gender", "Customer Type", "Type of Travel", "Class"]
IRREGULAR = ["Online boarding", "Inflight wifi service", "Gate location"]
# The 13 survey ratings, named exactly as the VIEW names them at positions 0..12. My first list
# used the raw-dataset spellings ("In-flight entertainment", "WiFi service", "In-flight service"),
# three of which do not exist in the view -- its spellings are "Inflight entertainment",
# "Inflight wifi service" and "Ease of Online booking". Declaring a column that is absent raises
# KeyError, so the names are resolved against the view rather than trusted from the raw header.
SURVEY13 = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
            "Gate location", "Food and drink", "Online boarding", "Seat comfort",
            "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
            "Checkin service", "Cleanliness"]
ARMS = {
    "L0": ([], "control: no categorical declaration"),
    "L1": (META4, "META4 as native categories"),
    "L2": (META4 + IRREGULAR, "META4 + the three zero/N-A sensitive surveys"),
    "L3": (META4 + SURVEY13, "META4 + all 13 survey ratings (predeclared, gated)"),
}


def lg(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype="float64"), -35, 35)))


def te_hash(Xf, te_pos) -> str:
    """Content hash of the 48 te_ columns for ONE fold. Per fold, never across folds."""
    import hashlib
    a = np.ascontiguousarray(Xf[:, te_pos], dtype="float64")
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def census(m) -> dict:
    """Split census straight from dump_model(); categorical splits carry decision_type '=='."""
    d = m.dump_model()
    import collections
    acc: list[tuple] = []

    def walk(n):
        if "split_feature" in n:
            acc.append((int(n["split_feature"]), n.get("decision_type")))
            walk(n["left_child"])
            walk(n["right_child"])

    for t in d["tree_info"]:
        walk(t["tree_structure"])
    types = collections.Counter(a[1] for a in acc)
    pf = collections.Counter(a[0] for a in acc)
    return {"n_splits": len(acc), "decision_types": dict(types),
            "n_categorical_splits": types.get("==", 0), "splits_per_feature": dict(pf)}


def fit_arm(Xf, yf, Xv, cat_idx, seed, es_X=None, es_y=None, n_rounds=None):
    """Replicates scripts/run_views.py::_fit_lgbm_es exactly, plus the categorical declaration."""
    import lightgbm as lgb
    p = dict(CHAMPION_PARAMS)
    p.update(CAT_PARAMS)
    if n_rounds is not None:
        p["n_estimators"] = int(n_rounds)
    p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    if n_rounds is None:
        ds = lgb.Dataset(Xf, label=yf, categorical_feature=list(cat_idx) if cat_idx else "auto")
        dv = lgb.Dataset(es_X, label=es_y, reference=ds)
        m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                      callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
        return m, int(m.best_iteration)
    ds = lgb.Dataset(Xf, label=yf, categorical_feature=list(cat_idx) if cat_idx else "auto")
    m = lgb.train(p, ds, num_boost_round=int(n_rounds))
    return m, int(n_rounds)


# ------------------------------------------------------------------ safety assertions
def assert_safety(arm, Xf, Xv, names, cat_cols, te_ref):
    """The 10 predeclared schema/safety checks. Any failure aborts before training.

    `te_ref` is the PER-FOLD control hash of the 48 te_ columns. Comparing te_ across FOLDS is
    meaningless -- they are fold-safe target encodings, refitted per fold, so fold 1's te_ columns
    are supposed to differ from fold 2's. My first version captured a single reference from
    whichever fold ran first and then compared later folds against it, so it aborted with "te_
    columns changed vs control" on a perfectly correct run. The reference is now keyed by fold, and
    the stronger property is checked directly: every arm in a fold receives the SAME assembled
    matrix object, so the treatment cannot have perturbed feature construction at all.
    """
    fails: list[str] = []

    def req(ok, msg):
        if not ok:
            fails.append(msg)

    # 3  feature count unchanged
    req(Xf.shape[1] == 285, f"feature count {Xf.shape[1]} != 285")
    # 3b the column NAMES and order are identical to the control: declaring categorical changes
    #    handling only, never the matrix layout.
    req(len(names) == 285 and len(set(names)) == 285, "column names are not 285 distinct values")
    # 2/4 the te_ block is present and byte-identical to this fold's control
    te_pos = [i for i, n in enumerate(names) if n.startswith("te_")]
    req(len(te_pos) == 48, f"te_ column count {len(te_pos)} != 48")
    req(te_hash(Xf, te_pos) == te_ref, "te_ columns differ from this fold's control")
    # 2b no appended duplicate: a declared categorical must be an EXISTING column
    pos = [names.index(c) for c in cat_cols]
    req(len(set(pos)) == len(pos), "duplicate column positions among the declared categoricals")
    # 5/6/7 category codes: non-negative integers, no NaN, and no code in the apply rows that was
    #    absent from the fit rows (which would let LightGBM invent a level at prediction time).
    for c, j in zip(cat_cols, pos):
        for tag, arr in (("fit", Xf[:, j]), ("val", Xv[:, j])):
            col = arr.astype("float64")
            req(not np.isnan(col).any(), f"{c}: NaN in {tag} rows")
            req(bool(np.all(col >= 0)), f"{c}: negative code in {tag} rows")
            req(bool(np.all(col == np.floor(col))), f"{c}: non-integer code in {tag} rows")
        unseen = set(np.unique(Xv[:, j]).tolist()) - set(np.unique(Xf[:, j]).tolist())
        req(not unseen, f"{c}: codes in val absent from fit: {sorted(unseen)}")
    # 7 survey zero must remain a REAL level, distinct from missingness.
    for c in cat_cols:
        if c in SURVEY13:
            req(0.0 in set(np.unique(Xf[:, names.index(c)].tolist())),
                f"{c}: level 0 is not present, so 'zero' may not be distinct from missing")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="L0,L1,L2")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tag", default="p12")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    if args.report:
        return report(args, tr, y, yi)

    folds = get_scheme(CHAMPION_SCHEME, yi, tr[ID_COL]).folds
    vb = ViewBuilder(tr, te, CHAMPION_VIEW)
    vb.build_static()

    print("=" * 100)
    print("PHASE 12 -- native categorical splits inside the winning LightGBM extra_trees family")
    print("=" * 100)
    print(f"  champion member {CHAMPION_MEMBER}: view={CHAMPION_VIEW} scheme={CHAMPION_SCHEME} "
          f"seed={CHAMPION_SEED}")
    print(f"  params: extra_trees=True lr=0.02 num_leaves=127 min_child_samples=40 "
          f"colsample=0.8 subsample=0.8 reg_lambda=1.0 max_bin=255 ES_patience={ES_PATIENCE}")
    print(f"  categorical params pinned: {CAT_PARAMS}")
    print("  declared prior: cat_smooth=10 and cat_l2=10 make a categorical split a SHRUNK target")
    print("  statistic with a prior -- the same smoothing family the 48 te_ columns already supply,")
    print("  so the expected effect is SMALL rather than new information. Stated before running.")
    print(f"  {'arm':<5}{'fold':>5}{'cat cols':>9}{'feat':>6}{'AUC':>12}{'delta':>10}"
          f"{'iters':>7}{'cat splits':>11}{'sec':>7}")
    print("  " + "-" * 89)

    # TWO fold-index coincidences, both invisible on fold 0, both found only because the L0
    # reproduction check runs on EVERY fold rather than the first:
    #   (a) the SEED. run_xt_zoo.py:112 passes the zoo entry's single seed and never adds the fold
    #       index. Using seed+k matches on fold 0 (1+0 == 1) and diverges thereafter.
    #   (b) the INNER SEED of the cross-fitted target encodings. run_xt_zoo.py:111 calls
    #       vb.assemble(fit, y, val, None, inner_seed=k) -- the FOLD INDEX. Using inner_seed=0
    #       matches on fold 0 and diverges on every other fold, because the inner cross-fit split
    #       for training rows is seeded by it.
    # Fixing (a) alone still left folds 1 and 2 mismatched by +4.5e-5 and -6.6e-5, which is what
    # exposed (b). A fold-0-only screen passes both bugs.
    out = {"tag": args.tag, "champion_member": CHAMPION_MEMBER,
           "champion_params": CHAMPION_PARAMS, "cat_params": CAT_PARAMS,
           "arms": {}, "folds_run": [],
           "protocol": "inner 10% carve of the FIT rows for early stopping; the model trains on "
                       "the other 90% and predicts the OUTER fold at best_iteration, exactly as "
                       "scripts/run_zoo.py::run_gbdt does. No outer-validation label reaches ES. "
                       "The SEED IS FIXED at 1 on every fold (run_xt_zoo.py:112 passes the entry "
                       "seed and never adds k) and inner_seed IS the fold index (run_xt_zoo.py:111). "
                       "Both fold-index coincidences are invisible on fold 0, which is why the L0 "
                       "reproduction check runs on every fold.",
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()[:12]}
    # te_ (target-encoding) columns are refitted PER FOLD, so a cross-fold reference is meaningless.
    # Keying the reference by fold is what makes the te_ invariance check meaningful: it asserts
    # every arm of a given fold was handed identical features, which is the property that actually
    # matters. The first version captured one reference from whichever fold ran first and compared
    # later folds against it, so it aborted with "te_ columns changed vs control" on a correct run.
    te_ref_by_fold: dict[int, str] = {}
    first_names: list[str] | None = None

    for aname in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
        if aname not in ARMS:
            raise SystemExit(f"unknown arm {aname}; known {sorted(ARMS)}")
        cat_cols, note = ARMS[aname]
        for k in [int(x) for x in args.folds.split(",")]:
            out["folds_run"].append(k)
            fit_idx = np.where(folds != k)[0]
            val_idx = np.where(folds == k)[0]
            Xf, Xa, names = vb.assemble(fit_idx, yi, val_idx, None, inner_seed=k)
            Xv = Xa["val"]
            te_pos = [i for i, n in enumerate(names) if n.startswith("te_")]
            te_ref = te_hash(Xf, te_pos)
            if k in te_ref_by_fold and te_ref_by_fold[k] != te_ref:
                raise SystemExit(f"STOP: fold {k} features differ between arms -- the treatment "
                                 f"perturbed feature construction.")
            te_ref_by_fold[k] = te_ref
            out.setdefault("fold_matrices", {})[str(k)] = {
                "te_hash": te_ref, "n_te": len(te_pos), "n_features": int(Xf.shape[1]),
                "n_fit": int(Xf.shape[0]), "n_val": int(Xv.shape[0])}
            if first_names is None:
                first_names = list(names)
            elif list(names) != first_names:
                raise SystemExit("STOP: column names/order differ between arms.")

            cat_cols_here = [c for c in cat_cols if c in names]
            missing = [c for c in cat_cols if c not in names]
            if missing:
                raise SystemExit(f"{aname}: columns absent from the {CHAMPION_VIEW} view: {missing}")
            cat_idx = [names.index(c) for c in cat_cols_here]

            fails = assert_safety(aname, Xf, Xv, names, cat_cols_here, te_ref)
            if fails:
                print(f"\n  {aname} fold {k}: SAFETY CHECK FAILED, not training:")
                for f in fails:
                    print(f"    - {f}")
                raise SystemExit("STOP: safety checks failed.")

            # SEED IS FIXED, not seed+k. run_xt_zoo.py:112 passes the zoo entry's single seed and
            # never adds the fold index. seed+k coincides with the correct value on fold 0
            # (1+0 == 1) and diverges on every other fold, so a fold-0-only screen cannot detect it.
            itr, es = _inner_es_split(fit_idx, yi, CHAMPION_SEED)
            pos = {int(v): i for i, v in enumerate(fit_idx)}
            tr_l = np.array([pos[int(v)] for v in itr])
            es_l = np.array([pos[int(v)] for v in es])
            if set(tr_l.tolist()) & set(es_l.tolist()):
                raise SystemExit("STOP: inner-train and ES rows overlap.")
            if len(tr_l) + len(es_l) != len(fit_idx):
                raise SystemExit("STOP: the inner carve does not partition the fit rows.")

            t0 = time.time()
            m, best = fit_arm(Xf[tr_l], y[itr], Xv, cat_idx, CHAMPION_SEED, Xf[es_l], y[es])
            pred = m.predict(Xv, num_iteration=best)
            secs = time.time() - t0
            cen = census(m)
            auc = float(roc_auc_score(y[val_idx], pred))
            np.save(REPORTS / f"{args.tag}_{aname}_{CHAMPION_SCHEME}_f{k}.npy",
                    pred.astype("float32"))

            ref = store.load_oof(CHAMPION_MEMBER).astype("float64")
            a_ref = float(roc_auc_score(y[val_idx], ref[val_idx]))
            rec = {"arm": aname, "note": note, "fold": k, "scheme": CHAMPION_SCHEME,
                   "view": CHAMPION_VIEW, "seed": CHAMPION_SEED, "cat_cols": cat_cols_here,
                   "cat_idx": cat_idx, "n_features": int(Xf.shape[1]), "n_cat": len(cat_cols_here),
                   "n_te": sum(1 for n in names if n.startswith("te_")),
                   "best_iter": best, "es_hit_cap": bool(best >= CHAMPION_PARAMS["n_estimators"] - 1),
                   "auc": auc, "champion_stored_auc": a_ref,
                   "delta_vs_champion_e5": (auc - a_ref) * 1e5,
                   "corr_vs_champion": float(corr(lg(pred), lg(ref[val_idx]))),
                   "spearman_vs_champion": float(spearman(pred, ref[val_idx])),
                   "seconds": round(secs, 1),
                   "census": {k2: v for k2, v in cen.items() if k2 != "splits_per_feature"},
                   "splits_on_declared": {c: cen["splits_per_feature"].get(j, 0)
                                          for c, j in zip(cat_cols_here, cat_idx)},
                   "n_train_rows": int(len(tr_l)), "n_es_rows": int(len(es_l))}
            out["arms"].setdefault(aname, {})[f"f{k}"] = rec
            dc = f"{rec['delta_vs_champion_e5']:+.1f}e"
            print(f"  {aname:<5}{k:>5}{len(cat_cols_here):>9}{Xf.shape[1]:>6}{auc:>12.6f}{dc:>10}"
                  f"{best:>7}{cen['n_categorical_splits']:>11}{secs:>7.0f}")

    save_json(out, REPORTS / f"{args.tag}_runs.json")
    print("\n  wrote", REPORTS / f"{args.tag}_runs.json")
    return 0


def report(args, tr, y, yi) -> int:
    runs = json.loads((REPORTS / f"{args.tag}_runs.json").read_text(encoding="utf-8"))
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    L = {e: lg(store.load_oof(e).astype("float64")) for e in ids}
    base_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    assert float(np.abs(sg(base_logit).astype(np.float32).astype("float64") - v3).max()) == 0.0, \
        "authoritative v3 geometry check failed"

    folds = get_scheme(CHAMPION_SCHEME, yi, tr[ID_COL]).folds
    tgt = CHAMPION_MEMBER
    print("=" * 104)
    print("PHASE 12 REPORT -- fold 0, extra_trees family, categorical handling as the only variable")
    print("=" * 104)
    print(f"  v3 geometry expit(mean(logits)) verified bit-exact in float32: True")
    print(f"  replacing v3 slot {tgt!r} (its OOF is the matched control)\n")
    print(f"  {'arm':<5}{'fold':>5}{'cat':>5}{'AUC':>12}{'delta':>10}{'iters':>7}"
          f"{'catsplit':>10}{'corr champ':>12}{'spear champ':>12}{'v3 swap':>11}{'swap d':>9}{'sec':>7}")
    print("  " + "-" * 110)
    rows = []
    for aname, recs in runs["arms"].items():
        for tagk, r in recs.items():
            f = REPORTS / f"{args.tag}_{aname}_{r['scheme']}_f{r['fold']}.npy"
            if not f.exists():
                continue
            k = r["fold"]
            val = np.where(folds == k)[0]
            P = np.load(f).astype("float64")
            col = L[tgt].copy()
            col[val] = lg(P)
            outside = np.ones(len(col), dtype=bool)
            outside[val] = False
            assert np.array_equal(col[outside], L[tgt][outside]), "splice altered out-of-fold rows"
            assert np.array_equal(col[val], lg(P)), "spliced column != native logit"
            aft = sg(np.mean(np.column_stack(
                [col if e == tgt else L[e] for e in ids]), axis=1))
            a_b4 = float(roc_auc_score(y[val], sg(base_logit[val])))
            a_sw = float(roc_auc_score(y[val], aft[val]))
            rows.append({"arm": aname, "fold": k, "auc": r["auc"],
                         "delta_vs_champion_e5": r["delta_vs_champion_e5"],
                         "v3_before": a_b4, "v3_after": a_sw,
                         "swap_delta_e5": (a_sw - a_b4) * 1e5,
                         "corr_champ": r["corr_vs_champion"],
                         "spear_champ": r["spearman_vs_champion"],
                         "n_cat": r["n_cat"], "best_iter": r["best_iter"],
                         "cat_splits": r["census"]["n_categorical_splits"],
                         "splits_on_declared": r["splits_on_declared"],
                         "seconds": r["seconds"]})
            print(f"  {aname:<5}{k:>5}{r['n_cat']:>5}{r['auc']:>12.6f}"
                  f"{r['delta_vs_champion_e5']:>+9.1f}e{r['best_iter']:>7}"
                  f"{r['census']['n_categorical_splits']:>10}"
                  f"{r['corr_vs_champion']:>12.5f}{r['spearman_vs_champion']:>12.5f}"
                  f"{a_sw:>11.6f}{(a_sw - a_b4) * 1e5:>+8.2f}e{r['seconds']:>7.0f}")

    # ---- L0 must reproduce the stored champion ON EVERY FOLD, or the harness is not the champion
    l0 = sorted((r for r in rows if r["arm"] == "L0"), key=lambda r: r["fold"])
    print()
    if l0:
        worst = max(abs(r["delta_vs_champion_e5"]) for r in l0)
        print(f"  L0 REPRODUCTION CHECK vs stored {tgt}, EVERY fold:")
        for r in l0:
            d = r["delta_vs_champion_e5"]
            print(f"    fold {r['fold']}: {d:+.3f}e-5  {'ok' if abs(d) < 1.0 else 'MISMATCH'}")
        print(f"    worst |delta| = {worst:.3f}e-5")
        if worst >= 1.0:
            print("    FAIL -- the harness is NOT the champion configuration; every delta above is")
            print("           void. The usual cause is the seed convention: the champion uses a")
            print("           FIXED seed on every fold, so seed+k matches on fold 0 and diverges")
            print("           after it, which a fold-0-only screen cannot detect.")
        else:
            print("    PASS -- the harness reproduces the champion configuration on every fold run.")
    else:
        print("  L0 not run: the reproduction check is MISSING and every delta is unvalidated.")

    print("\n  Fixed blend weights, DIAGNOSTIC ONLY (not tuned, not validated):")
    val = np.where(folds == 0)[0]
    a_b4 = float(roc_auc_score(y[val], sg(base_logit[val])))
    print(f"    v3 fold-0 control = {a_b4:.6f}")
    print(f"    {'arm':<5}" + "".join(f"{('w=' + str(w)):>13}" for w in (0.05, 0.15, 0.30)))
    for r in rows:
        if r["arm"] == "L0":
            continue
        P = np.load(REPORTS / f"{args.tag}_{r['arm']}_{CHAMPION_SCHEME}_f{r['fold']}.npy"
                    ).astype("float64")
        line = f"    {r['arm']:<5}"
        for w in (0.05, 0.15, 0.30):
            col = L[tgt].copy()
            col[val] = (1 - w) * L[tgt][val] + w * lg(P)
            aft = sg(np.mean(np.column_stack(
                [col if e == tgt else L[e] for e in ids]), axis=1))
            line += f"{(float(roc_auc_score(y[val], aft[val])) - a_b4) * 1e5:>+12.2f}e"
        print(line)

    best = max(rows, key=lambda r: r["delta_vs_champion_e5"]) if rows else None
    trigger5 = [r for r in rows if r["delta_vs_champion_e5"] >= 5.0]
    trigger15 = [r for r in rows if r["swap_delta_e5"] >= 1.5]
    print(f"\n  PROMOTION (predeclared)")
    print(f"    standalone >= +5e-5           : {[r['arm'] for r in trigger5] or 'none'}")
    print(f"    marginal blend >= +1.5e-5     : {[r['arm'] for r in trigger15] or 'none'}")
    verdict = ("PROMOTE to fold 1" if (trigger5 or trigger15) else
               "CLOSE the hypothesis: no treatment reached +5e-5 standalone or +1.5e-5 marginal. "
               "Do NOT rescue this with a hyperparameter zoo, and do NOT run L3.")
    print(f"    -> {verdict}")
    save_json({"rows": rows, "trigger5": [r["arm"] for r in trigger5],
               "trigger15": [r["arm"] for r in trigger15], "verdict": verdict,
               "best_arm": best["arm"] if best else None,
               "best_delta_e5": best["delta_vs_champion_e5"] if best else None},
              REPORTS / f"{args.tag}_report.json")
    print("\n  wrote", REPORTS / f"{args.tag}_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())