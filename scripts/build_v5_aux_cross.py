"""Build v5_aux_cross: v3 with 10 slots replaced by aux-equipped counterparts, on the TEST set.

PREDECLARED BEFORE ANY PUBLIC SCORE WAS CONSULTED
-------------------------------------------------
  role          Champion A candidate
  OOF           0.961523, +1.49e-5 versus v3
  folds         5/5 positive, paired t +3.97
  slots         6 extra_trees LightGBM, 3 XGBoost, 1 CatBoost (native-cat representation)
  admission     MISSES the predeclared +1.5e-5 gate by 0.6%. The gate is not moved. Submitted
                because the user authorised sanity-check submissions and the sign consistency is the
                cleanest this campaign has produced.
  expectation   public movement UNKNOWN, and predicted to be below public resolution, which is
                about +/-2e-4 paired. No specific score is predicted and none will be tuned toward.

TEST-TIME PROTOCOL
------------------
Each counterpart is refit on ALL 699,635 training labels, because a test model that used only 80% of
them would be strictly worse than v3's own test models, all of which use 100% (that is exactly what
v4_fulldata measures). The iteration count is the MEDIAN of that slot's five cross-validated
best_iterations, which is the policy scripts/run_views.py::_fit_full_predict_* documents.

THE AUX FEATURES AT TEST TIME MIRROR OOF EXACTLY
-------------------------------------------------
OOF used cross-fitted aux predictions for the fit rows and a full-fit aux model for the applied rows.
If the test model were given differently-distributed aux features than it was calibrated on, the
gain measured OOF would not be the gain realised at test. So: training rows get 5-fold inner
cross-fitted aux predictions over the training set; test rows get predictions from aux models fitted
on ALL training rows. Same construction, same label-freeness, same expected-value form, no
satisfaction label anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, ROOT, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from scripts.run_phase12 import lg, sg  # noqa: E402
from scripts.run_phase13 import RATINGS, build_aux  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

XT_SLOTS = ["xt_xt_d127_s1", "xt_xt_d63", "xt_xt_d255", "xt_xt_d127_cs05",
            "xt_xt_d127_ss06", "xt_xt_d127_bin63"]
XGB_ARM = {"prod5_xgb_full_primary": "X0", "xt_xgb_lossguide": "X1", "zoo_xgb_d6": "X2"}
XGB_PARAMS = {
    "X0": {},
    "X1": dict(learning_rate=0.03, max_depth=0, grow_policy="lossguide", max_leaves=127,
               min_child_weight=8, colsample_bytree=0.8),
    "X2": dict(learning_rate=0.04, max_depth=6, min_child_weight=5, colsample_bytree=0.85,
               subsample=0.9, reg_lambda=5.0, reg_alpha=0.05),
}
CAT_MEMBER, CAT_SEED, CAT_PARAMS = "z3_cat_d8_s2", 3, dict(learning_rate=0.04, depth=8)
SUB_DIR = ROOT / "submissions"


def _iters_from_json(tag: str) -> dict:
    p = REPORTS / f"{tag}_runs.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out: dict = {}
    for fk, rec in d.get("arms", {}).items():
        for var in rec:
            for kk, v in rec[var].items():
                out.setdefault(fk, {})[v["member"]] = int(v["iters"])
    return out


def _iters_from_logs() -> dict:
    """Recover per-slot iteration counts from the run logs.

    Run-scoped JSON is overwritten per --folds invocation -- the same hazard that bit p13b and again
    bit p14 -- so the logs are the durable record. Every fold/arm/iteration triple that produced a
    saved prediction vector is recovered from them, and the count of recovered folds per slot is
    asserted, because a median over two folds would silently differ from a median over five.
    """
    import re
    out: dict = {}
    # p14 aux rows: arm fold variant nfeat AUC d corr spear swap ITERS SEC (vs stored ...)
    pat = re.compile(
        r"^\s+(\S+)\s+(\d)\s+aux\s+\d+\s+[\d.]+\s+\S+\s+[\d.]+\s+[\d.]+\s+\S+\s+(\d+)\s+(\d+)\s+\(vs stored")
    for lg in (REPORTS / "p14_f12.log", REPORTS / "p14_f34.log", REPORTS / "p14_f0.log"):
        if not lg.exists():
            continue
        for line in lg.read_text(encoding="utf-8").splitlines():
            m = pat.match(line)
            if m:
                out.setdefault(f"f{m.group(2)}", {})[f"ARM_{m.group(1)}"] = int(m.group(3))
    # p13b slot rows: "slot NAME seed S standalone  +DDe-5  iters N  corr w/ original C"
    pat2 = re.compile(r"^\s+slot\s+(\S+)\s+seed\s+\d+\s+standalone\s+[+\-0-9.e]+\s+iters\s+(\d+)")
    for lg in (REPORTS / "p13b.log", REPORTS / "p13b34.log"):
        if not lg.exists():
            continue
        for line in lg.read_text(encoding="utf-8").splitlines():
            m = pat2.match(line)
            if m:
                out.setdefault("p13", {}).setdefault(m.group(1), []).append(int(m.group(2)))
    return out


def median_ints() -> dict:
    """Median cross-validated best_iteration per slot.

    Sourced from the run JSON where it survives, and from the logs otherwise. Both sources are
    merged and the number of folds recovered per slot is reported, because a test-time refit length
    taken from 2 folds instead of 5 is a silently different model.
    """
    out: dict = {}
    p14 = _iters_from_json("p14")
    for fk, m in p14.items():
        for slot, it in m.items():
            if slot in XGB_ARM or slot == CAT_MEMBER:
                out.setdefault(slot, []).append(it)
    lg = _iters_from_logs()
    for fk, m in lg.items():
        for slot, it in m.items():
            if slot.startswith("ARM_"):
                arm = slot[4:]
                member = next((k for k, v in XGB_ARM.items() if v == arm), None)
                if member is None and arm == "C1":
                    member = CAT_MEMBER
                if member:
                    out.setdefault(member, []).append(it)
    for slot, lst in lg.get("p13", {}).items():
        out.setdefault(slot, []).extend(lst)
    miss = [k for k in XT_SLOTS + list(XGB_ARM) + [CAT_MEMBER] if len(out.get(k, [])) < 5]
    if miss:
        raise SystemExit(f"STOP: fewer than 5 fold iteration counts recovered for {miss}. "
                         f"A median over fewer folds is a different model; refusing to guess.")
    return {k: int(np.median(sorted(v))) for k, v in out.items() if k in
            XT_SLOTS + list(XGB_ARM) + [CAT_MEMBER]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aux-rounds", type=int, default=250)
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    yi = tr[TARGET].values.astype("int8")
    y = tr[TARGET].values.astype("float64")
    ntr, nte = len(tr), len(te)
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]

    mi = median_ints()
    print("  median cross-validated iteration counts (the test-time refit length):")
    for k in XT_SLOTS + list(XGB_ARM) + [CAT_MEMBER]:
        print(f"    {k:<26} {mi[k]}")

    # ---- assemble the FULL train+test frame the way the aux builder needs ---------------
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    all_idx = np.arange(ntr + nte)
    # ViewBuilder keeps TWO static frames: static_tr (train rows) and static_te (train+test). The
    # `val` apply set indexes static_tr, so it cannot carry test rows; test rows come from
    # static_te via the FOURTH argument, which run_zoo.py:219 passes as np.arange(ntr, ntr+nte).
    # Two earlier attempts indexed past the end of static_tr before this.
    from src.common import TARGET as TGT
    Xf, Xa, names = vb.assemble(np.arange(ntr), yi, np.arange(ntr), all_idx[ntr:], inner_seed=0)
    Xtr_raw, Xte_raw = Xf, Xa["test"]
    assert Xtr_raw.shape[0] == ntr and Xte_raw.shape[0] == nte, (Xtr_raw.shape, Xte_raw.shape)
    print(f"  train frame {Xtr_raw.shape}, test frame {Xte_raw.shape}")

    # ---- aux features, mirroring the OOF construction ---------------------------------
    print("  building aux features over train+test (label-free, 5-fold inner cross-fit)")
    aux_tr, aux_te, info = build_aux(Xtr_raw, Xte_raw, names, 1, rounds=args.aux_rounds,
                                     inner_folds=5)
    Xtr = np.column_stack([Xtr_raw, aux_tr]).astype("float64")
    Xte = np.column_stack([Xte_raw, aux_te]).astype("float64")
    print(f"    aux mean out-of-fold accuracy {np.mean([i['aux_oof_acc'] for i in info]):.4f}")
    assert Xtr.shape[1] == 298 and Xte.shape[1] == 298, (Xtr.shape, Xte.shape)

    # ---- fit each counterpart on ALL training labels at its median iteration count -------
    import lightgbm as lgb
    import xgboost as xgb
    from scripts.run_phase12 import CAT_PARAMS
    preds = {}
    for slot in XT_SLOTS:
        n = mi[slot]
        p = dict(objective="binary", metric="auc", n_estimators=n, learning_rate=0.02,
                 num_leaves=127, min_child_samples=40, colsample_bytree=0.8, subsample=0.8,
                 subsample_freq=1, reg_lambda=1.0, max_bin=255, verbose=-1, n_jobs=8,
                 extra_trees=True, random_state=1, bagging_seed=2, feature_fraction_seed=3)
        m = lgb.train(p, lgb.Dataset(Xtr, label=y), num_boost_round=n)
        preds[slot] = m.predict(Xte)
        print(f"    {slot:<26} lgb  iters {n:<5} done")

    from scripts.run_views import _fit_xgb_es
    XGB_BASE = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
                    max_depth=8, min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=2.0, max_bin=256, tree_method="hist", device="cuda", n_jobs=8)
    for member, arm in XGB_ARM.items():
        n = mi[member]
        seed = {"prod5_xgb_full_primary": 1, "xt_xgb_lossguide": 12, "zoo_xgb_d6": 8}[member]
        p = dict(XGB_BASE)
        p.update(XGB_PARAMS[arm])
        p.update(n_estimators=n, random_state=seed)
        p.pop("early_stopping_rounds", None)
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr, y, verbose=False)
        preds[member] = m.predict_proba(Xte)[:, 1]
        print(f"    {member:<26} xgb  iters {n:<5} done")

    from catboost import CatBoostClassifier
    from scripts.native_cat import attach, cat_frame, cat_indices, default_cat_cols
    cols_all = [n for n in names] + [f"aux_ev_{i}" for i in range(13)]
    catcols = [c for c in default_cat_cols(True) if c in names]
    F, cn = attach(Xtr, cols_all, cat_frame(tr, catcols, np.arange(ntr)))
    V, _ = attach(Xte, cols_all, cat_frame(te, catcols, np.arange(nte)))
    p = dict(iterations=mi[CAT_MEMBER], learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
             random_seed=CAT_SEED, thread_count=8, verbose=0, allow_writing_files=False,
             boosting_type="Plain")
    m = CatBoostClassifier(**p)
    m.fit(F, y, cat_features=cat_indices(F, cn), verbose=0)
    preds[CAT_MEMBER] = m.predict_proba(V)[:, 1]
    print(f"    {CAT_MEMBER:<26} cat  iters {mi[CAT_MEMBER]:<5} done")

    # ---- splice into v3's TEST predictions, keeping 59 slots and equal-logit geometry ----
    T = {e: lg(store.load_test(e).astype("float64")) for e in ids}
    cols = [lg(preds[e]) if e in preds else T[e] for e in ids]
    assert len(cols) == 59, f"member count changed to {len(cols)}"
    v5t = sg(np.mean(np.column_stack(cols), axis=1))
    v3t = store.load_test("blend_v3_final").astype("float64")
    print(f"\n  v5_aux_cross test prediction")
    print(f"    slots replaced      : {len(preds)}  (member count stays 59, equal-logit)")
    print(f"    test corr vs v3     : {corr(lg(v5t), lg(v3t)):.5f}")
    print(f"    test spearman vs v3 : {spearman(v5t, v3t):.5f}")
    print(f"    mean |test p change|: {np.abs(v5t - v3t).mean():.3e}")
    print(f"    EXPECTED public movement: UNKNOWN, likely below public resolution (~+/-2e-4 paired)")

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub = te[[ID_COL]].copy()
    sub[TGT] = v5t
    p_out = SUB_DIR / "v5_aux_cross.csv"
    sub.to_csv(p_out, index=False)
    sha = hashlib.sha256(p_out.read_bytes()).hexdigest()[:16]
    print(f"    wrote {p_out}  rows {len(sub)}  sha16 {sha}")
    save_json({"name": "v5_aux_cross", "role": "Champion A candidate",
               "oof": 0.961523, "delta_vs_v3_e5": 1.491, "folds_positive": "5/5", "paired_t": 3.97,
               "gate": "+1.5e-5 MISSED by 0.6%; gate not moved",
               "slots_replaced": sorted(preds), "n_members": 59, "geometry": "expit(mean(logits))",
               "median_iterations": mi, "test_corr_vs_v3": float(corr(lg(v5t), lg(v3t))),
               "test_spearman_vs_v3": float(spearman(v5t, v3t)),
               "expected_public_movement": "UNKNOWN, likely below public resolution",
               "sha16": sha, "rows": int(len(sub)),
               "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()[:12]},
              REPORTS / "v5_aux_cross_manifest.json")
    print(f"    wrote {REPORTS / 'v5_aux_cross_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())