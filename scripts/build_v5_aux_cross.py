"""Repaired ten-slot v5 inference, preserving every banked finalist.

Require exactly five corrected OOF iteration records per counterpart. Refit
these ten slots on all labels with their own configurations and median TREE
COUNTS; retain the other 49 stored v3 test vectors. Auxiliary training features
use three inner folds, matching OOF. Save every substituted test vector.

Legacy v5's OOF gate passed on mean paired fold gain (+1.53545e-5), but its
submitted test configurations did not reproduce the OOF recipe. The repaired
candidate needs its own scorecard; it does not inherit the legacy public score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, ROOT, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from scripts.run_phase12 import lg, sg  # noqa: E402

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


def median_ints(record_dir="reports/sol_repair") -> dict:
    """Median tree count from exact fold/treatment/member records; fail closed."""
    out = {}
    arms = {s: s for s in XT_SLOTS}
    arms.update(XGB_ARM)
    arms[CAT_MEMBER] = "C1"
    for member, arm in arms.items():
        counts = []
        for fold in range(5):
            path = ROOT / record_dir / f"{arm}_A0_f{fold}.json"
            if not path.exists():
                raise SystemExit(f"STOP: missing exact fold/treatment iteration record: {path}")
            rec = json.loads(path.read_text(encoding="utf-8"))
            c = rec["contract"]
            if c["fold"] != fold or c["stage"] != 0 or c["spec"]["member"] != member:
                raise SystemExit(f"STOP: iteration record is for the wrong slot/fold/stage: {path}")
            # n_trees is a count; XGB/CatBoost zero-based indices were converted by the runner.
            counts.append(int(rec["n_trees"]))
        out[member] = int(np.median(counts))
    return out


def xt_refit_params(slot, n):
    from scripts.run_phase12 import CHAMPION_PARAMS, CAT_PARAMS
    from scripts.run_phase13b import SLOTS, SLOT_SEEDS
    seed = SLOT_SEEDS[slot]
    p = dict(CHAMPION_PARAMS)
    p.update(CAT_PARAMS)
    p.update(SLOTS[slot])
    p.update(n_estimators=n, random_state=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aux-rounds", type=int, default=250)
    ap.add_argument("--name", default="v5_aux_cross_repaired")
    ap.add_argument("--iteration-records", default="reports/sol_repair")
    args = ap.parse_args()

    mi = median_ints(args.iteration_records)
    if args.dry_run:
        print(json.dumps({"name": args.name, "median_tree_counts": mi,
                          "xt_params": {s: xt_refit_params(s, mi[s]) for s in XT_SLOTS}}, indent=2))
        return 0
    if (SUB_DIR / f"{args.name}.csv").exists():
        raise SystemExit("STOP: output submission exists; use a new name to preserve finalists")
    tr, te = load_cached_parquet()
    yi = tr[TARGET].values.astype("int8")
    y = tr[TARGET].values.astype("float64")
    ntr, nte = len(tr), len(te)
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]

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
    print("  building aux features (label-free, 3-fold inner cross-fit, matching OOF)")
    from scripts.sol_aux_cache import probability_cache
    from src.common import arr_sha256, ARTIFACTS
    pf, pv, aux_manifest = probability_cache(Xtr_raw, Xte_raw, names,
             tr[ID_COL].to_numpy(), te[ID_COL].to_numpy(), arr_sha256(tr[ID_COL].to_numpy()),
             rounds=args.aux_rounds, inner_folds=3, label="test refit")
    aux_tr, aux_te = pf @ np.arange(6), pv @ np.arange(6)
    info = aux_manifest["ratings"]
    Xtr = np.column_stack([Xtr_raw, aux_tr]).astype("float64")
    Xte = np.column_stack([Xte_raw, aux_te]).astype("float64")
    print(f"    aux mean out-of-fold accuracy {np.mean([i['aux_oof_acc'] for i in info]):.4f}")
    assert Xtr.shape[1] == 298 and Xte.shape[1] == 298, (Xtr.shape, Xte.shape)

    # ---- fit each counterpart on ALL training labels at its median iteration count -------
    import lightgbm as lgb
    import xgboost as xgb
    from scripts.run_phase12 import CAT_PARAMS
    preds = {}
    configs = {}
    pred_dir = ARTIFACTS / args.name
    pred_dir.mkdir(exist_ok=True)
    for slot in XT_SLOTS:
        n = mi[slot]
        p = xt_refit_params(slot, n)
        configs[slot] = p
        m = lgb.train(p, lgb.Dataset(Xtr, label=y), num_boost_round=n)
        preds[slot] = m.predict(Xte)
        np.save(pred_dir / f"{slot}_test.npy", preds[slot])
        print(f"    {slot:<26} lgb  iters {n:<5} done")

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
        configs[member] = p
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr, y, verbose=False)
        preds[member] = m.predict_proba(Xte)[:, 1]
        np.save(pred_dir / f"{member}_test.npy", preds[member])
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
    configs[CAT_MEMBER] = {**p, "cat_columns": catcols}
    m.fit(F, y, cat_features=cat_indices(F, cn), verbose=0)
    preds[CAT_MEMBER] = m.predict_proba(V)[:, 1]
    np.save(pred_dir / f"{CAT_MEMBER}_test.npy", preds[CAT_MEMBER])
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
    from src.submission.make import build
    p_out = build(v5t, args.name, notes="S0 repaired v5 configs; OOF from corrected counterparts; not automatically submitted", members=ids)
    sha = hashlib.sha256(p_out.read_bytes()).hexdigest()[:16]
    print(f"    wrote {p_out}  rows {len(te)}  sha16 {sha}")
    save_json({"name": args.name, "role": "repaired candidate pending OOF scorecard",
               "slots_replaced": sorted(preds), "n_members": 59, "geometry": "expit(mean(logits))",
               "median_iterations": mi, "test_corr_vs_v3": float(corr(lg(v5t), lg(v3t))),
               "params_by_slot": configs, "aux_manifest": aux_manifest,
               "test_slot_sha256": {s: arr_sha256(p) for s, p in preds.items()},
               "test_ids_sha256": arr_sha256(te[ID_COL].to_numpy()),
               "training_policy": "10 corrected slots use all labels; other 49 retain v3 fold-averaged test vectors",
               "test_spearman_vs_v3": float(spearman(v5t, v3t)),
               "expected_public_movement": "UNKNOWN, likely below public resolution",
               "sha16": sha, "rows": int(len(te)),
               "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()[:12]},
              REPORTS / f"{args.name}_manifest.json")
    print(f"    wrote {REPORTS / f'{args.name}_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
