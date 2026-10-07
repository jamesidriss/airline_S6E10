"""Frozen SOL-A ablations and corrected v5 OOF counterparts.

Every fold writes its own immutable contract/report. A0 reuses the historical
auxiliary recipe (250 rounds, three inner folds); all downstream TE priors are
now strictly cross-fitted. No public score enters training or selection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (ARTIFACTS, ID_COL, REPORTS, TARGET, arr_sha256, file_sha256,
                        git_commit, load_cached_parquet, save_json)
from src.features.view import ViewBuilder
from src.features.aux_distribution import signatures, fit_surprise_threshold
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.run_views import _inner_es_split
from scripts.run_phase13 import RATINGS
from scripts.run_phase13b import SLOTS, SLOT_SEEDS, fit_slot
from scripts.run_phase14 import XGB_ARMS, CAT_ARMS, fit_xgb, fit_cat
from scripts.sol_aux_cache import probability_cache
from scripts.audit_sol_state import reconstruct_v5
from scripts.run_phase14c import counter_path


def model_spec(arm):
    if arm in SLOTS:
        return {"member": arm, "seed": SLOT_SEEDS[arm], "params": SLOTS[arm], "family": "lgbm_xt"}
    if arm in XGB_ARMS:
        member, seed, params = XGB_ARMS[arm]
        return {"member": member, "seed": seed, "params": params, "family": "xgb"}
    member, seed, params, native = CAT_ARMS[arm]
    return {"member": member, "seed": seed, "params": params, "family": "cat", "native": native}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--stages", default="0,1,2")
    ap.add_argument("--arms", default="X2")
    ap.add_argument("--tag", default="sol_a")
    ap.add_argument("--repair", action="store_true")
    args = ap.parse_args()
    arms = list(SLOTS) + ["X0", "X1", "X2", "C1"] if args.repair else args.arms.split(",")
    stages = [0] if args.repair else [int(x) for x in args.stages.split(",")]
    tr, te = load_cached_parquet()
    y = tr[TARGET].to_numpy(dtype="int8")
    folds = get_scheme("primary", y, tr[ID_COL]).folds
    _, champion, _ = reconstruct_v5(y, folds)
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    source = {p: file_sha256(p) for p in ("scripts/run_sol_a.py", "scripts/run_phase13.py",
               "src/features/aux_distribution.py", "src/features/s6e10.py", "src/features/view.py")}
    root = ARTIFACTS / args.tag
    root.mkdir(exist_ok=True)
    report_dir = REPORTS / args.tag
    report_dir.mkdir(exist_ok=True)
    for k in map(int, args.folds.split(",")):
        start = time.monotonic()
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        assert not np.intersect1d(fi, va).size and len(fi) + len(va) == len(y)
        Xf, Xa, names = vb.assemble(fi, y, va, None, inner_seed=k)
        Xv = Xa["val"]
        pf, pv, cache = probability_cache(Xf, Xv, names, tr[ID_COL].to_numpy()[fi],
                    tr[ID_COL].to_numpy()[va], arr_sha256(folds), label=f"fold{k}")
        rpos = [names.index(r) for r in RATINGS]
        rf, rv = Xf[:, rpos], Xv[:, rpos]
        threshold = fit_surprise_threshold(pf, rf)
        af0, _ = signatures(pf, rf, 0)
        av0, _ = signatures(pv, rv, 0)
        oldf = np.load(REPORTS / f"p13b_auxfit_f{k}.npy")
        oldv = np.load(REPORTS / f"p13b_auxval_f{k}.npy")
        # This is independent reproduction of the historical upstream models, not
        # comparison of two transformations of the same saved array.
        ev_gap = max(float(np.abs(af0 - oldf).max()), float(np.abs(av0 - oldv).max()))
        if ev_gap > 1e-10:
            raise RuntimeError(f"STOP: auxiliary expectation control failed ({ev_gap})")
        for arm in arms:
            spec = model_spec(arm)
            old_slot_pred = np.load(counter_path(spec['member'], k))
            # Historical p13b/p14 use seed 1 for the ES partition, independently
            # of each estimator's seed. Freeze that convention for comparability.
            itr, es = _inner_es_split(fi, y, 1)
            tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
            assert np.array_equal(fi[tl], itr) and np.array_equal(fi[el], es)
            assert not np.intersect1d(itr, es).size
            a0_pred = None
            for stage in stages:
                af, an = signatures(pf, rf, stage, threshold)
                av, _ = signatures(pv, rv, stage, threshold)
                A, V = np.column_stack([Xf, af]).astype(float), np.column_stack([Xv, av]).astype(float)
                cols = list(names) + an
                contract = {"fold": k, "stage": stage, "view": "full", "spec": spec,
                            "fit_ids_sha256": arr_sha256(tr[ID_COL].to_numpy()[itr]),
                            "early_stop_ids_sha256": arr_sha256(tr[ID_COL].to_numpy()[es]),
                            "validation_ids_sha256": arr_sha256(tr[ID_COL].to_numpy()[va]),
                            "outer_fit_ids_sha256": arr_sha256(tr[ID_COL].to_numpy()[fi]),
                            "fold_sha256": arr_sha256(folds), "feature_names": cols,
                            "feature_fit_sha256": arr_sha256(A), "feature_val_sha256": arr_sha256(V),
                            "aux_fingerprint": cache["fingerprint"], "source_sha256": source,
                            "test_policy": "same inner-three-fold aux recipe; full-label refit at five-fold median tree count",
                            "te_protocol": "outer-fit-only competition targets; FIT rows crossfit tables AND prior",
                            "train_rows": len(itr), "validation_rows": len(va), "es_rows": len(es),
                            "historical_aux_ev_max_gap": ev_gap}
                from hashlib import sha256
                fingerprint = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
                rp = report_dir / f"{arm}_A{stage}_f{k}.json"
                pp = root / f"{arm}_A{stage}_f{k}.npy"
                if rp.exists():
                    rec = json.loads(rp.read_text(encoding="utf-8"))
                    if rec["fingerprint"] != fingerprint:
                        raise RuntimeError(f"existing result has a different contract: {rp}; use a new tag")
                    pred = np.load(pp)
                    assert arr_sha256(pred) == rec["prediction_sha256"]
                    print(f"reuse {arm} A{stage} fold{k}", flush=True)
                else:
                    t0 = time.monotonic()
                    if arm in SLOTS:
                        pred, best = fit_slot(A[tl], y[itr], V, spec["seed"], spec["params"], A[el], y[es])
                        trees = best
                    elif arm.startswith("X"):
                        pred, best = fit_xgb(A[tl], y[itr], V, spec["seed"], spec["params"], A[el], y[es])
                        trees = best + 1  # XGB best_iteration is zero based
                    else:
                        from scripts.native_cat import attach, cat_frame, default_cat_cols
                        catcols = [c for c in default_cat_cols(True) if c in names]
                        F, cn = attach(A[tl], cols, cat_frame(tr, catcols, itr))
                        E, _ = attach(A[el], cols, cat_frame(tr, catcols, es))
                        VV, _ = attach(V, cols, cat_frame(tr, catcols, va))
                        pred, best = fit_cat(F, y[itr], VV, spec["seed"], spec["params"], cn, E, y[es])
                        trees = best + 1  # CatBoost best_iteration is zero based
                    pred = np.asarray(pred, dtype="float32")
                    assert pred.shape == (len(va),) and np.isfinite(pred).all()
                    assert ((pred >= 0) & (pred <= 1)).all()
                    np.save(pp, pred)
                    rec = {"contract": contract, "fingerprint": fingerprint, "git": git_commit(),
                           "auc": float(roc_auc_score(y[va], pred)), "n_trees": trees,
                           "seconds": time.monotonic() - t0, "prediction_sha256": arr_sha256(pred),
                           "prediction_path": str(pp), "hardware": "RTX 5070 Ti 16GB; CPU eight threads",
                           "status": "POSITIVE_UNCONFIRMED"}
                    save_json(rec, rp)
                if stage == 0:
                    a0_pred = pred
                if a0_pred is None:
                    a0_pred = np.load(root / f"{arm}_A0_f{k}.npy")
                delta = rec["auc"] - float(roc_auc_score(y[va], a0_pred))
                new_champion = champion[va] + (logit(pred) - logit(old_slot_pred)) / 59
                marginal = float(roc_auc_score(y[va], new_champion) - roc_auc_score(y[va], champion[va]))
                rec.update(delta_vs_A0=delta, operational_v5_slot_delta=marginal,
                           logit_corr_vs_A0=float(np.corrcoef(logit(pred), logit(a0_pred))[0, 1]))
                save_json(rec, rp)
                print(f"{arm} A{stage} fold{k}: AUC {rec['auc']:.9f}; delta A0 {delta:+.9f}; v5 marginal {marginal:+.9f}; trees {rec['n_trees']}; {rec['seconds']:.1f}s", flush=True)
                if stage in (1, 2) and delta < 0 and marginal <= 0 and not args.repair:
                    rec["status"] = "NEGATIVE_STOP_LADDER"
                    save_json(rec, rp)
                    print("STOP ladder: negative standalone and ensemble; no blind feature expansion", flush=True)
                    break
                del A, V, af, av
            del a0_pred
        save_json({"completed_fold": k, "seconds": time.monotonic() - start, "git": git_commit()}, root / "progress.json")
        del Xf, Xv, Xa, pf, pv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
