"""Phase 20 H19-C -- full 5-fold promotion gate for smoothed nearest-key target features.

Completion of the fold-0 PATH verdict (reports/phase20/h19c_fold0_screen.json). For each of the
5 primary folds k: fit control (rich view) and treatment (rich view + nk block) on folds!=k, eval
on fold k, fold-safe (rich-view TE + nearest-key crossfit rebuilt from the fit rows only). Report
the per-fold paired member deltas and apply the frozen member-level promotion gate.

Protocol is FROZEN in configs/phase20/h19c_full5_prereg.json (committed a3444b8 BEFORE any folds
1-4 treatment evaluation).  Control and treatment are identical rich-view LGBM members; the ONLY
difference is the 3-column nk block.  The portfolio (11th-member) proxy is reported INFORMATIONALLY
only -- it underweights a feature block and is NOT the gate quantity.  CPU-only; 0 GPU; no submission.

Run:  .\\.venv\\Scripts\\python.exe scripts/phase20/h19c_full5_gate.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, save_json  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402
from scripts.evaluate_sol_foundation import classical  # noqa: E402
from scripts.replay_sol_classical import frozen_roles  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.phase19.h19a_difficulty import assert_partition, fit_lgbm_weighted  # noqa: E402
from src.phase20.h19c_nearest_key import (  # noqa: E402
    RATING_COLS, NearestKeyTE, build_segment_flags, rating_key_ints,
)

OUT = REPORTS / "phase20"
ART = ARTIFACTS / "phase17_B1"
PREREG = ROOT / "configs" / "phase20" / "h19c_full5_prereg.json"
PREREG_SHA256 = "475e90defa2f8c71a7be0ea993b9b85db83e3f5ff476868eb20840e633dae361"
CHAMPION_PORTFOLIO_SHA256 = "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552"
CHAMPION_ROUTE_SHA256 = "bcb2811329d3686882450e93435647f8f7ab65abfc48c1a1b51db1e9d1ef95e0"

MEMBER_PARAMS = {"objective": "binary", "metric": "auc", "n_estimators": 6000,
                 "learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 40,
                 "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
                 "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True,
                 "verbose": -1, "n_jobs": 8}
MEMBER_SEED = 20261010
INNER_ES_SEED = 1
NK_LAM, NK_MIN_COUNT, NK_INNER_FOLDS, NK_INNER_SEED = 30.0, 2, 5, 0


def log(msg: str) -> None:
    print(f"[h19c-full5] {msg}", flush=True)


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def _segment_auc(y, p, mask):
    ys, ps = y[mask], p[mask]
    if len(np.unique(ys)) < 2:
        return None
    return float(roc_auc_score(ys, ps))


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def decide(mean_delta: float, paired_se: float, positive_count: int) -> tuple[str, dict]:
    """Frozen promotion-gate rules from configs/phase20/h19c_full5_prereg.json."""
    gate_mean = mean_delta >= 1.5e-5
    gate_se = mean_delta >= 2.5 * paired_se
    gate_pos = positive_count >= 4
    if gate_mean and gate_se and gate_pos:
        verdict = "PROMOTE"
    elif mean_delta <= 0 or positive_count <= 2:
        verdict = "DEAD"
    else:
        verdict = "INCONCLUSIVE"
    decision = {
        "mean_paired_delta": mean_delta, "paired_se": paired_se,
        "positive_fold_count": positive_count,
        "gates": {"mean_ge_1p5e-5": bool(gate_mean), "mean_ge_2p5_se": bool(gate_se),
                  "positive_ge_4of5": bool(gate_pos)},
        "thresholds": {"mean": 1.5e-5, "se_multiple": 2.5, "min_positive_folds": 4},
        "verdict": verdict,
        "note": "member-level promotion gate; PROMOTE recommends the portfolio-level deployment test "
                "(nk added to all 10 aux members) as a SEPARATE, heavier, separately-preregistered "
                "experiment -- it does NOT auto-launch it.",
    }
    return verdict, decision


def main() -> int:
    t0 = time.time()
    if (OUT / "h19c_full5_gate.json").exists():
        raise FileExistsError("Preserve measured H19-C full-5 result")
    OUT.mkdir(parents=True, exist_ok=True)

    prereg_sha = file_sha256(PREREG)
    assert prereg_sha == PREREG_SHA256, f"prereg changed: {prereg_sha} != {PREREG_SHA256}"

    tr, te, y, folds, vectors, tests, proof = bank()
    ids = tr.id.to_numpy()
    roles = [r for r in frozen_roles() if r["aux_arm"]]
    n = len(y)
    ntr, nte = len(tr), len(te)
    ks = sorted(np.unique(folds).astype(int).tolist())
    assert ks == [0, 1, 2, 3, 4], ks
    log(f"rows={n} folds={ks}")

    # champion authorities (hash-asserted, all folds)
    route_full = np.full(n, np.nan, dtype=np.float64)
    port_full = np.full(n, np.nan, dtype=np.float64)
    for k in ks:
        fl = np.flatnonzero(folds == k)
        route_full[fl] = np.load(ART / f"f{k}" / "prediction.npy")
        port_full[fl] = np.load(ART / f"f{k}" / "portfolio.npy")
    assert np.isfinite(route_full).all() and np.isfinite(port_full).all()
    assert arr_sha256(port_full.astype("float32")) == CHAMPION_PORTFOLIO_SHA256, "portfolio hash drift"
    assert arr_sha256(route_full.astype("float32")) == CHAMPION_ROUTE_SHA256, "route hash drift"

    # segment flags (transductive, label-free, atlas definition) -- computed once
    comb = np.concatenate([tr[RATING_COLS].to_numpy(dtype=np.int64),
                           te[RATING_COLS].to_numpy(dtype=np.int64)], axis=0)
    is_dup_all = build_segment_flags(pd.DataFrame(comb, columns=RATING_COLS))
    is_dup_tr = is_dup_all[:len(tr)]

    # base-6 rating keys + raw ratings (label-free) for all train rows, once
    key_all = rating_key_ints(tr)
    R_all = tr[RATING_COLS].to_numpy(dtype=np.int64)

    vb = ViewBuilder(tr, te, "full")
    vb.build_static()

    oof_ctrl = np.zeros(n, dtype=np.float64)
    oof_treat = np.zeros(n, dtype=np.float64)
    route_oof = np.zeros(n, dtype=np.float64)
    aux_oof = np.zeros(n, dtype=np.float64)
    per_fold = []

    for k in ks:
        fit_idx = np.flatnonzero(folds != k)
        val_idx = np.flatnonzero(folds == k)
        assert_partition(fit_idx, val_idx, folds, k)

        # rich-view base features (fold-safe TE rebuilt from fit rows only)
        Xf, Xa, _ = vb.assemble(fit_idx, y, val_idx, None, inner_seed=MEMBER_SEED)
        Xval = Xa["val"]
        assert Xf.shape[0] == len(fit_idx) and Xval.shape[0] == len(val_idx)

        # nearest-key block (fold-safe): crossfit within fit rows, apply to fold-k rows
        nkte = NearestKeyTE(lam=NK_LAM, min_count=NK_MIN_COUNT,
                            inner_folds=NK_INNER_FOLDS, inner_seed=NK_INNER_SEED)
        nk_fit = nkte.crossfit(key_all[fit_idx], y[fit_idx].astype(float), R_all[fit_idx])
        nk_val = nkte.apply(key_all[fit_idx], y[fit_idx].astype(float),
                            key_all[val_idx], R_all[val_idx])
        assert nk_fit.shape == (len(fit_idx), 3) and nk_val.shape == (len(val_idx), 3)

        # identical inner-ES split (carved from fit rows only) for control + treatment
        trn_g, es_g = _inner_es_split(fit_idx, y, INNER_ES_SEED)
        tl = np.searchsorted(fit_idx, trn_g)
        el = np.searchsorted(fit_idx, es_g)
        ones = np.ones(len(tl), dtype=np.float64)

        log(f"fold {k}: fit={len(fit_idx)} val={len(val_idx)} training control+treatment ...")
        Xtr_c, Xes_c = Xf[tl], Xf[el]
        Xtr_t = np.hstack([Xf[tl], nk_fit[tl]])
        Xes_t = np.hstack([Xf[el], nk_fit[el]])
        Xval_t = np.hstack([Xval, nk_val])  # treatment predicts on base+nk, matching its train width
        ctrl_pred, it_c = fit_lgbm_weighted(Xtr_c, y[trn_g], Xval, MEMBER_PARAMS, MEMBER_SEED,
                                            Xes_c, y[es_g], sample_weight=ones)
        treat_pred, it_t = fit_lgbm_weighted(Xtr_t, y[trn_g], Xval_t, MEMBER_PARAMS, MEMBER_SEED,
                                             Xes_t, y[es_g], sample_weight=ones)
        assert ctrl_pred.shape == (len(val_idx),) and treat_pred.shape == (len(val_idx),)
        assert np.isfinite(ctrl_pred).all() and np.isfinite(treat_pred).all()
        oof_ctrl[val_idx] = ctrl_pred
        oof_treat[val_idx] = treat_pred

        # champion references for this fold (bit-exact composition check)
        route_k = route_full[val_idx]
        port_k = port_full[val_idx]
        aux_k_logits, _ = classical(k, "sol_clean_aux10_primary", roles, ids, folds, y,
                                    proof["data_sha256"])
        recomposed = (1.0 / (1.0 + np.exp(-(logit(route_k) + aux_k_logits) / 2.0))).astype("float32")
        assert np.array_equal(recomposed, port_k.astype("float32")), f"champion composition drift f{k}"
        route_oof[val_idx] = route_k
        aux_oof[val_idx] = aux_k_logits

        # per-fold metrics
        yk = y[val_idx]
        auc_c = _auc(yk, ctrl_pred)
        auc_t = _auc(yk, treat_pred)
        dupk = is_dup_tr[val_idx]
        uniqk = ~dupk
        u_c = _segment_auc(yk, ctrl_pred, uniqk)
        u_t = _segment_auc(yk, treat_pred, uniqk)
        d_c = _segment_auc(yk, ctrl_pred, dupk)
        d_t = _segment_auc(yk, treat_pred, dupk)
        # informational portfolio proxy (11th-member; underweights a feature block)
        aux11_c = (10.0 * aux_k_logits + logit(ctrl_pred)) / 11.0
        aux11_t = (10.0 * aux_k_logits + logit(treat_pred)) / 11.0
        port_c = 1.0 / (1.0 + np.exp(-(logit(route_k) + aux11_c) / 2.0))
        port_t = 1.0 / (1.0 + np.exp(-(logit(route_k) + aux11_t) / 2.0))
        pf = {
            "fold": k, "ctrl_auc": auc_c, "treat_auc": auc_t, "delta": auc_t - auc_c,
            "unique": {"n": int(uniqk.sum()), "ctrl": u_c, "treat": u_t,
                       "delta": (u_t - u_c) if (u_c is not None and u_t is not None) else None},
            "dup": {"n": int(dupk.sum()), "ctrl": d_c, "treat": d_t,
                    "delta": (d_t - d_c) if (d_c is not None and d_t is not None) else None},
            "portfolio_proxy": {"base_auc": _auc(yk, port_k), "ctrl_auc": _auc(yk, port_c),
                                "treat_auc": _auc(yk, port_t),
                                "delta": _auc(yk, port_t) - _auc(yk, port_c)},
            "corr_treat_route": _pearson(treat_pred, route_k),
            "corr_treat_aux": _pearson(treat_pred, 1.0 / (1.0 + np.exp(-aux_k_logits))),
            "ctrl_best_iter": it_c, "treat_best_iter": it_t,
        }
        per_fold.append(pf)
        log(f"  fold {k}: ctrl={auc_c:.10f} treat={auc_t:.10f} delta={auc_t-auc_c:+.3e} "
            f"unique_delta={pf['unique']['delta']}")
        del Xf, Xtr_c, Xes_c, Xtr_t, Xes_t, nk_fit
        gc.collect()

    # aggregate promotion statistics
    deltas = np.array([p["delta"] for p in per_fold], dtype=np.float64)
    mean_delta = float(deltas.mean())
    paired_se = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
    positive_count = int((deltas > 0).sum())
    verdict, decision = decide(mean_delta, paired_se, positive_count)

    # OOF-level standalone + informational portfolio
    auc_ctrl_oof = _auc(y, oof_ctrl)
    auc_treat_oof = _auc(y, oof_treat)
    uniq_all = ~is_dup_tr
    dup_all = is_dup_tr
    oof_unique = {"ctrl": _segment_auc(y, oof_ctrl, uniq_all), "treat": _segment_auc(y, oof_treat, uniq_all)}
    oof_dup = {"ctrl": _segment_auc(y, oof_ctrl, dup_all), "treat": _segment_auc(y, oof_treat, dup_all)}
    port_proxy_ctrl = 1.0 / (1.0 + np.exp(-(logit(route_oof) + (10.0 * aux_oof + logit(oof_ctrl)) / 11.0) / 2.0))
    port_proxy_treat = 1.0 / (1.0 + np.exp(-(logit(route_oof) + (10.0 * aux_oof + logit(oof_treat)) / 11.0) / 2.0))

    log(f"per-fold deltas: {[f'{d:+.3e}' for d in deltas]}")
    log(f"mean={mean_delta:+.3e} se={paired_se:.3e} positive={positive_count}/5")
    log(f"OOF ctrl={auc_ctrl_oof:.10f} treat={auc_treat_oof:.10f} delta={auc_treat_oof-auc_ctrl_oof:+.3e}")
    log(f"OOF unique: ctrl={oof_unique['ctrl']} treat={oof_unique['treat']}")
    log(f"OOF dup:    ctrl={oof_dup['ctrl']} treat={oof_dup['treat']}")
    log(f"VERDICT: {verdict}")

    np.save(OUT / "h19c_full5_oof_ctrl.npy", oof_ctrl.astype("float32"))
    np.save(OUT / "h19c_full5_oof_treat.npy", oof_treat.astype("float32"))

    res = {
        "experiment": "P20-H19C-FULL5", "hypothesis_id": "H19-C", "experiment_number": 45,
        "status": "COMPLETE", "verdict": verdict, "error": None,
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "prereg": str(PREREG.relative_to(ROOT)), "prereg_sha256": prereg_sha,
        "parent_fold0": "reports/phase20/h19c_fold0_screen.json (PATH_TO_REPLICATION)",
        "scope": {"scheme": "primary", "folds": ks, "n_train": ntr, "n_test": nte},
        "features": {"block": "nk", "params": {"lam": NK_LAM, "min_count": NK_MIN_COUNT,
                                               "inner_folds": NK_INNER_FOLDS, "inner_seed": NK_INNER_SEED},
                     "base_view": "full (rich, fold-safe TE)", "only_difference_vs_control": "presence of nk block"},
        "members": {"params": MEMBER_PARAMS, "member_seed": MEMBER_SEED, "inner_es_seed": INNER_ES_SEED,
                    "per_fold_best_iter": [{"fold": p["fold"], "ctrl": p["ctrl_best_iter"],
                                            "treat": p["treat_best_iter"]} for p in per_fold]},
        "per_fold": per_fold,
        "promotion": {"paired_fold_deltas": deltas.tolist(), "mean_paired_delta": mean_delta,
                      "paired_se": paired_se, "positive_fold_count": positive_count,
                      "gate": {"mean_threshold": 1.5e-5, "se_multiple": 2.5, "min_positive_folds": 4}},
        "oof_standalone": {"ctrl_auc": auc_ctrl_oof, "treat_auc": auc_treat_oof,
                           "delta": auc_treat_oof - auc_ctrl_oof,
                           "unique": oof_unique, "dup": oof_dup},
        "portfolio_proxy_informational": {
            "note": "11th-member equal-weight proxy; UNDERWEIGHTS a feature block; NOT the gate quantity",
            "ctrl_auc": _auc(y, port_proxy_ctrl), "treat_auc": _auc(y, port_proxy_treat),
            "delta": _auc(y, port_proxy_treat) - _auc(y, port_proxy_ctrl),
        },
        "decision": decision,
        "champion_hashes": {"portfolio": CHAMPION_PORTFOLIO_SHA256, "route": CHAMPION_ROUTE_SHA256,
                            "composition_bit_exact_per_fold": True},
        "budget": {"gpu_jobs": 0, "submission": 0},
        "oof_hashes": {"ctrl": arr_sha256(oof_ctrl.astype("float32")),
                       "treat": arr_sha256(oof_treat.astype("float32"))},
    }
    save_json(res, OUT / "h19c_full5_gate.json")
    log(f"saved {OUT / 'h19c_full5_gate.json'} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
