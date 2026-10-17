"""Phase 20 H19-C -- fold-0 screen for smoothed nearest-key target features.

Answers ONE question (fold 0 only): does aggregating the smoothed target means of
the FIT keys at Hamming distance 1 (a distinct mechanism from the closed H18-C
exact-key TE) improve a rich-view LightGBM member, with the gain concentrated in
the unique-rating-key weak region (36% of rows, atlas AUC 0.95032)?

Protocol is FROZEN in configs/phase20/h19c_prereg.json (committed 6e186d6 BEFORE
any treatment evaluation).  Control and treatment are identical rich-view LGBM
members; the ONLY difference is the 3-column nk block (crossfit for FIT rows,
apply for EVAL rows from the FIT table only).  Anti-leakage: no fold-0 label in
features/targets/early-stopping; segment flags are transductive + label-free.
CPU-only; 0 GPU jobs; no submission.

Run:  .\\.venv\\Scripts\\python.exe scripts/phase20/h19c_fold0_screen.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
from src.phase19.h19a_difficulty import (  # noqa: E402
    assert_no_fold0, assert_partition, fit_lgbm_weighted,
)
from src.phase20.h19c_nearest_key import (  # noqa: E402
    RATING_COLS, NearestKeyTE, build_segment_flags, rating_key_ints,
)

OUT = REPORTS / "phase20"
ART = ARTIFACTS / "phase17_B1"
PREREG = ROOT / "configs" / "phase20" / "h19c_prereg.json"
PREREG_SHA256 = "493065a22e0d9486494e7285f359659008d3ca4410cf9ad565540cff57430990"
CHAMPION_PORTFOLIO_SHA256 = "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552"
CHAMPION_ROUTE_SHA256 = "bcb2811329d3686882450e93435647f8f7ab65abfc48c1a1b51db1e9d1ef95e0"

# ---- frozen params (mirror prereg) ----
MEMBER_PARAMS = {"objective": "binary", "metric": "auc", "n_estimators": 6000,
                 "learning_rate": 0.02, "num_leaves": 127, "min_child_samples": 40,
                 "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
                 "reg_lambda": 1.0, "max_bin": 255, "extra_trees": True,
                 "verbose": -1, "n_jobs": 8}
MEMBER_SEED = 20261010
INNER_ES_SEED = 1
# nearest-key estimator (frozen in prereg)
NK_LAM, NK_MIN_COUNT, NK_INNER_FOLDS, NK_INNER_SEED = 30.0, 2, 5, 0


def log(msg: str) -> None:
    print(f"[h19c] {msg}", flush=True)


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def _segment_auc(y, p, mask) -> tuple[float | None, int]:
    ys, ps = y[mask], p[mask]
    if len(np.unique(ys)) < 2:
        return None, int(len(ys))
    return float(roc_auc_score(ys, ps)), int(len(ys))


def main() -> int:
    t0 = time.time()
    if (OUT / "h19c_fold0_screen.json").exists():
        raise FileExistsError("Preserve measured H19-C result")
    OUT.mkdir(parents=True, exist_ok=True)

    prereg_sha = file_sha256(PREREG)
    assert prereg_sha == PREREG_SHA256, f"prereg changed: {prereg_sha} != {PREREG_SHA256}"

    tr, te, y, folds, vectors, tests, proof = bank()
    ids = tr.id.to_numpy()
    roles = [r for r in frozen_roles() if r["aux_arm"]]
    n = len(y)
    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert_partition(FIT, EVAL, folds, 0)
    log(f"rows={n} FIT={len(FIT)} EVAL={len(EVAL)}")

    # ---- 1) rich-view base features (fold-safe TE identical to H19-A) ----
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    Xf, Xa, names = vb.assemble(FIT, y, EVAL, None, inner_seed=MEMBER_SEED)
    Xv = Xa["val"]
    assert Xf.shape[0] == len(FIT) and Xv.shape[0] == len(EVAL)
    log(f"rich view base: FIT={Xf.shape} EVAL={Xv.shape}")

    # ---- 2) nearest-key block (fold-safe): crossfit FIT, apply EVAL ----
    key_all = rating_key_ints(tr)                      # label-free base-6 keys (all tr rows)
    R_all = tr[RATING_COLS].to_numpy(dtype=np.int64)
    fit_key, fit_R = key_all[FIT], R_all[FIT]
    eval_key, eval_R = key_all[EVAL], R_all[EVAL]
    nkte = NearestKeyTE(lam=NK_LAM, min_count=NK_MIN_COUNT,
                        inner_folds=NK_INNER_FOLDS, inner_seed=NK_INNER_SEED)
    nk_fit = nkte.crossfit(fit_key, y[FIT].astype(float), fit_R)      # (len(FIT),3)
    nk_eval = nkte.apply(fit_key, y[FIT].astype(float), eval_key, eval_R)  # (len(EVAL),3)
    assert nk_fit.shape == (len(FIT), 3) and nk_eval.shape == (len(EVAL), 3)
    assert np.isfinite(nk_fit).all() and np.isfinite(nk_eval).all()
    log(f"nk block built: fit={nk_fit.shape} eval={nk_eval.shape} "
        f"nk1_fit_mean={nk_fit[:,0].mean():.4f} nk1_eval_mean={nk_eval[:,0].mean():.4f}")

    # ---- 3) control (base) vs treatment (base + nk), identical otherwise ----
    trn, es = _inner_es_split(FIT, y, INNER_ES_SEED)
    assert_no_fold0(trn, folds, "member trn")
    assert_no_fold0(es, folds, "member es")
    assert len(np.intersect1d(trn, es)) == 0
    tl, el = np.searchsorted(FIT, trn), np.searchsorted(FIT, es)

    Xtr_c, Xes_c, Xv_c = Xf[tl], Xf[el], Xv
    Xtr_t = np.hstack([Xf[tl], nk_fit[tl]])
    Xes_t = np.hstack([Xf[el], nk_fit[el]])
    Xv_t = np.hstack([Xv, nk_eval])
    ones = np.ones(len(trn), dtype=np.float64)

    log("training control (base rich view) ...")
    ctrl_pred, it_c = fit_lgbm_weighted(Xtr_c, y[trn], Xv_c, MEMBER_PARAMS, MEMBER_SEED,
                                        Xes_c, y[es], sample_weight=ones)
    log("training treatment (base + nk) ...")
    treat_pred, it_t = fit_lgbm_weighted(Xtr_t, y[trn], Xv_t, MEMBER_PARAMS, MEMBER_SEED,
                                         Xes_t, y[es], sample_weight=ones)
    assert ctrl_pred.shape == (len(EVAL),) and treat_pred.shape == (len(EVAL),)
    assert np.isfinite(ctrl_pred).all() and np.isfinite(treat_pred).all()
    log(f"fit done: ctrl_iter={it_c} treat_iter={it_t}")
    del Xf, Xtr_c, Xes_c, Xtr_t, Xes_t
    import gc
    gc.collect()

    # ---- 4) champion authorities (hash-asserted, composition bit-exact) ----
    route_full = np.full(n, np.nan, dtype=np.float64)
    port_full = np.full(n, np.nan, dtype=np.float64)
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        route_full[fl] = np.load(ART / f"f{k}" / "prediction.npy")
        port_full[fl] = np.load(ART / f"f{k}" / "portfolio.npy")
    assert np.isfinite(route_full).all() and np.isfinite(port_full).all()
    assert arr_sha256(port_full.astype("float32")) == CHAMPION_PORTFOLIO_SHA256, "portfolio hash drift"
    assert arr_sha256(route_full.astype("float32")) == CHAMPION_ROUTE_SHA256, "route hash drift"
    route0, port0 = route_full[EVAL], port_full[EVAL]
    aux0_logits, _ = classical(0, "sol_clean_aux10_primary", roles, ids, folds, y,
                               proof["data_sha256"])
    recomposed = (1.0 / (1.0 + np.exp(-(logit(route0) + aux0_logits) / 2.0))).astype("float32")
    assert np.array_equal(recomposed, port0.astype("float32")), "champion composition drift"
    aux0_prob = 1.0 / (1.0 + np.exp(-aux0_logits))

    # ---- 5) segment flags (transductive, label-free, atlas definition) ----
    comb = np.concatenate([tr[RATING_COLS].to_numpy(dtype=np.int64),
                           te[RATING_COLS].to_numpy(dtype=np.int64)], axis=0)
    import pandas as pd
    comb_df = pd.DataFrame(comb, columns=RATING_COLS)
    is_dup_all = build_segment_flags(comb_df)          # length len(tr)+len(te)
    is_dup_tr = is_dup_all[:len(tr)]
    dup0 = is_dup_tr[EVAL]                              # True = duplicate rating key
    unique0 = ~dup0

    # ---- 6) metrics ----
    y0 = y[EVAL]
    auc_ctrl = _auc(y0, ctrl_pred)
    auc_treat = _auc(y0, treat_pred)
    delta = auc_treat - auc_ctrl
    auc_route = _auc(y0, route0)
    auc_base = _auc(y0, port0)

    u_ctrl, n_u = _segment_auc(y0, ctrl_pred, unique0)
    u_treat, _ = _segment_auc(y0, treat_pred, unique0)
    d_ctrl, n_d = _segment_auc(y0, ctrl_pred, dup0)
    d_treat, _ = _segment_auc(y0, treat_pred, dup0)
    seg_u_delta = (u_treat - u_ctrl) if (u_ctrl is not None and u_treat is not None) else None
    seg_d_delta = (d_treat - d_ctrl) if (d_ctrl is not None and d_treat is not None) else None

    # ---- 7) correlations with champion halves ----
    def _pearson(a, b):
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        return float(np.corrcoef(a, b)[0, 1])
    corr = {nm: {"pearson_route": _pearson(mp, route0), "pearson_aux": _pearson(mp, aux0_prob)}
            for nm, mp in (("control", ctrl_pred), ("treatment", treat_pred))}

    # ---- 8) portfolio: equal-weight 11th aux member ----
    ctrl_logit, treat_logit = logit(ctrl_pred), logit(treat_pred)
    aux11_ctrl = (10.0 * aux0_logits + ctrl_logit) / 11.0
    aux11_treat = (10.0 * aux0_logits + treat_logit) / 11.0
    port_ctrl = 1.0 / (1.0 + np.exp(-(logit(route0) + aux11_ctrl) / 2.0))
    port_treat = 1.0 / (1.0 + np.exp(-(logit(route0) + aux11_treat) / 2.0))
    auc_port_base = _auc(y0, port0)
    auc_port_ctrl = _auc(y0, port_ctrl)
    auc_port_treat = _auc(y0, port_treat)
    port_delta = auc_port_treat - auc_port_ctrl

    # ---- 9) decision (frozen rules) ----
    verdict, decision = decide(delta, seg_u_delta, port_delta)

    log(f"AUC ctrl={auc_ctrl:.10f} treat={auc_treat:.10f} delta={delta:.3e}")
    log(f"unique-key: n={n_u} ctrl={u_ctrl} treat={u_treat} delta={seg_u_delta}")
    log(f"dup-key:    n={n_d} ctrl={d_ctrl} treat={d_treat} delta={seg_d_delta}")
    log(f"portfolio: base={auc_port_base:.10f} ctrl={auc_port_ctrl:.10f} "
        f"treat={auc_port_treat:.10f} delta={port_delta:.3e}")
    log(f"VERDICT: {verdict}")

    # ---- 10) persist ----
    np.save(OUT / "h19c_ctrl_pred.npy", ctrl_pred.astype("float32"))
    np.save(OUT / "h19c_treat_pred.npy", treat_pred.astype("float32"))
    res = {
        "experiment": "P20-H19C", "hypothesis_id": "H19-C", "experiment_number": 45,
        "status": "COMPLETE", "verdict": verdict, "error": None,
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "prereg": str(PREREG.relative_to(ROOT)), "prereg_sha256": prereg_sha,
        "scope": {"fold": 0, "fit_folds": [1, 2, 3, 4], "scheme": "primary", "n_fit": len(FIT), "n_eval": len(EVAL)},
        "features": {"block": "nk", "cols": ["nk1", "nk1_logc", "nk1_nk"],
                     "params": {"lam": NK_LAM, "min_count": NK_MIN_COUNT,
                                "inner_folds": NK_INNER_FOLDS, "inner_seed": NK_INNER_SEED},
                     "fit_convention": "crossfit within FIT (no row own label); EVAL apply from FIT table only",
                     "base_view": "full (rich, fold-safe TE)"},
        "members": {"params": MEMBER_PARAMS, "member_seed": MEMBER_SEED,
                    "inner_es_seed": INNER_ES_SEED, "only_difference": "presence of nk block",
                    "ctrl_best_iter": it_c, "treat_best_iter": it_t},
        "segments": {
            "unique": {"n": n_u, "ctrl_auc": u_ctrl, "treat_auc": u_treat, "delta": seg_u_delta},
            "dup": {"n": n_d, "ctrl_auc": d_ctrl, "treat_auc": d_treat, "delta": seg_d_delta},
            "definition": "transductive label-free atlas is_dup (key count>1 over tr+te)"},
        "standalone": {"ctrl_auc": auc_ctrl, "treat_auc": auc_treat, "delta": delta,
                       "route_auc": auc_route, "base_portfolio_auc": auc_base},
        "correlations": corr,
        "portfolio": {"composition": "aux11=(10*aux0+member_logit)/11; port=sigmoid((logit(route)+aux11)/2)",
                      "base_auc": auc_port_base, "ctrl_auc": auc_port_ctrl,
                      "treat_auc": auc_port_treat, "delta": port_delta},
        "decision": decision,
        "champion_hashes": {"portfolio": CHAMPION_PORTFOLIO_SHA256, "route": CHAMPION_ROUTE_SHA256,
                            "composition_bit_exact": True},
        "budget": {"gpu_jobs": 0, "submission": 0},
        "prediction_hashes": {"ctrl": arr_sha256(ctrl_pred.astype("float32")),
                              "treat": arr_sha256(treat_pred.astype("float32"))},
    }
    save_json(res, OUT / "h19c_fold0_screen.json")
    log(f"saved {OUT / 'h19c_fold0_screen.json'} in {time.time()-t0:.1f}s")
    return 0


def decide(delta: float, seg_u_delta: float | None, port_delta: float) -> tuple[str, dict]:
    """Frozen decision rules from configs/phase20/h19c_prereg.json."""
    d = delta
    su = seg_u_delta
    pd_ = port_delta
    if d <= 0:
        verdict = "DEAD"
    elif d < 3e-5 and (su is None or su <= 0):
        verdict = "DEAD"
    elif d >= 3e-5 and (su is not None and su > 0) and pd_ >= 0:
        verdict = "PATH_TO_REPLICATION"
    else:
        verdict = "INCONCLUSIVE"
    decision = {
        "d": d, "seg_u_delta": su, "port_delta": pd_,
        "thresholds": {"path_d": 3e-5, "path_seg_u": 0.0, "path_port": 0.0,
                       "dead_d": 0.0, "dead_if_d_lt": 3e-5},
        "rule_applied": verdict,
        "note": "fold 0 cannot validate the full 5-fold promotion gate; PATH only recommends folds 1-4 for future consideration, does not launch them",
    }
    return verdict, decision


if __name__ == "__main__":
    raise SystemExit(main())
