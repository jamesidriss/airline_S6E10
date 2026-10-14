"""Phase 19 H19-A -- fold-0 falsifier for difficulty-targeted reweighted LightGBM.

Answers ONE question (fold 0 only): does upweighting covariate-locatable hard rows
make a rich-view LightGBM member that is genuinely more predictive (and more often
correct on the champion's shared-fail pairs) than the same member with uniform
weights -- or does it merely identify hard observations?

Protocol is FROZEN in configs/phase19/h19a_prereg.json (committed ffcb28b BEFORE
any treatment evaluation).  Anti-leakage: difficulty weights are rebuilt strictly
inside FIT (folds 1-4) from inner-cross-fitted models that never see fold 0; the
Atlas diagnostics (per_row_burden, pred_a, ...) are FORBIDDEN (their producing
models trained on fold-0 labels).  CPU-only; 0 GPU jobs; no submission.

Run:  .\\.venv\\Scripts\\python.exe scripts\\phase19\\h19a_fold0_falsifier.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, save_json  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402
from scripts.evaluate_sol_foundation import classical  # noqa: E402
from scripts.replay_sol_classical import frozen_roles  # noqa: E402
from scripts.run_views import _fit_lgbm_es, _inner_es_split  # noqa: E402
from scripts.phase19.build_shared_error_atlas import two_class_counts  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.phase19.h19a_difficulty import (  # noqa: E402
    assert_no_fold0, assert_partition, assert_weight_valid,
    fit_lgbm_weighted, pearson, rank01, sample_pair_stats, spearman,
    weights_from_score,
)

OUT = REPORTS / "phase19"
ART = ARTIFACTS / "phase19" / "h19a"
PREREG = ROOT / "configs" / "phase19" / "h19a_prereg.json"
PREREG_SHA256 = "d02a8b74be49423ad40c3167d54a983eb8f87d783d1bba919a0e656ccfab5b09"
CHAMPION_PORTFOLIO_SHA256 = "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552"
CHAMPION_ROUTE_SHA256 = "bcb2811329d3686882450e93435647f8f7ab65abfc48c1a1b51db1e9d1ef95e0"

# ---- frozen params (mirror prereg) ----
JUDGE_A = {"n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 127,
           "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
           "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255,
           "extra_trees": True, "verbose": -1, "n_jobs": 8}
JUDGE_B = {"n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 63,
           "min_child_samples": 60, "colsample_bytree": 0.6, "subsample": 0.8,
           "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255,
           "extra_trees": False, "verbose": -1, "n_jobs": 8}
JUDGE_SEED_A, JUDGE_SEED_B = 11, 22
DIFF_PARAMS = {"n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 96,
               "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
               "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255,
               "verbose": -1, "n_jobs": 8}
DIFF_SEED = 1201
MEMBER_PARAMS = {"n_estimators": 6000, "learning_rate": 0.02, "num_leaves": 127,
                 "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
                 "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255,
                 "extra_trees": True, "verbose": -1, "n_jobs": 8}
MEMBER_SEED = 20261010
INNER_ES_SEED = 1
LAM, W_LO, W_HI = 0.5, 0.25, 2.0
SAMPLE_N, SAMPLE_SEED = 50_000_000, 20261019
INNER_FOLDS = [1, 2, 3, 4]  # FIT folds


def log(msg: str) -> None:
    print(f"[h19a] {msg}", flush=True)


# ----------------------------------------------------------------- audit
def anti_leakage_audit(tr, te, y, folds, vb, roles, ids, proof) -> dict:
    """Prove the Atlas diagnostics for folds 1-4 are fold-0 contaminated.

    1. set proof   : classical(k) / champion route for fold k train on folds != k,
                     which contains fold 0 for k in 1..4.
    2. numeric demo: a model trained on {0,2,3,4} vs {2,3,4} gives DIFFERENT
                     fold-1 predictions (static, label-free features) -> fold-0
                     labels provably change fold-k-row predictions.
    """
    log("audit: set proof + numeric fold-0 contamination demo")
    set_proof = {}
    for k in range(1, 5):
        fi = np.flatnonzero(folds != k)          # exactly classical(k)/route OOF fit set
        f0 = np.flatnonzero(folds == 0)
        in_fit = np.intersect1d(fi, f0)
        set_proof[str(k)] = {
            "fit_folds": sorted(np.unique(folds[fi]).astype(int).tolist()),
            "contains_fold0": bool(len(in_fit)),
            "n_fold0_rows_in_training": int(len(in_fit)),
        }
        assert len(in_fit) == len(f0), f"fold {k}: fold-0 not fully in classical(k) fit"

    # numeric demo: static (label-free) features, isolate the label channel
    import lightgbm as lgb
    static = vb.static_tr
    f1 = folds == 1
    train_with0 = folds != 1
    train_without0 = (folds != 1) & (folds != 0)
    p = dict(objective="binary", metric="auc", n_estimators=200, learning_rate=0.1,
             num_leaves=63, min_child_samples=40, verbose=-1, n_jobs=8,
             random_state=7, extra_trees=True)
    def _fit(mask):
        ds = lgb.Dataset(static[mask], label=y[mask])
        m = lgb.train(p, ds, num_boost_round=p["n_estimators"])
        return m.predict(static[f1])
    pw, po = _fit(train_with0), _fit(train_without0)
    corr = pearson(pw, po)
    mdiff = float(np.abs(pw - po).mean())
    numeric = {
        "fold": 1,
        "model_with_fold0_trained_on_folds": sorted(np.unique(folds[train_with0]).astype(int).tolist()),
        "model_without_fold0_trained_on_folds": sorted(np.unique(folds[train_without0]).astype(int).tolist()),
        "pearson_of_fold1_predictions": corr,
        "mean_abs_diff_fold1_predictions": mdiff,
        "max_abs_diff_fold1_predictions": float(np.abs(pw - po).max()),
        "interpretation": "predictions differ => fold-0 labels changed fold-1-row "
                          "predictions => any fold-k (k>=1) diagnostic derived from "
                          "them carries fold-0 label information",
    }
    assert corr < 1.0 - 1e-9 and mdiff > 0.0, "numeric contamination demo failed"

    blocking = (
        "FAITHFUL TARGET BLOCKED: the preregistered difficulty target is the champion "
        "SHARED difficulty (route=TabPFN AND aux both fail). The champion route is TabPFN "
        "(GPU). Rebuilding it fold-0-safely for folds 1-4 requires 4 TabPFN trainings inside "
        "FIT = heavy GPU, explicitly out of scope for H19-A (CPU-only, no heavy GPU). The "
        "champion aux for folds 1-4 is itself fold-0-contaminated (trains on folds != k). "
        "Therefore the faithful shared difficulty CANNOT be recalculated strictly safely within "
        "the H19-A budget. This falsifier proceeds with an explicitly documented fold-0-safe "
        "CPU PROXY shared difficulty (two inner-cross-fitted LightGBM judges) -- see prereg "
        "documented_deviations DEV-1."
    )
    return {
        "set_proof": set_proof,
        "numeric_demo": numeric,
        "blocking_diagnostic": blocking,
        "forbidden_as_training_weights": [
            "reports/phase19/per_row_burden.parquet", "atlas pred_a", "atlas pred_b",
            "atlas stored route (folds 1-4)", "atlas stored aux (folds 1-4)",
        ],
        "conclusion": "Atlas diagnostics for folds 1-4 are fold-0-contaminated and were NOT "
                      "used as training weights anywhere in this run.",
    }


# ------------------------------------------------- fold-0-safe difficulty
def rebuild_difficulty(vb, y, folds, fit_idx) -> tuple[np.ndarray, dict]:
    """Rebuild the difficulty score for FIT rows, fold-0-safely, CPU-only.

    Phase 1  judges   : for inner fold j, two LGBMs trained on FIT\\{j} -> fold j.
    Phase 2  target   : shared-misranking burden between the judges inside fold j,
                        fold-wise top-quintile flag (Atlas quintile convention).
    Phase 3  difficulty model : covariate-only LightGBM predicting the flag,
                        inner-cross-fit within FIT (train FIT\\{j} -> fold j).

    Returns (d[fit_idx] difficulty score, provenance dict).  Never sees fold 0.
    """
    import gc
    n = len(y)
    fit_set = np.zeros(n, dtype=bool)
    fit_set[fit_idx] = True
    judge_a = np.full(n, np.nan, dtype=np.float64)
    judge_b = np.full(n, np.nan, dtype=np.float64)

    # ---- Phase 1: judges (inner cross-fit within FIT) ----
    for j in INNER_FOLDS:
        fi = np.flatnonzero(fit_set & (folds != j))   # FIT \ {j}
        va = np.flatnonzero(folds == j)
        assert_no_fold0(fi, folds, f"judge fi (j={j})")
        assert len(np.intersect1d(fi, va)) == 0
        itr, es = _inner_es_split(fi, y, JUDGE_SEED_A + j)
        assert_no_fold0(itr, folds, f"judge trn (j={j})")
        assert_no_fold0(es, folds, f"judge es (j={j})")
        xf, xa, _ = vb.assemble(fi, y, va, None, inner_seed=j)
        tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
        Xtr, Xes, Xva = xf[tl], xf[el], xa["val"]
        pa, _ = fit_lgbm_weighted(Xtr, y[itr], Xva, JUDGE_A, JUDGE_SEED_A, Xes, y[es])
        pb, _ = fit_lgbm_weighted(Xtr, y[itr], Xva, JUDGE_B, JUDGE_SEED_B, Xes, y[es])
        judge_a[va], judge_b[va] = pa, pb
        del xf, xa, Xtr, Xes, Xva
        gc.collect()
        log(f"  judges inner fold {j} done")
    assert np.isfinite(judge_a[fit_set]).all() and np.isfinite(judge_b[fit_set]).all()

    # judge correlation (how 'shared' is two-CPU-judge shared fail)
    corr_ab = pearson(judge_a[fit_set], judge_b[fit_set])

    # ---- Phase 2: shared-misranking burden -> fold-wise top-quintile flag ----
    flag = np.zeros(n, dtype=np.int8)
    per_fold = {}
    for j in INNER_FOLDS:
        m = folds == j
        yy = y[m]
        ra, rb = rankdata(judge_a[m]), rankdata(judge_b[m])
        c = two_class_counts(yy, ra, rb)
        burden = c["burden_neg"].astype(np.int64) + c["burden_pos"].astype(np.int64)
        thr = float(np.quantile(burden, 0.8))
        flag[m] = (burden >= thr).astype(np.int8)
        per_fold[str(j)] = {"rows": int(m.sum()), "shared_ge": int(c["shared_ge"]),
                            "flag_rate": float(flag[m].mean()), "burden_q80": thr}
    assert 0.15 < float(flag[fit_set].mean()) < 0.25, \
        f"flag balance {float(flag[fit_set].mean())}"

    # ---- Phase 3: covariate difficulty model (inner cross-fit within FIT) ----
    d = np.full(n, np.nan, dtype=np.float64)
    for j in INNER_FOLDS:
        fi = np.flatnonzero(fit_set & (folds != j))
        va = np.flatnonzero(folds == j)
        itr, es = _inner_es_split(fi, flag.astype(np.int8), DIFF_SEED + j)
        assert_no_fold0(fi, folds, f"diff fi (j={j})")
        xf, xa, _ = vb.assemble(fi, y, va, None, inner_seed=j)
        tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
        Xtr, Xes, Xva = xf[tl], xf[el], xa["val"]
        dp, _ = fit_lgbm_weighted(Xtr, flag[itr], Xva, DIFF_PARAMS, DIFF_SEED, Xes, flag[es])
        d[va] = dp
        del xf, xa, Xtr, Xes, Xva
        gc.collect()
        log(f"  difficulty model inner fold {j} done")
    assert np.isfinite(d[fit_set]).all(), "difficulty score not finite"

    prov = {
        "judge_pearson": corr_ab,
        "judge_B": "inner-cross-fit within FIT, never fold 0",
        "target": "fold-wise top-quintile shared-misranking burden (two judges)",
        "flag_rate_fit": float(flag[fit_set].mean()),
        "per_fold_target": per_fold,
        "difficulty_score_mean": float(d[fit_set].mean()),
        "difficulty_score_std": float(d[fit_set].std()),
        "fold0_seen_by_any_supervised_fit": False,
    }
    return d[fit_set], prov


# ------------------------------------------------------------------ decision
def decide(d_delta, s_rate, s_se, port_delta) -> tuple[str, dict]:
    s_sig = bool(s_rate >= 2.0 * s_se)
    if d_delta < 3e-5 and s_rate <= 0:
        verdict = "DEAD"
    elif d_delta >= 3e-5 and s_sig and port_delta >= 0:
        verdict = "PATH_TO_REPLICATION"
    else:
        verdict = "INCONCLUSIVE"
    detail = {
        "d_delta": d_delta, "convincing_single_fold_gain": 3e-5,
        "s_rate": s_rate, "s_se": s_se, "s_sig": s_sig,
        "portfolio_delta": port_delta, "verdict": verdict,
        "rule": "DEAD: d<3e-5 AND s_rate<=0 | PATH: d>=3e-5 AND s>=2SE AND port>=0 | else INCONCLUSIVE",
    }
    return verdict, detail


# --------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    if (OUT / "h19a_fold0_falsifier.json").exists():
        raise FileExistsError("Preserve measured H19-A result")
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    # prereg is the frozen contract; verify hash matches the committed value
    prereg_sha = file_sha256(PREREG)
    assert prereg_sha == PREREG_SHA256, f"prereg changed: {prereg_sha} != {PREREG_SHA256}"

    tr, te, y, folds, vectors, tests, proof = bank()
    ids = tr.id.to_numpy()
    roles = [r for r in frozen_roles() if r["aux_arm"]]
    n = len(y)
    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert_partition(FIT, EVAL, folds, 0)
    log(f"rows={n} FIT={len(FIT)} EVAL={len(EVAL)} folds={sorted(np.unique(folds).astype(int).tolist())}")

    vb = ViewBuilder(tr, te, "full")
    vb.build_static()

    # 1) anti-leakage audit
    audit = anti_leakage_audit(tr, te, y, folds, vb, roles, ids, proof)

    # 2) fold-0-safe difficulty + weights
    d_fit, prov = rebuild_difficulty(vb, y, folds, FIT)
    w_fit, winfo = weights_from_score(d_fit, lam=LAM, lo=W_LO, hi=W_HI)
    assert_weight_valid(w_fit, lo=W_LO, hi=W_HI)
    # reproducibility: same input -> identical weights
    w_fit2, _ = weights_from_score(d_fit, lam=LAM, lo=W_LO, hi=W_HI)
    assert np.array_equal(w_fit, w_fit2), "weight not reproducible"
    w = np.full(n, np.nan, dtype=np.float64)
    w[FIT] = w_fit
    log(f"difficulty rebuilt; weight mean={winfo['mean']:.6f} "
        f"range=[{winfo['min']:.4f},{winfo['max']:.4f}]")

    # 3) control + treatment members (identical except weights)
    Xf, Xa, names = vb.assemble(FIT, y, EVAL, None, inner_seed=MEMBER_SEED)
    Xv = Xa["val"]
    assert Xf.shape[0] == len(FIT) and Xv.shape[0] == len(EVAL)
    trn, es = _inner_es_split(FIT, y, INNER_ES_SEED)
    assert_no_fold0(trn, folds, "member trn")
    assert_no_fold0(es, folds, "member es")
    assert len(np.intersect1d(trn, es)) == 0
    tl, el = np.searchsorted(FIT, trn), np.searchsorted(FIT, es)
    Xtr, Xes = Xf[tl], Xf[el]

    w_trn = w[trn]
    assert_weight_valid(w_trn, lo=W_LO, hi=W_HI, tol=1e-6) if False else None
    assert np.isfinite(w_trn).all() and (w_trn > 0).all(), "training weights invalid"
    ones_trn = np.ones(len(trn), dtype=np.float64)

    log("training control (uniform) ...")
    ctrl_pred, it_ctrl = fit_lgbm_weighted(Xtr, y[trn], Xv, MEMBER_PARAMS, MEMBER_SEED,
                                           Xes, y[es], sample_weight=ones_trn)
    log("training treatment (difficulty weights) ...")
    treat_pred, it_treat = fit_lgbm_weighted(Xtr, y[trn], Xv, MEMBER_PARAMS, MEMBER_SEED,
                                             Xes, y[es], sample_weight=w_trn)
    assert ctrl_pred.shape == (len(EVAL),) and treat_pred.shape == (len(EVAL),)
    assert np.isfinite(ctrl_pred).all() and np.isfinite(treat_pred).all()
    del Xf, Xtr, Xes
    import gc
    gc.collect()

    # 4) champion authorities. The frozen hashes are over the FULL OOF arrays
    #    (all 5 folds); load + assert them, then extract fold 0 for the eval.
    route_full = np.full(n, np.nan, dtype=np.float64)
    port_full = np.full(n, np.nan, dtype=np.float64)
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        route_full[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "prediction.npy")
        port_full[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "portfolio.npy")
    assert np.isfinite(route_full).all() and np.isfinite(port_full).all()
    assert arr_sha256(port_full.astype("float32")) == CHAMPION_PORTFOLIO_SHA256, "portfolio hash drift"
    assert arr_sha256(route_full.astype("float32")) == CHAMPION_ROUTE_SHA256, "route hash drift"
    route0, port0 = route_full[EVAL], port_full[EVAL]
    # fold-0 aux10 mean logits (fold-0-safe: classical(0) trains on folds 1-4)
    aux0_logits, _ = classical(0, "sol_clean_aux10_primary", roles, ids, folds, y,
                               proof["data_sha256"])
    # champion identity: composition bit-exact
    recomposed = (1.0 / (1.0 + np.exp(-(logit(route0) + aux0_logits) / 2.0))).astype("float32")
    assert np.array_equal(recomposed, port0.astype("float32")), "champion composition drift"
    aux0_prob = 1.0 / (1.0 + np.exp(-aux0_logits))

    # 5) exact standalone metrics
    y0 = y[EVAL]
    auc_ctrl = float(roc_auc_score(y0, ctrl_pred))
    auc_treat = float(roc_auc_score(y0, treat_pred))
    delta = auc_treat - auc_ctrl
    auc_route = float(roc_auc_score(y0, route0))
    auc_base = float(roc_auc_score(y0, port0))
    log(f"AUC control={auc_ctrl:.10f} treatment={auc_treat:.10f} delta={delta:.3e}")

    # exact shared-fail total (champion fold-0 route vs aux)
    c0 = two_class_counts(y0, rankdata(route0), rankdata(aux0_prob))
    shared_exact = {"shared_ge": int(c0["shared_ge"]), "shared_strict": int(c0["shared_strict"]),
                    "route_strict": int(c0["shared_strict"] + c0["route_only"]),
                    "aux_strict": int(c0["shared_strict"] + c0["aux_only"]),
                    "route_only": int(c0["route_only"]), "aux_only": int(c0["aux_only"]),
                    "pairs": int((y0 == 1).sum() * (y0 == 0).sum())}

    # 6) frozen pair sampling (mechanism: shared vs elsewhere, treat vs control)
    log(f"sampling {SAMPLE_N:,} fold-0 pairs (seed {SAMPLE_SEED}) ...")
    pairs = sample_pair_stats(y0, route0, aux0_prob, treat_pred, ctrl_pred,
                              n_pairs=SAMPLE_N, seed=SAMPLE_SEED)

    # 7) correlations with both champion halves
    corr = {}
    for nm, mp in (("control", ctrl_pred), ("treatment", treat_pred)):
        corr[nm] = {
            "pearson_route": pearson(mp, route0), "spearman_route": spearman(mp, route0),
            "pearson_aux": pearson(mp, aux0_prob), "spearman_aux": spearman(mp, aux0_prob),
        }

    # 8) frozen portfolio: equal-weight 11th aux member
    ctrl_logit, treat_logit = logit(ctrl_pred), logit(treat_pred)
    aux11_ctrl = (10.0 * aux0_logits + ctrl_logit) / 11.0
    aux11_treat = (10.0 * aux0_logits + treat_logit) / 11.0
    port_ctrl = 1.0 / (1.0 + np.exp(-(logit(route0) + aux11_ctrl) / 2.0))
    port_treat = 1.0 / (1.0 + np.exp(-(logit(route0) + aux11_treat) / 2.0))
    auc_port_base = float(roc_auc_score(y0, port0))
    auc_port_ctrl = float(roc_auc_score(y0, port_ctrl))
    auc_port_treat = float(roc_auc_score(y0, port_treat))
    port_delta = auc_port_treat - auc_port_ctrl

    # 9) decision (frozen rules)
    s_rate = float(pairs["shared"]["net_rate"])
    s_se = float(pairs["shared"]["net_rate_se"])
    verdict, decision = decide(delta, s_rate, s_se, port_delta)
    log(f"shared net_rate={s_rate:.3e} se={s_se:.3e} -> {verdict}")

    # ---- save artifacts ----
    np.save(ART / "control_pred.npy", ctrl_pred.astype("float32"))
    np.save(ART / "treatment_pred.npy", treat_pred.astype("float32"))
    np.save(ART / "fold0_ids.npy", ids[EVAL])
    np.save(ART / "fit_difficulty_score.npy", d_fit.astype("float32"))
    np.save(ART / "fit_weights.npy", w_fit.astype("float64"))

    elapsed = time.time() - t0
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H19-A",
        "title": "difficulty-targeted reweighted LGBM member -- fold-0 falsifier",
        "git": git_commit(),
        "start_sha": "b5b83a4a05338228c86c021071a84e08fab5d7dc",
        "prereg": "configs/phase19/h19a_prereg.json",
        "prereg_sha256": prereg_sha,
        "prereg_committed_before_results": True,
        "budget": {"gpu_jobs_new": 0, "cpu_seconds": elapsed,
                   "submissions_used": 0, "no_folds_1_4": True},
        "scope": {"eval_fold": 0, "fit_folds": [1, 2, 3, 4], "scheme": "primary"},
        "anti_leakage_audit": audit,
        "difficulty_reconstruction": prov,
        "weights": {**winfo, "applied_to": "member training rows only",
                    "control_weight": "all-ones (uniform)"},
        "members": {
            "control": {"best_iteration": it_ctrl, "auc": auc_ctrl},
            "treatment": {"best_iteration": it_treat, "auc": auc_treat},
            "identical_except": "sample_weight (uniform vs difficulty)",
            "params": MEMBER_PARAMS, "seed": MEMBER_SEED, "inner_es_seed": INNER_ES_SEED,
        },
        "standalone": {
            "control_auc": auc_ctrl, "treatment_auc": auc_treat, "delta": delta,
            "route_auc": auc_route, "champion_portfolio_auc": auc_base,
        },
        "shared_fail_exact": shared_exact,
        "pair_sampling": pairs,
        "correlations": corr,
        "portfolio": {
            "composition": "equal-weight 11th aux member: aux11=(10*aux10+member_logit)/11",
            "base_auc": auc_port_base, "plus_control_auc": auc_port_ctrl,
            "plus_treatment_auc": auc_port_treat, "delta_treat_vs_ctrl": port_delta,
        },
        "decision": decision,
        "verdict": verdict,
        "verdict_vocabulary": ["DEAD", "INCONCLUSIVE", "PATH_TO_REPLICATION"],
        "folds_1_4_recommendation": (
            "PATH: consider folds 1-4 for FUTURE replication (NOT launched here)" if verdict == "PATH_TO_REPLICATION"
            else "no folds 1-4 replication recommended"
        ),
        "artifacts": {
            "control_pred": arr_sha256(ctrl_pred.astype("float32")),
            "treatment_pred": arr_sha256(treat_pred.astype("float32")),
            "fold0_ids": arr_sha256(ids[EVAL]),
            "fit_weights": arr_sha256(w_fit.astype("float64")),
        },
        "elapsed_seconds": elapsed,
    }
    save_json(result, OUT / "h19a_fold0_falsifier.json")
    write_markdown(result, OUT / "h19a_fold0_falsifier.md")
    log(f"done in {elapsed:.1f}s  verdict={verdict}")
    return 0


def write_markdown(r: dict, path: Path) -> None:
    s, pf, ps, dec = r["standalone"], r["portfolio"], r["pair_sampling"], r["decision"]
    sh = ps["shared"]
    lines = [
        "# H19-A -- fold-0 falsifier: difficulty-targeted reweighted LightGBM",
        "",
        f"- **verdict:** `{r['verdict']}`",
        f"- start SHA `{r['start_sha'][:12]}` -> run SHA `{r['git'][:12]}`",
        f"- prereg `configs/phase19/h19a_prereg.json` sha256 `{r['prereg_sha256'][:16]}...` "
        f"committed **before** any result",
        f"- eval fold 0 only; FIT folds 1-4; scheme `primary`; GPU jobs **0**; submissions **0**",
        "",
        "## Anti-leakage",
        "",
        f"- Atlas diagnostics for folds 1-4 are **fold-0-contaminated** "
        f"(set proof: classical(k) trains on folds != k which contains fold 0; numeric demo: "
        f"fold-1 predictions pearson {r['anti_leakage_audit']['numeric_demo']['pearson_of_fold1_predictions']:.6f} "
        f"when fold 0 is in/out of training).",
        "- NOT used as training weights: `per_row_burden.parquet`, `pred_a`, `pred_b`, stored route/aux.",
        "- difficulty rebuilt **inside FIT only** (inner cross-fit, never fold 0).",
        f"- judge A/B pearson: {r['difficulty_reconstruction']['judge_pearson']:.6f} "
        f"(two fold-0-safe CPU LGBM judges).",
        "",
        "> **blocking diagnostic (faithful target):** " + r["anti_leakage_audit"]["blocking_diagnostic"],
        "",
        "## Standalone (fold 0, exact AUC)",
        "",
        "| member | AUC |",
        "|---|---|",
        f"| control (uniform) | {s['control_auc']:.10f} |",
        f"| treatment (difficulty) | {s['treatment_auc']:.10f} |",
        f"| **delta** | **{s['delta']:.3e}** |",
        f"| champion route | {s['route_auc']:.10f} |",
        f"| champion portfolio | {s['champion_portfolio_auc']:.10f} |",
        "",
        "## Shared-fail pair mechanism (frozen sampling)",
        "",
        f"- sampled {ps['n_pairs']:,} pairs, seed {ps['seed']}; shared rate "
        f"{ps['sampled_shared_rate']:.4f} +- {ps['sampled_shared_se']:.4f}",
        f"- exact shared-fail pairs (fold 0): {r['shared_fail_exact']['shared_ge']:,} "
        f"(strict {r['shared_fail_exact']['shared_strict']:,})",
        f"- on shared-fail pairs: treatment acc {sh['acc_treat']:.4f}+-{sh['acc_treat_se']:.4f} "
        f"vs control acc {sh['acc_ctrl']:.4f}+-{sh['acc_ctrl_se']:.4f}",
        f"- corrected {sh['corrected']:,} / degraded {sh['degraded']:,} / net rate "
        f"{sh['net_rate']:.3e}+-{sh['net_rate_se']:.3e}",
        f"- elsewhere (non-shared): net rate {ps['non_shared']['net_rate']:.3e}"
        f"+-{ps['non_shared']['net_rate_se']:.3e} (new errors introduced)",
        "",
        "## Correlations with champion halves",
        "",
        "| member | pearson route | pearson aux |",
        "|---|---|---|",
        f"| control | {r['correlations']['control']['pearson_route']:.6f} "
        f"| {r['correlations']['control']['pearson_aux']:.6f} |",
        f"| treatment | {r['correlations']['treatment']['pearson_route']:.6f} "
        f"| {r['correlations']['treatment']['pearson_aux']:.6f} |",
        "",
        "## Frozen portfolio (equal-weight 11th aux member)",
        "",
        f"- base (aux10) {pf['base_auc']:.10f}; +control {pf['plus_control_auc']:.10f}; "
        f"+treatment {pf['plus_treatment_auc']:.10f}",
        f"- **delta (treat vs control): {pf['delta_treat_vs_ctrl']:.3e}**",
        "",
        "## Decision (frozen rules)",
        "",
        f"- d = {dec['d_delta']:.3e} (bar {dec['convincing_single_fold_gain']:.0e}); "
        f"s_rate = {dec['s_rate']:.3e} +- {dec['s_se']:.3e} (sig={dec['s_sig']}); "
        f"portfolio delta = {dec['portfolio_delta']:.3e}",
        f"- **{dec['verdict']}** -- {dec['rule']}",
        "",
        f"## Recommendation",
        "",
        f"- {r['folds_1_4_recommendation']}",
        f"- budget: {r['budget']['cpu_seconds']:.0f}s CPU, {r['budget']['gpu_jobs_new']} GPU jobs, "
        f"{r['budget']['submissions_used']} submissions.",
        "",
        "> fold 0 alone cannot validate the full promotion gates "
        "(>=1.5e-5 mean paired, >=2.5x paired SE, >=4/5 folds).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
