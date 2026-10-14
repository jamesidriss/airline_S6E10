"""H18-B wave 1 — additive boosted-stump member (EBM/GAM stand-in) vs champion v7.

Pre-registered in configs/phase18/h18b_stump_prereg.json (read and hash-checked
at startup). Primary fold 0, CPU only. The certified pipeline is replayed
verbatim (full view, aux EV features, inner ES split seed=1); the ONLY new
element is ONE additional member: LightGBM with num_leaves=2/max_depth=1 so
every tree is a single split — the ensemble is a pure per-feature step-function
sum (Friedman boosted stumps), i.e. an additive model with no interactions.

Fidelity gates before fitting: rebuilt features hash-equal the certified
contracts, library versions equal, EV gap <= 1e-10 (inside add_expected_values).
Control = certified aux10 chain via classical(); champion identity
compose(route2_f0, aux_ctrl) == certified portfolio.npy asserted exact.

Aggregation (predeclared): aux11 = (10 * aux_logits_ctrl + logit(p_stump)) / 11.
Primary metric: portfolio fold-0 delta with exact pair-credit identity.
Advance bar: >= +3.0e-5, else close at wave 1.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.models.resource_guard import inference_guard  # noqa: E402
from scripts.run_views import _inner_es_split, _fit_lgbm_es  # noqa: E402
from scripts.replay_sol_classical import frozen_roles, add_expected_values  # noqa: E402
from scripts.evaluate_sol_foundation import classical  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402
from scripts.phase16_pair_metrics import pair_rescue_damage  # noqa: E402
from scripts.score_sol_foundation import correlations  # noqa: E402

PREREG = ROOT / "configs" / "phase18" / "h18b_stump_prereg.json"
ADVANCE_BAR = 3.0e-5
FOLD = 0
SEED = 1
STUMP_OVERRIDES = {
    "num_leaves": 2, "max_depth": 1, "learning_rate": 0.05,
    "colsample_bytree": 1.0, "subsample": 1.0, "subsample_freq": 0,
    "min_child_samples": 40, "max_bin": 255, "n_estimators": 6000,
}


def main() -> int:
    out_path = ROOT / "reports" / "phase18" / "h18b_f0.json"
    if out_path.exists():
        raise FileExistsError("Preserve measured result")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["hypothesis_id"] == "H18-B" and prereg["status"] == "PREREGISTERED_BEFORE_ANY_FIT"
    assert prereg["model_spec"]["overrides"] == STUMP_OVERRIDES, "overrides drifted from prereg"
    assert prereg["model_spec"]["seed"] == SEED

    from src.common import load_cached_parquet  # noqa: E402

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    roles = [r for r in frozen_roles() if r["aux_arm"]]
    assert len(roles) == 10
    folds = get_scheme("primary", y, ids).folds
    fi, va = np.flatnonzero(folds != FOLD), np.flatnonzero(folds == FOLD)
    itr, es = _inner_es_split(fi, y, SEED)
    tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
    assert np.array_equal(fi[tl], itr) and np.array_equal(fi[el], es)

    # ---- certified pipeline replay: view + auxiliary EV -------------------------
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    xf, xa, names = vb.assemble(fi, y, va, None, inner_seed=0)
    xv = xa["val"]
    del vb, xa
    gc.collect()
    xf, xv, aux_proof = add_expected_values(xf, xv, names, ids[fi], ids[va],
                                            arr_sha256(folds), FOLD, "primary")
    names = list(names) + [f"aux_ev_{j}" for j in range(13)]
    assert aux_proof["independent_ev_max_gap"] is not None and aux_proof["independent_ev_max_gap"] <= 1e-10

    certified_reports = {}
    for role in roles:
        r = json.loads((REPORTS / "sol_clean_aux10_primary" / f"{role['member']}_f{FOLD}.json")
                       .read_text(encoding="utf-8"))
        c = r["contract"]
        assert c["feature_fit_sha256"] == arr_sha256(xf), f"{role['member']}: rebuilt fit features drift"
        assert c["feature_val_sha256"] == arr_sha256(xv), f"{role['member']}: rebuilt val features drift"
        assert c["feature_names"] == names, f"{role['member']}: feature names drift"
        libs = {p: version(p) for p in ("numpy", "lightgbm", "xgboost", "catboost", "scikit-learn")}
        assert c["libraries"] == libs, f"{role['member']}: library drift {libs}"
        certified_reports[role["member"]] = r

    contract = {
        "hypothesis_id": "H18-B", "fold": FOLD, "git": git_commit(),
        "prereg_sha256": file_sha256(PREREG),
        "feature_fit_sha256": arr_sha256(xf), "feature_val_sha256": arr_sha256(xv),
        "feature_names": names,
        "fit_ids_sha256": arr_sha256(ids[fi]), "val_ids_sha256": arr_sha256(ids[va]),
        "fold_sha256": arr_sha256(folds),
        "model": "lightgbm boosted stumps (additive)",
        "overrides": STUMP_OVERRIDES, "seed": SEED,
        "aux_ev_proof": aux_proof,
        "certified_reports_sha256": {m: file_sha256(REPORTS / "sol_clean_aux10_primary" / f"{m}_f{FOLD}.json")
                                     for m in certified_reports},
    }
    work = ARTIFACTS / "phase18" / "h18b_f0"
    work.mkdir(parents=True, exist_ok=True)
    save_json({**contract, "status": "FIDELITY_GATES_PASSED_FITTING"}, work / "progress.json")

    # ---- additive stump member --------------------------------------------------
    t0 = time.monotonic()
    with inference_guard(work, contract, max_seconds=3000, min_available_gib=4):
        p_stump, best = _fit_lgbm_es(xf[tl], y[itr], xv, STUMP_OVERRIDES, SEED,
                                     xf[el], y[es])
    p_stump = np.asarray(p_stump, dtype="float32")
    fit_seconds = time.monotonic() - t0
    assert p_stump.shape == (len(va),) and np.isfinite(p_stump).all()
    assert ((p_stump >= 0) & (p_stump <= 1)).all()
    np.save(work / "prediction.npy", p_stump)
    labels = y[va]

    # ---- control chain + champion identity --------------------------------------
    _, _, _, _, vectors, _, proof = bank()
    aux_ctrl, aux_proofs = classical(FOLD, "sol_clean_aux10_primary", roles, ids, folds, y,
                                     proof["data_sha256"])
    compose = lambda a, al: (1 / (1 + np.exp(-(logit(a) + al) / 2))).astype("float32")
    assert np.array_equal((1 / (1 + np.exp(-aux_ctrl))).astype("float32"),
                          vectors["aux10"][va]), "aux10 chain drift"
    route2 = np.load(ARTIFACTS / "phase17_B1" / f"f{FOLD}" / "prediction.npy")
    champion = np.load(ARTIFACTS / "phase17_B1" / f"f{FOLD}" / "portfolio.npy")
    assert np.array_equal(compose(route2, aux_ctrl), champion), "champion identity drift"

    # ---- predeclared append aggregation: aux11 = (10*ctrl + stump) / 11 ---------
    aux11 = (aux_ctrl * 10.0 + logit(p_stump)) / 11.0
    portfolio_new = compose(route2, aux11)
    labels = y[va]
    auc_port_new = float(roc_auc_score(labels, portfolio_new))
    auc_port_champ = float(roc_auc_score(labels, champion))
    delta = auc_port_new - auc_port_champ
    pairs = pair_rescue_damage(labels, portfolio_new, champion)
    assert abs(pairs["net_auc_gain"] - delta) < 1e-14

    # ---- secondary: standalone + diversity evidence -----------------------------
    auc_stump = float(roc_auc_score(labels, p_stump))
    corr_champ = correlations(p_stump, champion, labels)
    corr_aux = correlations(p_stump, vectors["aux10"][va], labels)
    corr_route = correlations(p_stump, route2, labels)
    auc_aux11 = float(roc_auc_score(labels, (1 / (1 + np.exp(-aux11))).astype("float32")))
    auc_aux_ctrl = float(roc_auc_score(labels, vectors["aux10"][va]))

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H18-B", "fold": FOLD,
        "prereg_sha256": file_sha256(PREREG),
        "git": git_commit(),
        "contract": contract,
        "stump_fit_seconds": fit_seconds,
        "stump_best_iteration": int(best),
        "stump_auc": auc_stump,
        "stump_correlation": {"champion_portfolio": corr_champ, "aux10": corr_aux,
                              "route2": corr_route},
        "aux10_auc_control": auc_aux_ctrl, "aux11_auc": auc_aux11,
        "aux10_delta_via_append": auc_aux11 - auc_aux_ctrl,
        "portfolio_auc_control": auc_port_champ, "portfolio_auc_treatment": auc_port_new,
        "portfolio_delta": delta,
        "exact_pair_rescue_damage": pairs,
        "advance_bar": ADVANCE_BAR,
        "advance_pass": bool(delta >= ADVANCE_BAR),
        "diversity_clause_note": ("low correlation recorded as champion-B evidence only; "
                                  "does not extend the wave or promote anything"),
        "status": "COMPLETE",
        "verdict": ("H18B_WAVE1_PASS_EXTEND_TO_FOLDS_1_4" if delta >= ADVANCE_BAR
                    else "H18B_WAVE1_FAIL_CLOSE_NO_FURTHER_FITS"),
    }
    save_json(result, out_path)
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "exp_id": "phase18_h18b_f0", "kind": "PHASE18_H18B_WAVE1",
            "ts": result["utc"], "report": str(out_path.relative_to(ROOT)),
            "report_sha256": file_sha256(out_path),
            "paired_fold_deltas": [delta], "auc": auc_port_new,
            "complete_folds": [FOLD], "full_oof": False,
            "verdict": result["verdict"]}) + "\n")
    print(json.dumps({k: result[k] for k in ("stump_auc", "stump_correlation", "aux10_delta_via_append",
                                             "portfolio_delta", "advance_pass", "verdict")},
                     indent=2))
    return 0


if __name__ == "__main__":
    main()
