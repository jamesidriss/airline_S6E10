"""H18-C wave 1 — ratings-tuple-key fold-safe TE appended to the aux10 members.

Pre-registered in configs/phase18/h18c_keyte_prereg.json (read and hash-checked
at startup). Primary fold 0, all 10 aux members, CPU only. The certified
pipeline is replayed verbatim from scripts/replay_sol_classical.py (view build,
aux EV features, inner ES split seed=1, inner TE seed=0, per-family fitting
branch); the ONLY change is 4 appended FoldSafeTE columns for the 17-column
rating-tuple key. Fidelity gates before any treatment fit:
  1. rebuilt features hash-equal the certified feature_fit/val_sha256;
  2. library versions equal the certified contracts;
  3. auxiliary EV gap <= 1e-10 (inside add_expected_values);
  4. control replay of xt_xt_d127_ss06 without TE is bit-identical to its
     certified prediction (exact-refit determinism).
Primary metric: portfolio fold-0 delta vs champion v7 (compose(route2_fold0,
aux_logits_new) vs certified portfolio.npy), exact identity asserted.
Advance bar: >= +3.0e-5 portfolio delta, else close at wave 1.
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
from src.features.s6e10 import FoldSafeTE  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.models.resource_guard import inference_guard, ResourcePreflightError  # noqa: E402
from scripts.run_views import _inner_es_split, _fit_cat_es  # noqa: E402
from scripts.run_phase13 import RATINGS, fit_slot  # noqa: E402
from scripts.run_phase14 import fit_xgb, fit_cat  # noqa: E402
from scripts.native_cat import attach, cat_frame, default_cat_cols  # noqa: E402
from scripts.replay_sol_classical import frozen_roles, resolved_params  # noqa: E402
from scripts.evaluate_sol_foundation import classical  # noqa: E402

PREREG = ROOT / "configs" / "phase18" / "h18c_keyte_prereg.json"
ADVANCE_BAR = 3.0e-5
FOLD = 0
META4 = ["Gender", "Customer Type", "Type of Travel", "Class"]
TE_MEMBER_FIDELITY = "xt_xt_d127_ss06"


def main() -> int:
    out_path = ROOT / "reports" / "phase18" / "h18c_f0.json"
    if out_path.exists():
        raise FileExistsError("Preserve measured result")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["hypothesis_id"] == "H18-C" and prereg["status"] == "PREREGISTERED_BEFORE_ANY_FIT"

    from src.common import load_cached_parquet  # noqa: E402

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    roles = [r for r in frozen_roles() if r["aux_arm"]]
    assert len(roles) == 10 and all(r["scheme"] == "primary" and r["aux_arm"] for r in roles)
    folds = get_scheme("primary", y, ids).folds
    fi, va = np.flatnonzero(folds != FOLD), np.flatnonzero(folds == FOLD)
    itr, es = _inner_es_split(fi, y, 1)  # aux es_seed_policy == '1'
    tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
    assert np.array_equal(fi[tl], itr) and np.array_equal(fi[el], es)
    inner = 0  # te_inner_seed_policy 'fold' -> k

    # ---- certified pipeline replay: view + auxiliary EV -------------------------
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    xf, xa, names = vb.assemble(fi, y, va, None, inner_seed=inner)
    xv = xa["val"]
    del vb, xa
    gc.collect()
    from scripts.replay_sol_classical import add_expected_values  # noqa: E402

    xf, xv, aux_proof = add_expected_values(xf, xv, names, ids[fi], ids[va],
                                            arr_sha256(folds), FOLD, "primary")
    names = list(names) + [f"aux_ev_{j}" for j in range(13)]
    assert aux_proof["independent_ev_max_gap"] is not None and aux_proof["independent_ev_max_gap"] <= 1e-10

    certified = {}
    for role in roles:
        r = json.loads((REPORTS / "sol_clean_aux10_primary" / f"{role['member']}_f{FOLD}.json")
                       .read_text(encoding="utf-8"))
        c = r["contract"]
        assert c["feature_fit_sha256"] == arr_sha256(xf), f"{role['member']}: rebuilt fit features drift"
        assert c["feature_val_sha256"] == arr_sha256(xv), f"{role['member']}: rebuilt val features drift"
        assert c["feature_names"] == names, f"{role['member']}: feature names drift"
        libs = {p: version(p) for p in ("numpy", "lightgbm", "xgboost", "catboost", "scikit-learn")}
        assert c["libraries"] == libs, f"{role['member']}: library drift {libs}"
        p_ctrl = np.load(ARTIFACTS / "sol_clean_aux10_primary" / f"{role['member']}_f{FOLD}.npy")
        assert arr_sha256(p_ctrl) == r["prediction_sha256"]
        ctrl_ids = np.load(ARTIFACTS / "sol_clean_aux10_primary" / f"{role['member']}_f{FOLD}_ids.npy")
        assert np.array_equal(ctrl_ids, ids[va])
        certified[role["member"]] = (p_ctrl, r)

    # ---- key17 TE features (fold-safe) -----------------------------------------
    key_cols = list(RATINGS) + META4
    key = tr[key_cols].astype(str).agg("|".join, axis=1)
    assert key.nunique() > 1000
    te_obj = FoldSafeTE(keys={}, smooths=(10.0, 20.0, 100.0), inner_folds=5, inner_seed=0)
    k_fit = key.iloc[fi].reset_index(drop=True)
    k_val = key.iloc[va].reset_index(drop=True)
    xf_key, nm_fit = te_obj.crossfit({"key17": k_fit}, y[fi])
    xv_key, nm_val = te_obj.apply({"key17": k_fit}, y[fi], {"key17": k_val})
    assert xf_key.shape == (len(fi), 4) and xv_key.shape == (len(va), 4)
    assert np.isfinite(xf_key).all() and np.isfinite(xv_key).all()
    names2 = names + ["te_key17_s10", "te_key17_s20", "te_key17_s100", "te_key17_cnt"]
    xf2 = np.column_stack([xf, xf_key]).astype("float32")
    xv2 = np.column_stack([xv, xv_key]).astype("float32")
    del xf_key, xv_key
    gc.collect()

    work = ARTIFACTS / "phase18" / "h18c_f0"
    work.mkdir(parents=True, exist_ok=True)
    contract = {
        "hypothesis_id": "H18-C", "fold": FOLD, "git": git_commit(),
        "prereg_sha256": file_sha256(PREREG),
        "feature_fit_sha256": arr_sha256(xf2), "feature_val_sha256": arr_sha256(xv2),
        "feature_names": names2, "key_columns": key_cols,
        "key_unique_fit": int(k_fit.nunique()), "key_dup_rate_fit": float(1 - k_fit.nunique() / len(k_fit)),
        "fit_ids_sha256": arr_sha256(ids[fi]), "val_ids_sha256": arr_sha256(ids[va]),
        "fold_sha256": arr_sha256(folds),
        "aux_ev_proof": aux_proof,
        "certified_reports_sha256": {m: file_sha256(REPORTS / "sol_clean_aux10_primary" / f"{m}_f{FOLD}.json")
                                     for m in certified},
    }
    save_json({**contract, "status": "FIDELITY_GATES_PASSED_FITTING"}, work / "progress.json")

    # ---- fidelity control replay: one member, no TE ----------------------------
    role0 = roles[0]
    assert role0["member"] == TE_MEMBER_FIDELITY
    pred_ctrl, best_ctrl = fit_slot(xf[tl], y[itr], xv, role0["seed"], role0["overrides"],
                                    xf[el], y[es])
    pred_ctrl = np.asarray(pred_ctrl, dtype="float32")
    ctrl_report = certified[role0["member"]][1]
    assert arr_sha256(pred_ctrl) == ctrl_report["prediction_sha256"], \
        "Control replay is not bit-identical to the certified prediction; environment drifted"
    assert int(best_ctrl) == ctrl_report["n_trees"]
    del pred_ctrl
    gc.collect()

    # ---- treatment: all 10 members with the 4 appended TE columns ---------------
    results, start = {}, time.monotonic()
    for role in roles:
        member = role["member"]
        params = resolved_params(role)
        mroot = work / member
        mroot.mkdir(exist_ok=True)
        contract_m = {**contract, "role": role, "params": params}
        if (mroot / "prediction.npy").exists():
            raise FileExistsError(f"Preserve existing treatment fit for {member}")

        def fit_member(role=role, mroot=mroot):
            if role["aux_arm"] == "C1":
                cats = default_cat_cols(True)
                f, cn = attach(xf2[tl], names2, cat_frame(tr, cats, itr))
                e, _ = attach(xf2[el], names2, cat_frame(tr, cats, es))
                v, _ = attach(xv2, names2, cat_frame(tr, cats, va))
                p, b = fit_cat(f, y[itr], v, role["seed"], role["overrides"], cn, e, y[es])
                del f, e, v
                return p, b
            function = fit_slot if role["family"] == "lgbm" else (
                fit_xgb if role["family"] == "xgb" else _fit_cat_es)
            return function(xf2[tl], y[itr], xv2, role["seed"], role["overrides"], xf2[el], y[es])

        t0 = time.monotonic()
        with inference_guard(mroot, contract_m, max_seconds=2700, min_available_gib=4):
            p, best = fit_member()
        p = np.asarray(p, dtype="float32")
        assert p.shape == (len(va),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        np.save(mroot / "prediction.npy", p)
        p_ctrl, r_ctrl = certified[member]
        auc_t = float(roc_auc_score(y[va], p))
        auc_c = float(roc_auc_score(y[va], p_ctrl))
        results[member] = {
            "family": role["family"], "auc_control": auc_c, "auc_treatment": auc_t,
            "delta": auc_t - auc_c, "n_trees": int(best),
            "seconds": time.monotonic() - t0,
            "prediction_sha256": arr_sha256(p),
        }
        print(f"{member}: control {auc_c:.9f} -> treat {auc_t:.9f} "
              f"({auc_t - auc_c:+.3e}) in {results[member]['seconds']:.0f}s", flush=True)
        del p
        gc.collect()

    # ---- operational assembly ---------------------------------------------------
    aux_logits_new = np.mean([logit(np.load(work / m / "prediction.npy")) for m in results], axis=0)
    from scripts.phase16_common import bank  # noqa: E402

    _, _, _, _, vectors, _, proof = bank()
    aux_logits_ctrl2, aux_proofs = classical(FOLD, "sol_clean_aux10_primary", roles, ids, folds, y,
                                             proof["data_sha256"])
    compose = lambda a, al: (1 / (1 + np.exp(-(logit(a) + al) / 2))).astype("float32")
    route2 = np.load(ARTIFACTS / "phase17_B1" / f"f{FOLD}" / "prediction.npy")
    champion = np.load(ARTIFACTS / "phase17_B1" / f"f{FOLD}" / "portfolio.npy")
    assert np.array_equal(compose(route2, aux_logits_ctrl2), champion), "champion identity drift"
    portfolio_new = compose(route2, aux_logits_new)
    labels = y[va]
    auc_port_new = float(roc_auc_score(labels, portfolio_new))
    auc_port_champ = float(roc_auc_score(labels, champion))
    delta = auc_port_new - auc_port_champ
    from scripts.phase16_pair_metrics import pair_rescue_damage  # noqa: E402

    pairs = pair_rescue_damage(labels, portfolio_new, champion)
    assert abs(pairs["net_auc_gain"] - delta) < 1e-14

    aux_auc_new = float(roc_auc_score(labels, (1 / (1 + np.exp(-aux_logits_new))).astype("float32")))
    aux_auc_ctrl = float(roc_auc_score(labels, vectors["aux10"][va]))
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "H18-C", "fold": FOLD,
        "prereg_sha256": file_sha256(PREREG),
        "git": git_commit(),
        "contract": contract,
        "fidelity_control_replay": {
            "member": TE_MEMBER_FIDELITY,
            "certified_prediction_sha256": ctrl_report["prediction_sha256"],
            "bit_identical": True, "n_trees": int(best_ctrl),
        },
        "members": results,
        "aux10_auc_control": aux_auc_ctrl, "aux10_auc_treatment": aux_auc_new,
        "aux10_delta": aux_auc_new - aux_auc_ctrl,
        "portfolio_auc_control": auc_port_champ, "portfolio_auc_treatment": auc_port_new,
        "portfolio_delta": delta,
        "exact_pair_rescue_damage": pairs,
        "advance_bar": ADVANCE_BAR,
        "advance_pass": bool(delta >= ADVANCE_BAR),
        "total_seconds": time.monotonic() - start,
        "status": "COMPLETE",
        "verdict": ("H18C_WAVE1_PASS_EXTEND_TO_FOLDS_1_4" if delta >= ADVANCE_BAR
                    else "H18C_WAVE1_FAIL_CLOSE_NO_FURTHER_FITS"),
    }
    save_json(result, out_path)
    with (ROOT / "experiments" / "ledger.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "exp_id": "phase18_h18c_f0", "kind": "PHASE18_H18C_WAVE1",
            "ts": result["utc"], "report": str(out_path.relative_to(ROOT)),
            "report_sha256": file_sha256(out_path),
            "paired_fold_deltas": [delta], "auc": auc_port_new,
            "complete_folds": [FOLD], "full_oof": False,
            "verdict": result["verdict"]}) + "\n")
    print(json.dumps({k: result[k] for k in ("aux10_delta", "portfolio_delta", "advance_pass",
                                             "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    main()
