"""H19-B: frozen aux10 deployment-policy experiment (single full-FIT vs bag-of-five).

Pseudo-test = primary fold 4; FIT = primary folds 0-3. Control reuses the certified
full-FIT fold-4 aux10 members (no retraining). Treatment trains 10 roles x 5 inner-fold
bag members (~80% of FIT each), aggregates by equal mean-logit, then the complete v7
composition sigmoid((logit(route4)+aux_logits)/2).

Frozen by configs/phase19/h19b_prereg.json (sha256 37f4d7cd...), committed BEFORE any
treatment evaluation. No pseudo-test label enters any fitting function. No v7 mutation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import (ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit,
                        load_cached_parquet, save_json)
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.replay_sol_classical import frozen_roles, resolved_params, PROTOCOL
from scripts.evaluate_sol_foundation import classical
from src.phase19.h19b_bag5 import (INNER_PARTITIONS_SEED, PSEUDOTEST_FOLD, bag_fit_rows,
                                   build_inner_partitions, exact_pair_flips,
                                   fit_ids_manifest, train_bag_member)

PREREG_SHA256 = "37f4d7cd3ad142b20cb3a211df1d6a115287e11b1e9627e4e8f473aa4d82ec9a"
CHAMPION_ROUTE_SHA256 = None  # filled from h19a module at runtime
TAG = "phase19_h19b_bag5"
SOURCES = ('scripts/replay_sol_classical.py', 'scripts/run_views.py', 'scripts/run_phase12.py',
           'scripts/run_phase13.py', 'scripts/run_phase13b.py', 'scripts/run_phase14.py',
           'scripts/native_cat.py', 'scripts/run_sol_a.py', 'scripts/sol_aux_cache.py',
           'src/features/aux_distribution.py', 'src/features/view.py', 'src/features/s6e10.py',
           'src/models/resource_guard.py', 'src/phase19/h19b_bag5.py')


def log(msg):
    print(f"[h19b] {msg}", flush=True)


def load_route_and_portfolio(fold_idx):
    """Certified TabPFN route + champion portfolio on the pseudo-test fold, hash-checked."""
    from src.common import ARTIFACTS as A
    route = np.load(A / "phase17_B1" / f"f{PSEUDOTEST_FOLD}" / "prediction.npy")
    port = np.load(A / "phase17_B1" / f"f{PSEUDOTEST_FOLD}" / "portfolio.npy")
    assert route.shape == port.shape == (len(fold_idx),)
    return route, port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="Train ONE bag member (fold-4-safe) to measure cost; AUC recorded but "
                         "must NOT select roles. Stops before the full 50.")
    ap.add_argument("--max-bags", type=int, default=5, help="bags per role (frozen=5)")
    args = ap.parse_args()

    prereg = Path("configs/phase19/h19b_prereg.json")
    assert file_sha256(prereg) == PREREG_SHA256, "prereg hash drift; protocol not frozen"
    log(f"prereg hash verified {PREREG_SHA256[:12]}")

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    fit_idx = np.flatnonzero(folds != PSEUDOTEST_FOLD)
    val_idx = np.flatnonzero(folds == PSEUDOTEST_FOLD)
    fold_sha = arr_sha256(folds)
    log(f"rows={len(y)} FIT={len(fit_idx)} (folds 0-3) PSEUDO_TEST={len(val_idx)} (fold 4)")

    roles = [r for r in frozen_roles() if r["aux_arm"]]
    assert len(roles) == 10
    data_hash = {s: file_sha256(f"data/raw/{s}.csv") for s in ("train", "test")}
    sources = {p: file_sha256(p) for p in SOURCES}

    # ---- control: reuse certified full-FIT fold-4 members; verify reproduction --------
    t0 = time.monotonic()
    aux_ctrl_logits, _ = classical(PSEUDOTEST_FOLD, "sol_clean_aux10_primary", roles,
                                   ids, folds, y, data_hash)
    # prove the control is a true single full-FIT refit: each member trained on folds!=4,
    # which for the primary scheme is exactly FIT={0,1,2,3}; bag members must be subsets.
    fit_pop_sizes = {}
    for role in roles:
        rp = REPORTS / "sol_clean_aux10_primary" / f"{role['member']}_f{PSEUDOTEST_FOLD}.json"
        c = json.loads(rp.read_text(encoding="utf-8"))["contract"]
        fit_pop_sizes[role["member"]] = (c["train_rows"] + c["es_rows"], c["fold_sha256"])
        assert c["fold_sha256"] == fold_sha, f"{role['member']} not on the primary fold scheme"
        assert c["train_rows"] + c["es_rows"] == len(fit_idx), \
            f"{role['member']} control fit population != FIT size"
    route4, port4 = load_route_and_portfolio(val_idx)
    recomposed = (1.0 / (1.0 + np.exp(-(logit(route4) + aux_ctrl_logits) / 2.0))).astype("float32")
    identity_gap = float(np.abs(recomposed - port4).max())
    assert identity_gap == 0.0, f"control portfolio not bit-exact: gap {identity_gap}"
    auc_ctrl_aux = float(roc_auc_score(y[val_idx], 1 / (1 + np.exp(-aux_ctrl_logits))))
    auc_ctrl_port = float(roc_auc_score(y[val_idx], port4))
    log(f"control verified bit-exact (gap {identity_gap}); full-FIT={len(fit_idx)} rows; "
        f"aux AUC={auc_ctrl_aux:.10f} portfolio AUC={auc_ctrl_port:.10f} ({time.monotonic()-t0:.1f}s)")

    # ---- treatment inner partitions -------------------------------------------------
    parts = build_inner_partitions(fit_idx, y, n_splits=args.max_bags, seed=INNER_PARTITIONS_SEED)
    manifest = fit_ids_manifest(fit_idx, parts)
    manifest["pseudotest_excluded"] = bool(
        not any(np.intersect1d(p, val_idx).size for p in parts))
    assert manifest["pseudotest_excluded"], "pseudo-test rows leaked into an inner partition"

    root = ARTIFACTS / TAG
    reports = REPORTS / TAG
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    save_json(manifest, reports / "bag_fit_manifest.json")

    if args.pilot:
        # cost measurement only: first role, first bag. AUC is logged but not used to select.
        role = roles[0]
        params = resolved_params(role)
        bf = bag_fit_rows(fit_idx, parts, 0)
        log(f"PILOT role={role['member']} bag0 fit_rows={len(bf)} (cost measurement only)")
        t1 = time.monotonic()
        _, rec = train_bag_member(role, params, bf, val_idx, y, ids, tr, te, 0,
                                  root, reports, sources, data_hash, fold_sha, 0)
        elapsed = time.monotonic() - t1
        log(f"PILOT done in {elapsed:.1f}s n_trees={rec['n_trees']} "
            f"(AUC logged={rec['auc']:.6f}, not used for selection)")
        save_json({"status": "PILOT_COMPLETE_COST_ONLY", "role": role["member"],
                   "elapsed_seconds": elapsed, "n_trees": rec["n_trees"],
                   "fit_rows": len(bf), "git": git_commit(),
                   "note": "single-member cost probe; AUC must not select roles"},
                  reports / "pilot_cost.json")
        return 0

    # ---- full 10 roles x 5 bag run with atomic per-member checkpoints ----------------
    aux_per_role = {}
    member_seconds = {}
    for r_i, role in enumerate(roles):
        params = resolved_params(role)
        role_logits = []
        for b in range(args.max_bags):
            stem = f"{role['member']}_bag{b}"
            rp = reports / (stem + ".json")
            pp = root / (stem + ".npy")
            if rp.exists() and pp.exists():
                rec = json.loads(rp.read_text(encoding="utf-8"))
                pred = np.load(pp)
                assert arr_sha256(pred) == rec["prediction_sha256"], f"resume hash {stem}"
                log(f"  resume {stem}")
            else:
                bf = bag_fit_rows(fit_idx, parts, b)
                pred, rec = train_bag_member(role, params, bf, val_idx, y, ids, tr, te, 0,
                                             root, reports, sources, data_hash, fold_sha, b)
            role_logits.append(np.clip(pred.astype("float64"), 1e-7, 1 - 1e-7))
            member_seconds[stem] = rec["seconds"]
        # store the raw probability members for aggregation below
        aux_per_role[role["member"]] = role_logits
        log(f"role {r_i+1}/10 {role['member']} done")

    # aggregate: mean-logit over 5 members per role, then mean-logit over 10 roles
    role_agg = {m: mean_logit_from_logprobs(lst) for m, lst in aux_per_role.items()}
    aux_bag_logits = np.mean(np.column_stack(list(role_agg.values())), axis=1)
    port_treat = (1.0 / (1.0 + np.exp(-(logit(route4) + aux_bag_logits) / 2.0))).astype("float32")
    auc_treat_aux = float(roc_auc_score(y[val_idx], 1 / (1 + np.exp(-aux_bag_logits))))
    auc_treat_port = float(roc_auc_score(y[val_idx], port_treat))
    log(f"treatment aux AUC={auc_treat_aux:.10f} portfolio AUC={auc_treat_port:.10f}")

    save_json({"aux_ctrl_logits_sha256": arr_sha256(aux_ctrl_logits),
               "aux_bag_logits_sha256": arr_sha256(aux_bag_logits),
               "port_ctrl_sha256": arr_sha256(port4),
               "port_treat_sha256": arr_sha256(port_treat),
               "member_seconds": member_seconds, "git": git_commit()},
              reports / "predictions_index.json")
    np.save(root / "aux_ctrl_logits.npy", aux_ctrl_logits)
    np.save(root / "aux_bag_logits.npy", aux_bag_logits)
    np.save(root / "port_ctrl.npy", port4)
    np.save(root / "port_treat.npy", port_treat)
    np.save(root / "pseudotest_ids.npy", ids[val_idx])
    log("full run complete; see reports for evaluation")
    return 0


def mean_logit_from_logprobs(logits_list):
    """Equal mean over members of already-logit vectors (no re-logit)."""
    return np.mean(np.column_stack(logits_list), axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
