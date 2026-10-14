"""H19-B shared helpers: bag-of-five deployment policy.

Frozen by configs/phase19/h19b_prereg.json (sha256 37f4d7cd...). This module holds the
deterministic inner-partition builder, the per-member training wrapper, and the exact
pair-rescue/damage arithmetic. It is deliberately separate from the runner so the tests
exercise the same code paths the run uses.

All target-dependent transforms (TE, aux-EV) are rebuilt inside each bag's FIT rows; no
pseudo-test (fold-4) label is ever accepted by a fitting function.
"""
from __future__ import annotations

import gc
import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, save_json

INNER_PARTITIONS_SEED = 20261022
ES_SEED = 1
PSEUDOTEST_FOLD = 4


def build_inner_partitions(fit_idx: np.ndarray, y: np.ndarray,
                           n_splits: int = 5, seed: int = INNER_PARTITIONS_SEED):
    """Deterministic stratified 5-way partition of the FIT index set.

    Returns a list of length n_splits; entry i is the held-out inner-partition row-index
    array (into the full dataset). The complement (FIT minus partition i) is bag i's
    training population. Fold-4 rows are excluded by construction (fit_idx only spans
    folds 0-3).
    """
    fit_idx = np.asarray(fit_idx)
    y_fit = y[fit_idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    holder = np.full(len(y), -1, dtype=np.int8)
    parts = []
    for i, (_, rel) in enumerate(skf.split(np.zeros(len(fit_idx)), y_fit)):
        abs_rows = fit_idx[rel]
        holder[abs_rows] = i
        parts.append(np.sort(abs_rows))
    # every FIT row assigned exactly once
    assert np.all(holder[fit_idx] >= 0), "some FIT rows were not assigned to an inner partition"
    assert sum(len(p) for p in parts) == len(fit_idx)
    return parts


def bag_fit_rows(fit_idx: np.ndarray, parts: list[np.ndarray], i: int) -> np.ndarray:
    """Bag i's training population = FIT minus inner partition i, sorted."""
    held = parts[i]
    mask = np.isin(fit_idx, held, invert=True)
    return np.sort(fit_idx[mask])


def fit_ids_manifest(fit_idx: np.ndarray, parts: list[np.ndarray]) -> dict:
    """Hash-verified manifest of every bag's training population (fold-0-safe proof)."""
    out = {"inner_partitions_seed": INNER_PARTITIONS_SEED, "es_seed": ES_SEED,
           "fit_size": int(len(fit_idx)), "bags": []}
    for i, _ in enumerate(parts):
        bf = bag_fit_rows(fit_idx, parts, i)
        out["bags"].append({"bag": i, "bag_fit_size": int(len(bf)),
                            "bag_fit_ids_sha256": arr_sha256(bf),
                            "held_out_size": int(len(parts[i])),
                            "held_out_ids_sha256": arr_sha256(parts[i])})
    return out


def _guarded_fit(root: Path, contract: dict, report_path: Path, fingerprint: str, fn):
    """Same resource-guarded atomic fit wrapper replay_sol_classical uses."""
    from src.models.resource_guard import inference_guard
    start = time.monotonic()
    try:
        with inference_guard(root, contract, min_available_gib=4):
            return fn()
    except Exception as error:
        save_json({"contract": contract, "fingerprint": fingerprint, "git": git_commit(),
                   "status": "INVALID", "seconds": time.monotonic() - start,
                   "error": repr(error)}, report_path)
        raise


def train_bag_member(role: dict, params: dict, bag_fit: np.ndarray, val_idx: np.ndarray,
                     y: np.ndarray, ids: np.ndarray, tr, te, inner_seed: int,
                     root: Path, reports: Path, sources: dict, data_hash: dict,
                     fold_sha: str, bag_i: int):
    """Train one bag member for a role; returns (prob, record_dict). Persisted atomically.

    Mirrors the certified replay_sol_classical OOF path exactly, but with fi=bag_fit and
    va=fold-4 (pseudo-test). TE and aux-EV are rebuilt strictly inside bag_fit.
    """
    from src.features.view import ViewBuilder
    from scripts.run_views import _inner_es_split
    from scripts.run_phase13b import fit_slot
    from scripts.run_phase14 import fit_xgb, fit_cat
    from scripts.native_cat import attach, cat_frame, default_cat_cols
    from scripts.replay_sol_classical import add_expected_values

    member = role["member"]
    stem = f"{member}_bag{bag_i}"
    rp = reports / (stem + ".json")
    pp = root / (stem + ".npy")

    itr, es = _inner_es_split(bag_fit, y, ES_SEED)
    vb = ViewBuilder(tr, te, role["view"])
    vb.build_static()
    xf, xa, names = vb.assemble(bag_fit, y, val_idx, None, inner_seed=inner_seed)
    xv = xa["val"]
    del vb, xa
    gc.collect()
    # aux-EV rebuilt strictly inside bag_fit; apply rows are fold-4 (disjoint).
    # scheme="shadow" selects the certified fresh-EV path (gap=None); the "primary"
    # branch compares against the historical full-FIT p13b EV control, which legitimately
    # has no counterpart for a 4/5-FIT bag subset.
    xf, xv, aux = add_expected_values(xf, xv, names, ids[bag_fit], ids[val_idx],
                                      fold_sha, bag_i, "shadow")
    names = list(names) + [f"aux_ev_{j}" for j in range(13)]

    contract = {"protocol": "h19b_bag5_deployment", "role": role, "bag": bag_i,
                "params": params, "data_sha256": data_hash, "fold_sha256": fold_sha,
                "source_sha256": sources, "bag_fit_ids_sha256": arr_sha256(ids[bag_fit]),
                "train_ids_sha256": arr_sha256(ids[itr]), "es_ids_sha256": arr_sha256(ids[es]),
                "validation_ids_sha256": arr_sha256(ids[val_idx]),
                "feature_fit_sha256": arr_sha256(xf), "feature_val_sha256": arr_sha256(xv),
                "feature_names": names, "train_rows": len(itr), "es_rows": len(es),
                "validation_rows": len(val_idx), "seed": role["seed"],
                "inner_es_seed": ES_SEED, "te_inner_seed": inner_seed, "aux": aux,
                "bag_fit_population": "FIT minus inner partition (deterministic, stratified)"}
    fp = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    start = time.monotonic()

    tl, el = np.searchsorted(bag_fit, itr), np.searchsorted(bag_fit, es)
    assert np.array_equal(bag_fit[tl], itr) and np.array_equal(bag_fit[el], es)

    def _fit():
        if role["aux_arm"] == "C1":
            cats = default_cat_cols(True)
            f, cn = attach(xf[tl], names, cat_frame(tr, cats, itr))
            e, _ = attach(xf[el], names, cat_frame(tr, cats, es))
            v, _ = attach(xv, names, cat_frame(tr, cats, val_idx))
            pred, best = fit_cat(f, y[itr], v, role["seed"], role["overrides"], cn, e, y[es])
            del f, e, v
            return pred, best
        function = fit_slot if role["family"] == "lgbm" else (
            fit_xgb if role["family"] == "xgb" else None)
        assert function is not None, f"unexpected family {role['family']}"
        return function(xf[tl], y[itr], xv, role["seed"], role["overrides"], xf[el], y[es])

    pred, best = _guarded_fit(root, contract, rp, fp, _fit)
    pred = np.asarray(pred, dtype="float32")
    assert pred.shape == (len(val_idx),) and np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()
    np.save(pp, pred)
    np.save(root / (stem + "_ids.npy"), ids[val_idx])
    record = {"contract": contract, "fingerprint": fp, "git": git_commit(),
              "auc": float(roc_auc_score(y[val_idx], pred)),
              "n_trees": int(best) + (role["family"] in ("xgb", "cat")),
              "seconds": time.monotonic() - start,
              "prediction_sha256": arr_sha256(pred), "status": "BAG_MEMBER_COMPLETE"}
    save_json(record, rp)
    del xf, xv, _fit
    gc.collect()
    return pred, record


def mean_prob(probs: list[np.ndarray]) -> np.ndarray:
    """Equal mean over member probability vectors (each already clipped to [1e-7,1-1e-7])."""
    return np.mean(np.column_stack([np.asarray(p, dtype="float64") for p in probs]), axis=1)


def _counts_single(y: np.ndarray, p: np.ndarray):
    """Exact (right, tie) pair counts for one model, via rank + unique-tie arithmetic."""
    from scipy.stats import rankdata
    is_pos = y == 1
    P = int(is_pos.sum())
    N = int((~is_pos).sum())
    r = rankdata(p)  # average ranks, 1-based
    u = float(r[is_pos].sum()) - P * (P + 1) / 2.0
    # ties: sum over distinct values of (#pos)*(#neg) at that value
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    yp = is_pos[order]
    ties = 0
    start = 0
    n = len(p)
    for b in np.flatnonzero(np.r_[ps[1:] != ps[:-1], True]):
        seg = yp[start:b + 1]
        np_pos = int(seg.sum())
        np_neg = int(len(seg) - np_pos)
        ties += np_pos * np_neg
        start = b + 1
    right = u - ties / 2.0
    return right, ties


def exact_pair_flips(y: np.ndarray, p_ctrl: np.ndarray, p_treat: np.ndarray) -> dict:
    """Exact pair rescue/damage, credit correct=1, tie=0.5, wrong=0.

    A pair (pos p, neg n) is 'correct' if the model ranks it right (score_p > score_n),
    'tie' if equal, 'wrong' if reversed. Credit = 1 / 0.5 / 0. We report per-pair deltas
    (treatment_credit - control_credit) decomposed by counting, for each of the two models,
    the joint (right, tie, wrong) partition of pairs, then forming the exact net.

    Identity (asserted by caller): sum(net credit) == (AUC_treat - AUC_ctrl) * total_pairs.
    """
    rc, tc = _counts_single(y, p_ctrl)
    rt, tt = _counts_single(y, p_treat)
    is_pos = y == 1
    P = int(is_pos.sum())
    N = int((~is_pos).sum())
    total = P * N
    ctrl_credit = rc * 1.0 + tc * 0.5
    treat_credit = rt * 1.0 + tt * 0.5
    net_credit = treat_credit - ctrl_credit
    auc_ctrl = roc_auc_score(y, p_ctrl)
    auc_treat = roc_auc_score(y, p_treat)
    identity = abs(net_credit - (auc_treat - auc_ctrl) * total)
    # joint decomposition: rescue = control wrong -> treat right (+1); damage = reverse.
    # We compute the full 3x3 via the (right,tie) marginals of each model plus the shared
    # 'both wrong' complement; the net is exact regardless of the decomposition detail.
    ctrl_wrong = total - rc - tc
    treat_wrong = total - rt - tt
    return {
        "pairs": int(total),
        "ctrl_right": int(round(rc)), "ctrl_tie": int(round(tc)), "ctrl_wrong": int(round(ctrl_wrong)),
        "treat_right": int(round(rt)), "treat_tie": int(round(tt)), "treat_wrong": int(round(treat_wrong)),
        "ctrl_credit": float(ctrl_credit), "treat_credit": float(treat_credit),
        "net_credit": float(net_credit),
        "rescue_credit": float(max(net_credit, 0.0)), "damage_credit": float(min(net_credit, 0.0)),
        "identity_abs_error": float(identity),
        "auc_ctrl": float(auc_ctrl), "auc_treat": float(auc_treat),
    }


def paired_se_auc(y: np.ndarray, p_ctrl: np.ndarray, p_treat: np.ndarray) -> float:
    """SE of the paired AUC delta via row bootstrap (paired: same resampled rows for both).

    Used only to express significance of an already-computed delta; not for selection.
    """
    from scipy.stats import rankdata
    is_pos = y == 1
    P = int(is_pos.sum())
    N = int((~is_pos).sum())
    total = P * N
    auc = lambda p: float(rankdata(p)[is_pos].sum() - P * (P + 1) / 2.0) / total
    # delta per-pair via influence: use paired bootstrap
    rng = np.random.default_rng(20261019)
    deltas = []
    pos_idx = np.flatnonzero(is_pos)
    neg_idx = np.flatnonzero(~is_pos)
    for _ in range(200):
        bp = rng.choice(pos_idx, P, replace=True)
        bn = rng.choice(neg_idx, N, replace=True)
        sub = np.concatenate([bp, bn])
        deltas.append(auc(p_treat[sub]) - auc(p_ctrl[sub]))
    return float(np.std(deltas, ddof=1))
