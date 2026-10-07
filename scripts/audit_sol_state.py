"""Independent S0 arithmetic/integrity audit; never retrains or overwrites finalists."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (ARTIFACTS, ID_COL, RAW, REPORTS, TARGET, arr_sha256,
                        file_sha256, git_commit, load_cached_parquet, save_json)
from src.submission import store
from src.validation.compare import logit, spearman
from src.validation.folds import get_scheme
from scripts.run_phase14c import XT_SLOTS, XGB_SLOTS, CAT_SLOTS, counter_path


def members():
    d = json.loads((REPORTS / "finalist_v3_final.json").read_text(encoding="utf-8"))
    return [m["exp_id"] for m in d["members"]]


def reconstruct_v5(y, folds):
    ids = members()
    assert len(ids) == len(set(ids)) == 59
    base = np.zeros(len(y), dtype=float)
    replacement = np.zeros(len(y), dtype=float)
    hashes = {}
    for member in ids:
        p = store.load_oof(member)
        assert p.shape == y.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        z = logit(p)
        base += z / len(ids)
        if member in XT_SLOTS + XGB_SLOTS + CAT_SLOTS:
            q = np.full(len(y), np.nan)
            hashes[member] = {}
            for k in range(5):
                path = counter_path(member, k)
                v = np.load(path)
                mask = folds == k
                assert v.shape == (int(mask.sum()),), (path, v.shape)
                assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
                q[mask] = v
                hashes[member][str(k)] = {"path": str(path.relative_to(REPORTS.parent)),
                                         "sha256": arr_sha256(v), "rows": len(v)}
            replacement += (logit(q) - z) / len(ids)
    return base, base + replacement, hashes


def pair_diagnostic(y, a, b, seed=20261010, n=2_000_000):
    rng = np.random.default_rng(seed)
    pi = rng.choice(np.flatnonzero(y == 1), n, replace=True)
    ni = rng.choice(np.flatnonzero(y == 0), n, replace=True)
    aa = np.sign(a[pi] - a[ni])
    bb = np.sign(b[pi] - b[ni])
    rescued = (aa > 0) & (bb < 0)
    damaged = (aa < 0) & (bb > 0)
    return {"seed": seed, "pairs": n, "rescue_rate": float(rescued.sum() / (bb < 0).sum()),
            "damage_rate": float(damaged.sum() / (bb > 0).sum()),
            "rescue_count": int(rescued.sum()), "damage_count": int(damaged.sum()),
            "sampled_delta_with_ties": float(np.mean((aa - bb) / 2)),
            "warning": "Monte Carlo pair diagnostic, not exact AUC arithmetic"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/sol_s0_audit.json")
    args = ap.parse_args()
    tr, te = load_cached_parquet()
    y = tr[TARGET].to_numpy(dtype="int8")
    original = json.loads((REPORTS / "finalist_v3_final.json").read_text(encoding="utf-8"))
    raw_hashes = {p.name: file_sha256(p) for p in RAW.glob("*.csv")}
    for name, h in original["data_hashes"].items():
        assert raw_hashes[name].startswith(h), f"raw data changed: {name}"
    raw_tr, raw_te = pd.read_csv(RAW / "train.csv"), pd.read_csv(RAW / "test.csv")
    pd.testing.assert_frame_equal(tr, raw_tr, check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(te, raw_te, check_dtype=False, check_exact=True)
    assert tr[ID_COL].is_unique and te[ID_COL].is_unique
    assert not np.intersect1d(tr[ID_COL], te[ID_COL]).size
    schemes = {s: get_scheme(s, y, tr[ID_COL]) for s in ("primary", "shadow", "block10")}
    folds = schemes["primary"].folds
    base, new, hashes = reconstruct_v5(y, folds)
    stored = store.load_oof("blend_v3_final")
    from scipy.special import expit
    # NumPy and scipy expit can differ below float32 resolution; the stored vector is float32.
    assert np.array_equal(expit(base).astype("float32"), stored), "v3 blend does not reproduce"
    fa = [float(roc_auc_score(y[folds == k], new[folds == k])) for k in range(5)]
    fb = [float(roc_auc_score(y[folds == k], base[folds == k])) for k in range(5)]
    d = np.subtract(fa, fb)
    se = float(d.std(ddof=1) / np.sqrt(5))
    out = {"git": git_commit(), "accessed_utc": datetime.now(timezone.utc).isoformat(),
           "arithmetic_status": "PASS", "campaign_contract_status": "FAIL_REQUIRES_REPAIR",
           "data_sha256": raw_hashes, "train_ids_sha256": arr_sha256(tr[ID_COL].to_numpy()),
           "test_ids_sha256": arr_sha256(te[ID_COL].to_numpy()),
           "fold_sha256": {s: arr_sha256(v.folds) for s, v in schemes.items()},
           "v3_auc": float(roc_auc_score(y, base)), "v5_auc": float(roc_auc_score(y, new)),
           "pooled_delta": float(roc_auc_score(y, new) - roc_auc_score(y, base)),
           "v3_fold_auc": fb, "v5_fold_auc": fa, "paired_fold_deltas": d.tolist(),
           "mean_paired_delta": float(d.mean()), "paired_se": se,
           "paired_t": float(d.mean() / se), "positive_folds": int((d > 0).sum()),
           "oof_logit_corr": float(np.corrcoef(base, new)[0, 1]),
           "oof_spearman": spearman(base, new), "counterpart_artifacts": hashes,
           "pair_diagnostic": pair_diagnostic(y, new, base), "submissions": {}}
    for name in ("v3_final", "v4_fulldata", "v5_aux_cross"):
        p = REPORTS.parent / "submissions" / f"{name}.csv"
        sub = pd.read_csv(p)
        assert list(sub) == [ID_COL, TARGET]
        assert np.array_equal(sub[ID_COL], te[ID_COL])
        v = sub[TARGET].to_numpy()
        assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
        out["submissions"][name] = {"rows": len(v), "file_sha256": file_sha256(p),
                                    "prediction_sha256": arr_sha256(v)}
    v5man = json.loads((REPORTS / "v5_aux_cross_manifest.json").read_text(encoding="utf-8"))
    assert out["submissions"]["v5_aux_cross"]["file_sha256"].startswith(v5man["sha16"])
    p3 = pd.read_csv(REPORTS.parent / "submissions/v3_final.csv")[TARGET].to_numpy()
    p5 = pd.read_csv(REPORTS.parent / "submissions/v5_aux_cross.csv")[TARGET].to_numpy()
    out["test_logit_corr_v5_v3"] = float(np.corrcoef(logit(p5), logit(p3))[0, 1])
    out["test_spearman_v5_v3"] = spearman(p5, p3)
    out["defects"] = [
        {"id": "TE_PRIOR_OWN_LABEL", "status": "code fixed; historical vectors not retrained",
         "detail": "inner TE tables excluded apply labels but their smoothing prior used all outer-fit labels",
         "maximum_own_prior_contribution_primary": 1 / int((folds != 0).sum()),
         "scope": "fit-row self exclusion fails; no outer-validation label enters TE"},
        {"id": "V5_XT_CONFIG_COLLAPSE", "detail": "six test counterparts all used seed 1 and d127 defaults"},
        {"id": "V5_ITERATION_MIX", "detail": "median merged controls, treatments, duplicate folds and both CatBoost representations; fold 0 records absent"},
        {"id": "V5_AUX_POLICY", "detail": "OOF inner crossfit 3 folds, test inner crossfit 5 folds"},
        {"id": "V3_TEST_POPULATION_CLAIM", "detail": "v3 test members are fold averages, not uniformly 100%-label refits; v4 explicitly refits 32 members"},
        {"id": "P14C_DOUBLE_LOGIT", "detail": "corr_v3 clipped mean logits as probabilities; corrected here and in code"},
        {"id": "FOLD_HASH_MISSING", "detail": "registry verified y hash but not fold bytes or ordered ids; existing arrays now hash bound without regeneration"},
        {"id": "TEST_RUNNER_FAILURE_RESET", "detail": "stochastic suite failures were zeroed before aggregate exit; fixed"},
        {"id": "TEST_RECONSTRUCTION_EVIDENCE", "detail": "v5 per-slot test vectors were never saved; CSV hash reproduces but ten test-slot arithmetic cannot be independently reconstructed"},
    ]
    out["limitations"] = ["Legacy per-fold vectors have no ordered row-id sidecars. Length, fold mapping, file hashes and v3 exact blend are checked; retrospective alignment cannot be fully proved.",
                          "v4 has no distinct OOF vector: v3 is its CV proxy, not a measured v4 OOF improvement."]
    dest = ARTIFACTS / "sol_s0"
    dest.mkdir(exist_ok=True)
    np.save(dest / "v3_logit_oof.npy", base)
    np.save(dest / "v5_logit_oof.npy", new)
    out["oof_artifact_sha256"] = {"v3": arr_sha256(base), "v5": arr_sha256(new)}
    save_json(out, args.output)
    print(json.dumps({k: out[k] for k in ("arithmetic_status", "campaign_contract_status", "v3_auc", "v5_auc", "pooled_delta", "mean_paired_delta", "paired_t", "positive_folds")}))
    print("Defects:", ", ".join(d["id"] for d in out["defects"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
