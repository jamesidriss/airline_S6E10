"""P20-FINALIST-READINESS audit (items 91/94) -- prereg 304179ee, frozen before results.

H91: seed-vs-subset variance decomposition on multi-seed OOF arrays.
H94: champion stability vs preprocessing variants (per-fold equal-logit blend + Spearman).

Read-only: loads saved OOF arrays from src/submission/store.py, the immutable
primary fold registry, and the labeled dataset. No training, no submission,
no public-LB consultation. Logs (not probabilities) are used for Spearman and
for the equal-logit blend; AUC is rank-based so monotone transforms do not
change it.

Usage:
  ./.venv/Scripts/python.exe scripts/phase20/finalist_readiness_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import TARGET, load_raw  # noqa: E402
from src.submission.store import load_oof, list_all  # noqa: E402

OUT = ROOT / "reports" / "phase20" / "finalist_readiness"
FOLDS_PATH = ROOT / "data" / "cache" / "folds" / "primary_5f_seed20261010.npy"

# Multi-seed groups: identical recipe, different seed only (from store meta).
SEED_GROUPS = {
    "lgbm_xt_f10": ["z3_xt_f10", "z3_xt_f10_s2", "z4_xt_f10_s3", "z4_xt_f10_s4"],
    "realmlp_full": ["prod5_realmlp_full_primary_s1_e6", "prod5_realmlp_full_primary_s7_e6",
                     "prod5_realmlp_full_primary_s123_e6",
                     "prod5_realmlp_full_primary_s2024_e6"],
}

# Reference = best certified blend IN THE STORE. NOTE: the actual champion
# v7_sol_tabpfn_internal2_aux10 (OOF 0.961853) is NOT archived in the prediction
# store -- only its test prediction is (artifacts/v7_sol_tabpfn_internal2_aux10_test).
# This is a documentation gap for the finalist audit (see report framing_note).
CHAMPION = "sol_route_aux10_primary_final"
VARIANTS = [
    "view_lgbm_raw_primary",        # 21 raw
    "view_lgbm_raw_ext_primary",     # 75 raw+ext
    "view_lgbm_raw_ogsurf_primary",  # 189 raw+ogs+surf
    "view_lgbm_core3_primary",       # 225 core3
    "view_lgbm_core3_ogsurf_primary",  # 393 core3+ogs+surf
    "view_lgbm_full_primary",        # 285 full (TE)
    "view_cat_full_primary",
    "view_xgb_full_primary",
    "xt_xt_f10",                     # strongest single blend member
]


def logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def per_fold_auc(oof: np.ndarray, y: np.ndarray, folds: np.ndarray) -> list[float]:
    return [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in range(5)]


def h91(y: np.ndarray, folds: np.ndarray) -> dict:
    out = {}
    for name, members in SEED_GROUPS.items():
        mats, overall = [], []
        for m in members:
            oof = load_oof(m)
            mats.append(per_fold_auc(oof, y, folds))
            overall.append(float(roc_auc_score(y, oof)))
        a = np.asarray(mats)  # (n_seeds, 5)
        mu = a.mean()
        seed_eff = a.mean(axis=1) - mu          # per-seed mean deviation
        fold_eff = a.mean(axis=0) - mu          # per-fold mean deviation
        resid = a - mu - seed_eff[:, None] - fold_eff[None, :]
        n_seeds, n_folds = a.shape
        # variance components of the total per-cell variance (two-way, no replication)
        seed_var = float((seed_eff ** 2).mean()) * n_folds
        subset_var = float((fold_eff ** 2).mean()) * n_seeds
        resid_var = float((resid ** 2).mean())
        out[name] = {
            "members": members,
            "overall_auc": dict(zip(members, [round(v, 6) for v in overall])),
            "per_fold_auc": {m: [round(v, 6) for v in row] for m, row in zip(members, mats)},
            "seed_variance_component": round(seed_var, 10),
            "subset_variance_component": round(subset_var, 10),
            "residual_variance": round(resid_var, 10),
            "seed_over_subset_ratio": round(seed_var / subset_var, 3) if subset_var > 0 else None,
        }
    return out


def h94(y: np.ndarray, folds: np.ndarray) -> dict:
    champ = logit(load_oof(CHAMPION))
    champ_p = load_oof(CHAMPION)
    champ_fold_auc = per_fold_auc(champ_p, y, folds)
    rows = {}
    stable_all = True
    for v in VARIANTS:
        try:
            vp = load_oof(v)
        except KeyError:
            rows[v] = {"error": "not in store"}
            continue
        vlog = logit(vp)
        blend = (champ + vlog) / 2.0
        per_fold = []
        for k in range(5):
            m = folds == k
            ca = champ_fold_auc[k]
            ba = float(roc_auc_score(y[m], blend[m]))
            sp = float(spearmanr(vlog[m], champ[m]).statistic)
            per_fold.append({
                "fold": k,
                "champion_auc": round(ca, 6),
                "blend_auc": round(ba, 6),
                "blend_delta": round(ba - ca, 8),
                "spearman": round(sp, 6),
            })
        min_delta = min(r["blend_delta"] for r in per_fold)
        min_sp = min(r["spearman"] for r in per_fold)
        stable = min_delta >= -5e-5 and min_sp > 0.98
        stable_all = stable_all and stable
        rows[v] = {
            "per_fold": per_fold,
            "min_blend_delta": round(min_delta, 8),
            "min_spearman": round(min_sp, 6),
            "stable": stable,
        }
    return {
        "champion": CHAMPION,
        "framing_note": (
            "Reference is sol_route_aux10_primary_final (OOF 0.961766), the best certified "
            "blend in the prediction store. The actual champion v7_sol_tabpfn_internal2_aux10 "
            "(OOF 0.961853) is NOT archived in the store -- only its test prediction "
            "(artifacts/v7_sol_tabpfn_internal2_aux10_test). Documentation gap: the champion "
            "OOF must be archived before a true champion-stability audit (DEP 94/96)."),
        "champion_overall_auc": round(float(roc_auc_score(y, champ_p)), 6),
        "champion_fold_auc": [round(v, 6) for v in champ_fold_auc],
        "stability_rule": "every fold: blend_delta >= -5e-5 AND spearman > 0.98",
        "variants": rows,
        "champion_stable": stable_all,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_raw()
    y = ds.train[TARGET].to_numpy()
    folds = np.load(FOLDS_PATH)
    assert len(y) == len(folds) == 699635

    have = {r["exp_id"] for r in list_all()}
    missing = [m for g in SEED_GROUPS.values() for m in g if m not in have]
    missing += [v for v in [CHAMPION, *VARIANTS] if v not in have]
    if missing:
        raise SystemExit(f"missing from store: {missing}")

    report = {
        "prereg": "configs/phase20/finalist_readiness_prereg.json",
        "prereg_sha256": "304179ee1bdec11361971261c427784ae9c77297fbdda9ebe99ddfcb8e29d574",
        "h91_seed_vs_subset": h91(y, folds),
        "h94_stability": h94(y, folds),
    }
    path = OUT / "finalist_readiness_audit.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    # console summary
    for name, d in report["h91_seed_vs_subset"].items():
        print(f"H91 {name}: seed_var={d['seed_variance_component']:.2e} "
              f"subset_var={d['subset_variance_component']:.2e} "
              f"ratio={d['seed_over_subset_ratio']}")
    h94r = report["h94_stability"]
    print(f"H94 champion {h94r['champion']} AUC={h94r['champion_overall_auc']} "
          f"stable={h94r['champion_stable']}")
    for v, d in h94r["variants"].items():
        if "error" in d:
            print(f"  {v}: MISSING")
            continue
        print(f"  {v}: min_delta={d['min_blend_delta']:+.2e} "
              f"min_spearman={d['min_spearman']:.4f} stable={d['stable']}")
    print(f"report -> {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
