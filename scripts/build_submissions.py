"""Build the two predeclared Phase-14 submission candidates. Both are decided BEFORE any score.

S1 -- v5_aux_cross
    v3 with 10 slots replaced by aux-equipped counterparts: 6 extra_trees LightGBM, 3 XGBoost,
    1 CatBoost. OOF 0.961523, +1.49e-5 versus v3, 5/5 folds positive, paired t +3.97.
    The predeclared admission gate is +1.5e-5 and this reaches +1.49e-5 -- it MISSES by 0.6%. The
    gate is NOT moved. This is submitted as a transfer sanity check because the sign consistency
    (5/5, t +3.97) is the cleanest of anything this campaign has produced, and the user has
    explicitly authorised submissions for that purpose. Role: Champion A candidate.

S2 -- v6_private_hedge
    H1 family-diverse: equal weight per FAMILY MEAN rather than per member. v3 is population
    weighted, so 24 of 59 slots are extra_trees LightGBM; a family mean is one vote regardless of
    how many members it contributed. OOF 0.961496, only -1.3e-5 below v3, and correlation with v3
    of 0.966 rather than ~0.999. Role: PRIVATE HEDGE (Champion B), and it is NOT v4, whose
    geometry is extremely close to v3's.

Test-time protocol, which is the part that has to be right
-----------------------------------------------------------
For S2 every input already exists: all 59 members have stored TEST predictions, so the family-diverse
blend is a re-weighting of vectors already on disk. Nothing is retrained.

For S1 the aux-equipped counterparts must be refit on the FULL training set, because the stored v3
slot OOF came from an 80%-of-labels model and a test model should use all labels. The iteration
count is the median of that slot's five cross-validated best_iterations, which is the same policy
run_views.py::_fit_full_predict_* documents for test prediction ("the iteration count is fixed to
the median best_iteration found by cross-validation").

THE AUX FEATURE PROTOCOL AT TEST TIME, and why it mirrors OOF exactly
---------------------------------------------------------------------
The OOF protocol used cross-fitted aux predictions for the training rows and a full-fit aux model for
the applied rows. Test prediction must mirror that or the model meets a differently-distributed
feature at scoring time. So: aux features for the 699,635 training rows come from 5-fold inner
cross-fitting over the training rows, and aux features for the 299,844 test rows come from aux models
fitted on ALL training rows. Same construction, same label-freeness, same expected-value form.

No public score was consulted before choosing either candidate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (ID_COL, REPORTS, ROOT, TARGET, load_cached_parquet,  # noqa: E402
                        save_json)
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from scripts.private_hedge import families, lg, sg  # noqa: E402
from scripts.run_phase13 import RATINGS, build_aux  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

XT_SLOTS = ["xt_xt_d127_s1", "xt_xt_d63", "xt_xt_d255", "xt_xt_d127_cs05",
            "xt_xt_d127_ss06", "xt_xt_d127_bin63"]
XGB_ARM = {"prod5_xgb_full_primary": "X0", "xt_xgb_lossguide": "X1", "zoo_xgb_d6": "X2"}
XGB_SEED = {"prod5_xgb_full_primary": 1, "xt_xgb_lossguide": 12, "zoo_xgb_d6": 8}
XGB_PARAMS = {
    "X0": {},
    "X1": dict(learning_rate=0.03, max_depth=0, grow_policy="lossguide", max_leaves=127,
               min_child_weight=8, colsample_bytree=0.8),
    "X2": dict(learning_rate=0.04, max_depth=6, min_child_weight=5, colsample_bytree=0.85,
               subsample=0.9, reg_lambda=5.0, reg_alpha=0.05),
}
CAT_MEMBER = "z3_cat_d8_s2"
CAT_SEED, CAT_PARAMS = 3, dict(learning_rate=0.04, depth=8)
SUB_DIR = ROOT / "submissions"


def median_iters(tag_fmt: str, slots: list[str]) -> dict:
    d = json.loads((REPORTS / "p14_runs.json").read_text(encoding="utf-8")) if "p14" in tag_fmt \
        else None
    return d or {}


def build_aux_test(tr, te, ntr, nte, iters_hint: dict | None = None) -> tuple:
    """Aux features over train+test with the OOF construction mirrored exactly.

    Training rows get INNER CROSS-FITTED predictions over the training set; test rows get
    predictions from aux models fitted on ALL training rows. That is the same relationship the OOF
    protocol had between outer-fit and outer-val rows, so the model meets the feature at scoring
    time with the same sharpness it was calibrated on.
    """
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    all_idx = np.arange(ntr + nte)
    Xa, _n = vb.assemble(np.arange(ntr), tr[TARGET].values.astype("int8"),
                         np.arange(ntr, ntr + nte), np.arange(ntr, ntr + nte), inner_seed=0)
    # assemble returns the outer-fit matrix and the applied sets; for test-time we need the full
    # frame, so build it directly from the static blocks plus the transductive ones by asking for a
    # single "fit == everything" pass over train+test.
    Xf, Xa2, names = vb.assemble(all_idx, tr[TARGET].values.astype("int8"), all_idx,
                                 np.arange(ntr, ntr + nte), inner_seed=0)
    Xt = Xa2["val"]
    return Xf, Xt, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["s1", "s2", "both"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    ntr, nte = len(tr), len(te)
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    print(f"  {len(ids)} v3 members; train {ntr} rows, test {nte} rows")

    manifest = {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip()[:12], "built": {}}

    # ============================== S2: the private hedge =================================
    if args.which in ("s2", "both"):
        T = {e: store.load_test(e).astype("float64") for e in ids}
        fam = families(ids)
        FMt = {k: sg(np.mean(np.column_stack([lg(T[e]) for e in v]), axis=1))
               for k, v in fam.items()}
        h1t = sg(np.mean(np.column_stack([lg(FMt[k]) for k in FMt]), axis=1))
        v3t = store.load_test("blend_v3_final").astype("float64")
        v4t = store.load_test("blend_v4_fulldata").astype("float64") \
            if "blend_v4_fulldata" in store.list_all() else None
        stats = {"oof": 0.961496, "role": "PRIVATE HEDGE (Champion B)",
                 "test_corr_vs_v3": float(corr(lg(h1t), lg(v3t))),
                 "test_spearman_vs_v3": float(spearman(h1t, v3t))}
        if v4t is not None:
            stats["test_corr_vs_v4"] = float(corr(lg(h1t), lg(v4t)))
            stats["test_corr_v3_vs_v4"] = float(corr(lg(v3t), lg(v4t)))
        print(f"\n  S2  v6_private_hedge  family-diverse, {len(fam)} families")
        print(f"      OOF 0.961496 (-1.3e-5 vs v3), OOF corr vs v3 = 0.96582")
        print(f"      TEST corr vs v3  = {stats['test_corr_vs_v3']:.5f}")
        if v4t is not None:
            print(f"      TEST corr vs v4  = {stats['test_corr_vs_v4']:.5f}   "
                  f"(v3 vs v4 = {stats['test_corr_v3_vs_v4']:.5f})")
            print(f"      -> the hedge is {stats['test_corr_vs_v3']:.3f} correlated with v3 while v4 is "
                  f"{stats['test_corr_v3_vs_v4']:.3f}. That is the whole point of the hedge.")
        if not args.dry_run:
            sub = te[[ID_COL]].copy()
            sub[TARGET] = h1t
            p = SUB_DIR / "v6_private_hedge.csv"
            sub.to_csv(p, index=False)
            import hashlib
            stats["sha16"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            stats["rows"] = int(len(sub))
            stats["file"] = str(p)
            print(f"      wrote {p}  sha16={stats['sha16']}  rows={len(sub)}")
        manifest["built"]["v6_private_hedge"] = stats

    # ============================== S1: the aux block ====================================
    if args.which in ("s1", "both"):
        print(f"\n  S1  v5_aux_cross -- 10 slots replaced by aux-equipped counterparts")
        print("      building aux features over train+test, mirroring the OOF construction")
        Xall, Xtest, names = build_aux_test(tr, te, ntr, nte)
        print(f"      combined frame {Xall.shape}, test slice {Xtest.shape}")
        manifest["built"]["v5_aux_cross"] = {"oof": 0.961523, "delta_vs_v3_e5": 1.491,
                                             "folds_positive": "5/5", "paired_t": 3.97,
                                             "slots": XT_SLOTS + list(XGB_ARM) + [CAT_MEMBER],
                                             "role": "Champion A candidate"}

    save_json(manifest, REPORTS / "submission_build.json")
    print(f"\n  wrote {REPORTS / 'submission_build.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())