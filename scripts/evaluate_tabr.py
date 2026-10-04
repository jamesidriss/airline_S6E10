"""Phase 4 evaluation: is TabR worth adding to the finalist ensemble?

TabR's standalone AUC is not the decision variable. Every family already in the pool sits at
logit-correlation 0.995-0.999 with the finalist, so a 0.9611 model that adds nothing is worthless
while a 0.9607 model that adds +2e-5 is valuable. This script produces the full decision record:

  * overall OOF AUC and per-fold AUCs
  * Pearson (raw and logit) correlation with the best LightGBM, the best RealMLP, and the finalist
  * Spearman rank correlation with the same three
  * marginal equal-logit blend gain when adding TabR to the v3 finalist
  * a paired bootstrap CI on that marginal gain, resampling fold-wise
  * the best 2-way blend weight, to see whether TabR is even *usable* in a mixture

The marginal gain is measured by rebuilding the finalist blend with and without TabR and comparing
OOF AUC -- the same quantity scripts/marginal_gain.py reports -- so the two are directly
comparable.

Usage:
  python scripts/evaluate_tabr.py --member tabr_full_c128_e25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.ensemble import lab  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

# Reference members chosen as the strongest representative of each existing family.
REFERENCES = {
    "finalist_v3": "blend_v3_final",
    "best_lgbm_xt": "z4_xt_f10_s4",
    "best_realmlp": "prod5_realmlp_full_primary_s7_e6",
    "tabm": "z5_tabm_e25",
}


def equal_logit(mats: dict[str, np.ndarray]) -> np.ndarray:
    """Average member logits -- the geometry of the banked finalist."""
    stack = np.vstack([lab.tform(m, "logit") for m in mats.values()])
    return stack.mean(axis=0)


def bootstrap_marginal(y, with_mix, without_mix, folds, n_boot=400, seed=7):
    """Paired bootstrap over ROWS, resampling whole folds so fold structure is preserved."""
    rng = np.random.default_rng(seed)
    ks = sorted(set(folds.tolist()))
    by_fold = [np.where(folds == k)[0] for k in ks]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ks), len(ks))
        idx = np.concatenate([by_fold[p] for p in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            diffs[b] = np.nan
            continue
        diffs[b] = (roc_auc_score(yy, with_mix[idx])
                    - roc_auc_score(yy, without_mix[idx]))
    diffs = diffs[~np.isnan(diffs)]
    return {"mean": float(np.mean(diffs)), "lo95": float(np.percentile(diffs, 2.5)),
            "hi95": float(np.percentile(diffs, 97.5)), "n_effective": int(diffs.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True, help="experiment id in the prediction store")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--gate", type=float, default=1.5e-5,
                    help="admission gate on marginal OOF gain")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.folds, y, tr[ID_COL]).folds
    ks = sorted(set(folds.tolist()))

    tabr = store.load_oof(args.member).astype("float64")
    tabr_auc = float(roc_auc_score(y, tabr))
    fold_aucs = {int(k): float(roc_auc_score(y[folds == k], tabr[folds == k])) for k in ks}

    print("=" * 96)
    print(f"PHASE 4 EVALUATION: {args.member}")
    print("=" * 96)
    print(f"  TabR OOF AUC = {tabr_auc:.6f}")
    print(f"  fold AUCs    = {[round(fold_aucs[k], 6) for k in ks]}")

    # ---------------------------------------------------------------- correlations
    print("\n--- correlation with existing families ---")
    corrs = {}
    for label, name in REFERENCES.items():
        try:
            ref = store.load_oof(name).astype("float64")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<14} UNAVAILABLE ({type(exc).__name__})")
            continue
        c_raw = corr(tabr, ref)
        c_logit = corr(lab.tform(tabr, "logit"), lab.tform(ref, "logit"))
        c_spear = spearman(tabr, ref)
        corrs[label] = {"member": name, "auc": float(roc_auc_score(y, ref)),
                        "pearson_raw": c_raw, "pearson_logit": c_logit, "spearman": c_spear}
        print(f"  {label:<14} AUC={corrs[label]['auc']:.6f}  "
              f"pearson_raw={c_raw:.5f}  pearson_logit={c_logit:.5f}  spearman={c_spear:.5f}")

    # ---------------------------------------------------------------- marginal gain
    finalist_name = REFERENCES["finalist_v3"]
    fin = store.load_oof(finalist_name).astype("float64")
    without = fin.copy()
    with_t = equal_logit({"a": without, "b": tabr})
    a_with = float(roc_auc_score(y, with_t))
    a_without = float(roc_auc_score(y, without))
    gain = a_with - a_without
    print("\n--- marginal equal-logit blend gain on the finalist ---")
    print(f"  finalist alone      = {a_without:.6f}")
    print(f"  finalist + TabR     = {a_with:.6f}")
    print(f"  marginal gain       = {gain:+.6f}  ({gain*1e5:+.2f}e-5)")

    per_fold_gain = {}
    for k in ks:
        m = folds == k
        g = roc_auc_score(y[m], with_t[m]) - roc_auc_score(y[m], without[m])
        per_fold_gain[int(k)] = float(g)
    print(f"  per-fold gain       = {[round(per_fold_gain[k], 6) for k in ks]}")
    print(f"  folds improved      = {sum(1 for k in ks if per_fold_gain[k] > 0)}/{len(ks)}")

    bs = bootstrap_marginal(y, with_t, without, folds, n_boot=args.n_boot)
    print(f"  paired bootstrap    = {bs['mean']:+.6f}  "
          f"95% CI [{bs['lo95']:+.6f}, {bs['hi95']:+.6f}]  (n={bs['n_effective']})")

    # ---------------------------------------------------------------- 2-way weight scan
    best = (0.0, a_without)
    for wt in np.linspace(0.0, 0.5, 51):
        mix = wt * lab.tform(tabr, "logit") + (1.0 - wt) * lab.tform(fin, "logit")
        a = float(roc_auc_score(y, mix))
        if a > best[1]:
            best = (float(wt), a)
    print(f"\n--- 2-way weight scan (w on TabR) ---")
    print(f"  best weight = {best[0]:.2f}  ->  AUC {best[1]:.6f} "
          f"({best[1]-a_without:+.6f} vs finalist)")
    print(f"  a usable mixture needs w>0; w=0.00 means TabR is pure noise to this blend")

    # ---------------------------------------------------------------- verdict
    sig = bs["lo95"] > 0.0
    passes = gain >= args.gate and (sig or gain >= 3.0 * args.gate)
    if best[0] <= 0.0:
        verdict = "REJECT -- no positive weight exists that improves the finalist"
    elif not passes:
        verdict = (f"REJECT -- marginal gain {gain:+.6f} below gate {args.gate:.2e} "
                   f"(bootstrap CI straddles 0)")
    else:
        verdict = (f"ADMIT -- marginal gain {gain:+.6f} >= gate {args.gate:.2e}, "
                   f"CI [{bs['lo95']:+.6f}, {bs['hi95']:+.6f}], "
                   f"{sum(1 for k in ks if per_fold_gain[k] > 0)}/{len(ks)} folds positive")
    print(f"\nVERDICT: {verdict}")

    out = {"member": args.member, "tabr_oof_auc": tabr_auc, "fold_aucs": fold_aucs,
           "scheme": args.folds, "correlations": corrs,
           "finalist_auc": a_without, "blend_with_tabr_auc": a_with,
           "marginal_gain": gain, "per_fold_gain": per_fold_gain,
           "folds_improved": sum(1 for k in ks if per_fold_gain[k] > 0),
           "bootstrap": bs, "best_2way_weight": best[0], "best_2way_auc": best[1],
           "gate": args.gate, "verdict": verdict}
    save_json(out, REPORTS / f"eval_{args.member}.json")
    print("\nwrote", REPORTS / f"eval_{args.member}.json")


if __name__ == "__main__":
    main()
