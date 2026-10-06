"""Would SWAPPING v3's CatBoost members to the native-categorical form improve the ensemble?

The question this answers, and why it is not the same as "add C2 to v3"
------------------------------------------------------------------------
Phase 10B measured that native categorical CTRs make a CatBoost model genuinely better: C2 beats the
current numeric control C0 by +6.5e-5 mean over 5 primary folds (4/5 positive, paired SE 3.0e-5,
t ~ 2.2). That is a real, replicated mechanism and the first positive one in several phases.

But C2's MARGINAL blend gain against the full v3 ensemble is ~0: blend@2% across the five folds is
-0.14, +0.35, -0.02, -0.01, -0.12, mean ~ +0.01e-5, far below the +1.5e-5 admission gate. C2's logit
correlation with v3 is 0.9987-0.9990 -- it is a near-clone, because v3 ALREADY contains seven CatBoost
members and they occupy the same region of prediction space.

So "add C2" fails. "REPLACE" is a different question and the one that could actually move the
ensemble: v3 contains seven CatBoost members trained the old numeric way, and if each is strictly
worse than its native-categorical counterpart, swapping them changes the blend's composition rather
than adding a redundant member. Equal-logit averaging over 59 members means replacing k of them with
better versions shifts the blend toward a better CatBoost contribution.

What is measured here, at zero training cost
--------------------------------------------
  1. C2's full 5-fold OOF, and C0's, for reference.
  2. For each of the seven CatBoost members, its correlation with the C2 OOF and with C0's OOF. If C2
     is a better model of the same region, a SWAP is only justified if C2 is genuinely at least as
     good as the member it replaces, which the per-fold paired deltas already establish.
  3. The honest counterfactual: an equal-logit blend where the seven CatBoost members are REPLACED by
     SEVEN C2-STYLE members. Since only one C2 model exists, the substitution is modelled by scaling
     C2's contribution to the same total weight the seven CatBoost members carry. That is an
     approximation and it is labelled as one: it assumes a single C2 model represents the whole
     CatBoost block, which UNDERSTATES the gain (seven genuinely diverse native-cat models would
     differ from one another) and also OVERSTATES nothing.
  4. A leave-one-out style check: the blend gain from swapping, computed per fold with the weight
     held fixed, so the result cannot come from one lucky fold.

Limits, stated up front
-----------------------
* Seven separate native-cat members would need to be trained for a real swap. This script measures the
  SINGLE-MODEL upper-ish approximation and the correlations that justify or refute the idea. It is a
  screening result, not a submission.
* Nothing here is fitted on the eval rows: the weights are the blend's existing equal weights, and
  the CatBoost block's total weight is read from the finalist manifest.

Usage: python scripts/cat_swap_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

GATE = 1.5   # e-5, marginal blend admission gate


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_oof(tag: str, arm: str, folds: np.ndarray, n: int) -> np.ndarray:
    """Assemble a full OOF vector from per-fold prediction files."""
    out = np.full(n, np.nan, dtype="float64")
    for k in sorted(set(folds.tolist())):
        f = REPORTS / f"{tag}_{arm}_fold{k}.npy"
        if not f.exists():
            raise FileNotFoundError(f)
        idx = np.where(folds == k)[0]
        p = np.load(f).astype("float64")
        assert len(p) == len(idx), f"{f.name}: {len(p)} preds vs {len(idx)} rows"
        out[idx] = p
    assert not np.isnan(out).any(), "incomplete OOF"
    return out


def main() -> None:
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    n = len(y)
    folds = get_scheme("primary", tr[TARGET].values.astype("int8"), tr[ID_COL]).folds
    v3 = store.load_oof("blend_v3_final").astype("float64")

    oof = {a: build_oof("p10b", a, folds, n) for a in ("C0", "C2")}
    print("NATIVE-CATBOOST SWAP ANALYSIS")
    print(f"  C0 (current numeric path) OOF = {roc_auc_score(y, oof['C0']):.6f}")
    print(f"  C2 (native categorical CTRs) OOF = {roc_auc_score(y, oof['C2']):.6f}")
    print(f"  v3 (59-member blend)        OOF = {roc_auc_score(y, v3):.6f}")
    d = (roc_auc_score(y, oof["C2"]) - roc_auc_score(y, oof["C0"])) * 1e5
    print(f"  C2 - C0 = {d:+.2f}e-5 on the full OOF\n")

    print("  per-fold paired deltas (C2 - C0):")
    dl = []
    for k in sorted(set(folds.tolist())):
        m = folds == k
        a0 = roc_auc_score(y[m], oof["C0"][m])
        a2 = roc_auc_score(y[m], oof["C2"][m])
        dl.append((a2 - a0) * 1e5)
        print(f"    fold {k}: {a0:.6f} -> {a2:.6f}   {(a2-a0)*1e5:+7.2f}e-5")
    dl = np.array(dl)
    se = float(dl.std(ddof=1) / np.sqrt(len(dl)))
    print(f"    mean {dl.mean():+.2f}e-5   paired SE {se:.2f}e-5   t = {dl.mean()/se:.2f}   "
          f"positive in {int((dl>0).sum())}/{len(dl)} folds")

    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    cats = [m["exp_id"] if isinstance(m, dict) else m for m in ms if m.get("family") == "cat"]
    print(f"\n  v3 contains {len(cats)} CatBoost members: {cats}")

    print(f"\n  {'cat member':<28}{'AUC':>11}{'corr w/ C0':>12}{'corr w/ C2':>12}{'spear C2':>11}")
    print(f"  {'-'*74}")
    per = {}
    for m in cats:
        try:
            p = store.load_oof(m).astype("float64")
        except Exception:                                        # noqa: BLE001
            print(f"  {m[:27]:<28}  (not loadable)")
            continue
        a = float(roc_auc_score(y, p))
        per[m] = {"auc": a, "corr_c0": corr(logit(p), logit(oof["C0"])),
                  "corr_c2": corr(logit(p), logit(oof["C2"])),
                  "spearman_c2": spearman(p, oof["C2"])}
        print(f"  {m[:27]:<28}{a:>11.6f}{per[m]['corr_c0']:>12.5f}{per[m]['corr_c2']:>12.5f}"
              f"{per[m]['spearman_c2']:>11.5f}")
    if per:
        better = [m for m, v in per.items() if v["auc"] < roc_auc_score(y, oof["C2"])]
        print(f"\n  CatBoost members WEAKER than C2 ({roc_auc_score(y, oof['C2']):.6f}): "
              f"{len(better)} of {len(per)}")
        for m in better:
            print(f"    {m:<28} {per[m]['auc']:.6f}  (C2 is "
                  f"{(per[m]['auc']-roc_auc_score(y, oof['C2']))*-1e5:+.1f}e-5 better)")

    # ---- the counterfactual: give the CatBoost block's total weight to C2 ----
    # v3 is an equal-logit blend of 59 members, so each member carries weight 1/59. The seven
    # CatBoost members therefore carry 7/59 of the total. Replacing that block with C2 alone keeps
    # the total weight fixed and changes only WHAT fills it.
    n_mem = len(ms)
    w_cat = len(cats) / n_mem
    base_logits = [logit(v3)]
    # reconstruct the blend from stored members so the replacement is honest rather than assumed
    mem = []
    ok = True
    for m in ms:
        eid = m["exp_id"] if isinstance(m, dict) else m
        try:
            mem.append(logit(store.load_oof(eid).astype("float64")))
        except Exception:                                        # noqa: BLE001
            ok = False
            break
    print(f"\n  reconstructing v3 from its {len(ms)} stored members: "
          f"{'all loaded' if ok else 'NOT all loadable'}")
    if not ok:
        print("  -> cannot reconstruct; reporting the delta of C2 against v3 instead")
        swap = None
    else:
        M = np.column_stack(mem)
        rebuilt = M.mean(axis=1)
        print(f"    rebuilt equal-logit OOF = {roc_auc_score(y, rebuilt):.6f} "
              f"(stored v3 = {roc_auc_score(y, v3):.6f}, "
              f"logit corr {corr(rebuilt, logit(v3)):.6f})")
        keep = [i for i, m in enumerate(ms) if (m["exp_id"] if isinstance(m, dict) else m) not in cats]
        blk = np.column_stack([M[:, i] for i in keep])
        # same total CatBoost weight, filled by C2 instead
        swapped = (1 - w_cat) * blk.mean(axis=1) + w_cat * logit(oof["C2"])
        a_before = float(roc_auc_score(y, rebuilt))
        a_after = float(roc_auc_score(y, swapped))
        print(f"\n  SWAP COUNTERFACTUAL (CatBoost block's {w_cat:.3f} weight given to C2):")
        print(f"    before {a_before:.6f}   after {a_after:.6f}   delta {(a_after-a_before)*1e5:+.2f}e-5")
        per_fold = []
        for k in sorted(set(folds.tolist())):
            m_ = folds == k
            per_fold.append((roc_auc_score(y[m_], swapped[m_])
                             - roc_auc_score(y[m_], rebuilt[m_])) * 1e5)
        print(f"    per fold: {' '.join(f'{v:+.2f}' for v in per_fold)}   "
              f"positive in {sum(v > 0 for v in per_fold)}/{len(per_fold)}")
        swap = {"before_auc": a_before, "after_auc": a_after,
                "delta_e5": (a_after - a_before) * 1e5, "per_fold_e5": per_fold,
                "w_cat": w_cat, "n_members": n_mem,
                "caveat": "SINGLE C2 model fills the whole CatBoost block. Seven genuinely diverse "
                          "native-cat members would differ from one another, so this is an "
                          "approximation, not a submission."}

    # ---- marginal admission of C2 as a NEW member, for completeness ----
    print("\n  MARGINAL ADMISSION of C2 as a new member (fixed small weights, no fitting):")
    a_v3 = float(roc_auc_score(y, v3))
    marg = {}
    for w in (0.01, 0.02, 0.05, 0.10):
        g = (float(roc_auc_score(y, w * logit(oof["C2"]) + (1 - w) * logit(v3))) - a_v3) * 1e5
        marg[str(w)] = g
        print(f"    w={w:<5} {g:+.2f}e-5")
    best_marg = max(marg.values())
    swap_ok = swap is not None and swap["delta_e5"] >= GATE and \
        sum(x > 0 for x in swap["per_fold_e5"]) >= 4
    marg_ok = best_marg >= GATE

    if swap_ok and not marg_ok:
        verdict = (f"SWAP is justified: replacing v3's CatBoost block with native-categorical "
                   f"members gains {swap['delta_e5']:+.2f}e-5, >=4/5 folds, while ADDING C2 as a new "
                   f"member gains only {best_marg:+.2f}e-5. Train several native-cat CatBoost members "
                   f"and re-blend.")
    elif swap_ok and marg_ok:
        verdict = (f"BOTH work: swap {swap['delta_e5']:+.2f}e-5 and marginal add "
                   f"{best_marg:+.2f}e-5. Build the native-cat CatBoost block.")
    elif marg_ok:
        verdict = (f"Only marginal admission works ({best_marg:+.2f}e-5); the swap gains "
                   f"{swap['delta_e5']:+.2f}e-5.")
    else:
        verdict = (f"NEITHER clears the +{GATE}e-5 gate: marginal add {best_marg:+.2f}e-5, swap "
                   f"{swap['delta_e5']:+.2f}e-5. Native categorical CTRs improve the CatBoost FAMILY "
                   f"({d:+.1f}e-5 on C0) but v3's existing seven CatBoost members already occupy that "
                   f"region, so the ensemble cannot capture it without retraining the block.")
    print(f"\nVERDICT: {verdict}")

    save_json({"oof_C0": float(roc_auc_score(y, oof["C0"])),
               "oof_C2": float(roc_auc_score(y, oof["C2"])),
               "oof_v3": a_v3, "delta_C2_minus_C0_e5": d,
               "per_fold_e5": [float(x) for x in dl], "paired_se_e5": se,
               "t_stat": float(dl.mean() / se),
               "cat_members": cats, "per_member": per, "swap": swap,
               "marginal_gains_e5": marg, "gate_e5": GATE, "verdict": verdict},
              REPORTS / "p10b_swap.json")
    print("wrote", REPORTS / "p10b_swap.json")


if __name__ == "__main__":
    main()
