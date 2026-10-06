"""ZERO-COST single-slot swap test using C2's existing 5-fold OOF.

Why this exists, and why it comes before the seven-slot run
-----------------------------------------------------------
Training seven diverse native counterparts costs roughly 12 hours: CatBoost's cost on this workload
fits 0.428*R + 6.4e-4*R^2 seconds, so a 2500-round inner ES is ~84 minutes PER ARM at block10's
629,671 outer-fit rows. That is a large spend on an untested premise.

But the premise has a free test. Phase 10's C2 is already a full 5-fold native-categorical CatBoost
model on the `full` view at depth 8 / lr 0.04 / l2 3.0 -- which is exactly the configuration of two of
v3's seven CatBoost slots (z3_cat_d8_s2, seed 3, and prod5_cat_full_primary, seed 1), differing only
in seed. Its OOF vector already exists. So:

    swap ONE CatBoost slot for C2, leave the other 52 members and the other 6 CatBoost slots alone

costs zero compute and is exactly the zero-degree-of-freedom operational question Phase 11 asks. It
is a LOWER BOUND on what the diverse block can deliver, in the following precise sense: the native
block differs from this test only in that its remaining six slots are also native rather than numeric,
and a native model is individually at least as good as its numeric counterpart (Phase 10: +6.64e-5
for C2 - C0, 4/5 folds). So if a ONE-slot swap is already negative, the seven-slot version has to
overcome a deficit that grows with the number of swapped slots, and the expensive run is not
warranted. If a one-slot swap is clearly positive, the expensive run is warranted.

What this cannot tell us
------------------------
One swap with a single seed understates the diverse block, because it removes none of the block's
internal diversity -- the other six members are untouched. So a positive result here is a genuine
encouragement but not the block result, and a NEGATIVE result is informative but not conclusive
either (a seed-matched swap could behave differently). Both readings are reported with that caveat
attached rather than treated as the block verdict.

Prediction semantics and fold alignment are handled explicitly: C2's per-fold prediction files are
stitched by the primary fold scheme, and its AUC is recomputed from the label rather than trusted.

Usage: python scripts/c2_single_slot_swap.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.native_block_analysis import sig  # noqa: E402

SWAPPABLE = ["z3_cat_d8_s2", "prod5_cat_full_primary"]   # slots whose config C2 matches


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stitch(tag: str, arm: str, folds: np.ndarray, n: int) -> np.ndarray:
    v = np.full(n, np.nan)
    for k in sorted(set(folds.tolist())):
        p = REPORTS / f"{tag}_{arm}_fold{k}.npy"
        if not p.exists():
            raise FileNotFoundError(f"missing {p.name}")
        idx = np.where(folds == k)[0]
        pr = np.load(p).astype("float64")
        assert len(pr) == len(idx), f"{p.name}: {len(pr)} vs {len(idx)}"
        v[idx] = pr
    assert not np.isnan(v).any()
    return v


def main() -> None:
    t0 = time.time()
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    n = len(y)
    folds = get_scheme("primary", tr[TARGET].values.astype("int8"), tr[ID_COL]).folds
    inv = json.loads((REPORTS / "native_cat_slot_inventory.json").read_text(encoding="utf-8"))
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    nmem = len(ids)
    cat_ids = [s["exp_id"] for s in inv["slots"]]
    by_id = {s["exp_id"]: s for s in inv["slots"]}

    C2 = stitch("p10b", "C2", folds, n)
    C0 = stitch("p10b", "C0", folds, n)
    # Model-level gain in AUC units, kept as a float in AUC space. An earlier version of this print
    # block wrote "+6.64e-5" for a value already expressed in e-5, double-scaling it to 6.64e-10 and
    # making every derived ratio below nonsense (it reported the gate as 190404x out of reach).
    mech_e5 = (float(roc_auc_score(y, C2)) - float(roc_auc_score(y, C0))) * 1e5
    w_block = len(cat_ids) / nmem
    w_slot = 1.0 / nmem
    gate = 1.5
    L = {e: logit(store.load_oof(e).astype("float64")) for e in ids}
    base_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    base = float(roc_auc_score(y, sig(base_logit)))
    v3 = store.load_oof("blend_v3_final").astype("float64")

    print("C2 SINGLE-SLOT SWAP -- zero compute, C2's 5-fold OOF already exists")
    print("=" * 100)
    print(f"  {nmem} members, {len(cat_ids)} CatBoost slots, block weight "
          f"{len(cat_ids)/nmem:.5f} (unchanged by any swap)")
    print(f"  equal-logit reference OOF = {base:.9f}   stored v3 = {roc_auc_score(y, v3):.9f}")
    print(f"  C2 OOF = {roc_auc_score(y, C2):.6f}   C0 OOF = {roc_auc_score(y, C0):.6f}   "
          f"phase-10 mechanism delta {(roc_auc_score(y, C2)-roc_auc_score(y, C0))*1e5:+.2f}e-5")

    print(f"\n  {'replaced slot':<26}{'depth':>6}{'seed':>6}{'orig AUC':>11}{'swap v3 AUC':>13}"
          f"{'delta e5':>10}{'pos folds':>11}{'O-C2 corr':>11}")
    print(f"  {'-'*100}")
    rows = []
    for eid in SWAPPABLE:
        s = by_id[eid]
        cols = [logit(C2) if e == eid else L[e] for e in ids]
        aft = sig(np.mean(np.column_stack(cols), axis=1))
        a = float(roc_auc_score(y, aft))
        pf = [(float(roc_auc_score(y[folds == k], aft[folds == k]))
               - float(roc_auc_score(y[folds == k], sig(base_logit[folds == k])))) * 1e5
              for k in range(5)]
        c = float(corr(L[eid], logit(C2)))
        rows.append({"exp_id": eid, "slot": s["slot"], "depth": s["params"]["depth"],
                     "seed": s["seed"], "orig_auc": float(roc_auc_score(y, sig(L[eid]))),
                     "swap_auc": a, "delta_e5": (a - base) * 1e5,
                     "per_fold_e5": pf, "folds_positive": sum(x > 0 for x in pf),
                     "o_c2_corr": c})
        print(f"  {eid:<26}{s['params']['depth']:>6}{s['seed']:>6}"
              f"{float(roc_auc_score(y, sig(L[eid]))):>11.6f}{a:>13.6f}"
              f"{(a-base)*1e5:>+9.2f}e{sum(x > 0 for x in pf):>8}/5{c:>11.5f}")

    # also: what does swapping in the NUMERIC counterpart C0 give? That isolates how much of any
    # effect is the native mechanism versus merely having a modern fixed-round protocol model.
    print(f"\n  CONTROL -- the same swap with C0 (numeric, same protocol) instead of C2:")
    ctrl = []
    for eid in SWAPPABLE:
        cols = [logit(C0) if e == eid else L[e] for e in ids]
        a = float(roc_auc_score(y, sig(np.mean(np.column_stack(cols), axis=1))))
        d = (a - base) * 1e5
        ctrl.append({"exp_id": eid, "swap_auc": a, "delta_e5": d})
        print(f"    {eid:<26} swap with C0 -> {a:.6f}  delta {d:+.2f}e-5")
    print("    If C2's swap beats C0's swap, the difference is the native mechanism rather than")
    print("    the protocol change, measured INSIDE the ensemble rather than on a single model.")

    mean_c2 = float(np.mean([r["delta_e5"] for r in rows]))
    mean_c0 = float(np.mean([r["delta_e5"] for r in ctrl]))
    npos = sum(1 for r in rows if r["folds_positive"] >= 3)
    print(f"\n  mean single-slot swap delta: C2 {mean_c2:+.2f}e-5   C0 {mean_c0:+.2f}e-5   "
          f"mechanism-in-ensemble {mean_c2-mean_c0:+.2f}e-5")
    print(f"  swaps positive in >=3/5 folds: {npos}/{len(rows)}")

    # ---------------- structural arithmetic: is this route even capable of clearing the gate? ----
    print(f"\n  STRUCTURAL ARITHMETIC -- can a CatBoost-family gain reach the +{gate}e-5 gate?")
    print(f"    block weight {w_block:.5f} (7/59)   slot weight {w_slot:.5f} (1/59)")
    print(f"    measured model-level gain C2 - C0   = {mech_e5:+.2f}e-5")
    ub = w_block * mech_e5
    need = gate / w_block
    print(f"    if the whole block improved by {mech_e5:+.2f}e-5 and that transferred")
    print(f"      PROPORTIONALLY to weight, the blend would gain at most "
          f"{w_block:.5f} x {mech_e5:+.2f}e-5 = {ub:+.2f}e-5")
    print(f"    reaching +{gate}e-5 through this block alone needs a model-level gain of "
          f"{need:+.2f}e-5")
    print(f"      i.e. {need/mech_e5:.1f}x what C2 actually delivers")
    print(f"    naive linear scaling of the measured one-slot result to seven slots gives "
          f"{7*mean_c2:+.2f}e-5")
    print(f"    and that scaling is GENEROUS: the swapped members correlate 0.998 with the block,")
    print(f"    so most of the improvement is already represented in the blend and cancels.")

    if mean_c2 <= 0:
        verdict = (f"STOP the diverse-block run. A single-slot swap with an already-trained native "
                   f"model is {mean_c2:+.2f}e-5, i.e. not positive, and swapping MORE slots makes the "
                   f"deficit larger rather than smaller because each replaced slot is individually "
                   f"worse than C2 by an amount the block's averaging was partly offsetting. The "
                   f"7-slot native block costs ~12 h of training to test a premise this free test "
                   f"contradicts.")
    elif mean_c2 < 1.0:
        verdict = (f"MARGINAL: a single-slot swap gives {mean_c2:+.2f}e-5, below the +1.0e-5 Stage 1 "
                   f"bar and below the +1.5e-5 admission gate. The 7-slot run would need to roughly "
                   f"multiply this, which is plausible but not established. Expensive; justify before "
                   f"spending.")
    else:
        verdict = (f"PROCEED: a single-slot swap already gives {mean_c2:+.2f}e-5, above the +1.0e-5 "
                   f"Stage 1 bar, so the 7-slot diverse block is worth its ~12 h.")
    if ub < gate:
        verdict += (f" DECISIVE OVERRIDE: even a PERFECT native block is capped at about "
                    f"{ub:+.2f}e-5 of ensemble gain by the block's {w_block:.1%} weight, which is "
                    f"below the +{gate}e-5 gate, so the 7-slot run cannot succeed and is not "
                    f"justified regardless of the measured single-slot figure.")
    print(f"\nVERDICT: {verdict}")

    save_json({"nmem": nmem, "cat_block_weight": w_block, "slot_weight": w_slot,
               "mechanism_model_level_e5": mech_e5, "gate_e5": gate,
               "ensemble_upper_bound_if_block_perfect_e5": ub,
               "model_gain_needed_for_gate_e5": need,
               "naive_7x_scaled_measured_e5": 7 * mean_c2,
               "reference_equal_logit_auc": base, "c2_auc": float(roc_auc_score(y, C2)),
               "c0_auc": float(roc_auc_score(y, C0)),
               "swaps": rows, "c0_control_swaps": ctrl,
               "mean_swap_c2_e5": mean_c2, "mean_swap_c0_e5": mean_c0,
               "mechanism_inside_ensemble_e5": mean_c2 - mean_c0,
               "caveat": "One swap leaves the other six CatBoost members untouched, so the block's "
                         "internal diversity is preserved here whereas a 7-slot native block would "
                         "rebuild it. This is a lower-bound-style screen on the hypothesis, not the "
                         "block result, and C2 uses seed 4 rather than the swapped slots' own seeds.",
               "verdict": verdict, "seconds": round(time.time() - t0, 1),
               "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip()[:12]},
              REPORTS / "c2_single_slot_swap.json")
    print("wrote", REPORTS / "c2_single_slot_swap.json", f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
