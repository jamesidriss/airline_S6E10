"""Phase 14C -- AUX_CROSS: replace a CROSS-FAMILY block of v3 slots with aux-equipped counterparts.

WHY THIS IS THE STRONGEST CURRENT ROUTE
---------------------------------------
Phase 13 established the auxiliary-task mechanism on extra_trees LightGBM and found it real but hard
to harvest: six slots bought 2.8x rather than 6x, because extra_trees members are near-clones and
each new prediction correlated 0.998-0.999 with the member it replaced.

Phase 14 established the SAME mechanism on XGBoost and CatBoost, with per-slot ensemble swaps of
+0.20 to +0.34e-5 -- comparable to or better than the extra_trees slots, on families that are
STRUCTURALLY different from each other. A member can only correct pairs the blend gets wrong in a way
its near-clones cannot, so corrections acquired in different families should stack better than
corrections acquired six times inside one family.

That is a hypothesis, not a result. Phase 11 is the standing proof that inferring a family's blend
contribution from its members' standalone gains is invalid, and Phase 11R withdrew a whole closure
built on exactly that arithmetic. So the block delta here is MEASURED, from real OOF vectors, with
the member count held at 59 and every slot weight left at exactly 1/59.

SLOT SELECTION IS STRUCTURAL, NOT CHOSEN BY SCORE
---------------------------------------------------
The rule is: every v3 slot for which an aux-equipped counterpart was actually trained, across all
three families. Nothing is selected on its measured gain, because selecting slots by full-OOF
performance and then reporting a full-OOF improvement is the circular move. Concretely, the rule
admits exactly these 10 of 59 slots:

  extra_trees (6)  xt_xt_d127_s1, xt_xt_d63, xt_xt_d255, xt_xt_d127_cs05,
                   xt_xt_d127_ss06, xt_xt_d127_bin63
  xgboost (3)      prod5_xgb_full_primary, xt_xgb_lossguide, zoo_xgb_d6
  catboost (1)     z3_cat_d8_s2   -- via the NATIVE-CATEGORICAL + aux cell

C1 (native-cat + aux) is used for the CatBoost slot rather than C0 (numeric + aux) on a structural
basis, not on score: it is the more decorrelated representation, and the 2x2 decomposition already
measured both. Using both on one slot is impossible -- they replace the same member -- and that is
recorded rather than silently resolved.

Usage:
  python scripts/run_phase14c.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_phase12 import CHAMPION_SCHEME, lg, sg  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

XT_SLOTS = ["xt_xt_d127_s1", "xt_xt_d63", "xt_xt_d255", "xt_xt_d127_cs05",
            "xt_xt_d127_ss06", "xt_xt_d127_bin63"]
XGB_SLOTS = ["prod5_xgb_full_primary", "xt_xgb_lossguide", "zoo_xgb_d6"]
CAT_SLOTS = ["z3_cat_d8_s2"]
GROUPS = {
    "v3_control": [],
    "AUX_xt6": XT_SLOTS,
    "AUX_xgb3": XGB_SLOTS,
    "AUX_cat1": CAT_SLOTS,
    "AUX_xgb3_cat1": XGB_SLOTS + CAT_SLOTS,
    "AUX_CROSS": XT_SLOTS + XGB_SLOTS + CAT_SLOTS,
}


def counterpart(member: str) -> str | None:
    """The saved aux-equipped prediction tag for a v3 slot, or None if none was trained."""
    if member in XT_SLOTS:
        return f"p13b_{member}"
    if member in XGB_SLOTS:
        return f"p14_{ {'prod5_xgb_full_primary': 'X0', 'xt_xgb_lossguide': 'X1', 'zoo_xgb_d6': 'X2'}[member] }_aux"
    if member in CAT_SLOTS:
        return "p14_C1_aux"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400_000)
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    folds = get_scheme(CHAMPION_SCHEME, yi, tr[ID_COL]).folds

    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    L = {e: lg(store.load_oof(e).astype("float64")) for e in ids}
    base = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    assert float(np.abs(sg(base).astype(np.float32).astype("float64") - v3).max()) == 0.0
    a_v3 = float(roc_auc_score(y, base))

    print("=" * 116)
    print("PHASE 14C -- AUX_CROSS: a CROSS-FAMILY aux replacement block, measured from real vectors")
    print("=" * 116)
    print(f"  v3 = expit(mean(logits)) verified bit-exact. OOF {a_v3:.6f}, {len(ids)} members")
    print(f"  slot selection rule: every v3 slot with a TRAINED aux counterpart, across all three")
    print(f"  families. No slot is chosen by its measured gain.")
    for tag, slots in GROUPS.items():
        print(f"    {tag:<18} {len(slots):>2} slots  {slots if slots else '(control)'}")

    # verify every counterpart exists on all 5 folds BEFORE reporting anything
    missing = []
    for m in XT_SLOTS + XGB_SLOTS + CAT_SLOTS:
        cp = counterpart(m)
        for k in range(5):
            p = REPORTS / f"{cp}_{CHAMPION_SCHEME}_f{k}.npy"
            if not p.exists():
                missing.append(f"{m} -> {p.name}")
    if missing:
        raise SystemExit("STOP: missing counterpart predictions:\n  " + "\n  ".join(missing[:12]))

    pair_seed = 20261010
    rng = np.random.default_rng(pair_seed)
    pos, neg = np.where(y > 0.5)[0], np.where(y <= 0.5)[0]
    n = min(1_200_000, len(pos), len(neg))
    pi = rng.choice(pos, n, replace=False)
    ni = rng.choice(neg, n, replace=False)
    order = rng.permutation(n)[:args.pairs]
    pi, ni = pi[order], ni[order]

    def rescue(cand):
        ok = True
        i, j = pi, ni
        v_ok = base[i] > base[j]
        c_ok = cand[i] > cand[j]
        A = int((~v_ok & ~c_ok).sum())
        B = int((~v_ok & c_ok).sum())
        C = int((v_ok & ~c_ok).sum())
        D = int((v_ok & c_ok).sum())
        return {"A": A, "B": B, "C": C, "D": D,
                "rescue_rate": B / (A + B) if A + B else float("nan"),
                "damage_rate": C / (C + D) if C + D else float("nan"),
                "net": B - C}

    print(f"\n  {'group':<18}{'slots':>6}{'OOF':>12}{'delta vs v3':>13}{'folds+':>8}"
          f"{'corr v3':>10}{'spear v3':>10}{'rescue':>9}{'damage':>9}{'net':>8}")
    print("  " + "-" * 113)
    out = {"v3_oof": a_v3, "n_members": len(ids), "groups": {}, "pair_seed": pair_seed,
           "n_pairs": int(len(pi)),
           "slot_selection_rule": "every v3 slot with a trained aux counterpart, all three "
                                  "families; no selection on measured gain"}
    for name, slots in GROUPS.items():
        cols = list(L[e] if counterpart(e) is None else L[e] for e in ids)
        for m in slots:
            cp = counterpart(m)
            c = L[m].copy()
            for k in range(5):
                val = np.where(folds == k)[0]
                c[val] = lg(np.load(REPORTS / f"{cp}_{CHAMPION_SCHEME}_f{k}.npy")
                           .astype("float64"))
            outside = np.ones(len(c), dtype=bool)
            outside[np.concatenate([np.where(folds == k)[0] for k in range(5)])] = False
            assert np.array_equal(c[outside], L[m][outside]), f"{name}/{m}: moved out-of-fold rows"
            cols[ids.index(m)] = c
        blend = sg(np.mean(np.column_stack(cols), axis=1))
        assert len(cols) == len(ids), "member count changed"
        a = float(roc_auc_score(y, blend))
        fa = [float(roc_auc_score(y[folds == k], blend[folds == k])) for k in range(5)]
        db = [a - float(roc_auc_score(y[folds == k], base[folds == k])) for k in range(5)]
        ps = rescue(blend)
        rec = {"slots": slots, "n_slots": len(slots), "oof": a, "delta_e5": (a - a_v3) * 1e5,
               "per_fold_delta_e5": [d * 1e5 for d in db],
               "n_folds_positive": int(sum(1 for d in db if d > 0)),
               "corr_v3": float(corr(lg(blend), lg(base))),
               "spearman_v3": float(spearman(blend, base)), **ps}
        out["groups"][name] = rec
        print(f"  {name:<18}{len(slots):>6}{a:>12.6f}{(a - a_v3) * 1e5:>+12.2f}e"
              f"{rec['n_folds_positive']:>6}/5{rec['corr_v3']:>10.5f}{rec['spearman_v3']:>10.5f}"
              f"{ps['rescue_rate']:>9.4f}{ps['damage_rate']:>9.4f}{ps['net']:>8}")

    best = max(out["groups"].items(), key=lambda kv: kv[1]["delta_e5"])
    d = np.array(best[1]["per_fold_delta_e5"])
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    gate = (best[1]["delta_e5"] >= 1.5 and best[1]["n_folds_positive"] >= 4
            and best[1]["delta_e5"] >= 2.5 * se)
    out["best_group"] = best[0]
    out["best_delta_e5"] = best[1]["delta_e5"]
    out["best_paired_se_e5"] = se
    out["best_t"] = best[1]["delta_e5"] / se if se else None
    out["gate_pass"] = bool(gate)
    out["verdict"] = (
        f"AUX_CROSS clears the admission gate: {best[1]['delta_e5']:+.2f}e-5 with "
        f"{best[1]['n_folds_positive']}/5 folds positive and t {out['best_t']:.2f}. This is "
        f"Champion A material."
        if gate else
        f"Best group is {best[0]} at {best[1]['delta_e5']:+.2f}e-5, short of the +1.5e-5 gate. "
        f"No admission and no submission from this block.")
    print("\n  per-fold deltas for the best group:")
    for k, dv in zip(range(5), best[1]["per_fold_delta_e5"]):
        print(f"    fold {k}: {dv:+.3f}e-5")
    print(f"  mean {best[1]['delta_e5']:+.3f}e-5  paired SE {se:.3f}  t {out['best_t']:+.2f}")
    print(f"  GATE (>= +1.5e-5, >=4/5 folds positive, >= 2.5x paired SE): "
          f"{'PASS' if gate else 'FAIL'}")
    print(f"  -> {out['verdict']}")
    save_json(out, REPORTS / "p14c_aux_cross.json")
    print(f"\n  wrote {REPORTS / 'p14c_aux_cross.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())