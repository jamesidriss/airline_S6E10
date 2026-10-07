"""Private-hedge construction and the pair-complementarity metric.

WHY THIS EXISTS
---------------
Public is 20% of test; private is 80%. A second finalist that is a near-clone of the first buys
almost nothing, which is why v4 is NOT automatically the hedge: its prediction geometry is extremely
close to v3's. A hedge is judged on how many of the pairs the champion gets WRONG it rescues, and how
many it breaks -- not on correlation and not on AUC.

THE METRIC, and why correlation is the wrong tool
-------------------------------------------------
For a fixed sample of positive-negative row pairs, split the candidates' correctness into

    A = both wrong
    B = v3 wrong, candidate correct      <- RESCUED
    C = v3 correct, candidate wrong      <- DAMAGED
    D = both correct

    rescue_rate = B / (A + B)      of the pairs v3 gets wrong, the share the candidate fixes
    damage_rate = C / (C + D)      of the pairs v3 gets right, the share the candidate breaks

rescue_rate is the number that matters for a hedge, and damage_rate is the price. A candidate can be
strongly correlated with v3 and still rescue a lot, because it only has to fix SOME of the wrongly
ordered pairs, not be different everywhere.

The pair sample is drawn ONCE from a fixed seed, persisted, and hashed. It is evaluation only -- no
target enters any feature or model through this module.

NESTED SELECTION, so the frontier is honest
-------------------------------------------
H2 selects a blend on TRAINING folds subject to a correlation cap, then reports it on the HELD-OUT
fold. Selecting and reporting on the same rows would produce a frontier that is optimistic by
construction, which is the same error as reading a full-OOF-tuned alpha as an honest result. The
correlation caps are fixed at 0.999 / 0.997 / 0.995 BEFORE any is evaluated, and family weights live
on a coarse 0.10 simplex, never as 59 free parameters.

Usage:
  python scripts/private_hedge.py --report
  python scripts/private_hedge.py --nested
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PAIR_SEED = 20261010
N_PAIRS = 400_000
CORR_CAPS = (0.999, 0.997, 0.995)      # predeclared before evaluation
# A SMALL PREDECLARED SET of family-weight vectors, not an enumerated simplex. My first version
# swept itertools.product over an 11-level grid in 6 dimensions -- 1.77e6 weight vectors per cap per
# fold -- which is not a heavy computation, it is an impossible one, and it timed out after an hour
# having reported nothing about the hedges that mattered. The brief asks for "a small fixed set" and
# "at most 5-7 family components", which is what this is. Every vector is stated before evaluation
# and none is chosen by looking at a held-out fold.
CANDIDATE_WEIGHTS = {
    "all6_equal":        None,   # filled in as equal weight over every family present
    "no_tabm":           None,   # equal weight, tabm dropped
    "no_tabm_no_cat":    None,
    "xt_heavy":          (0.40, 0.10, 0.10, 0.10, 0.20, 0.10),
    "xt_heavy_no_tabm":  (0.40, 0.12, 0.12, 0.12, 0.24, 0.00),
    "xt_half":           (0.50, 0.10, 0.10, 0.10, 0.10, 0.10),
    "xt_third":          (0.34, 0.11, 0.11, 0.11, 0.22, 0.11),
    "balanced5":         (0.25, 0.15, 0.15, 0.15, 0.20, 0.10),
    "det_heavy":         (0.25, 0.25, 0.20, 0.10, 0.10, 0.10),
    "neural_heavy":      (0.25, 0.10, 0.10, 0.15, 0.30, 0.10),
}


def lg(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sg(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype="float64"), -35, 35)))


def families(ids: list[str]) -> dict:
    """Group v3 members into families. Family MEANS are the hedge primitives, not raw members.

    v3 is population-weighted, so 24 of 59 slots are extra_trees LightGBM and the LightGBM families
    together dominate. Averaging within family first, then weighting families, changes the geometry:
    a family mean is one vote regardless of how many members it contributed. That is the single
    cheapest way to buy structural diversity, and S17's near-identical result was with the OLD
    member-weighted construction, so it has to be re-verified rather than assumed.
    """
    meta = {}
    for line in Path("experiments/ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d.get("exp_id"):
                meta[d["exp_id"]] = d
    out: dict[str, list[str]] = {}
    for e in ids:
        d = meta.get(e, {})
        fam = d.get("family", "?")
        if fam == "lgbm":
            fam = "lgbm_xt" if (d.get("params") or {}).get("extra_trees") else "lgbm_det"
        out.setdefault(fam, []).append(e)
    return out


def pair_stats(cand: np.ndarray, v3: np.ndarray, y: np.ndarray, pairs) -> dict:
    """A/B/C/D over a FIXED positive-negative pair sample."""
    i, j = pairs
    yi, yj = y[i], y[j]
    ok = yi != yj
    i, j = i[ok], j[ok]
    pos, neg = (yi == 1), (yi == 0)
    pi, ni = i[pos], j[pos]                     # pi positive, ni negative
    v_ok = v3[pi] > v3[ni]                      # v3 ranks the positive above the negative
    c_ok = cand[pi] > cand[ni]
    A = int((~v_ok & ~c_ok).sum())
    B = int((~v_ok & c_ok).sum())
    C = int((v_ok & ~c_ok).sum())
    D = int((v_ok & c_ok).sum())
    rescue = B / (A + B) if (A + B) else float("nan")
    damage = C / (C + D) if (C + D) else float("nan")
    return {"A_both_wrong": A, "B_rescued": B, "C_damaged": C, "D_both_right": D,
            "n_pairs": int(A + B + C + D),
            "rescue_rate": rescue, "damage_rate": damage,
            "net_rescued": B - C,
            "agreement": (B + D) / max(A + B + C + D, 1)}


def make_pairs(y: np.ndarray, n_pos=1_200_000, seed=PAIR_SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y > 0.5)[0]
    neg = np.where(y <= 0.5)[0]
    n = min(n_pos, len(pos), len(neg))
    pi = rng.choice(pos, n, replace=False)
    ni = rng.choice(neg, n, replace=False)
    idx = np.arange(n)
    rng.shuffle(idx)
    take = min(N_PAIRS, n)
    return pi[idx[:take]], ni[idx[:take]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--nested", action="store_true")
    args = ap.parse_args()

    tr, _ = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", yi, tr[ID_COL]).folds

    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    L = {e: lg(store.load_oof(e).astype("float64")) for e in ids}
    base = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    assert float(np.abs(sg(base).astype(np.float32).astype("float64") - v3).max()) == 0.0
    a_v3 = float(roc_auc_score(y, base))
    fam = families(ids)
    print("=" * 110)
    print("PRIVATE HEDGE CONSTRUCTION -- family-level geometry and pair complementarity")
    print("=" * 110)
    print(f"  v3 = expit(mean(logits)) verified bit-exact. OOF {a_v3:.6f}, {len(ids)} members")
    print("  family composition of v3:")
    for k, v in sorted(fam.items(), key=lambda kv: -len(kv[1])):
        print(f"    {k:<12} {len(v):>3} slots")
    pairs = make_pairs(y)
    import hashlib
    ph = hashlib.sha256(np.stack(pairs).tobytes()).hexdigest()[:16]
    print(f"  pair sample: {len(pairs[0])} pairs, seed {PAIR_SEED}, sha16 {ph} (evaluation only)")
    out = {"v3_oof": a_v3, "pair_seed": PAIR_SEED, "pair_sha16": ph,
           "n_pairs": len(pairs[0]), "corr_caps": list(CORR_CAPS),
           "families": {k: v for k, v in fam.items()}}

    # ---------------- family means: the hedge primitives ----------------------------------
    FM = {k: sg(np.mean(np.column_stack([L[e] for e in v]), axis=1)) for k, v in fam.items()}
    print("\n  FAMILY MEANS (one vote per family, regardless of member count)")
    print(f"  {'family':<12}{'n':>4}{'OOF':>12}{'gap vs v3':>12}{'corr v3':>10}{'spear v3':>10}")
    for k in sorted(FM, key=lambda k: -roc_auc_score(y, FM[k])):
        a = float(roc_auc_score(y, FM[k]))
        print(f"  {k:<12}{len(fam[k]):>4}{a:>12.6f}{(a - a_v3) * 1e5:>+11.1f}e"
              f"{float(corr(lg(FM[k]), lg(base))):>10.5f}{float(spearman(FM[k], base)):>10.5f}")

    # ---------------- H0: drop the dominant extra_trees family ---------------------------
    nonxt = [k for k in FM if k != "lgbm_xt"]
    h0 = sg(np.mean(np.column_stack([lg(FM[k]) for k in nonxt]), axis=1))
    # ---------------- H1: equal weight per strong family, all families ---------------------
    h1 = sg(np.mean(np.column_stack([lg(FM[k]) for k in FM]), axis=1))
    cands = {"v3": base, "H0_nonXT": h0, "H1_family_diverse": h1}
    print("\n  PREDECLARED HEDGE CANDIDATES")
    print(f"  {'candidate':<20}{'OOF':>12}{'gap vs v3':>12}{'fold sd':>10}{'corr v3':>10}"
          f"{'spear v3':>10}{'rescue':>9}{'damage':>9}{'net':>9}")
    for name, c in cands.items():
        a = float(roc_auc_score(y, c))
        fa = [float(roc_auc_score(y[folds == k], c[folds == k])) for k in range(5)]
        ps = pair_stats(c, base, y, pairs)
        out[name] = {"oof": a, "gap_e5": (a - a_v3) * 1e5, "fold_aucs": fa,
                     "fold_sd": float(np.std(fa, ddof=1)),
                     "corr_v3": float(corr(lg(c), lg(base))),
                     "spearman_v3": float(spearman(c, base)), **ps}
        print(f"  {name:<20}{a:>12.6f}{(a - a_v3) * 1e5:>+11.1f}e{out[name]['fold_sd']:>10.5f}"
              f"{out[name]['corr_v3']:>10.5f}{out[name]['spearman_v3']:>10.5f}"
              f"{ps['rescue_rate']:>9.4f}{ps['damage_rate']:>9.4f}{ps['net_rescued']:>9}")

    # ---------------- H2: nested, correlation-capped family selection ----------------------
    print("\n  H2 -- NESTED correlation-capped family selection (caps predeclared above)")
    print("  selection on 4 training folds, reported on the held-out fold. No full-OOF tuning.")
    keys = sorted(FM)
    print(f"    family order for the weight vectors: {keys}")
    # Materialise the three data-independent members of the candidate set.
    CANDIDATE_WEIGHTS["all6_equal"] = tuple(1.0 / len(keys) for _ in keys)
    for drop in ("tabm", "cat"):
        CANDIDATE_WEIGHTS["no_tabm" if drop == "tabm" else "no_tabm_no_cat"] = tuple(
            (1.0 / (len(keys) - 1)) if k != drop else 0.0 for k in keys)
    rows = []
    for held in range(5):
        tr_mask = folds != held
        te_mask = folds == held
        best = None
        for cap in CORR_CAPS:
            for name, wv in CANDIDATE_WEIGHTS.items():
                if wv is None or len(wv) != len(keys):
                    continue
                w = np.array(wv, dtype="float64")
                s = w.sum()
                if s <= 0:
                    continue
                w = w / s
                c = sg(np.column_stack([lg(FM[k]) for k in keys]) @ w)
                cc = float(corr(lg(c[tr_mask]), lg(base[tr_mask])))
                if cc > cap:
                    continue
                at = float(roc_auc_score(y[tr_mask], c[tr_mask]))
                if best is None or at > best[0]:
                    best = (at, cap, cc, w.copy(), name)
        if best is None:
            print(f"    fold {held}: no candidate satisfies even the loosest cap 0.999")
            continue
        at, cap, cc, w, nm = best
        c = sg(np.column_stack([lg(FM[k]) for k in keys]) @ w)
        ae = float(roc_auc_score(y[te_mask], c[te_mask]))
        a_v3e = float(roc_auc_score(y[te_mask], base[te_mask]))
        nz = {k: round(float(x), 3) for k, x in zip(keys, w) if x > 0}
        print(f"    fold {held}: picked {nm:<16} cap {cap}  train AUC {at:.6f}  "
              f"held-out {ae:.6f}  (v3 {a_v3e:.6f}, {((ae - a_v3e) * 1e5):+.1f}e-5)  {nz}")
        rows.append({"held": held, "picked": nm, "cap": cap, "train_auc": at,
                     "heldout_auc": ae, "v3_heldout": a_v3e, "delta_e5": (ae - a_v3e) * 1e5,
                     "train_corr": cc, "weights": nz})
    if rows:
        d = np.array([r["delta_e5"] for r in rows])
        print(f"    nested mean delta vs v3 on held-out folds: {d.mean():+.2f}e-5, "
              f"{(d > 0).sum()}/{len(d)} positive")
        out["h2_nested"] = rows
        out["h2_nested_mean_delta_e5"] = float(d.mean())

    save_json(out, REPORTS / "private_hedge.json")
    print("\n  wrote", REPORTS / "private_hedge.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())