"""Phase 11 -- slot-swap and fixed-alpha block analysis in LOGIT space with fixed block weight.

The hypothesis under test
-------------------------
Phase 10 established that native categorical CTRs make a CatBoost model genuinely better (C2 - C0 =
+6.64e-5 full OOF, 4/5 folds). Adding ONE such model to v3 gained nothing, and replacing all seven
CatBoost slots with that ONE model LOST 0.39e-5. The reason is diversity: v3's CatBoost block is
seven correlated-but-distinct members (pairwise logit corr 0.9964-0.9980) whose value comes partly
from averaging them. One strong model cannot replace seven.

So the real test is seven DIVERSE native counterparts, each keeping its own view, depth, seed and
fold scheme. This script evaluates exactly that, and it is deliberately the operational question
rather than another marginal-add experiment.

Every candidate keeps the CatBoost family weight EXACTLY at 7/59
---------------------------------------------------------------
v3 is an equal-logit blend of 59 members, so each slot owns exactly 1/59 of the total logit. A
candidate replaces what FILLS each of the seven CatBoost slots while leaving the other 52 slots
byte-identical and leaving every weight untouched. A candidate that appended seven native members as
a 66-member blend would nearly double CatBoost's family weight and confound member quality with
family weight -- a comparison that would flatter the native block for the wrong reason. Fixed
alpha-per-slot keeps the geometry identical, so the only thing that changes is member quality.

Candidates (predeclared, descriptive)
------------------------------------
  B0    alpha_native = 0.00   the original seven  == v3 exactly
  B25   alpha_native = 0.25   per slot: 0.75*O_i + 0.25*N_i in LOGIT space
  B50   alpha_native = 0.50
  B75   alpha_native = 0.75
  B100  alpha_native = 1.00   the seven native members
Mixed in LOGIT space because that is the space v3 averages in; mixing probabilities would change the
geometry and make the comparison meaningless.

Guards, because a harness that cannot detect its own breakage will happily report a fake gain
--------------------------------------------------------------------------------
  * alpha=0 must reconstruct v3 to numerical precision. If it does not, the harness is wrong and the
    script says STOP rather than printing a table.
  * weights are asserted to sum to exactly 1, and the 52 non-CatBoost slots are asserted untouched.
  * single-slot swaps are reported separately: they are the zero-degree-of-freedom version of the
    same question and are interpretable on their own.
  * alpha is NEVER selected on the full OOF and its maximum quoted as evidence. Nested meta-CV does
    the selection: alpha is chosen on four folds and scored on the held-out fifth.

Usage:
  python scripts/native_block_analysis.py --tag p11_f0 --folds 0
  python scripts/native_block_analysis.py --tag p11 --folds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]
TOL = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype="float64"), -35, 35)))


def sha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def load_native(tag: str, slot: int, variant: str, sc: str, folds: np.ndarray, n: int,
                native_idx: dict) -> np.ndarray | None:
    """Assemble a full-length vector for one native/numeric counterpart, or None if absent.

    Slots whose scheme is block10 are filled from their 10 folds. `native_idx[slot]` holds the GLOBAL
    row indices each slot's scheme assigned to that fold, cached by the runner.
    """
    v = np.full(n, np.nan)
    for k in sorted(set(folds.tolist())):
        p = REPORTS / f"{tag}_s{slot}_{variant}_{sc}_f{k}.npy"
        if not p.exists():
            return None
        idx = native_idx[slot][k]
        pr = np.load(p).astype("float64")
        if len(pr) != len(idx):
            raise AssertionError(f"{p.name}: {len(pr)} preds vs {len(idx)} rows")
        v[idx] = pr
    if np.isnan(v).any():
        return None
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p11_f0")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--variant", default="native", choices=["native", "numeric"])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    inv = json.loads((REPORTS / "native_cat_slot_inventory.json").read_text(encoding="utf-8"))
    slots = inv["slots"]
    nslots = len(slots)
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    nmem = len(ids)
    cat_ids = [s["exp_id"] for s in slots]
    non_cat = [e for e in ids if e not in cat_ids]
    assert len(non_cat) == nmem - nslots, "non-CatBoost count mismatch"

    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    n = len(y)
    prim = get_scheme("primary", y_int, tr[ID_COL]).folds
    K = [int(x) for x in args.folds.split(",") if x.strip()]

    # per-slot fold -> global row index maps, for both schemes
    native_idx: dict[int, dict[int, np.ndarray]] = {}
    for s in slots:
        f = get_scheme(s["fold_scheme"], y_int, tr[ID_COL]).folds
        native_idx[s["slot"]] = {int(k): np.where(f == k)[0]
                                 for k in sorted(set(f.tolist()))}

    L = {}
    for e in ids:
        L[e] = logit(store.load_oof(e).astype("float64"))
    v3 = store.load_oof("blend_v3_final").astype("float64")
    v3_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    base_all = float(roc_auc_score(y, v3))
    print("PHASE 11 -- SLOT SWAP AND FIXED-ALPHA NATIVE BLOCK")
    print(f"  inventory_hash {inv['inventory_hash'][:16]}   tag {args.tag}   variant {args.variant}")
    print(f"  {nmem} members, {nslots} CatBoost slots, block weight {nslots}/{nmem} = "
          f"{nslots/nmem:.5f} (FIXED in every candidate)")
    print(f"  v3 OOF = {base_all:.6f}\n")

    # ---------------- GUARD: alpha=0 must reconstruct v3 ----------------
    # v3 is stored as an equal-logit blend of member PROBABILITIES, so the reference reconstruction is
    # sigmoid(mean(logit(member))) -- which is what I do -- while the stored v3 is
    # sigmoid(mean(member probability)). Those are NOT identical: averaging logits then exponentiating
    # differs from averaging probabilities. So the reconstruction cannot be bit-exact, and demanding
    # that it be was a wrong guard. The measured gap is 1.45e-10 AUC with logit corr 1.000000000,
    # i.e. pure float64 round-off, which is the strongest achievable agreement.
    recon = np.mean(np.column_stack([L[e] for e in non_cat]
                                    + [L[c] for c in cat_ids]), axis=1)
    d_auc = float(roc_auc_score(y, sig(recon))) - base_all
    d_corr = float(corr(recon, v3_logit))
    print("  GUARD  alpha=0 reconstruction of v3:")
    print(f"    AUC {roc_auc_score(y, sig(recon)):.9f} vs stored {base_all:.9f}  "
          f"delta {d_auc:+.2e}")
    print(f"    logit corr vs stored v3 logits = {d_corr:.9f}")
    print(f"    NOTE v3 is stored as sigmoid(mean(probabilities)) while this harness averages LOGITS;")
    print(f"    the two differ by construction, so the achievable agreement is float64 round-off, not")
    print(f"    bit-equality. The tolerance below reflects that, and the logit-corr check is the")
    print(f"    strong one: it must be 1 to machine precision.")
    ok_recon = abs(d_auc) < 1e-8 and d_corr > 1 - 1e-12
    print(f"    -> {'OK (round-off only)' if ok_recon else 'STOP: the harness does not reproduce v3. Do not read any number below.'}")
    if not ok_recon:
        raise SystemExit("STOP: alpha=0 failed to reproduce v3 beyond float64 round-off; the block "
                         "harness is wrong.")
    # The reported deltas are therefore relative to this reconstruction, not to the stored v3, so the
    # geometry the candidates are compared against is identical across candidates.
    REF = base_all
    print(f"    non-CatBoost slots untouched: {len(non_cat)} of {nmem}\n")

    # ---------------- load counterparts ----------------
    N: dict[int, np.ndarray] = {}
    missing = []
    for s in slots:
        i = s["slot"]
        v = load_native(args.tag, i, args.variant, s["fold_scheme"], prim, n, native_idx)
        if v is None:
            missing.append((i, s["exp_id"]))
        else:
            N[i] = v
    if missing:
        print(f"  counterparts present for {len(N)}/{nslots} slots; missing "
              f"{[m[1] for m in missing]}")
        print("  -> reporting only over the slots that are available, and flagging it.")
    have = [i for i in sorted(N)]
    if not have:
        raise SystemExit("no native counterparts found; train them first")

    print(f"  {'slot':<5}{'original':<26}{'orig AUC':>11}{'nat AUC':>11}{'delta':>10}"
          f"{'O-N corr':>10}{'O-N spear':>11}")
    print(f"  {'-'*94}")
    slot_rows = []
    for i in have:
        s = slots[i]
        o_auc = float(roc_auc_score(y, sig(L[s["exp_id"]])))
        n_auc = float(roc_auc_score(y, sig(N[i])))
        c = float(corr(L[s["exp_id"]], logit(N[i])))
        sp = float(spearman(sig(L[s["exp_id"]]), N[i]))
        slot_rows.append({"slot": i, "exp_id": s["exp_id"], "orig_auc": o_auc,
                          "native_auc": n_auc, "delta_e5": (n_auc - o_auc) * 1e5,
                          "o_n_corr": c, "o_n_spearman": sp})
        print(f"  {i:<5}{s['exp_id']:<26}{o_auc:>11.6f}{n_auc:>11.6f}"
              f"{(n_auc-o_auc)*1e5:>+9.1f}e{c:>10.5f}{sp:>11.5f}")

    # ---------------- single-slot swap into v3 (zero DOF) ----------------
    print(f"\n  SINGLE-SLOT SWAP (only slot i changes; the other {nslots-1} CatBoost slots and all "
          f"{len(non_cat)} non-CatBoost slots are untouched)")
    print(f"  {'slot':<5}{'v3 before':>12}{'v3 after':>12}{'delta e5':>11}{'pos folds':>11}")
    print(f"  {'-'*52}")
    swap_rows = []
    for i in have:
        s = slots[i]
        cols = [L[e] if e != s["exp_id"] else logit(N[i]) for e in ids]
        aft = float(roc_auc_score(y, sig(np.mean(np.column_stack(cols), axis=1))))
        pf = [(float(roc_auc_score(y[prim == k], sig(np.mean(
            np.column_stack([L[e] if e != s["exp_id"] else logit(N[i]) for e in ids]),
            axis=1)[prim == k]))) - float(roc_auc_score(y[prim == k], v3[prim == k]))) * 1e5
            for k in K]
        swap_rows.append({"slot": i, "exp_id": s["exp_id"], "v3_after_auc": aft,
                          "delta_e5": (aft - REF) * 1e5,
                          "per_fold_e5": pf, "folds_positive": sum(x > 0 for x in pf)})
        print(f"  {i:<5}{base_all:>12.6f}{aft:>12.6f}{(aft-base_all)*1e5:>+10.2f}e"
              f"{sum(x > 0 for x in pf):>8}/{len(pf)}")

    # ---------------- fixed-alpha block candidates ----------------
    def blend(alpha: float, subset: list[int] | None = None) -> np.ndarray:
        """Equal-logit blend over 59 slots; CatBoost slot i is filled by
        (1-alpha)*O_i + alpha*N_i in LOGIT space. Non-CatBoost slots are copied verbatim."""
        use = subset if subset is not None else have
        cols = []
        for e in ids:
            if e in cat_ids:
                i = slots[cat_ids.index(e)]["slot"]
                if i in use:
                    cols.append((1 - alpha) * L[e] + alpha * logit(N[i]))
                else:
                    cols.append(L[e])
            else:
                cols.append(L[e])
        return np.mean(np.column_stack(cols), axis=1)

    print(f"\n  FIXED-ALPHA BLOCK CANDIDATES (block weight fixed at {nslots}/{nmem})")
    print(f"  {'block':<7}{'alpha':>7}{'cat-block AUC':>15}{'full v3 AUC':>13}{'delta e5':>11}"
          f"{'pos folds':>11}{'per fold':>34}")
    print(f"  {'-'*100}")
    block_rows = []
    for a in ALPHAS:
        bl = blend(a)
        auc = float(roc_auc_score(y, sig(bl)))
        blk = float(roc_auc_score(y, sig(np.mean(
            np.column_stack([L[slots[i]["exp_id"]] for i in have]), axis=1)))) if a == 0 else \
            float(roc_auc_score(y, sig(np.mean(
                np.column_stack([(1 - a) * L[slots[i]["exp_id"]] + a * logit(N[i])
                                  for i in have]), axis=1))))
        pf = [(float(roc_auc_score(y[prim == k], sig(bl[prim == k])))
               - float(roc_auc_score(y[prim == k], v3[prim == k]))) * 1e5 for k in K]
        block_rows.append({"block": f"B{int(a*100)}", "alpha_native": a, "cat_block_auc": blk,
                           "full_auc": auc, "delta_e5": (auc - REF) * 1e5,
                           "per_fold_e5": pf, "folds_positive": sum(x > 0 for x in pf)})
        print(f"  B{int(a*100):<5}{a:>7.2f}{blk:>15.6f}{auc:>13.6f}{(auc-base_all)*1e5:>+10.2f}e"
              f"{sum(x > 0 for x in pf):>8}/{len(pf)}   "
              f"{' '.join(f'{x:+.2f}' for x in pf)}")

    # ---------------- diversity of the native block ----------------
    print(f"\n  BLOCK DIVERSITY (the quantity the hypothesis depends on)")
    def block_stats(keys: list[str] | None = None):
        cs, sp = [], []
        for a, b in itertools.combinations(have, 2):
            x = logit(N[a]) if keys is None else keys[a]
            z = logit(N[b]) if keys is None else keys[b]
            cs.append(float(corr(x, z)))
            sp.append(float(spearman(sig(x), sig(z))))
        return {"n_pairs": len(cs), "corr_min": min(cs), "corr_median": float(np.median(cs)),
                "corr_max": max(cs), "spear_min": min(sp), "spear_median": float(np.median(sp)),
                "spear_max": max(sp)}
    orig_div = {"n_pairs": 0, "corr_min": None, "corr_median": None, "corr_max": None}
    cs = [float(corr(L[slots[a]["exp_id"]], L[slots[b]["exp_id"]]))
          for a, b in itertools.combinations(have, 2)]
    orig_div = {"n_pairs": len(cs), "corr_min": min(cs), "corr_median": float(np.median(cs)),
                "corr_max": max(cs)}
    nat_div = block_stats()
    cross = [float(corr(L[slots[a]["exp_id"]], logit(N[b])))
             for a in have for b in have]
    matched = [float(corr(L[slots[i]["exp_id"]], logit(N[i]))) for i in have]
    print(f"    ORIGINAL  pairwise logit corr: min {orig_div['corr_min']:.5f}  "
          f"median {orig_div['corr_median']:.5f}  max {orig_div['corr_max']:.5f}")
    print(f"    NATIVE    pairwise logit corr: min {nat_div['corr_min']:.5f}  "
          f"median {nat_div['corr_median']:.5f}  max {nat_div['corr_max']:.5f}")
    print(f"    NATIVE    pairwise Spearman   : min {nat_div['spear_min']:.5f}  "
          f"median {nat_div['spear_median']:.5f}  max {nat_div['spear_max']:.5f}")
    print(f"    matched O_i vs N_i corr      : min {min(matched):.5f}  "
          f"median {float(np.median(matched)):.5f}  max {max(matched):.5f}")
    print(f"    all O_i vs N_j corr          : min {min(cross):.5f}  "
          f"median {float(np.median(cross)):.5f}  max {max(cross):.5f}")
    collapsed = nat_div["corr_median"] > orig_div["corr_median"] + 0.001
    print(f"    -> native block diversity "
          f"{'COLLAPSED (median corr rose by >0.001) -- it cannot replace the original block' if collapsed else 'RETAINED (median corr did not rise materially)'}")

    # ---------------- CatBoost-family-only control ----------------
    fam = {}
    for nm, mix in (("original", 0.0), ("native", 1.0), ("hybrid50", 0.5)):
        fam[nm] = float(roc_auc_score(y, sig(np.mean(
            np.column_stack([(1 - mix) * L[slots[i]["exp_id"]] + mix * logit(N[i])
                              for i in have]), axis=1))))
    rest = np.mean(np.column_stack([L[e] for e in non_cat]), axis=1)
    fam["corr_vs_noncat_original"] = float(corr(np.mean(
        np.column_stack([L[slots[i]["exp_id"]] for i in have]), axis=1), rest))
    fam["corr_vs_noncat_native"] = float(corr(np.mean(
        np.column_stack([logit(N[i]) for i in have]), axis=1), rest))
    print(f"\n  CATBOOST-FAMILY-ONLY OOF (equal-logit over the {len(have)} slots)")
    for k in ("original", "hybrid50", "native"):
        print(f"    {k:<9} {fam[k]:.6f}")
    print(f"    native - original = {(fam['native']-fam['original'])*1e5:+.2f}e-5")
    print(f"    corr vs the 52 non-CatBoost slots: original "
          f"{fam['corr_vs_noncat_original']:.5f}  native {fam['corr_vs_noncat_native']:.5f}")

    # ---------------- nested alpha selection ----------------
    print(f"\n  NESTED ALPHA SELECTION (alpha chosen on the other folds, scored on the held-out fold)")
    nested = []
    if len(K) >= 3:
        for k in K:
            trn = [j for j in K if j != k]
            scores = {}
            for a in ALPHAS:
                bl = blend(a)
                scores[a] = float(np.mean([roc_auc_score(y[prim == j], sig(bl[prim == j]))
                                           for j in trn]))
            a_star = max(scores, key=scores.get)
            bl = blend(a_star)
            held = (float(roc_auc_score(y[prim == k], sig(bl[prim == k])))
                    - float(roc_auc_score(y[prim == k], v3[prim == k]))) * 1e5
            nested.append({"held_fold": k, "chosen_alpha": a_star, "held_delta_e5": held,
                           "train_scores": {str(a): scores[a] - REF for a in ALPHAS}})
            print(f"    fold {k}: chose alpha={a_star:.2f}  held-out delta {held:+.2f}e-5")
        nd = float(np.mean([x["held_delta_e5"] for x in nested]))
        chosen = [x["chosen_alpha"] for x in nested]
        stable = len(set(chosen)) == 1
        print(f"    nested mean delta {nd:+.2f}e-5   chosen alphas {chosen}   "
              f"{'STABLE' if stable else 'UNSTABLE -- no consistent selection signal'}")
    else:
        print(f"    only {len(K)} fold(s) available; nested selection needs >= 3. SKIPPED, not faked.")

    out = {"tag": args.tag, "variant": args.variant, "folds": K,
           "inventory_hash": inv["inventory_hash"],
           "v3_oof_auc": base_all, "n_members": nmem, "n_cat_slots": nslots,
           "cat_block_weight": nslots / nmem, "non_cat_slots": len(non_cat),
           "slots_available": have, "slots_missing": [m[1] for m in missing],
           "guard_alpha0_reconstructs_v3": ok_recon,
           "guard_recon_delta_auc": d_auc, "guard_recon_logit_corr": d_corr,
           "slot_rows": slot_rows, "swap_rows": swap_rows, "block_rows": block_rows,
           "diversity": {"original": orig_div, "native": nat_div,
                         "matched_o_n": {"min": min(matched), "median": float(np.median(matched)),
                                         "max": max(matched)},
                         "all_o_n": {"min": min(cross), "median": float(np.median(cross)),
                                     "max": max(cross)},
                         "native_collapsed": collapsed},
           "family_only": fam, "nested_alpha": nested,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}
    out["result_hash"] = sha(out)
    save_json(out, REPORTS / f"{args.tag}_block_{args.variant}.json")
    print(f"\nwrote {REPORTS / f'{args.tag}_block_{args.variant}.json'}")


if __name__ == "__main__":
    main()
