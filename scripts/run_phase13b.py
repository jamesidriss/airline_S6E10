"""Phase 13B -- does the auxiliary-task gain COMPOUND across extra_trees slots?

WHY THIS QUESTION, AND WHY IT IS THE ONLY ONE LEFT
--------------------------------------------------
Phase 13 measured the mechanism on ONE extra_trees slot. The result:
  standalone  +12.25e-5 mean, t +3.11, 5/5 positive   -- the largest standalone effect in this
             campaign, far above the +2e-5 to +6e-5 of Phases 11 and 12
  MARGINAL on +0.27e-5 mean, t +4.24, 5/5 positive     -- the first consistently POSITIVE and
             statistically clean marginal in the whole campaign, but 5.6x short of the +1.5e-5 gate

A single slot is 1/59 of the blend, so a per-slot marginal of +0.27e-5 is exactly what a per-slot
mechanism should produce. The predeclared §8 route for a validated arm is to test replacing one OR
SEVERAL members, using actual OOF vectors. If the per-slot marginal is stable and the slots are
genuinely decorrelated, several aux-equipped slots should compound; if they correct the same pairs,
they will not. Only measurement can say which, and Phase 11 is the standing proof that inferring this
from family weights is invalid.

A CHAOS FINDING THAT GOVERNS HOW THESE NUMBERS MUST BE READ
----------------------------------------------------------
A settings-sensitivity probe on fold 0 reran the aux build at 400 rounds / 5 inner folds instead of
250 / 3. The auxiliary models' out-of-fold accuracy was IDENTICAL (mean 0.6998 both times, matching
to three decimals per rating), yet the champion's standalone AUC moved 0.961320 -> 0.961047, a swing
of 27e-5. The features were stable; the model was not.

That is a property of extra_trees: randomised thresholds and per-node feature subsampling make the
fit CHAOTIC under perturbations far below any level that changes a feature's meaning. So at this scale
a single fold's standalone delta is not reproducible, and the campaign's habit of reading standalone
deltas is unsafe for this family.

The MARGINAL is unaffected: +0.31e-5 and +0.24e-5 at the two settings. Pair ranking is robust to
tree-level chaos in a way a single model's AUC is not. That makes the measured marginal the quantity
to trust, and it is the quantity this script maximises information about.

Usage:
  python scripts/run_phase13b.py --folds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_phase12 import (CHAMPION_PARAMS, CHAMPION_SCHEME, CHAMPION_VIEW,  # noqa: E402
                                 lg, sg, te_hash)
from scripts.run_phase13 import RATINGS, build_aux  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# Real extra_trees members of v3 on the full view under the primary scheme, with the exact params
# each was registered with in scripts/run_xt_zoo.py. Each gets its OWN seed, so the slots stay
# decorrelated for the reason the blend relies on.
# Real extra_trees members of v3 on the full view under the primary scheme. Seeds are READ FROM THE
# LEDGER, not parsed out of the exp_id. My first version derived them with
# `int(s.rsplit("_s", 1)[1])`, which crashes on xt_xt_d127_bin63 ("127_bin63" is not an int) and,
# worse, silently falls back to the loop counter for slots with no "_s" at all -- giving
# xt_xt_d255 seed 2 when its real seed is 4. A silently wrong seed is exactly the failure that made
# Phase 12's fold-0 screen untrustworthy, so the values below are stated explicitly and pinned.
SLOT_SEEDS = {"xt_xt_d127_s1": 1, "xt_xt_d63": 3, "xt_xt_d255": 4, "xt_xt_d127_cs05": 5,
              "xt_xt_d127_ss06": 7, "xt_xt_d127_bin63": 8}
SLOTS = {
    "xt_xt_d127_s1": dict(learning_rate=0.02, num_leaves=127, extra_trees=True),
    "xt_xt_d255": dict(learning_rate=0.02, num_leaves=255, min_child_samples=80, extra_trees=True),
    "xt_xt_d127_bin63": dict(learning_rate=0.02, num_leaves=127, max_bin=63, extra_trees=True),
    "xt_xt_d127_ss06": dict(learning_rate=0.02, num_leaves=127, subsample=0.6, extra_trees=True),
    "xt_xt_d127_cs05": dict(learning_rate=0.02, num_leaves=127, colsample_bytree=0.5,
                            extra_trees=True),
    "xt_xt_d63": dict(learning_rate=0.03, num_leaves=63, extra_trees=True),
}


def fit_slot(Xf, yf, Xv, seed, params, es_X, es_y):
    import lightgbm as lgb
    from scripts.run_phase12 import CAT_PARAMS, ES_PATIENCE
    p = dict(CHAMPION_PARAMS)
    p.update(CAT_PARAMS)
    p.update(params)
    p.update(random_state=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    ds = lgb.Dataset(Xf, label=yf, categorical_feature="auto")
    dv = lgb.Dataset(es_X, label=es_y, reference=ds)
    m = lgb.train(p, ds, num_boost_round=p["n_estimators"], valid_sets=[dv],
                  callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])
    return m.predict(Xv, num_iteration=int(m.best_iteration)), int(m.best_iteration)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--tag", default="p13b")
    ap.add_argument("--aux-rounds", type=int, default=250)
    ap.add_argument("--aux-inner-folds", type=int, default=3)
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    yi = tr[TARGET].values.astype("int8")
    folds = get_scheme(CHAMPION_SCHEME, yi, tr[ID_COL]).folds
    vb = ViewBuilder(tr, te, CHAMPION_VIEW)
    vb.build_static()

    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    L = {e: lg(store.load_oof(e).astype("float64")) for e in ids}
    base_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    assert float(np.abs(sg(base_logit).astype(np.float32).astype("float64") - v3).max()) == 0.0
    present = sorted((s for s in SLOTS if s in ids), key=lambda s: SLOT_SEEDS[s])
    # Cross-check the hard-coded seeds against the ledger, so a stale literal cannot silently make a
    # slot its own non-counterpart.
    _led = {}
    for _ln in Path("experiments/ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if _ln.strip():
            _d = json.loads(_ln)
            if _d.get("exp_id") in SLOT_SEEDS:
                _led[_d["exp_id"]] = _d.get("seed")
    wrong = {s: (SLOT_SEEDS[s], _led[s]) for s in present
             if s in _led and _led[s] != SLOT_SEEDS[s]}
    if wrong:
        raise SystemExit(f"STOP: slot seeds disagree with the ledger: {wrong}")
    print("=" * 108)
    print("PHASE 13B -- do the auxiliary-task gains COMPOUND across extra_trees slots?")
    print("=" * 108)
    print(f"  v3 geometry verified bit-exact. {len(ids)} members; {len(present)} extra_trees "
          f"slots present in v3:")
    for s in present:
        print(f"    {s:<22} seed {SLOT_SEEDS[s]:<3} {SLOTS[s]}")
    print("  The auxiliary features are built ONCE PER FOLD and shared by every slot, so the cost is")
    print("  13 aux fits per fold plus one extra_trees fit per slot.\n")

    out = {"tag": args.tag, "slots": present, "folds": {}, "aux_rounds": args.aux_rounds,
           "aux_inner_folds": args.aux_inner_folds,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()[:12]}

    for k in [int(x) for x in args.folds.split(",")]:
        fit_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]
        Xf, Xa, names = vb.assemble(fit_idx, yi, val_idx, None, inner_seed=k)
        Xv = Xa["val"]
        te_pos = [i for i, n in enumerate(names) if n.startswith("te_")]
        te0 = te_hash(Xf, te_pos)

        cache = REPORTS / f"{args.tag}_auxfit_f{k}.npy"
        vcache = REPORTS / f"{args.tag}_auxval_f{k}.npy"
        if cache.exists() and vcache.exists():
            aux_fit, aux_val = np.load(cache), np.load(vcache)
            aux_sec = 0.0
            print(f"  fold {k}: aux features loaded from cache")
        else:
            t0 = time.time()
            aux_fit, aux_val, _info = build_aux(Xf, Xv, names, 1, rounds=args.aux_rounds,
                                                inner_folds=args.aux_inner_folds)
            aux_sec = time.time() - t0
            np.save(cache, aux_fit)
            np.save(vcache, aux_val)
            print(f"  fold {k}: aux features built in {aux_sec:.0f}s")

        Xf_aug = np.column_stack([Xf, aux_fit]).astype("float64")
        Xv_aug = np.column_stack([Xv, aux_val]).astype("float64")
        assert te_hash(Xf_aug, te_pos) == te0, "adding aux columns perturbed the te_ block"

        itr, es = _inner_es_split(fit_idx, yi, 1)
        pos = {int(v): j for j, v in enumerate(fit_idx)}
        tr_l = np.array([pos[int(v)] for v in itr])
        es_l = np.array([pos[int(v)] for v in es])

        per = {}
        cols = {s: lg(L[s].copy()) for s in present}
        for si, s in enumerate(present, start=1):
            seed = SLOT_SEEDS[s]
            # The seed must equal the one the stored member was produced with, or the slot is not
            # its own counterpart and the comparison is meaningless.
            assert seed == SLOT_SEEDS[s], "slot seed mismatch"
            pred, best = fit_slot(Xf_aug[tr_l], y[itr], Xv_aug, seed, SLOTS[s],
                                  Xf_aug[es_l], y[es])
            a_new = float(roc_auc_score(y[val_idx], pred))
            a_old = float(roc_auc_score(y[val_idx], sg(L[s][val_idx])))
            c = L[s].copy()
            c[val_idx] = lg(pred)
            per[s] = {"seed": seed, "best_iter": best, "auc_with_aux": a_new,
                      "auc_original": a_old, "standalone_e5": (a_new - a_old) * 1e5,
                      "corr_with_original": float(corr(lg(pred), L[s][val_idx]))}
            cols[s] = c
            np.save(REPORTS / f"{args.tag}_{s}_{CHAMPION_SCHEME}_f{k}.npy",
                    pred.astype("float32"))
            print(f"    slot {s:<22} seed {seed:<3} standalone {per[s]['standalone_e5']:+7.2f}e-5"
                  f"  iters {best:<5} corr w/ original {per[s]['corr_with_original']:.5f}")

        # ---- the actual question: cumulative multi-slot marginal, from real vectors ----
        a_b4 = float(roc_auc_score(y[val_idx], sg(base_logit[val_idx])))
        order, cum = [], {}
        for s in present:
            order.append(s)
            cc = [cols[e] if e in order else L[e] for e in ids]
            aft = sg(np.mean(np.column_stack(cc), axis=1))
            a_sw = float(roc_auc_score(y[val_idx], aft[val_idx]))
            cum[s] = (a_sw - a_b4) * 1e5
        print(f"    v3 fold-{k} control = {a_b4:.6f}")
        run = 0.0
        for s in order:
            run += cum[s] - (cum[order[order.index(s) - 1]] if order.index(s) else 0.0)
            print(f"    cumulative over {len(order) and order.index(s) + 1} slot(s) -> "
                  f"{cum[s]:+.3f}e-5")
        out["folds"][f"f{k}"] = {"v3_control": a_b4, "per_slot": per, "cumulative_marginal_e5": cum,
                                 "aux_seconds": round(aux_sec, 1)}
        print(f"    ALL {len(order)} aux slots swapped: {cum[order[-1]]:+.3f}e-5\n")

    # ---- pooled verdict ------------------------------------------------------------------
    ks = sorted(out["folds"], key=lambda x: out["folds"][x]["v3_control"])
    full = [out["folds"][f]["cumulative_marginal_e5"][present[-1]] for f in ks]
    print(f"  ALL-SLOT marginal per fold: {[f'{v:+.3f}' for v in full]}  "
          f"mean {np.mean(full):+.3f}e-5  "
          f"{(np.array(full) > 0).sum()}/{len(full)} positive")
    gate = float(np.mean(full)) >= 1.5 and (np.array(full) > 0).sum() >= max(4, len(full) - 1)
    out["all_slot_marginals_e5"] = full
    out["all_slot_mean_e5"] = float(np.mean(full))
    out["gate_pass"] = bool(gate)
    out["verdict"] = ("ADMIT the aux-equipped extra_trees block" if gate else
                      "marginal remains below the +1.5e-5 gate; no admission, no submission")
    print(f"  -> {out['verdict']}")
    save_json(out, REPORTS / f"{args.tag}_runs.json")
    print(f"  wrote {REPORTS / f'{args.tag}_runs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())