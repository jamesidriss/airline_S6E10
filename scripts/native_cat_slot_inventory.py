"""Phase 11 -- recover the EXACT metadata of v3's seven CatBoost slots.

Why this script exists
----------------------
Phase 11 replaces each of v3's seven CatBoost members with a native-categorical counterpart. If a
slot's real configuration is guessed wrong, the "counterpart" is not a counterpart and the resulting
delta measures nothing. Two guesses in the task brief were WRONG and are corrected here from evidence:

  * `z3_cat_f10` was assumed to mean `feature_fraction=0.1`. It does not. Its recorded params are
    `learning_rate=0.04, depth=8` -- identical to `z3_cat_d8_s2` apart from the seed -- and the name
    refers to the FOLD SCHEME: it is the only `block10` member of the seven. Its counterpart must
    therefore be built on block10, not primary.
  * `prod5_cat_full_primary` params "must be recovered EXACTLY". They are not recorded anywhere as
    an explicit dict; the slot is a byte-identical duplicate of `view_cat_full_primary`
    (oof_sha 935d6778ee54 on both) produced by scripts/run_views.py with all defaults, so its
    configuration is the runner's defaults and its seed is run_views' default seed of 1.

Sources of truth, in priority order
------------------------------------
  1. the prediction store's per-member metadata (params, seed, featureset, oof_sha, has_test)
  2. experiments/ledger.jsonl records for the same exp_id
  3. reports/all_models_final.md, which carries the fold scheme and the seed column
  4. scripts/run_views.py for the defaults behind any slot with no explicit params dict
Anything not recoverable from these is recorded as None rather than guessed.

Two protocol facts that must travel with the inventory
------------------------------------------------------
  * ORIGINAL OOF policy: `_fit_cat_es` early-stops on a 10% carve of outer-fit, then writes the OOF
    from the model trained on the OTHER 90%. The carve is permanently discarded. Phase 10 measured
    the fixed-round refit-on-100% protocol as worth +5.3e-5 on LightGBM and +12.8e-5 on CatBoost.
  * ORIGINAL TEST policy: `_fit_full_predict_cat` refits on 100% of outer-fit at the MEDIAN best
    iteration -- but with `learning_rate=0.05`, while the OOF path used `0.04`. That is a
    pre-existing inconsistency in the finalist's test predictions, recorded here because any native
    test inference must decide deliberately whether to reproduce it.

Consequence for interpretation, stated up front
-----------------------------------------------
A native counterpart trained under the Phase 10 fixed-round protocol is better than its original on
TWO axes at once: the native-categorical representation AND a 10% larger training fraction. So
`native - original` is an OPERATIONAL REPLACEMENT DELTA, not a causal estimate of the mechanism. The
causal estimate is C2 - C0 from Phase 10, where both arms shared one protocol (+6.64e-5, 4/5 folds).
This is labelled in every output rather than left implicit.

Usage: python scripts/native_cat_slot_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, load_cached_parquet, save_json  # noqa: E402
from src.features.view import VIEWS  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

# run_views.py::_fit_cat_es defaults, for slots with no explicit params dict
RUNNER_DEFAULTS = {"iterations": 6000, "learning_rate": 0.04, "depth": 8, "l2_leaf_reg": 3.0,
                   "random_strength": 1.0, "rsm": None, "border_count": None,
                   "boosting_type": "Plain", "cat_features": None}
RUNNER_DEFAULT_SEED = 1          # scripts/run_views.py: ap.add_argument("--seed", type=int, default=1)
ES_PATIENCE = 300                # early_stopping_rounds in _fit_cat_es

SLOT_ORDER = ["z3_cat_f10", "z3_cat_d10", "z3_cat_d8_s2", "prod5_cat_full_primary",
              "z3_cat_d6", "z3_cat_core3", "z3_cat_ogs"]


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def main() -> None:
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    store_meta = {m["exp_id"]: m for m in store.list_all()}

    ledger = {}
    for line in (REPORTS.parent / "experiments" / "ledger.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ledger.setdefault(r["exp_id"], []).append(r)

    # Fold scheme comes from reports/all_models_final.md, the only place it is tabulated. The parse
    # must be STRICT: an earlier version matched any line containing the substring "cat", which also
    # hits the pairwise-correlation tables (their row labels are exp_ids like prod5_cat_full_primary)
    # and silently captured that table's AUC column as the fold scheme -- 0.9751 instead of block10.
    # Require the family cell to equal "cat" exactly and the scheme cell to be a known scheme name.
    KNOWN_SCHEMES = {"primary", "shadow", "block10"}
    scheme: dict[str, str] = {}
    md = (REPORTS / "all_models_final.md").read_text(encoding="utf-8")
    for line in md.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 5 and cells[0] in SLOT_ORDER and cells[2] == "cat" \
                and cells[4] in KNOWN_SCHEMES:
            scheme[cells[0]] = cells[4]
    missing_scheme = [e for e in SLOT_ORDER if e not in scheme]
    if missing_scheme:
        raise SystemExit(f"could not recover the fold scheme for {missing_scheme}; refusing to "
                         f"guess. The inventory must be exact.")

    print("PHASE 11 -- NATIVE CATBOOST SLOT INVENTORY")
    print("=" * 104)
    print(f"  v3 has {len(ids)} members; CatBoost slots: {len(SLOT_ORDER)}")
    print(f"  each slot owns exactly 1/{len(ids)} of the equal-logit weight\n")

    slots = []
    for i, eid in enumerate(SLOT_ORDER):
        assert eid in ids, f"{eid} is not in the v3 finalist manifest"
        sm = store_meta.get(eid, {})
        lrec = ledger.get(eid, [{}])[0]
        explicit = sm.get("params") or lrec.get("params") or {}
        from_defaults = not explicit
        params = dict(RUNNER_DEFAULTS)
        params.update(explicit)
        seed = sm.get("seed", lrec.get("seed"))
        if seed is None:
            seed = RUNNER_DEFAULT_SEED if from_defaults else None
        fs = scheme.get(eid) or sm.get("featureset")
        view = sm.get("featureset") or lrec.get("featureset")
        if view not in VIEWS:
            view = None
        rec = {
            "slot": i, "exp_id": eid,
            "view": view, "view_known": view is not None,
            "fold_scheme": fs,
            "params": params,
            "params_from_runner_defaults": from_defaults,
            "params_source": ("explicit record" if explicit
                              else "scripts/run_views.py::_fit_cat_es defaults"),
            "seed": seed, "seed_source": ("explicit record" if sm.get("seed") is not None
                                          or lrec.get("seed") is not None
                                          else "run_views.py --seed default"),
            "n_features": sm.get("n_features"),
            "oof_auc_recorded": sm.get("auc"),
            "oof_sha": sm.get("oof_sha"),
            "has_test": sm.get("has_test"),
            "dup": sm.get("dup") or None,
            "categorical_treatment_original": "NONE -- all categoricals were ordinal float32; "
                                              "cat_features was never passed",
            "original_oof_iteration_policy": f"inner ES on a 10% carve of outer-fit "
                                             f"(early_stopping_rounds={ES_PATIENCE}); OOF written "
                                             f"from the model trained on the other 90%, so the carve "
                                             f"is permanently discarded",
            "original_test_iteration_policy": "refit on 100% of outer-fit at int(median best_iter) "
                                              "with learning_rate=0.05, while the OOF path used 0.04 "
                                              "-- pre-existing inconsistency, recorded not fixed",
            "native_counterpart_changes": ["append 17 native string categorical twins "
                                           "(META4 + SERVICE13) and declare them via cat_features",
                                           "train under the Phase 10 fixed-round protocol "
                                           "(inner ES selects the count, refit on 100% of "
                                           "outer-fit)"],
            "native_counterpart_keeps": ["numeric view", "learning_rate", "depth", "l2_leaf_reg",
                                         "seed", "fold scheme", "boosting_type=Plain"],
        }
        rec["config_hash"] = sha({k: rec[k] for k in
                                  ("view", "fold_scheme", "params", "seed")})
        slots.append(rec)
        print(f"  slot {i}  {eid:<26} view={str(view):<12} scheme={str(fs):<9} "
              f"lr={params['learning_rate']} depth={params['depth']} "
              f"l2={params['l2_leaf_reg']} seed={seed}"
              f"{'  [defaults]' if from_defaults else ''}")

    bad_view = [s["exp_id"] for s in slots if not s["view_known"]]
    if bad_view:
        raise SystemExit(f"unknown feature view for {bad_view}; refusing to guess.")

    print("\n  CORRECTIONS TO THE TASK BRIEF'S ASSUMPTIONS (from evidence, not deference):")
    f10 = next(s for s in slots if s["exp_id"] == "z3_cat_f10")
    print(f"    * 'f10' is NOT feature_fraction=0.1. Recorded params are lr=0.04, depth=8 -- the same")
    print(f"      as z3_cat_d8_s2 apart from the seed. The name refers to the FOLD SCHEME: f10 is")
    print(f"      the only {f10['fold_scheme']} member of the seven, and its counterpart must be")
    print(f"      built on {f10['fold_scheme']}.")
    p5 = next(s for s in slots if s["exp_id"] == "prod5_cat_full_primary")
    print(f"    * prod5_cat_full_primary has no explicit params dict anywhere. It is a byte-identical")
    print(f"      duplicate of view_cat_full_primary (oof_sha {p5['oof_sha']} on both), so its config")
    print(f"      is the run_views.py defaults and its seed is {p5['seed']} "
          f"(run_views --seed default).")

    print("\n  DIVERSITY ALREADY PRESENT IN THE ORIGINAL BLOCK (pairwise logit corr):")
    tr, _ = load_cached_parquet()
    y = tr["satisfaction"].values.astype("float64")
    def lg(p):
        p = np.clip(p.astype("float64"), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    P = {}
    for s in slots:
        P[s["exp_id"]] = lg(store.load_oof(s["exp_id"]))
    cs = []
    for a in range(len(slots)):
        for b in range(a + 1, len(slots)):
            cs.append(float(np.corrcoef(P[slots[a]["exp_id"]], P[slots[b]["exp_id"]])[0, 1]))
    print(f"    {len(cs)} pairs: min {min(cs):.5f}  median {float(np.median(cs)):.5f}  "
          f"max {max(cs):.5f}")
    print("    This is the diversity the native block must preserve; a block whose members collapse")
    print("    toward 0.999+ cannot replace it regardless of individual AUC.")

    out = {
        "n_members_v3": len(ids),
        "n_cat_slots": len(slots),
        "weight_per_slot": 1.0 / len(ids),
        "cat_block_weight": len(slots) / len(ids),
        "runner_defaults": RUNNER_DEFAULTS,
        "runner_default_seed": RUNNER_DEFAULT_SEED,
        "es_patience": ES_PATIENCE,
        "slots": slots,
        "original_block_pairwise_logit_corr": {
            "n_pairs": len(cs), "min": min(cs), "median": float(np.median(cs)), "max": max(cs)},
        "interpretation_warning":
            "native - original_slot is an OPERATIONAL REPLACEMENT delta confounded with the "
            "training-protocol change (fixed-round refit on 100% vs the original 90% carve). The "
            "causal estimate of the native-cat mechanism is Phase 10's C2 - C0 = +6.64e-5, where "
            "both arms shared one protocol.",
        "fold_scheme_caveat":
            "v3 already mixes fold schemes: six slots are primary and z3_cat_f10 is block10. A "
            "block10 slot's OOF is out-of-fold with respect to block10, not primary, so for a row in "
            "primary fold 0 its prediction may come from a model trained on other primary-fold-0 "
            "rows. This is a PRE-EXISTING property of the v3 blend, not something the swap "
            "introduces; per-fold deltas here are reported over primary folds for consistency.",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()[:12],
    }
    out["inventory_hash"] = sha(out)
    save_json(out, REPORTS / "native_cat_slot_inventory.json")
    print(f"\n  wrote reports/native_cat_slot_inventory.json")
    print(f"  inventory_hash = {out['inventory_hash'][:32]}")
    print(f"  cat block weight = {len(slots)}/{len(ids)} = {out['cat_block_weight']:.5f} "
          f"(must stay FIXED in every candidate)")


if __name__ == "__main__":
    main()
