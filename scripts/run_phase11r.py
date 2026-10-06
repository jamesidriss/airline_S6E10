"""Phase 11R -- staged, cost-controlled tests of the native-CatBoost hypothesis.

CORRECTION CARRIED BY THIS PHASE (append-only; the earlier claim is withdrawn)
---------------------------------------------------------------------------
Phase 11 concluded: "the CatBoost block is structurally capped at +0.79e-5, because
block_weight x model_level_gain = 0.11864 x 6.64e-5." THAT IS NOT AN UPPER BOUND and the
conclusion drawn from it is withdrawn.

ROC-AUC is a nonlinear functional of the prediction vector. In general

    AUC(blend with improved members) - AUC(old blend)

is NOT equal to, and is NOT bounded above by,

    family_weight * [AUC(new family model) - AUC(old family model)].

A member can disproportionately correct exactly those pairs the full ensemble currently ranks
incorrectly, which is the whole basis of ensemble complementarity; conversely it can add nothing
even when its standalone AUC rises. That is precisely why complementarity must be MEASURED from
actual prediction vectors rather than inferred from weights and standalone AUCs. The valid Phase 11
conclusion is only: the two zero-cost swaps tested produced essentially zero blend gain, making the
diverse-native-block hypothesis LOWER PRIORITY and requiring a staged cost-controlled test.

SECOND CORRECTION: those two swaps were APPROXIMATE ZERO-COMPUTE DIAGNOSTICS, not exact slot
replacements. Phase 10's C2 was trained with seed 4, so reusing it for `z3_cat_d8_s2` (seed 3) is not
that slot's counterpart. Replacing several diverse originals with one C2 prediction also destroys the
seed/view/depth diversity BY CONSTRUCTION, which is the very thing the hypothesis was about. C2
matches `prod5_cat_full_primary` on view/scheme/depth/lr/l2 but not on seed, so no slot is an exact
match and the earlier swaps are relabelled accordingly.

THIRD CORRECTION: v3's authoritative geometry is expit(mean(member LOGITS)), per
scripts/reproduce_finalist.py:122-152 -- verified against the store, where float32(expit(mean(logits)))
reproduces the stored vector with max abs difference EXACTLY 0.0, while mean(probabilities) differs by
3.53e-02 and has logit corr 0.978. Phase 11's guard asserted the wrong geometry in its printed note.

WHAT THIS PHASE DOES, IN THE PRE-DECLARED ORDER
-----------------------------------------------
  11R-A  C4 = C2 + exact categorical Age twin            (~75 levels)
  11R-B  C5 = C2 + exact categorical Flight Distance twin (~3,474 levels, route-like id)
          -- timing/memory probe FIRST, high-cardinality CTRs are the expensive case
  11R-C  C2_ctr2 = C2 with max_ctr_complexity = 2, after auditing what C2 actually built
  11R-D  THREE genuinely distinct exact counterparts, N_D6 / N_D10 / N_CORE3, each with its own
          ORIGINAL seed, plus the exact MINI_B50 / MINI_B100 four-slot replacements

Question A (can the mechanism be made materially stronger?) is answered by 11R-A/B/C.
Question B (do a few distinct counterparts preserve enough diversity to move the ensemble?) is
answered by 11R-D with ACTUAL prediction vectors.

COST DISCIPLINE, learned the hard way in Phase 11
-------------------------------------------------
CatBoost's per-round cost RISES with round count (measured: 0.4916 s/round at 100 rounds vs 0.6835 at
400 on the block10 workload), so a 40-round probe understated a full run by ~7x. Every arm here is
costed from a LONG timing anchor before training, and any arm projecting > 90 minutes is reported
rather than launched.

Usage:
  python scripts/run_phase11r.py --arm C4 --folds 0
  python scripts/run_phase11r.py --arm C5 --probe-only
  python scripts/run_phase11r.py --arm ctr2 --folds 0
  python scripts/run_phase11r.py --arm ND6,ND10,NCORE3 --folds 0
  python scripts/run_phase11r.py --report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.native_cat import (attach, cat_cardinality, cat_frame, cat_indices,  # noqa: E402
                                default_cat_cols, is_string_series, to_cat_series,
                                verify_no_target)
from scripts.run_views import _inner_es_split  # noqa: E402

ES_ROUNDS = 2500
ES_PATIENCE = 200
# Phase 10's C2 reference, for apples-to-apples deltas. Seed 4 is recorded so no later arm can
# silently differ from it on anything but the intended mechanism.
# Phase 10's C2 reference, for apples-to-apples deltas. Seed 4 is recorded so no later arm can
# silently differ from it on anything but the intended mechanism.
# Keys are spelled to match `arm_specs()` entries ("lr", "l2") so the C2REF arm can be built from
# this dict directly; the test suite compares against them by name. An earlier version used
# "learning_rate"/"l2_leaf_reg" here and "lr"/"l2" in the specs, which raised KeyError the moment a
# reference arm was actually constructed -- the mismatch was invisible while C2REF was unused.
C2_REF = {"view": "full", "depth": 8, "lr": 0.04, "l2": 3.0, "seed": 4, "scheme": "primary"}
MINI_SLOTS = {                      # exact counterpart -> original slot it replaces
    "C2": "z3_cat_d8_s2",           # replaced as a DIAGNOSTIC; seed differs, flagged in the report
    "ND6": "z3_cat_d6",
    "ND10": "z3_cat_d10",
    "NCORE3": "z3_cat_core3",
}
INV = REPORTS / "native_cat_slot_inventory.json"


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype="float64"), -35, 35)))


def sha(o) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def arm_specs() -> dict:
    inv = {s["exp_id"]: s for s in json.loads(INV.read_text(encoding="utf-8"))["slots"]}
    d6, d10, c3 = inv["z3_cat_d6"], inv["z3_cat_d10"], inv["z3_cat_core3"]
    return {
        # --- question A: can the mechanism be made stronger? ---
        "C4": {"view": "full", "depth": 8, "lr": 0.04, "l2": 3.0, "seed": C2_REF["seed"],
               "scheme": "primary", "extra_src": ["Age"],
               "note": "C2 + exact categorical Age twin; numeric Age retained"},
        "C5": {"view": "full", "depth": 8, "lr": 0.04, "l2": 3.0, "seed": C2_REF["seed"],
               "scheme": "primary", "extra_src": ["Flight Distance"],
               "note": "C2 + exact categorical Flight Distance twin; numeric FD retained"},
        "ctr2": {"view": "full", "depth": 8, "lr": 0.04, "l2": 3.0, "seed": C2_REF["seed"],
                 "scheme": "primary", "extra_src": [], "max_ctr_complexity": 2,
                 "note": "C2 with max_ctr_complexity=2; nothing else changes"},
        # --- question B: do genuinely distinct counterparts preserve diversity? ---
        "ND6": {"view": d6["view"], "depth": d6["params"]["depth"],
                "lr": d6["params"]["learning_rate"], "l2": d6["params"]["l2_leaf_reg"],
                "seed": d6["seed"], "scheme": d6["fold_scheme"], "extra_src": [],
                "note": "EXACT counterpart of z3_cat_d6 (own seed 1, depth 6, l2 5.0)"},
        "ND10": {"view": d10["view"], "depth": d10["params"]["depth"],
                 "lr": d10["params"]["learning_rate"], "l2": d10["params"]["l2_leaf_reg"],
                 "seed": d10["seed"], "scheme": d10["fold_scheme"], "extra_src": [],
                 "note": "EXACT counterpart of z3_cat_d10 (own seed 2, depth 10, l2 6.0, lr 0.03)"},
        "NCORE3": {"view": c3["view"], "depth": c3["params"]["depth"],
                   "lr": c3["params"]["learning_rate"], "l2": c3["params"]["l2_leaf_reg"],
                   "seed": c3["seed"], "scheme": c3["fold_scheme"], "extra_src": [],
                   "note": "EXACT counterpart of z3_cat_core3 (own seed 4, core3 view)"},
        # The C2 reference, restated as an arm so it can be retrained under THIS run's ES budget.
        # Phase 10's C2 was selected under a 6000-round budget (fold 0 selected ~1040-1300); an arm
        # trained under a 2500 cap that hits the cap is penalised in a way C2 was not, so comparing
        # against the stored C2 would manufacture a negative delta out of a budget difference.
        "C2REF": {"view": C2_REF["view"], "depth": C2_REF["depth"], "lr": C2_REF["lr"],
                  "l2": C2_REF["l2"], "seed": C2_REF["seed"], "scheme": C2_REF["scheme"],
                  "extra_src": [],
                  "note": "C2 reference retrained under THIS run's ES budget (budget-matched)"},
    }


def build(vb, tr, y_int, fit, val, extra_src):
    Xf, Xa, names = vb.assemble(fit, y_int, val, None, inner_seed=0)
    src = default_cat_cols(True) + list(extra_src)
    missing = [c for c in src if c not in tr.columns]
    if missing:
        raise KeyError(f"categorical source columns absent: {missing}")
    verify_no_target([f"ncat__{c}" for c in src])
    f, cn = attach(Xf, names, cat_frame(tr, src, fit))
    v, _ = attach(Xa["val"], names, cat_frame(tr, src, val))
    return f, v, cn, src


def fit_cat(frame, y, spec, cat_names, seed, n_rounds, es=None):
    from catboost import CatBoostClassifier
    p = {"learning_rate": spec["lr"], "depth": spec["depth"], "l2_leaf_reg": spec["l2"],
         "random_seed": int(seed), "thread_count": 8, "verbose": 0,
         "allow_writing_files": False, "boosting_type": "Plain"}
    if spec.get("max_ctr_complexity"):
        p["max_ctr_complexity"] = int(spec["max_ctr_complexity"])
    kw = {"cat_features": cat_indices(frame, cat_names)} if cat_names else {}
    if es is None:
        p["iterations"] = int(n_rounds)
        m = CatBoostClassifier(**p)
        m.fit(frame, y, **kw)
        return m, int(n_rounds), None
    p["iterations"] = int(es[2])
    p["eval_metric"] = "AUC"
    m = CatBoostClassifier(**p)
    m.fit(frame, y, eval_set=(es[0], es[1]), early_stopping_rounds=ES_PATIENCE, verbose=0, **kw)
    best = int(m.get_best_iteration() or es[2])
    return m, best, dict(model=m, hit_cap=best >= es[2] - 1, es_rounds=es[2])


def ctr_audit(m) -> dict:
    """What CTRs did the model ACTUALLY build? Reported, not assumed from parameter defaults."""
    out: dict = {}
    try:
        import tempfile
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as td:
            p = _P(td) / "m.json"
            m.save_model(str(p))
            blob = p.read_text(encoding="utf-8", errors="ignore")
        out["dump_counts"] = {k: blob.count(f'"{k}"') for k in
                              ("ctr_type", "ctr_data", "ctr_leaf_count", "one_hot",
                               "simple_ctr", "combinations_ctr", "Borders", "target_border_count")}
        out["n_ctr_leaf_descriptions"] = len(m.get_leaf_ctr_description() or {})
    except Exception as exc:                                     # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    try:
        out["cat_feature_count"] = len(m.get_cat_feature_indices())
    except Exception:                                            # noqa: BLE001
        out["cat_feature_count"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="C4")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tag", default="p11r")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--probe-rounds", type=int, default=400,
                    help="timing anchor length. Phase 11 proved a 40-round anchor understates cost "
                         "~7x because CatBoost's per-round cost rises with round count.")
    ap.add_argument("--budget-minutes", type=float, default=90.0)
    ap.add_argument("--project-iters", type=int, default=1100,
                    help="iteration count the cost projection assumes the ES fit reaches. MEASURED "
                         "on this workload: a correctly-configured fold-0 CatBoost early-stops at "
                         "1043. Projecting at the 2500 CAP instead made C5 and ctr2 project 121 and "
                         "132 minutes and skip both, when their real cost is ~44 and ~48 min.")
    ap.add_argument("--retrain-c2", action="store_true",
                    help="also train the C2 REFERENCE under this run's ES budget. REQUIRED for a "
                         "fair delta whenever the budget differs from the one C2 was originally "
                         "trained under, because a truncated ES penalises only the arm that hit the "
                         "cap.")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    specs = arm_specs()
    arms = [a.strip().upper() if a.strip().lower() != "ctr2" else "ctr2"
            for a in args.arm.split(",") if a.strip()]
    if args.retrain_c2 and "C2REF" not in arms:
        arms = ["C2REF"] + arms
    unknown = [a for a in arms if a not in specs]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known {sorted(specs)}")

    if args.report:
        return report(args, tr, y, y_int)

    print("PHASE 11R -- staged native-CatBoost tests")
    print("=" * 100)
    print(f"  ES budget {ES_ROUNDS} rounds, patience {ES_PATIENCE}; Plain only; Ordered banned")
    print(f"  timing anchor {args.probe_rounds} rounds (NOT 40: per-round cost rises with rounds)")
    print(f"  budget {args.budget_minutes:.0f} min per arm; an arm projecting above that is "
          f"reported, not launched\n")

    out = {"tag": args.tag, "es_rounds": ES_ROUNDS, "es_patience": ES_PATIENCE,
           "c2_reference": C2_REF, "arms": {},
           "corrections": {
               "structural_cap_withdrawn":
                   "Phase 11's claim that the CatBoost block is capped at +0.79e-5 was NOT a "
                   "mathematical upper bound. AUC is a nonlinear functional of the prediction "
                   "vector, and blend gain is not bounded by family_weight x standalone gain: a "
                   "member can disproportionately correct pairs the ensemble currently gets wrong. "
                   "The valid Phase 11 conclusion is only that the two zero-cost swaps gave "
                   "essentially zero gain, making this hypothesis lower priority.",
               "previous_swaps_relabelled":
                   "The Phase 11 swaps are APPROXIMATE ZERO-COMPUTE DIAGNOSTICS, not exact slot "
                   "replacements: C2 was trained with seed 4, so it is not the counterpart of any "
                   "slot whose original seed differs, and substituting one C2 for several diverse "
                   "originals destroys the diversity the hypothesis was about. No slot matches C2 "
                   "exactly.",
               "v3_geometry":
                   "Authoritative v3 = expit(mean(member LOGITS)) per reproduce_finalist.py:122-152. "
                   "Verified on the store: float32(expit(mean(logits))) reproduces it with max abs "
                   "difference exactly 0.0, while mean(probabilities) does NOT reproduce the store "
                   "-- it differs by 3.53e-02 with logit corr 0.978. Phase 11's guard printed the "
                   "wrong geometry and asserted a float64 round-off tolerance that happened to pass; "
                   "the correct guard is float64 within float32 eps AND the float32 cast bit-exact.",
               "es_subset_of_training_CONTAMINATED_EARLY_STOPPING":
                   "FOURTH CORRECTION, found by chasing an unexplained reproducibility gap. The first "
                   "C2REF run selected 2499 of 2500 rounds and hit the ES cap, while Phase 10's "
                   "identically-configured C2 selected 1043 under a 6000-round budget with patience "
                   "300. A shorter budget with SHORTER patience cannot plausibly stop later, so the "
                   "gap was not a budget effect. Cause: this harness fitted the ES model on the FULL "
                   "outer-fit frame while passing eval_set = f.iloc[es_l], making the early-stopping "
                   "set a SUBSET OF ITS OWN TRAINING DATA. CatBoost never early-stops in that "
                   "configuration, so every arm ran to the cap and its refit used a fixed cap-length "
                   "round count instead of a validated one. A direct control run -- same nominal "
                   "settings (2500 rounds, patience 200), ES model trained on the 90% inner-train "
                   "carve -- selected 1043, matching Phase 10, and two such identical runs both "
                   "returned 1043, confirming CatBoost is deterministic here and the 2499 was a bug "
                   "rather than noise. FIXED: the ES model now trains on f.iloc[itr_l] only, and the "
                   "harness refuses to run if the two row sets overlap or do not partition the outer "
                   "fit. Every fold-0 arm trained before this fix is flagged CONTAMINATED and its "
                   "numbers are not used.",
               "pre_fix_arms_contaminated": ["C2REF", "C4"],
           },
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}

    for aname in arms:
        spec = specs[aname]
        sc = spec["scheme"]
        folds = get_scheme(sc, y_int, tr[ID_COL]).folds
        vb = ViewBuilder(tr, te, spec["view"])
        vb.build_static()
        card = cat_cardinality(tr, default_cat_cols(True) + list(spec["extra_src"]))
        print(f"\n{'-'*100}\n{aname}: {spec['note']}\n  view={spec['view']} scheme={sc} "
              f"depth={spec['depth']} lr={spec['lr']} l2={spec['l2']} seed={spec['seed']}\n"
              f"  added categorical cardinality: "
              f"{ {c: card[c] for c in spec['extra_src']} }\n{'-'*100}")

        for k in [int(x) for x in args.folds.split(",")]:
            fit = np.where(folds != k)[0]
            val = np.where(folds == k)[0]
            f, v, cn, src = build(vb, tr, y_int, fit, val, spec["extra_src"])

            # ---- cost from a LONG anchor before committing ----
            t0 = time.time()
            fit_cat(f, y[fit], spec, cn, spec["seed"] + k, args.probe_rounds)
            per = (time.time() - t0) / args.probe_rounds
            # cost(R) ~ a*R + b*R^2 fitted through the anchor, as measured in Phase 11
            b = max(0.0, (per - 0.6 * (0.2905 * len(fit) / 559708)) / args.probe_rounds)
            a = per - b * args.probe_rounds
            est = lambda R: a * R + b * R * R
            # Cost projection. The FIRST version of this projected `est(ES_ROUNDS) + est(1200)`,
            # i.e. it assumed early stopping would never fire and the ES fit would run the whole
            # 2500-round cap. That assumption is wrong: on this workload the correctly-configured
            # model early-stops at ~1043 rounds (measured, and reproducible -- two identical
            # correctly-configured runs both returned 1043). Using the cap as if it were the
            # expected cost made C5 and ctr2 look like 121 and 132 minutes and skipped both, when
            # their real cost at the measured stopping point is roughly 44 and 48 minutes.
            # Project at the MEASURED stopping point, and report the cap-worst-case separately so
            # the upper bound stays visible. An upper bound used as a point estimate is a silent
            # skip of the experiment, which is what happened.
            es_R = min(ES_ROUNDS, int(args.project_iters + ES_PATIENCE))
            proj_min = (est(es_R) + est(args.project_iters)) / 60.0
            worst_min = (est(ES_ROUNDS) + est(ES_ROUNDS)) / 60.0
            print(f"  fold {k}: {f.shape[1]} feat ({len(cn)} cat)  anchor {args.probe_rounds} rounds "
                  f"= {per:.4f}s/round", flush=True)
            print(f"    fitted cost(R) = {a:.4f}*R + {b:.2e}*R^2", flush=True)
            print(f"    projected at the MEASURED stopping point (ES ~{es_R}, refit "
                  f"~{args.project_iters}) ~ {proj_min:.0f} min", flush=True)
            print(f"    worst case if ES never fires and runs the {ES_ROUNDS} cap ~ {worst_min:.0f} min",
                  flush=True)
            if args.probe_only:
                out["arms"].setdefault(aname, {})[f"probe_f{k}"] = {
                    "n_features": int(f.shape[1]), "n_cat": len(cn),
                    "anchor_rounds": args.probe_rounds, "sec_per_round": per,
                    "cost_a": a, "cost_b": b, "projected_minutes": proj_min,
                    "cat_cardinality": card}
                continue
            if proj_min > args.budget_minutes:
                print(f"    SKIPPED: projected {proj_min:.0f} min exceeds the "
                      f"{args.budget_minutes:.0f} min budget. Reported, not launched.")
                out["arms"].setdefault(aname, {})[f"skip_f{k}"] = {
                    "projected_minutes": proj_min, "budget_minutes": args.budget_minutes,
                    "reason": "cost probe exceeded the per-arm budget"}
                continue

            itr_g, es_g = _inner_es_split(fit, y_int, int(spec["seed"]) + k)
            pos = {int(vv): j for j, vv in enumerate(fit)}
            itr_l = np.array([pos[int(vv)] for vv in itr_g])
            es_l = np.array([pos[int(vv)] for vv in es_g])
            # BUG FOUND AND FIXED HERE. The ES model was previously fitted on the FULL outer-fit frame
            # `f` with eval_set = f.iloc[es_l], which made the early-stopping set a SUBSET OF ITS OWN
            # TRAINING DATA. CatBoost then never early-stopped -- training AUC keeps improving -- so
            # every arm ran to the round cap. Measured: C2REF selected 2499 of 2500 while the same
            # nominal config trained correctly selects 1043 (verified: two identical correct-config
            # runs both return 1043, so CatBoost IS deterministic here and the 2499 was not noise).
            # Every Phase 11R fold-0 arm trained before this fix has a contaminated iteration count.
            # The ES model must be trained on the 90% inner-train rows ONLY, matching Phase 10's
            # `_fit_cat_es`, and the eval set must be the disjoint 10% carve.
            if len(set(itr_l.tolist()) & set(es_l.tolist())):
                raise SystemExit("STOP: inner-train and ES row sets overlap; early stopping would be "
                                 "evaluated on rows the model trained on.")
            if len(itr_l) + len(es_l) != len(fit):
                raise SystemExit(f"STOP: inner-train ({len(itr_l)}) + ES ({len(es_l)}) != outer fit "
                                 f"({len(fit)}); the carve is not a partition.")
            t0 = time.time()
            _, n_iter, info = fit_cat(f.iloc[itr_l], y[itr_g], spec, cn, int(spec["seed"]) + k, 0,
                                      es=(f.iloc[es_l], y[es_g], ES_ROUNDS))
            t_es = time.time() - t0
            m, used, _ = fit_cat(f, y[fit], spec, cn, int(spec["seed"]) + k, n_iter)
            pred = m.predict_proba(v)[:, 1]
            t_refit = time.time() - t0 - t_es
            auc = float(roc_auc_score(y[val], pred))
            np.save(REPORTS / f"{args.tag}_{aname}_{sc}_f{k}.npy", pred.astype("float32"))
            rec = {"arm": aname, "fold": k, "fold_scheme": sc, "view": spec["view"],
                   "depth": spec["depth"], "lr": spec["lr"], "l2": spec["l2"],
                   "seed": spec["seed"], "extra_src": list(spec["extra_src"]),
                   "max_ctr_complexity": spec.get("max_ctr_complexity"),
                   "n_features": int(f.shape[1]), "n_cat": len(cn), "cat_cols": cn,
                   "cat_cardinality": {c: card[c] for c in src},
                   "cat_hash": sha(cn), "schema_hash": sha(list(f.columns)),
                   "inner_best_iter": int(n_iter), "refit_iter": int(used),
                   "es_hit_cap": bool(info["hit_cap"]) if info else None,
                   "n_rows_refit": int(len(fit)), "n_rows_eval": int(len(val)),
                   "auc": auc, "seconds_es": round(t_es, 1), "seconds_total": round(t_es + t_refit, 1),
                   "ctr_audit": ctr_audit(m),
                   "pred_sha": sha(pred.astype("float32").tolist()[:200])}
            out["arms"].setdefault(aname, {})[f"f{k}"] = rec
            print(f"    AUC={auc:.6f}  iters={used}  (ES {t_es:.0f}s, total {t_es+t_refit:.0f}s)"
                  f"{'  *** ES HIT CAP: NOT CONVERGED ***' if rec['es_hit_cap'] else ''}",
                  flush=True)
            print(f"    CTR audit: {rec['ctr_audit'].get('dump_counts')}", flush=True)

    save_json(out, REPORTS / f"{args.tag}_runs.json")
    print("\nwrote", REPORTS / f"{args.tag}_runs.json")


def report(args, tr, y, y_int) -> None:
    """Compare each arm to C2 on its own fold, then measure exact single-slot swaps."""
    runs = json.loads((REPORTS / f"{args.tag}_runs.json").read_text(encoding="utf-8"))
    inv = json.loads(INV.read_text(encoding="utf-8"))
    by_id = {s["exp_id"]: s for s in inv["slots"]}
    man = json.loads((REPORTS.parent / "reports" / "finalist_v3_final.json").read_text(
        encoding="utf-8"))
    ms = man["members"] if isinstance(man, dict) and "members" in man else man
    ids = [(m["exp_id"] if isinstance(m, dict) else m) for m in ms]
    nmem = len(ids)
    L = {e: logit(store.load_oof(e).astype("float64")) for e in ids}
    base_logit = np.mean(np.column_stack([L[e] for e in ids]), axis=1)
    v3 = store.load_oof("blend_v3_final").astype("float64")
    print("=" * 118)
    print("PHASE 11R REPORT -- each arm vs C2 on its own fold, then EXACT single-slot swaps")
    print("=" * 118)
    print(f"  authoritative v3 = expit(mean(member logits)); float32 cast is bit-exact: "
          f"{np.abs(sig(base_logit).astype(np.float32).astype('float64') - v3).max() == 0.0}")

    prim = get_scheme("primary", y_int, tr[ID_COL]).folds
    rows = []
    for aname, recs in runs["arms"].items():
        for tag, r in recs.items():
            if not tag.startswith("f"):
                continue
            f = REPORTS / f"{args.tag}_{aname}_{r['fold_scheme']}_f{r['fold']}.npy"
            if not f.exists():
                continue
            k = r["fold"]
            sc = r["fold_scheme"]
            fl = get_scheme(sc, y_int, tr[ID_COL]).folds
            val = np.where(fl == k)[0]
            P = np.load(f).astype("float64")
            # C2's own fold-k prediction for the same scheme, when it exists
            # Budget-matched reference: prefer a C2REF trained in the SAME run (same ES cap), because
            # Phase 10's stored C2 used a 6000-round budget while this run uses 2500. Comparing an
            # arm that hit the cap against a reference that did not would manufacture a negative delta
            # out of a budget difference rather than out of the mechanism.
            c2f = REPORTS / f"{args.tag}_C2REF_{sc}_f{k}.npy"
            ref_name = "C2REF (budget-matched)"
            if not c2f.exists():
                c2f = REPORTS / f"p10b_C2_{sc}_f{k}.npy"
                ref_name = "C2 (Phase 10, 6000-round budget)"
            d_c2 = None
            o_n = None
            ref_auc = None
            if c2f.exists():
                c2 = np.load(c2f).astype("float64")
                ref_auc = float(roc_auc_score(y[val], c2))
                d_c2 = (float(roc_auc_score(y[val], P)) - ref_auc) * 1e5
                o_n = float(corr(logit(c2), logit(P)))
            rows.append({"arm": aname, "fold": k, "scheme": sc, "auc": float(
                roc_auc_score(y[val], P)), "delta_vs_C2_e5": d_c2, "corr_vs_C2": o_n,
                         "reference": ref_name, "reference_auc": ref_auc,
                         "reference_es_hit_cap": (
                             runs["arms"].get("C2REF", {}).get(f"f{k}", {}) or {}
                         ).get("es_hit_cap"),
                         "corr_vs_v3": float(corr(logit(P), logit(v3[val]))),
                         "spearman_vs_v3": float(spearman(P, v3[val])),
                         "iters": r["refit_iter"], "seconds": r["seconds_total"],
                         "es_hit_cap": r.get("es_hit_cap"),
                         "n_cat": r["n_cat"], "ctr_audit": r.get("ctr_audit")})

    print(f"\n  {'arm':<8}{'fold':>5}{'AUC':>12}{'d vs ref':>10}{'ref':>9}{'ref cap':>9}"
          f"{'corr ref':>10}{'corr v3':>10}{'spear v3':>10}{'iters':>8}{'sec':>7}")
    print("  " + "-" * 105)
    for r in rows:
        dc = f"{r['delta_vs_C2_e5']:+.1f}e" if r["delta_vs_C2_e5"] is not None else "-"
        cc = f"{r['corr_vs_C2']:.5f}" if r["corr_vs_C2"] is not None else "-"
        cap = "HIT" if r.get("reference_es_hit_cap") else "-"
        print(f"  {r['arm']:<8}{r['fold']:>5}{r['auc']:>12.6f}{dc:>10}"
              f"{('C2REF' if 'budget-matched' in str(r.get('reference')) else 'C2p10'):>9}{cap:>9}"
              f"{cc:>10}{r['corr_vs_v3']:>10.5f}{r['spearman_vs_v3']:>10.5f}{r['iters']:>8}"
              f"{r['seconds']:>7.0f}")
    print("  'ref cap' = HIT means the REFERENCE also hit the ES cap, so both sides are truncated")
    print("  and the comparison is fair; '-' means only the arm was truncated, which would make its")
    print("  delta a pessimistic artifact of the budget rather than a property of the mechanism.")

    # ---- exact single-slot swaps, fold-0 only, using ACTUAL vectors ----
    print(f"\n  EXACT SINGLE-SLOT SWAPS on fold 0 (replace only slot O_i; other 58 untouched)")
    print(f"  {'arm':<8}{'replaces':<26}{'O_i AUC':>11}{'N_i AUC':>11}{'standalone d':>14}"
          f"{'O-N corr':>10}{'v3 before':>12}{'v3 swap':>12}{'swap d':>10}")
    print(f"  {'-'*116}")
    swaps = []
    for aname in MINI_SLOTS:
        tgt = MINI_SLOTS[aname]
        f = REPORTS / f"{args.tag}_{aname}_primary_f0.npy"
        if not f.exists():
            continue
        fl = get_scheme("primary", y_int, tr[ID_COL]).folds
        val = np.where(fl == 0)[0]
        P = np.load(f).astype("float64")
        cols = [logit(P) if e == tgt else L[e] for e in ids]
        aft = sig(np.mean(np.column_stack(cols), axis=1))
        a_sw = float(roc_auc_score(y[val], aft))
        a_b4 = float(roc_auc_score(y[val], sig(base_logit[val])))
        o_auc = float(roc_auc_score(y[val], sig(L[tgt][val])))
        n_auc = float(roc_auc_score(y[val], P))
        c = float(corr(L[tgt][val], logit(P)))
        swaps.append({"arm": aname, "replaces": tgt, "seed_matches_original":
                      runs["arms"][aname][f"f0"]["seed"] == by_id[tgt]["seed"],
                      "view_matches": runs["arms"][aname][f"f0"]["view"] == by_id[tgt]["view"],
                      "scheme_matches": runs["arms"][aname][f"f0"]["fold_scheme"]
                      == by_id[tgt]["fold_scheme"],
                      "o_auc": o_auc, "n_auc": n_auc,
                      "standalone_delta_e5": (n_auc - o_auc) * 1e5, "o_n_corr": c,
                      "v3_before": a_b4, "v3_after": a_sw, "swap_delta_e5": (a_sw - a_b4) * 1e5})
        print(f"  {aname:<8}{tgt:<26}{o_auc:>11.6f}{n_auc:>11.6f}"
              f"{(n_auc-o_auc)*1e5:>+13.1f}e{c:>10.5f}{a_b4:>12.6f}{a_sw:>12.6f}"
              f"{(a_sw-a_b4)*1e5:>+9.2f}e")

    # ---- MINI block: four real native predictions, both alphas ----
    mini = {}
    have = [a for a in MINI_SLOTS if (REPORTS / f"{args.tag}_{a}_primary_f0.npy").exists()]
    print(f"\n  MINI NATIVE BLOCK on fold 0 (available native arms: {have})")
    if len(have) >= 2:
        fl = get_scheme("primary", y_int, tr[ID_COL]).folds
        val = np.where(fl == 0)[0]
        a_b4 = float(roc_auc_score(y[val], sig(base_logit[val])))
        print(f"  {'block':<12}{'alpha':>7}{'v3 control':>13}{'block AUC':>12}{'delta e5':>11}")
        print(f"  {'-'*56}")
        for alpha, nm in ((1.0, "MINI_B100"), (0.5, "MINI_B50")):
            cols = []
            for e in ids:
                tgt = next((a for a in have if MINI_SLOTS[a] == e), None)
                if tgt is None:
                    cols.append(L[e])
                else:
                    P = np.load(REPORTS / f"{args.tag}_{tgt}_primary_f0.npy").astype("float64")
                    # the counterpart only covers fold 0, so outside it the original stands in.
                    # Reported on FOLD 0 ONLY for that reason.
                    v = logit(P) if e == tgt else L[e]
                    cols.append((1 - alpha) * L[e] + alpha * v if e == tgt else v)
            a = float(roc_auc_score(y[val], sig(np.mean(np.column_stack(cols), axis=1))[val]))
            mini[nm] = {"alpha_native": alpha, "slots": [MINI_SLOTS[a2] for a2 in have],
                        "v3_control_fold0": a_b4, "block_auc_fold0": a,
                        "delta_e5": (a - a_b4) * 1e5, "n_native": len(have)}
            print(f"  {nm:<12}{alpha:>7.2f}{a_b4:>13.6f}{a:>12.6f}{(a-b4)*1e5:>+10.2f}e")
        print(f"  NOTE: these are FOLD-0 numbers only, because each counterpart is trained on fold 0")
        print(f"  alone. A block AUC over all folds is not defined until fold 1+ are trained.")
    else:
        print("    fewer than two native arms present; mini block not computable yet")

    # ---- diversity of the native subset vs the original subset ----
    div = {}
    if len(have) >= 2:
        def stats(get):
            import itertools
            cs, sp = [], []
            for a, b in itertools.combinations(have, 2):
                Pa = np.load(REPORTS / f"{args.tag}_{a}_primary_f0.npy").astype("float64")
                Pb = np.load(REPORTS / f"{args.tag}_{b}_primary_f0.npy").astype("float64")
                cs.append(float(corr(logit(Pa), logit(Pb))))
                sp.append(float(spearman(Pa, Pb)))
            return {"n_pairs": len(cs), "corr_min": min(cs), "corr_median": float(np.median(cs)),
                    "corr_max": max(cs), "spear_min": min(sp),
                    "spear_median": float(np.median(sp)), "spear_max": max(sp)}
        div["native"] = stats(None)
        ocs = []
        import itertools
        for a, b in itertools.combinations(have, 2):
            ocs.append(float(corr(L[MINI_SLOTS[a]], L[MINI_SLOTS[b]])))
        div["original"] = {"n_pairs": len(ocs), "corr_min": min(ocs),
                           "corr_median": float(np.median(ocs)), "corr_max": max(ocs)}
        print(f"\n  DIVERSITY of the {len(have)}-arm subset (fold 0)")
        print(f"    ORIGINAL subset logit corr: min {div['original']['corr_min']:.5f}  "
              f"median {div['original']['corr_median']:.5f}  max {div['original']['corr_max']:.5f}")
        print(f"    NATIVE   subset logit corr: min {div['native']['corr_min']:.5f}  "
              f"median {div['native']['corr_median']:.5f}  max {div['native']['corr_max']:.5f}")
        print(f"    NATIVE   subset Spearman  : min {div['native']['spear_min']:.5f}  "
              f"median {div['native']['spear_median']:.5f}  max {div['native']['spear_max']:.5f}")
        collapse = div["native"]["corr_median"] > div["original"]["corr_median"] + 0.001
        div["collapsed"] = collapse
        print(f"    -> native diversity {'COLLAPSED' if collapse else 'RETAINED'}")

    # ---- stage-1 gate ----
    A = max((abs(r["delta_vs_C2_e5"]) for r in rows
             if r["delta_vs_C2_e5"] is not None), default=0.0)
    mech = [r for r in rows if r["delta_vs_C2_e5"] is not None and r["delta_vs_C2_e5"] >= 5.0]
    npos = sum(1 for s in swaps if s["swap_delta_e5"] > 0)
    block_best = max((abs(m["delta_e5"]) for m in mini.values()), default=0.0)
    gates = {
        "A_mini_block_ge_1e-5": any(m["delta_e5"] >= 1.0 for m in mini.values()),
        "B_two_of_three_swaps_positive_and_diversity_retained":
            npos >= 2 and (not div.get("collapsed", False)),
        "C_mechanism_strength_ge_5e-5": bool(mech),
    }
    print(f"\n  STAGE-1 GATE (predeclared)")
    print(f"    A mini block >= +1.0e-5            : {gates['A_mini_block_ge_1e-5']} "
          f"(best {block_best:+.2f}e-5)")
    print(f"    B >=2/3 swaps positive + diversity : {gates['B_two_of_three_swaps_positive_and_diversity_retained']}"
          f" ({npos}/{len(swaps)} positive)")
    print(f"    C mechanism >= +5e-5 over C2        : {gates['C_mechanism_strength_ge_5e-5']} "
          f"({len(mech)} arm(s))")
    passed = any(gates.values())
    print(f"    -> {'PROCEED to fold 1' if passed else 'CLOSE Phase 11: no gate met, no seven-slot training'}")

    save_json({"rows": rows, "swaps": swaps, "mini_block": mini, "diversity": div,
               "gates": gates, "passed": passed, "n_positive_swaps": npos,
               "best_block_delta_e5": block_best,
               "verdict": ("PROCEED to fold 1 on at least one predeclared gate" if passed
                           else "CLOSE Phase 11 -- no predeclared gate met; the seven-slot plan "
                                "is NOT started")},
              REPORTS / f"{args.tag}_report.json")
    print("\nwrote", REPORTS / f"{args.tag}_report.json")


if __name__ == "__main__":
    main()
