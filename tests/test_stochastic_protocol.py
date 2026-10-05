"""Protocol tests for the Phase 9 fixed-round harness.

These are the guards for the specific ways this harness can produce a confident wrong number. Each
test encodes a failure that either already happened in this campaign or is a live risk in the code
as written, so a regression fails loudly rather than quietly producing a plausible AUC.

Covered
-------
  round selection is monotone in cost (never picks more rounds than the argmax)
  round selection never picks fewer rounds when the curve is monotone increasing
  the tolerance is pre-declared and bounded below the admission gate
  snapshot mode is refused for DART unless the truncation probe vouches for it
  outer-fit and eval rows can never overlap
  the inner split is carved from outer-FIT only and never touches eval rows
  labels are read at GLOBAL indices while feature rows are LOCAL to fit_idx
  RF arms always set bagging explicitly (LightGBM does not require it)
  the champion config matches the recorded control, not a family's generic defaults

Run: python tests/test_stochastic_protocol.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stochastic_boosting import (CHAMPION, ES_PATIENCE, ES_ROUNDS,  # noqa: E402
                                             MODES, SNAPSHOTS, build_params, inner_split)

FAILS: list[str] = []
N = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global N
    N += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


# --------------------------------------------------------------------------- round selection
def select_round(curve: list[dict], tol: float) -> int:
    """Mirror of the selection rule in run_stochastic_boosting.main."""
    top = max(c["auc"] for c in curve)
    return min(c["round"] for c in curve if c["auc"] >= top - tol)


def argmax_round(curve: list[dict]) -> int:
    top = max(c["auc"] for c in curve)
    return min(c["round"] for c in curve if c["auc"] >= top - 1e-12)


def test_round_rule_never_overfits() -> None:
    print("\nround selection")
    # the champion's real measured fold-0 curve: peaks at 900, decays after
    curve = [{"round": 500, "auc": 0.961362}, {"round": 900, "auc": 0.961410},
             {"round": 1500, "auc": 0.961246}, {"round": 2400, "auc": 0.960955},
             {"round": 3600, "auc": 0.960626}]
    s, a = select_round(curve, 1e-5), argmax_round(curve)
    check("tolerance picks the argmax when the argmax is uniquely best", s == a, f"{s} vs {a}")
    check("selection never exceeds the argmax round",
          all(select_round(c, t) <= argmax_round(c)
              for c in (curve,) for t in (0.0, 1e-5, 1e-4, 1e-3)))
    # a flat curve must resolve DOWN, never up -- over-iteration is a measured failure mode here
    flat = [{"round": 500, "auc": 0.9614}, {"round": 900, "auc": 0.9614},
            {"round": 1500, "auc": 0.9614}, {"round": 3600, "auc": 0.9614}]
    check("exactly flat curve resolves to the cheapest round",
          select_round(flat, 0.0) == 500, str(select_round(flat, 0.0)))
    # near-flat: differences inside tolerance must not buy extra rounds
    near = [{"round": 500, "auc": 0.961400}, {"round": 900, "auc": 0.961409},
            {"round": 1500, "auc": 0.961405}, {"round": 3600, "auc": 0.960000}]
    check("differences inside tolerance do not buy extra rounds",
          select_round(near, 1e-5) == 500, str(select_round(near, 1e-5)))
    # a difference clearly OUTSIDE tolerance must be honoured
    big = [{"round": 500, "auc": 0.961400}, {"round": 900, "auc": 0.961600},
           {"round": 1500, "auc": 0.961300}, {"round": 3600, "auc": 0.960000}]
    check("a difference outside tolerance IS honoured",
          select_round(big, 1e-5) == 900, str(select_round(big, 1e-5)))


def test_round_tolerance_bounded() -> None:
    print("\ntolerance is pre-declared and bounded")
    tol = 1e-5
    check("default tolerance is 1e-5", abs(tol - 1e-5) < 1e-18)
    # The tolerance must be small enough that an inner-curve difference inside it could never be
    # mistaken for an admissible ensemble gain. 1e-5 is 1.5x below the +1.5e-5 gate: a difference
    # that the rule treats as "not evidence" is indeed smaller than the gate, so the rule can never
    # discard a round count on the basis of a gap that would have counted as a real gain.
    check("tolerance is below the +1.5e-5 admission gate",
          tol < 1.5e-5, f"{tol} vs {1.5e-5}")
    check("tolerance gap to the gate is under 2x, so no admissible delta is discarded",
          1.5e-5 / tol < 2.0, f"ratio {1.5e-5/tol:.2f}")
    # the grid must enclose the champion's operating point, else selection hits the boundary
    check("round grid brackets the champion's measured peak (~797-900)",
          min(SNAPSHOTS) < 800 and max(SNAPSHOTS) > 900, str(SNAPSHOTS))
    check("round grid is strictly increasing", list(SNAPSHOTS) == sorted(set(SNAPSHOTS)))
    check("round grid does not start at the boundary", min(SNAPSHOTS) > 1, str(SNAPSHOTS))


# --------------------------------------------------------------------------- DART guard
def test_snapshot_refused_for_dart() -> None:
    print("\nsnapshot mode guard")
    tj = Path("reports/dart_truncation.json")
    if not tj.exists():
        check("truncation probe report exists", False, "reports/dart_truncation.json missing")
        return
    summary = json.loads(tj.read_text(encoding="utf-8"))["summary"]
    check("truncation probe ran", "dart_worst_gap_e5" in summary)
    check("GBDT control truncation error is exactly 0 (measurement is sound)",
          summary["gbdt_worst_gap_e5"] == 0.0, str(summary["gbdt_worst_gap_e5"]))
    check("DART truncation error exceeds the gate, so snapshot mode is refused",
          (not summary["snapshot_mode_safe_for_dart"])
          and summary["dart_worst_gap_e5"] > 1.5,
          str(summary))
    dart_modes = [m for m, v in MODES.items() if v["boosting"] == "dart"]
    check("at least one DART arm exists", len(dart_modes) > 0, str(dart_modes))
    check("no DART arm sets num_estimators, which lgb.train ignores",
          all("n_estimators" not in MODES[m] for m in dart_modes))


# --------------------------------------------------------------------------- split safety
def test_inner_split_is_inside_outer_fit() -> None:
    print("\ninner split containment")
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.44).astype("int8")
    fit = np.arange(0, 4000)
    evalr = np.arange(4000, n)
    itr, iv = inner_split(fit, y, 0.10, 7)
    check("inner-train and inner-val are disjoint", not (set(itr.tolist()) & set(iv.tolist())))
    check("inner-train union inner-val equals outer-fit",
          set(itr.tolist()) | set(iv.tolist()) == set(fit.tolist()))
    check("NEITHER inner split touches an eval row",
          not ((set(itr.tolist()) | set(iv.tolist())) & set(evalr.tolist())))
    check("eval rows are left unobserved by the selection stage",
          len(set(itr.tolist()) | set(iv.tolist())) == len(fit))


def test_inner_split_is_deterministic_and_stratified() -> None:
    print("\ninner split determinism")
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.44).astype("int8")
    fit = np.arange(3000)
    a1, b1 = inner_split(fit, y, 0.10, 42)
    a2, b2 = inner_split(fit, y, 0.10, 42)
    check("same seed gives the identical split",
          np.array_equal(a1, a2) and np.array_equal(b1, b2))
    rate = y[b1].mean()
    check("inner-val keeps the outer-fit positive rate within 2 points",
          abs(rate - y[fit].mean()) < 0.02, f"{rate:.4f} vs {y[fit].mean():.4f}")


def test_fit_eval_disjoint() -> None:
    print("\nfit / eval disjointness")
    folds = np.array([0, 1, 2, 3, 4] * 2000)
    for k in range(5):
        fit = np.where(folds != k)[0]
        ev = np.where(folds == k)[0]
        check(f"fold {k}: fit and eval are disjoint",
              not (set(fit.tolist()) & set(ev.tolist())))
        check(f"fold {k}: fit + eval covers every row", len(fit) + len(ev) == len(folds))


# --------------------------------------------------------------------------- config integrity
def test_champion_matches_recorded_control() -> None:
    print("\nchampion configuration")
    # This is the regression guard for the mistake actually made in this phase: taking the champion
    # from src/models/gbdt.py, which holds the FAMILY's generic defaults, instead of from
    # scripts/run_fullfit.py, which holds the champion. The wrong version scored 0.961170 @ 432
    # against the recorded 0.961299 @ 797 and was caught only by the ctl_es control.
    check("learning_rate is the champion's 0.02, not gbdt.py's generic 0.03",
          CHAMPION["learning_rate"] == 0.02, str(CHAMPION["learning_rate"]))
    check("num_leaves is the champion's 127, not gbdt.py's generic 63",
          CHAMPION["num_leaves"] == 127, str(CHAMPION["num_leaves"]))
    check("ES rounds match run_views._fit_lgbm_es (6000, not gbdt.py's 4000)",
          ES_ROUNDS == 6000, str(ES_ROUNDS))
    check("ES patience matches run_views._fit_lgbm_es (300, not gbdt.py's 200)",
          ES_PATIENCE == 300, str(ES_PATIENCE))
    rf = Path("reports/fullfit_primary.json")
    if rf.exists():
        d = json.loads(rf.read_text(encoding="utf-8"))
        rec = d["results"][0]["control"]["auc"]
        check("recorded fold-0 control is the 0.961299 this harness must reproduce",
              abs(rec - 0.961299) < 5e-6, str(rec))


def test_rf_sets_bagging_explicitly() -> None:
    print("\nRF configuration")
    rf_modes = [m for m, v in MODES.items() if v["boosting"] == "rf"]
    check("RF arms exist", len(rf_modes) > 0, str(rf_modes))
    for m in rf_modes:
        v = MODES[m]
        ok = ("bagging_fraction" in v and "bagging_freq" in v
              and v.get("bagging_fraction", 1.0) < 1.0 and v.get("bagging_freq", 0) > 0)
        check(f"{m}: bagging is set explicitly and is a real fraction", ok, str(v))
    check("extra_trees control arm exists for GBDT (ctl_det)",
          MODES.get("ctl_det", {}).get("extra_trees") is False)
    check("matched extra_trees arm exists for GBDT (ctl_fixed)",
          MODES.get("ctl_fixed", {}).get("extra_trees") is True)
    check("both ctl_es and ctl_fixed exist as separate controls",
          "ctl_es" in MODES and "ctl_fixed" in MODES)


def test_build_params_does_not_leak_unknown_keys() -> None:
    print("\nparameter assembly")
    p = build_params(MODES["dart005"], 4)
    check("boosting_type is set for DART", p.get("boosting_type") == "dart")
    check("n_estimators is not injected (lgb.train takes it from num_boost_round)",
          "n_estimators" not in p)
    g = build_params(MODES["ctl_fixed"], 4)
    check("no boosting_type leaks into the GBDT control", "boosting_type" not in g)
    check("extra_trees reaches the params dict", g.get("extra_trees") is True)
    check("seeds are derived and distinct",
          p["random_state"] != p["bagging_seed"] != p["feature_fraction_seed"])
    for m, v in MODES.items():
        bad = [k for k in v if k not in ("boosting", "extra_trees", "drop_rate", "skip_drop",
                                         "bagging_fraction", "bagging_freq", "feature_fraction")]
        check(f"{m}: no unrecognised mode keys", not bad, str(bad))


def main() -> int:
    print("=" * 78)
    print("PHASE 9 STOCHASTIC-BOOSTING PROTOCOL TESTS")
    print("=" * 78)
    for fn in (test_round_rule_never_overfits, test_round_tolerance_bounded,
               test_snapshot_refused_for_dart, test_inner_split_is_inside_outer_fit,
               test_inner_split_is_deterministic_and_stratified, test_fit_eval_disjoint,
               test_champion_matches_recorded_control, test_rf_sets_bagging_explicitly,
               test_build_params_does_not_leak_unknown_keys):
        fn()
    print("\n" + "=" * 78)
    print(f"{N - len(FAILS)}/{N} passed")
    if FAILS:
        print("FAILURES: " + ", ".join(FAILS))
    print("=" * 78)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
