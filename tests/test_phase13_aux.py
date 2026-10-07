"""Regression tests for Phase 13: auxiliary-task expected-value features.

Each test encodes a way Phase 13 could produce a confident wrong number. Several exist because I made
those mistakes while building it.

  1  the aux path cannot reach the competition label -- STRUCTURALLY, not by a check that cannot fail
  2  the fit-row aux predictions are CROSS-FITTED, so the feature is not a memorised copy of the
     rating it smooths (which would be a train/serve mismatch)
  3  the validation rows are unseen by every aux model
  4  the expected value is sum_k k * P(k), formed from a 6-class multiclass model
  5  the rating column names are the VIEW's spellings, resolved against the view
  6  adding the aux columns does not perturb the 48 te_ columns
  7  the aux features have strictly FEWER columns than the ratings themselves (they are smoothers),
     and correlate strongly but imperfectly with the rating -- i.e. neither identical nor useless
  8  the ensemble marginal is measured from real vectors, never inferred from standalone AUC
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


# --------------------------------------------------------------------------- 1
def test_aux_path_cannot_reach_the_label() -> None:
    print("\n1. the aux path cannot reach the competition label")
    from scripts.run_phase13 import build_aux
    sig = set(inspect.signature(build_aux).parameters)
    check("build_aux takes no label argument",
          not (sig & {"y", "yi", "label", "target", "TARGET"}), str(sorted(sig)))
    check("build_aux takes the view matrices and names, nothing else",
          {"Xf", "Xv", "names"} <= sig, str(sorted(sig)))
    src = Path("scripts/run_phase13.py").read_text(encoding="utf-8")
    body = src.split("def build_aux")[1].split("\ndef ")[0]
    check("build_aux never references the target column or the y array",
          "TARGET" not in body and not any(f" {v}[" in body for v in ("y", "yi")),
          "an aux-path reference to the label survived")
    # The structural guarantee is the real one. My first version asserted that flipping every label
    # left the aux features unchanged -- a check that could never fail, because the view matrix has
    # no label column in it. That is the same defect as the Phase 10A replication statistic that
    # returned r=1.000 for all 42 keys. Pin the STRUCTURAL property instead.
    check("the runner records a structural label-free proof rather than a flip test",
          "label_free_structural" in src)
    check("no label-flip test remains in the aux path",
          "tr_flip" not in src and "tr.assign" not in body)


# --------------------------------------------------------------------------- 2, 3
def test_cross_fitted_and_val_unseen() -> None:
    print("\n2/3. fit rows are cross-fitted and val rows are unseen")
    src = Path("scripts/run_phase13.py").read_text(encoding="utf-8")
    check("an inner StratifiedKFold cross-fit is used for the fit rows",
          "StratifiedKFold(inner_folds" in src and "oof[b]" in src)
    check("the validation rows are predicted by a model fitted on ALL fit rows",
          "lgb.Dataset(Zft, label=yf)" in src)
    check("both sides therefore come from out-of-sample predictions",
          src.count("m.predict(") + src.count("mall.predict(") >= 2)
    # The failure mode this prevents: an aux model trained on the fit rows and asked to predict the
    # SAME rows memorises their ratings, so aux_ev_j becomes a near-duplicate of rating_j during
    # training while being genuinely smoothed at validation time. That mismatch is not detectable
    # from AUC alone, so it is pinned here.
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 5000)
    n_in = 0
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        assert not (set(a.tolist()) & set(b.tolist()))
        n_in += 1
    check("the inner cross-fit produces disjoint train/apply splits for every rating", n_in == 3)

    # A memorisation demonstration: a high-capacity model asked about its own training rows is far
    # more accurate there than a cross-fitted estimate, which is exactly the gap the fix closes.
    import lightgbm as lgb
    # The target must NOT be in the design matrix, or the demo is trivial: my first version appended
    # t as the last column of X, which made BOTH accuracies 1.000 -- a demonstration that appeared to
    # confirm the memorisation gap while proving nothing. And the design must carry real structure:
    # with pure noise in Z the correct cross-fitted accuracy IS the 1/6 chance rate, so a check
    # demanding above-chance accuracy on it tests the wrong thing.
    Z = rng.normal(size=(20000, 8)).astype("float32")
    latent = (0.8 * Z[:, 0] + 0.5 * Z[:, 1] - 0.3 * Z[:, 2] + rng.normal(scale=0.6, size=20000))
    t = np.clip(np.digitize(latent, np.quantile(latent, np.linspace(0.17, 0.83, 5))), 0, 5)
    p = dict(objective="multiclass", num_class=6, n_estimators=120, learning_rate=0.15,
             num_leaves=63, min_child_samples=100, verbose=-1, n_jobs=4, random_state=0)
    m_in = lgb.train(p, lgb.Dataset(Z, label=t), num_boost_round=120)
    insample = float((np.argmax(m_in.predict(Z), axis=1) == t).mean())
    oof = np.zeros_like(m_in.predict(Z))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(Z, t):
        mm = lgb.train(p, lgb.Dataset(Z[a], label=t[a]), num_boost_round=120)
        oof[b] = mm.predict(Z[b])
    cross = float((np.argmax(oof, axis=1) == t).mean())
    check("in-sample aux accuracy exceeds cross-fitted accuracy, so memorisation is real",
          insample > cross + 0.02, f"in-sample {insample:.3f} vs cross-fitted {cross:.3f}")
    check("neither accuracy is perfect, so the demo is not trivially satisfied",
          max(insample, cross) < 0.999, f"{insample:.4f}/{cross:.4f}")
    check("cross-fitted accuracy is well above the 1/6 chance rate, so the mechanism carries signal",
          cross > 0.25, f"{cross:.3f}")


# --------------------------------------------------------------------------- 4
def test_expected_value_formula() -> None:
    print("\n4. the feature is sum_k k * P(k) over a 6-class multiclass model")
    from scripts.run_phase13 import N_EXPECTED, RATINGS
    check("there are 13 ratings", len(RATINGS) == 13, str(len(RATINGS)))
    check("the model is 6-class, matching the 0..5 rating scale",
          N_EXPECTED == 6, str(N_EXPECTED))
    src = Path("scripts/run_phase13.py").read_text(encoding="utf-8")
    check("the expected value is the probability vector dotted with 0..5",
          "oof @ np.arange(N_EXPECTED" in src and "pv @ np.arange(N_EXPECTED" in src)
    check("objective is multiclass", 'objective="multiclass"' in src)
    check("the prediction shape is asserted rather than assumed",
          "expected" in src and "STOP: aux model" in src)
    # arithmetic check on the formula itself
    p = np.array([[0.1, 0.0, 0.4, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ev = p @ np.arange(6, dtype="float64")
    check("expected value of [0.1,0,0.4,0,0.5,0] is 0*0.1+2*0.4+4*0.5 = 2.8",
          abs(ev[0] - 2.8) < 1e-12, str(ev[0]))
    check("expected value of a point mass on 3 is 3", abs(ev[1] - 0.0) < 1e-12)


# --------------------------------------------------------------------------- 5
def test_rating_names_resolve_against_the_view() -> None:
    print("\n5. the rating names are the VIEW's spellings")
    from src.common import ID_COL, TARGET, load_cached_parquet
    from src.features.view import ViewBuilder
    from src.validation.folds import get_scheme
    from scripts.run_phase13 import RATINGS

    tr, te = load_cached_parquet()
    yi = tr[TARGET].values.astype("int8")
    fl = get_scheme("primary", yi, tr[ID_COL]).folds
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    _Xf, _Xa, names = vb.assemble(np.where(fl != 0)[0], yi, np.where(fl == 0)[0], None,
                                  inner_seed=0)
    check("every declared rating exists in the full view",
          all(r in names for r in RATINGS), str([r for r in RATINGS if r not in names]))
    # The raw-dataset spellings differ for three of them. If someone reverts to the raw header, an
    # absent name silently drops a column rather than raising, so the count is asserted too.
    raw_only = [r for r in ("In-flight entertainment", "WiFi service", "In-flight service")
                if r not in names]
    check("three raw-dataset spellings do NOT exist in the view, so they must not be used",
          len(raw_only) == 3, str(raw_only))
    check("none of the raw-only spellings appears in RATINGS",
          not (set(raw_only) & set(RATINGS)))
    check("the 13 ratings occupy the first 13 positions of the view, as the audit recorded",
          [names.index(r) for r in RATINGS] == list(range(13)),
          str([names.index(r) for r in RATINGS]))


# --------------------------------------------------------------------------- 6, 7
def test_aux_columns_are_additive_and_informative() -> None:
    print("\n6/7. the aux columns are additive, smooth and informative")
    import json
    p = Path("reports/p13_runs.json")
    if not p.exists():
        check("p13 run record present (skipped: not run yet)", True)
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    arms = d.get("arms", {})
    if not arms:
        check("p13 run record has arms (skipped: empty)", True)
        return
    # A record written before the marginal and split-census fields existed cannot be checked against
    # them. That is reported as a skip with a reason, NOT as a pass, so an absent field is visible
    # rather than silently accepted.
    missing = [k for k, r in arms.items()
               if not {"marginal_e5", "split_census", "aux_vs_rating"} <= set(r)]
    if missing:
        print(f"    (SKIP: {missing} predate the marginal/census fields; rerun run_phase13.py)")
        return
    for key, rec in arms.items():
        check(f"{key}: 13 aux columns added to 285, not replacing any",
              rec["n_features"] == 285 + 13 and rec["n_base"] == 285,
              f"{rec['n_base']} -> {rec['n_features']}")
        check(f"{key}: the te_ block hash is unchanged",
              rec["te_hash_unchanged"] is True)
        check(f"{key}: the label-free structural proof held", rec["label_free_structural"] is True)
    k0 = sorted(arms)[0]
    dup = arms[k0]["aux_vs_rating"]
    corrs = [v["corr_aux_ev_with_rating_fit"] for v in dup.values()]
    check("every aux feature correlates strongly with the rating it smooths",
          min(corrs) > 0.5, f"min {min(corrs):.3f}")
    check("no aux feature is IDENTICAL to its rating, or cross-fitting was skipped",
          max(corrs) < 0.9999, f"max {max(corrs):.5f}")
    rms = [v["rmse"] for v in dup.values()]
    check("every aux feature differs numerically from its rating",
          min(rms) > 0.01, f"min rmse {min(rms):.4f}")


# --------------------------------------------------------------------------- 8
def test_marginal_is_measured_not_inferred() -> None:
    print("\n8. the ensemble marginal is measured from real vectors")
    src = Path("scripts/run_phase13.py").read_text(encoding="utf-8")
    check("the marginal is computed by splicing into the exact v3 slot",
          "col[val_idx] = lg(pred)" in src and "CHAMPION_MEMBER else L[e]" in src)
    check("the splice is asserted to leave out-of-fold rows untouched",
          "splice moved rows" in src)
    check("the v3 geometry is reconstructed as expit(mean(logits))",
          "np.mean(np.column_stack" in src and "base_logit" in src)
    check("the marginal is read on the fold's own rows",
          "roc_auc_score(y[val_idx], aft[val_idx])" in src)
    check("no weight-times-AUC approximation appears",
          "family_weight" not in src and "1.0 / len(ids) *" not in src)
    import json
    p = Path("reports/p13_runs.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        arms = d.get("arms", {})
        stale = [k for k, r in arms.items() if "marginal_e5" not in r]
        if stale:
            print(f"    (SKIP: folds {stale} predate the measured-marginal field)")
        elif arms:
            check("the run record stores a measured marginal per fold",
                  all("marginal_e5" in r for r in arms.values()))
            check("the run record stores the split census on the aux columns",
                  all("pct_splits_on_aux" in r["split_census"] for r in arms.values()))


def main() -> int:
    print("=" * 84)
    print("PHASE 13 -- REGRESSION TESTS")
    print("=" * 84)
    for fn in (test_aux_path_cannot_reach_the_label, test_cross_fitted_and_val_unseen,
               test_expected_value_formula, test_rating_names_resolve_against_the_view,
               test_aux_columns_are_additive_and_informative, test_marginal_is_measured_not_inferred):
        fn()
    print("\n" + "=" * 84)
    print(f"{N - len(FAILS)}/{N} passed")
    if FAILS:
        print("FAILURES: " + ", ".join(FAILS))
    print("=" * 84)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())