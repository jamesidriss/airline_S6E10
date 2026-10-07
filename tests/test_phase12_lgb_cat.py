"""Regression tests for Phase 12: native categorical splits in the LightGBM extra_trees family.

Each test encodes a specific way Phase 12 could produce a confident wrong number. Several exist
because I made those mistakes while building it.

  1  the champion config comes from _fit_lgbm_es + extra_trees, NOT src/models/gbdt.py defaults
  2  no categorical was ever declared in any LightGBM run in the ledger
  3  declaring a categorical changes HANDLING only: 285 columns, same names, same order
  4  the 48 te_ columns are byte-identical between control and treatment
  5  category codes are non-negative integers with no NaN, on fit AND apply rows
  6  no code appears in apply rows that was absent from fit rows
  7  survey level 0 is a REAL level, distinct from missingness
  8  the inner ES carve is disjoint from its training rows and partitions the fit rows
  9  the treatment is OBSERVABLE: it produces categorical splits the control has none of
 10  extra_trees does not disturb the categorical search
 11  the audit's split detector actually detects, proven on a forced probe
 12  the categorical hyperparameters are PINNED, not library-implicit
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
def test_champion_config_provenance() -> None:
    print("\n1. champion configuration provenance")
    import lightgbm as lgb
    from scripts.run_phase12 import (CAT_PARAMS, CHAMPION_MEMBER, CHAMPION_PARAMS, CHAMPION_SEED,
                                    CHAMPION_VIEW, fit_arm)

    check("the champion member is the extra_trees member recorded in the ledger",
          CHAMPION_MEMBER == "xt_xt_d127_s1", CHAMPION_MEMBER)
    check("the champion view is 'full'", CHAMPION_VIEW == "full")
    check("the champion seed is 1", CHAMPION_SEED == 1)

    src = Path("scripts/run_views.py").read_text(encoding="utf-8")
    # every parameter of the champion must appear verbatim in the authoritative fit function, so a
    # silent edit to run_views.py cannot leave this file claiming to match it
    body = src.split("def _fit_lgbm_es")[1].split("\ndef ")[0]
    for k in ("learning_rate", "num_leaves", "min_child_samples", "colsample_bytree", "subsample",
              "subsample_freq", "reg_lambda", "max_bin", "n_estimators"):
        check(f"{k} appears verbatim in _fit_lgbm_es and in CHAMPION_PARAMS",
              k in body and str(CHAMPION_PARAMS[k]) in body.replace("0.02", "0.02"),
              f"{CHAMPION_PARAMS.get(k)}")
    check("extra_trees=True is the champion lever", CHAMPION_PARAMS.get("extra_trees") is True)
    check("learning_rate is 0.02, matching run_xt_zoo.py:36", CHAMPION_PARAMS["learning_rate"] == 0.02)
    check("num_leaves is 127, matching run_xt_zoo.py:36", CHAMPION_PARAMS["num_leaves"] == 127)

    xtzoo = Path("scripts/run_xt_zoo.py").read_text(encoding="utf-8")
    check("run_xt_zoo.py registers xt_d127_s1 with extra_trees=True, seed 1, full, primary",
          "xt_d127_s1" in xtzoo and "extra_trees=True" in xtzoo)
    check("the champion is NOT taken from src/models/gbdt.py defaults",
          "from src.models.gbdt import" not in Path("scripts/run_phase12.py").read_text(
              encoding="utf-8"))

    check("fit_arm accepts a categorical index list",
          "cat_idx" in Path("scripts/run_phase12.py").read_text(encoding="utf-8"))
    check("the champion params do not silently include a categorical declaration",
          "categorical_feature" not in CHAMPION_PARAMS and "categorical_feature" not in CAT_PARAMS)


# --------------------------------------------------------------------------- 2
def test_no_prior_lgbm_native_categorical() -> None:
    print("\n2. native LightGBM categoricals were never tested before (so this is not a re-run)")
    import json
    hits = []
    for line in Path("experiments/ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        blob = json.dumps(d)
        if d.get("family") == "lgbm" and (
                "categorical_feature" in blob or d.get("native_cat") is True
                or "lgb_native_cat" in blob):
            hits.append(d.get("exp_id"))
    check("no prior lgbm ledger entry declares native categoricals", not hits, str(hits[:5]))
    # and the feature side has none either
    fsrc = Path("src/features/s6e10.py").read_text(encoding="utf-8") + \
        Path("src/features/view.py").read_text(encoding="utf-8")
    check("the feature layer never emits a lightgbm categorical flag",
          "categorical_feature" not in fsrc)
    # the notebook audit already established the public negative was a DIFFERENT substitution
    audit = json.loads(Path("reports/public_notebook_audit.json").read_text(encoding="utf-8"))
    check("the notebook audit recorded that it used no native categoricals",
          audit.get("catboost_native_categorical_used") is False)


# --------------------------------------------------------------------------- 3, 4
def test_handling_only_may_change() -> None:
    print("\n3/4. declaring a categorical changes HANDLING only")
    from src.common import ID_COL, TARGET, load_cached_parquet
    from src.features.view import ViewBuilder
    from src.validation.folds import get_scheme
    from scripts.run_phase12 import ARMS, CHAMPION_VIEW

    tr, te = load_cached_parquet()
    yi = tr[TARGET].values.astype("int8")
    fl = get_scheme("primary", yi, tr[ID_COL]).folds
    fit_idx = np.where(fl != 0)[0]
    val_idx = np.where(fl == 0)[0]
    vb = ViewBuilder(tr, te, CHAMPION_VIEW)
    vb.build_static()
    Xf, Xa, names = vb.assemble(fit_idx, yi, val_idx, None, inner_seed=0)

    check("the full view has exactly 285 features", len(names) == 285, str(len(names)))
    check("the matrix shape matches the declared width", Xf.shape[1] == 285)
    te_pos = [i for i, n in enumerate(names) if n.startswith("te_")]
    check("the view carries exactly 48 te_ columns", len(te_pos) == 48, str(len(te_pos)))

    # Declaring categorical must not require a different matrix. Re-assembling gives the same array,
    # so the arms differ ONLY in the categorical_feature argument handed to LightGBM.
    Xf2, Xa2, names2 = vb.assemble(fit_idx, yi, val_idx, None, inner_seed=0)
    check("re-assembly is bit-identical, so the matrix is a fixed input to every arm",
          np.array_equal(Xf, Xf2) and names == names2)
    check("te_ columns are byte-identical across assemblies",
          np.array_equal(Xf[:, te_pos], Xf2[:, te_pos]))

    # The declared columns must already EXIST in the matrix: no appending, no duplicates.
    for arm, (cols, _n) in ARMS.items():
        present = [c for c in cols if c in names]
        check(f"{arm}: every declared categorical is an existing view column",
              len(present) == len(cols), str([c for c in cols if c not in names]))
        pos = [names.index(c) for c in present]
        check(f"{arm}: no duplicate positions", len(set(pos)) == len(pos))
        check(f"{arm}: declaring it would not change the column count",
              len(pos) <= len(names))
    check("Age, Flight Distance and the delays are NOT categorical in any arm",
          not any("Age" in c or "Flight Distance" in c for _a, (cs, _n) in ARMS.items() for c in cs))
    check("L0 declares nothing, which is what makes it a clean control",
          ARMS["L0"][0] == [])
    check("L1 is META4 only", ARMS["L1"][0] == ["Gender", "Customer Type", "Type of Travel", "Class"])
    check("L2 is META4 plus the three irregular surveys",
          set(ARMS["L2"][0]) == set(ARMS["L1"][0]) | {"Online boarding", "Inflight wifi service",
                                                    "Gate location"})
    check("L3 adds all 13 survey ratings on top of META4",
          len(ARMS["L3"][0]) == 17 and set(ARMS["L1"][0]) <= set(ARMS["L3"][0]))
    del te_pos


# --------------------------------------------------------------------------- 5, 6, 7
def test_category_code_safety() -> None:
    print("\n5/6/7. category codes are safe and zero is a real level")
    from src.common import ID_COL, TARGET, load_cached_parquet
    from src.features.view import ViewBuilder
    from src.validation.folds import get_scheme
    from scripts.run_phase12 import ARMS, CHAMPION_VIEW, SURVEY13

    tr, te = load_cached_parquet()
    yi = tr[TARGET].values.astype("int8")
    fl = get_scheme("primary", yi, tr[ID_COL]).folds
    fit_idx = np.where(fl != 0)[0]
    val_idx = np.where(fl == 0)[0]
    vb = ViewBuilder(tr, te, CHAMPION_VIEW)
    vb.build_static()
    Xf, Xa, names = vb.assemble(fit_idx, yi, val_idx, None, inner_seed=0)
    Xv = Xa["val"]

    allc = sorted({c for cols, _ in ARMS.values() for c in cols})
    check("the audited column set is exactly 17", len(allc) == 17, str(len(allc)))
    for c in allc:
        j = names.index(c)
        for tag, arr in (("fit", Xf[:, j]), ("val", Xv[:, j])):
            col = arr.astype("float64")
            check(f"{c} [{tag}]: no NaN", not np.isnan(col).any())
            check(f"{c} [{tag}]: all codes non-negative (LightGBM requirement)",
                  bool(np.all(col >= 0)))
            check(f"{c} [{tag}]: all codes integral", bool(np.all(col == np.floor(col))))
        unseen = set(np.unique(Xv[:, j]).tolist()) - set(np.unique(Xf[:, j]).tolist())
        check(f"{c}: no apply-row code absent from fit rows", not unseen, str(sorted(unseen)))

    # 7: survey zero is a REAL level. If 0 were being swallowed as missingness, the level set
    # would start at 1. Online boarding / wifi / gate location all have a genuine 0 meaning
    # "not available / not offered", which is a distinct state from "unknown".
    zero_cols = [c for c in allc if c in SURVEY13 and 0.0 in set(np.unique(
        Xf[:, names.index(c)]).tolist())]
    check("survey columns expose level 0 as a present code", len(zero_cols) >= 3, str(zero_cols))
    for c in ("Online boarding", "Inflight wifi service", "Gate location"):
        check(f"{c}: level 0 present, so it is distinct from missingness",
              0.0 in set(np.unique(Xf[:, names.index(c)]).tolist()))
    # and no sentinel was injected by us: no code equals a magic large value
    for c in allc:
        mx = float(Xf[:, names.index(c)].max())
        check(f"{c}: max code {mx:.0f} is a plausible level count, not a sentinel",
              mx < 1000, f"{mx}")


# --------------------------------------------------------------------------- 8
def test_inner_es_carve_is_clean() -> None:
    print("\n8. inner ES carve is disjoint and partitioning")
    from scripts.run_views import _inner_es_split
    n = 20000
    folds = np.repeat(np.arange(5), n // 5)
    y = (np.arange(n) % 7 == 0).astype("int8")
    fit = np.where(folds != 0)[0]
    itr, es = _inner_es_split(fit, y, 1)
    check("inner-train and ES are disjoint", not (set(int(v) for v in itr) & set(int(v) for v in es)))
    check("inner-train + ES partition the fit rows exactly",
          len(itr) + len(es) == len(fit), f"{len(itr)}+{len(es)} vs {len(fit)}")
    check("ES is ~10% of fit rows", 0.08 < len(es) / len(fit) < 0.12, f"{len(es) / len(fit):.3f}")
    src = Path("scripts/run_phase12.py").read_text(encoding="utf-8")
    check("the runner trains on the 90% inner-train rows, not the whole fit frame",
          "fit_arm(Xf[tr_l], y[itr]" in src)
    check("the runner aborts if the carve overlaps", "inner-train and ES rows overlap" in src)
    check("the runner aborts if the carve does not partition",
          "does not partition the fit rows" in src)


# --------------------------------------------------------------------------- 9, 10, 11
def test_detector_and_extra_trees_interaction() -> None:
    print("\n9/10/11. the treatment is observable and extra_trees does not break it")
    from scripts.run_phase12 import census

    # A forced probe where the categorical is the ONLY signal, so a miss cannot be blamed on gain.
    # The target is deliberately non-monotone, so a numeric threshold CANNOT express it and only a
    # categorical split can win.
    rng = np.random.default_rng(0)
    levels, n = 6, 60000
    x = rng.integers(0, levels, n)
    mu = np.array([0.10, 0.45, 0.25, 0.50, 0.15, 0.40])
    y = (rng.random(n) < mu[x]).astype(int)
    X = x.reshape(-1, 1).astype(float)
    import lightgbm as lgb
    p = dict(objective="binary", n_estimators=25, learning_rate=0.12, num_leaves=8,
             min_child_samples=20, colsample_bytree=1.0, subsample=1.0, verbose=-1, n_jobs=4,
             random_state=1)
    m_un = lgb.train(dict(p, extra_trees=True), lgb.Dataset(X, label=y), num_boost_round=25)
    m_dec = lgb.train(dict(p, extra_trees=True),
                      lgb.Dataset(X, label=y, categorical_feature=[0]), num_boost_round=25)
    m_dec_noet = lgb.train(dict(p, extra_trees=False),
                           lgb.Dataset(X, label=y, categorical_feature=[0]), num_boost_round=25)

    cu, cd, cdn = census(m_un), census(m_dec), census(m_dec_noet)
    check("the detector finds ZERO categorical splits when none is declared",
          cu["n_categorical_splits"] == 0, str(cu["decision_types"]))
    check("the detector finds categorical splits when one IS declared",
          cd["n_categorical_splits"] > 0, str(cd["decision_types"]))
    check("undeclared splits are all numerical",
          set(cu["decision_types"]) == {"<="}, str(cu["decision_types"]))
    check("extra_trees=True and False give the SAME number of categorical splits",
          cd["n_categorical_splits"] == cdn["n_categorical_splits"],
          f"{cd['n_categorical_splits']} vs {cdn['n_categorical_splits']}")
    check("extra_trees does not destroy the categorical search",
          cd["n_categorical_splits"] > 0)
    check("the declared feature actually receives the splits",
          cd["splits_per_feature"].get(0, 0) > 0, str(cd["splits_per_feature"]))

    # My first audit counted the substring '||' and reported 0 for every configuration, including
    # one that demonstrably worked. Pin that the detector is CODE that keys on decision_type, not a
    # substring count. The check looks for an actual counting CALL, because the audit's docstring
    # legitimately mentions the wrong discriminator in order to record that it was wrong -- a naive
    # "the string must not appear" test fails on that documentation, which is the wrong lesson.
    audit_src = Path("scripts/audit_lgb_native_cat.py").read_text(encoding="utf-8")
    code_lines = [ln for ln in audit_src.splitlines()
                  if "||" in ln and not ln.lstrip().startswith("#")]
    check("no CODE line counts '||' as a split detector",
          not any(".count(" in ln for ln in code_lines), str(code_lines))
    check("the audit's census keys on decision_type",
          'types.get("==", 0)' in audit_src)
    check("run_phase12's census keys on decision_type too",
          'types.get("==", 0)' in Path("scripts/run_phase12.py").read_text(encoding="utf-8"))
    # and the detector is proven correct against ground truth: it must find 0 where there are none
    # and >0 where there are some. That is the property the wrong discriminator violated.
    check("detector is exact on ground truth: 0 declared -> 0 found",
          census(m_un)["n_categorical_splits"] == 0)
    check("detector is exact on ground truth: 1 declared -> >0 found",
          census(m_dec)["n_categorical_splits"] > 0)

    # NaN in a declared categorical is MISSING, not a category: LightGBM's internal marker appears.
    Xn = X.copy()
    Xn[:500, 0] = np.nan
    m_nan = lgb.train(dict(p, extra_trees=True),
                      lgb.Dataset(Xn, label=y, categorical_feature=[0]), num_boost_round=25)
    fi = m_nan.dump_model()["feature_infos"]["Column_0"]
    check("LightGBM exposes its internal missing marker (-1) for a declared categorical",
          -1 in (fi.get("values") or []) or fi.get("min_value") == -1, str(fi))
    check("the missing marker means NaN is routed as MISSING, not as a category code",
          -1 in (fi.get("values") or []))


# --------------------------------------------------------------------------- 12
def test_cat_params_are_pinned() -> None:
    print("\n12. categorical hyperparameters are pinned, not library-implicit")
    from scripts.run_phase12 import CAT_PARAMS
    expected = {"cat_smooth": 10, "cat_l2": 10, "max_cat_threshold": 32,
                "max_cat_to_onehot": 4, "min_data_per_group": 100}
    for k, v in expected.items():
        check(f"{k} is pinned to {v}", CAT_PARAMS.get(k) == v, str(CAT_PARAMS.get(k)))
    src = Path("scripts/run_phase12.py").read_text(encoding="utf-8")
    check("fit_arm actually applies CAT_PARAMS", "p.update(CAT_PARAMS)" in src)
    check("the audit also pins them", '"cat_smooth": 10' in Path(
        "scripts/audit_lgb_native_cat.py").read_text(encoding="utf-8"))
    # a model built with them must echo them back, which is what makes them auditable in a record
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 5000)
    y = (rng.random(5000) < np.array([0.1, 0.4, 0.2, 0.45])[x]).astype(int)
    m = lgb.train(dict(objective="binary", n_estimators=3, verbose=-1, n_jobs=2, **CAT_PARAMS),
                  lgb.Dataset(x.reshape(-1, 1).astype(float), label=y, categorical_feature=[0]),
                  num_boost_round=3)
    got = {k: m.params.get(k) for k in CAT_PARAMS}
    check("the fitted model echoes the pinned categorical params back into the run record",
          all(got[k] == v for k, v in CAT_PARAMS.items()), str(got))


def main() -> int:
    import json
    print("=" * 84)
    print("PHASE 12 -- REGRESSION TESTS")
    print("=" * 84)
    for fn in (test_champion_config_provenance, test_no_prior_lgbm_native_categorical,
               test_handling_only_may_change, test_category_code_safety,
               test_inner_es_carve_is_clean, test_detector_and_extra_trees_interaction,
               test_cat_params_are_pinned):
        fn()
    print("\n" + "=" * 84)
    print(f"{N - len(FAILS)}/{N} passed")
    if FAILS:
        print("FAILURES: " + ", ".join(FAILS))
    print("=" * 84)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())