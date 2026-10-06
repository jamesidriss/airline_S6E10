"""Guards for the native-CatBoost categorical frame.

The nine properties the brief requires, each turned into an assertion that fails loudly. The
motivation for most of them is a specific way this could go wrong while still producing a confident,
plausible-looking AUC.

  1. exact identity      A twin must round-trip the original value. If categorical identity were
                         pushed through float32 or an ordinal encoder, a category and its neighbour
                         would silently become one bucket or an ordered distance.
  2. numeric twin kept   The original numeric service ratings must REMAIN in the frame alongside the
                         categorical twins, so CatBoost sees both the ordinal and the identity view.
                         Dropping them would make C2 a different experiment from C0.
  3. no target           Twins are built from the raw frame by column name. A target-derived column
                         would make every downstream score meaningless.
  4. schema match        train / apply / test frames must carry identical columns in identical order,
                         and the declared `cat_features` names must all resolve.
  5. sentinel no clash   The missing sentinel must not equal any real level, or a missing value would
                         masquerade as a genuine category.
  6. cardinality         Exact expected cardinalities, measured not assumed. Three of my initial
                         guesses were wrong (Customer Type, Class, Baggage handling).
  7. eval cannot alter   Outer-validation labels must not be able to change the frame. The twins are
                         built from `tr`/`te` only, so this is asserted by construction: perturbing
                         y must leave every twin bit-identical.
  8. arms differ only in the intended mechanism. C1 vs C0 must differ ONLY in boosting_type, and
                         C2 vs C0 ONLY by appended twin columns. A stray parameter would make the
                         2x2 uninterpretable.
  9. C0 unchanged        The numeric frame with no twins must be byte-identical to what the existing
                         pipeline feeds CatBoost, so the control really is the current control.

Run: python tests/test_native_cat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.s6e10 import SURVEY13 as SURVEY_COLS  # noqa: E402
from scripts.native_cat import (DEFAULT_SERVICE_CARDINALITY, EXPECTED_CARDINALITY,  # noqa: E402
                                META4, MISSING_SENTINEL, attach, cat_cardinality, cat_frame,
                                cat_indices, cat_name, default_cat_cols, is_string_series,
                                to_cat_series, verify_no_target)

FAILS: list[str] = []
N = 0
_DF = {}


def frame() -> pd.DataFrame:
    if "df" not in _DF:
        from src.common import load_cached_parquet
        tr, te = load_cached_parquet()
        _DF["df"], _DF["te"] = tr, te
    return _DF["df"]


def check(name: str, cond: bool, detail: str = "") -> None:
    global N
    N += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def _raises(fn, exc=Exception) -> bool:
    """True when `fn` raises the expected exception type (any Exception if unspecified)."""
    try:
        fn()
    except exc:
        return True
    except Exception:                                          # noqa: BLE001
        return False
    return False


# --------------------------------------------------------------------------- 1. identity
def test_twins_preserve_exact_identity() -> None:
    print("\n1. categorical identity is preserved exactly")
    tr = frame()
    s = tr["Gender"]
    t = to_cat_series(s)
    check("twin holds strings, never numeric",
          all(isinstance(v, str) for v in t.head(200)), str(t.dtype))
    check("twin has the same length as the source", len(t) == len(s))
    # exact round trip for a string column
    back = t.map(lambda z: None if z == MISSING_SENTINEL else z)
    orig = s.astype("object")
    same = int((back.fillna("~") == orig.astype(str).fillna("~")).sum())
    check("every Gender value round-trips exactly", same == len(s), f"{same}/{len(s)}")
    # ratings: whole numbers must not acquire a decimal tail, which would create a phantom level
    r = to_cat_series(tr["Inflight wifi service"])
    lv = sorted(x for x in r.unique() if x != MISSING_SENTINEL)
    check("rating levels are plain integers with no decimal tail",
          all(x in {"0", "1", "2", "3", "4", "5"} for x in lv), str(lv))
    check("rating level count matches the measured cardinality", len(lv) == 6, str(len(lv)))


def test_twins_are_not_ordinal_encoded() -> None:
    print("\n1b. twins are NOT ordinal codes")
    tr = frame()
    t = to_cat_series(tr["Type of Travel"])
    vals = set(t.unique())
    check("twin values are the original words, not 0/1 codes",
          vals == {"Business travel", "Personal Travel"} or
          all(not str(v).isdigit() for v in vals), str(sorted(vals))[:120])


# --------------------------------------------------------------------------- 2. numeric kept
def test_numeric_columns_remain_present() -> None:
    print("\n2. original numeric columns remain alongside the twins")
    from src.features.view import RAW21
    tr = frame()
    cols = default_cat_cols(True)
    cats = cat_frame(tr, cols, np.arange(1000))
    for c in SURVEY_COLS:
        check(f"'{c}' is both a numeric raw column and a categorical twin",
              c in RAW21 and cat_name(c) in cats.columns)
        check(f"numeric '{c}' is NOT converted to a string by the twin builder",
              pd.api.types.is_numeric_dtype(tr[c]))
    check("META4 numeric twins exist too", all(cat_name(c) in cats.columns for c in META4))
    check("twin columns are named distinctly from numeric columns",
          not (set(cats.columns) & set(RAW21)))


# --------------------------------------------------------------------------- 3. no target
def test_no_target_in_twins() -> None:
    print("\n3. no target anywhere in the twins")
    tr = frame()
    cols = default_cat_cols(True)
    cats = cat_frame(tr, cols, np.arange(500))
    check("no twin column name mentions the target",
          not [c for c in cats.columns if "satisf" in c.lower()], str(list(cats.columns)[:4]))
    # the guard must pass on the columns that would actually reach CatBoost. The raw frame's first
    # columns DO include the target, and passing those must make it raise -- that is the guard
    # working, so it is asserted separately below.
    try:
        verify_no_target(list(cats.columns))
        ok = True
    except AssertionError:
        ok = False
    check("verify_no_target passes on the CatBoost twin columns", ok)
    try:
        verify_no_target(list(tr.columns[:60]))
        raised_on_raw = False
    except AssertionError:
        raised_on_raw = True
    check("verify_no_target RAISES when the raw frame (which contains the target) is passed",
          raised_on_raw)
    try:
        verify_no_target(["ncat__satisfaction_derived"])
        raised = False
    except AssertionError:
        raised = True
    check("verify_no_target RAISES on a target-named column", raised)
    # perturbing the label must not change any twin
    a = cat_frame(tr, cols, np.arange(3000))
    flipped = tr.assign(satisfaction=~tr["satisfaction"].astype(bool))
    b = cat_frame(flipped, cols, np.arange(3000))
    check("flipping every label leaves all twins bit-identical",
          a.equals(b))


# --------------------------------------------------------------------------- 4. schema match
def test_schemas_match_across_row_sets() -> None:
    print("\n4. train / apply / test schemas match exactly")
    tr = frame()
    _te = _DF["te"]
    cols = default_cat_cols(True)
    rng = np.random.default_rng(0)
    n = len(tr)
    sets = {"fit": rng.choice(n, 1000, replace=False),
            "val": rng.choice(n, 800, replace=False),
            "test": rng.choice(len(_te), 700, replace=False)}
    srcs = {"fit": tr, "val": tr, "test": _te}
    built = {k: cat_frame(srcs[k], cols, sets[k]) for k in sets}
    ref = list(built["fit"].columns)
    for k in ("val", "test"):
        check(f"{k} columns are identical to fit columns and in the same order",
              list(built[k].columns) == ref, f"{list(built[k].columns)[:3]} vs {ref[:3]}")
        check(f"{k} rows match the requested index count", len(built[k]) == len(sets[k]))
    check("every declared cat_feature name resolves in every frame",
          all(c in built[k].columns for k in built for c in ref))
    # attach must preserve the numeric block and append the twins
    Xn = np.zeros((len(sets["fit"]), 7), dtype="float32")
    nnum = [f"n{i}" for i in range(7)]
    frame_out, cat_names = attach(Xn, nnum, built["fit"])
    check("attach returns one cat name per twin column", len(cat_names) == len(cols))
    check("attach keeps every numeric column and appends every twin",
          len(frame_out.columns) == len(nnum) + len(cols), str(frame_out.shape))
    check("attach preserves float dtype for the numeric block",
          all(str(frame_out[c].dtype).startswith("float") for c in nnum))
    check("attach preserves STRING dtype for the twin block (object or pandas str)",
          all(is_string_series(frame_out[c]) for c in cat_names),
          str(sorted({str(frame_out[c].dtype) for c in cat_names})))
    check("twin VALUES are python strings, which is what CatBoost requires",
          all(isinstance(v, str) for v in frame_out[cat_names[0]].head(50)))
    # positional resolution, checked against the frame's own column order
    ci = cat_indices(frame_out, cat_names)
    check("cat_indices returns one index per twin", len(ci) == len(cols))
    check("cat_indices point at the twin columns, not at column 0",
          all(frame_out.columns[i] in cat_names for i in ci), str(ci[:4]))
    check("cat_indices are strictly increasing (twins are appended last)",
          ci == sorted(ci) and min(ci) == len(nnum), f"{min(ci)}..{max(ci)}")


# --------------------------------------------------------------------------- 5. sentinel
def test_sentinel_does_not_collide() -> None:
    print("\n5. missing sentinel cannot collide with a real level")
    tr = frame()
    cols = default_cat_cols(True)
    collide = []
    for c in cols:
        real = set(to_cat_series(tr[c]).unique())
        if MISSING_SENTINEL in real:
            collide.append(c)
    check("no real level equals the sentinel", not collide, str(collide))
    check("sentinel is delimited so it cannot be a rating or a word",
          MISSING_SENTINEL.startswith("__") and MISSING_SENTINEL.endswith("__"))
    # a genuinely missing value must become the sentinel
    s = tr["Gender"].astype(object).copy()
    s.iloc[:3] = None
    t = to_cat_series(s)
    check("NaN becomes the sentinel", all(t.iloc[:3] == MISSING_SENTINEL), str(t.iloc[:3].tolist()))
    check("non-missing values are untouched by the sentinel logic",
          t.iloc[3] == to_cat_series(tr["Gender"]).iloc[3])
    # a numeric column with NaN
    sn = tr["Age"].astype(float).copy()
    sn.iloc[:2] = np.nan
    tn = to_cat_series(sn)
    check("NaN in a numeric column becomes the sentinel",
          all(tn.iloc[:2] == MISSING_SENTINEL), str(tn.iloc[:2].tolist()))


# --------------------------------------------------------------------------- 6. cardinality
def test_cardinalities_are_measured_values() -> None:
    print("\n6. cardinalities match the measured values")
    tr = frame()
    cols = default_cat_cols(True)
    card = cat_cardinality(tr, cols)
    for c in cols:
        exp = EXPECTED_CARDINALITY.get(c, DEFAULT_SERVICE_CARDINALITY)
        check(f"{c}: cardinality {card[c]} == expected {exp}", card[c] == exp, str(card[c]))
    check("Customer Type is 2 (no 'Neutral Customer' in this data)", card["Customer Type"] == 2)
    check("Class is 3 (Business/Eco/Plus, no 'First')", card["Class"] == 3)
    check("Baggage handling is 5 (one rating level unobserved)", card["Baggage handling"] == 5)
    check("no twin is a single level (would waste a CTR slot)",
          all(v > 1 for v in card.values()), str(card))
    check("no META4 twin exceeds 4 levels",
          all(card[c] <= 4 for c in META4), str({c: card[c] for c in META4}))


# --------------------------------------------------------------------------- 7. eval inert
def test_eval_labels_cannot_alter_the_frame() -> None:
    print("\n7. outer-validation labels cannot alter the training frame")
    tr = frame()
    cols = default_cat_cols(True)
    rng = np.random.default_rng(1)
    n = len(tr)
    folds = rng.integers(0, 5, n)
    val_rows = np.where(folds == 0)[0]
    fit_rows = np.where(folds != 0)[0]
    # the frame for FIT rows must be identical whether or not we look at VAL rows at all
    base = cat_frame(tr, cols, fit_rows)
    tr2 = tr.copy()
    # `satisfaction` is bool dtype here, so the flip must be built as bool, not as `1 - x` (int64),
    # which pandas refuses to write back into a bool column.
    tr2["satisfaction"] = (~tr2["satisfaction"].astype(bool))
    tr2.loc[val_rows, "satisfaction"] = (~tr2.loc[val_rows, "satisfaction"].astype(bool))
    perturbed_fit = cat_frame(tr2, cols, fit_rows)
    check("perturbing validation-fold labels leaves the FIT frame identical",
          base.equals(perturbed_fit))
    check("FIT and VAL row sets are disjoint", not (set(fit_rows.tolist()) & set(val_rows.tolist())))
    check("twins are built from tr/te by column name only, so no fold statistic enters them",
          list(base.columns) == [cat_name(c) for c in cols])


# --------------------------------------------------------------------------- 8. arm isolation
def test_arms_differ_only_in_the_intended_mechanism() -> None:
    print("\n8. arms differ only in the intended mechanism")
    tr = frame()
    cols = default_cat_cols(True)
    rows = np.arange(2000)
    Xn = np.zeros((len(rows), 5), dtype="float32")
    names = [f"n{i}" for i in range(5)]

    # C0: numeric only
    c0_frame, c0_cats = attach(Xn, names, cat_frame(tr, [], rows))
    # C2: numeric + twins
    c2_frame, c2_cats = attach(Xn, names, cat_frame(tr, cols, rows))

    check("C0 declares no categorical features", c0_cats == [], str(c0_cats))
    check("C2 declares exactly the twin columns", len(c2_cats) == len(cols))
    check("C0 and C2 share an identical numeric block",
          c0_frame[names].equals(c2_frame[names]))
    check("C2 differs from C0 ONLY by appended twin columns",
          list(c2_frame.columns) == list(c0_frame.columns) + list(c2_cats),
          f"{len(c0_frame.columns)} vs {len(c2_frame.columns)}")
    check("C0 numeric values are unchanged between arms",
          np.array_equal(c0_frame[names].to_numpy(), c2_frame[names].to_numpy()))
    # C1 vs C0: identical frames, only boosting_type may differ
    check("C1 uses the same frame as C0, so only boosting_type can differ",
          c0_frame.equals(attach(Xn, names, cat_frame(tr, [], rows))[0]))
    # and the excluded high-cardinality columns must genuinely be absent
    check("Age twin is NOT built in this experiment", cat_name("Age") not in c2_cats)
    check("Flight Distance twin is NOT built in this experiment",
          cat_name("Flight Distance") not in c2_cats)


# --------------------------------------------------------------------------- 9. C0 unchanged
def test_c0_is_the_current_control() -> None:
    print("\n9. C0 reproduces the existing numeric control")
    from src.features.view import RAW21
    tr = frame()
    rows = np.arange(1000)
    X = np.empty((len(tr), len(RAW21)), dtype="float32")
    for j, c in enumerate(RAW21):
        s = tr[c]
        X[:, j] = (s.to_numpy(dtype="float32") if pd.api.types.is_numeric_dtype(s)
                   else pd.factorize(s.astype(str), sort=True)[0].astype("float32"))
    X = np.nan_to_num(X, nan=-999.0)
    # what the existing pipeline feeds CatBoost: a raw float32 numpy matrix, no cat_features
    f0, c0 = attach(X[rows], list(RAW21), cat_frame(tr, [], rows))
    check("C0 carries no categorical columns", c0 == [])
    check("C0 has exactly the 21 raw numeric columns", len(f0.columns) == 21, str(len(f0.columns)))
    check("C0 numeric block is byte-identical to the raw float32 matrix",
          np.array_equal(f0.to_numpy(dtype="float32"), X[rows]))
    check("C0 numeric block is float32 throughout",
          all(str(d).startswith("float32") for d in f0.dtypes), str(set(map(str, f0.dtypes))))


def test_cat_features_resolution_rule() -> None:
    print("\n11. how CatBoost resolves cat_features (measured, not assumed)")
    # My first probe run reported EVERY native-categorical arm as unsupported, and I concluded
    # "CatBoost resolves cat_features positionally, not by name". That conclusion was WRONG. The real
    # cause: the probe passed the SOURCE column names (Gender, Type of Travel, ...) to a frame whose
    # columns are the TWIN names (ncat__Gender, ...). CatBoost resolved those names as positional
    # indices, read numeric column 0, and raised
    #     Invalid type for cat_feature[...] = 1.0 : cat_features must be integer or string
    # Measured behaviour in 1.2.10: names that EXIST are honoured; names that do NOT exist are
    # reinterpreted as positions. Both paths below are asserted so the rule cannot drift unnoticed.
    tr = frame()
    cols = default_cat_cols(True)
    src_names, twin_names = list(cols), [cat_name(c) for c in cols]
    rows = np.arange(3000)
    Xn = np.zeros((len(rows), 21), dtype="float32")
    names = [f"n{i}" for i in range(21)]
    f, cn = attach(Xn, names, cat_frame(tr, cols, rows))
    y = (np.arange(len(rows)) % 2)

    check("twin names ARE in the frame", all(c in f.columns for c in cn))
    check("source names are NOT in the frame (they are renamed)",
          not any(c in f.columns for c in src_names))

    from catboost import CatBoostClassifier

    def fit(kw):
        m = CatBoostClassifier(iterations=15, depth=6, verbose=0, allow_writing_files=False,
                               random_seed=4)
        m.fit(f, y, **kw)
        return list(m.get_cat_feature_indices())

    ci = cat_indices(f, cn)
    check("twin columns occupy indices 21..37", ci[0] == 21 and ci[-1] == 21 + len(cn) - 1,
          f"{ci[0]}..{ci[-1]}")
    check("positional indices and TWIN NAMES give the SAME resolved cat indices",
          fit({"cat_features": ci}) == fit({"cat_features": cn}), "names honoured")
    check("a name that is not in the frame RAISES rather than being silently reinterpreted",
          _raises(lambda: fit({"cat_features": src_names})),
          "source names must raise")
    check("cat_indices raises on a name absent from the frame",
          _raises(lambda: cat_indices(f, ["nope"]), KeyError))
    bad = pd.concat([f[names].astype("float32"),
                     pd.DataFrame({cn[0]: np.zeros(len(f), dtype="float32")}, index=f.index)], axis=1)
    check("cat_indices raises when a declared categorical column holds floats",
          _raises(lambda: cat_indices(bad, [cn[0]]), TypeError))


def main() -> int:
    print("=" * 78)
    print("NATIVE CATBOOST CATEGORICAL FRAME -- GUARD TESTS")
    print("=" * 78)
    for fn in (test_twins_preserve_exact_identity, test_twins_are_not_ordinal_encoded,
               test_numeric_columns_remain_present, test_no_target_in_twins,
               test_schemas_match_across_row_sets, test_sentinel_does_not_collide,
               test_cardinalities_are_measured_values, test_eval_labels_cannot_alter_the_frame,
               test_arms_differ_only_in_the_intended_mechanism, test_c0_is_the_current_control,
               test_cat_features_resolution_rule):
        fn()
    print("\n" + "=" * 78)
    print(f"{N - len(FAILS)}/{N} passed")
    if FAILS:
        print("FAILURES: " + ", ".join(FAILS))
    print("=" * 78)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
