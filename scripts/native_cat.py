"""CatBoost-specific frame with NATIVE categorical features.

Why this module exists
----------------------
Every CatBoost model this campaign has ever fitted went through

    ViewBuilder -> float32 numpy matrix -> CatBoostClassifier.fit(X, y)

with NO `cat_features` argument. Verified by audit: zero occurrences of `cat_features` or
CatBoost `boosting_type` anywhere in `src/` or `scripts/`. So `Gender`, `Customer Type`,
`Type of Travel`, `Class` and all thirteen service ratings were handed to CatBoost as **ordinal
floats**, and CatBoost's defining mechanism -- ordered target statistics for categorical features --
was never exercised. The best CatBoost ever recorded here is 0.9610555 (`z3_cat_d10`), below the
deterministic LightGBM, so "CatBoost is worse" was never actually tested; "CatBoost without
categoricals" was.

Two rules this module enforces
------------------------------
1. **Category identity must never pass through float32.** A float32 round-trip is lossy and, worse,
   makes ordinal and categorical indistinguishable to the model. Categorical twins are built as
   Python strings from the RAW frame and carried in a pandas object column, and are declared to
   CatBoost by NAME via `cat_features`, never coerced to a number.

2. **The twins must not touch the target.** They are read from `tr`/`te` by column name only. The
   existing `*__cat` columns inside the champion view are label-free exact-value twins and are left
   untouched, so arm C0 (no native twins) reproduces the current numeric control bit-for-bit.

Sentinel handling
-----------------
NaN becomes the literal string MISSING_SENTINEL. Service ratings are the integers 0-5 and the META4
values are a handful of words, so a double-underscore-delimited sentinel cannot collide with a real
level; `tests/test_native_cat.py` asserts that non-collision directly rather than assuming it.

Cardinality is reported, not assumed: SERVICE13 has 6 levels each (0-5), META4 has 2-3. Those are
exact and are asserted in tests. Flight Distance (~3,474 levels) and Age (~75) are deliberately NOT
included here -- high-cardinality CTRs are a separate, later experiment, and mixing them in would
make the 2x2 uninterpretable.

Usage:
  Xf, Xa, names, cat_names = build_native_cat_frame(vb, tr, te, fit_idx, apply_sets)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.s6e10 import SURVEY13  # noqa: E402

META4 = ["Gender", "Customer Type", "Type of Travel", "Class"]
MISSING_SENTINEL = "__MISSING__"

# Exact cardinalities, MEASURED on this dataset rather than assumed. My first guesses were wrong on
# three columns and the measurement caught them:
#   Customer Type      2, not 3 -- there is no "Neutral Customer" level in this competition's data
#   Class              3, not 4 -- Business / Eco / Plus; no "First"
#   Baggage handling   5, not 6 -- one rating level is unobserved
# The other ten service ratings do carry the full 0-5 range. Asserted in tests so a silent schema
# change cannot pass unnoticed.
EXPECTED_CARDINALITY = {
    "Gender": 2, "Customer Type": 2, "Type of Travel": 2, "Class": 3,
    "Baggage handling": 5,
}
DEFAULT_SERVICE_CARDINALITY = 6


def cat_name(col: str) -> str:
    """Column name for a categorical twin. Prefixed so it cannot collide with a numeric feature."""
    return f"ncat__{col}"


def to_cat_series(s: pd.Series) -> pd.Series:
    """Raw column -> exact-identity string series with an explicit missing sentinel.

    Values are stringified WITHOUT normalisation beyond stripping whitespace, so 'Female' and
    'female' would remain distinct -- which is correct for "exact original identity", and any such
    collision is surfaced by the cardinality test rather than silently merged here.
    """
    if pd.api.types.is_numeric_dtype(s):
        # ratings are integers stored as float; render whole numbers without a decimal tail so
        # 0.0 and "0" cannot become two different categories
        v = s.astype("float64")
        out = np.where(np.isnan(v), MISSING_SENTINEL,
                       np.where(np.abs(v - np.round(v)) < 1e-9,
                                np.char.mod("%.0f", np.round(v)),
                                np.char.mod("%.6g", v)))
        return pd.Series(out, index=s.index, dtype=object)
    out = s.astype("object").where(s.notna(), MISSING_SENTINEL).map(
        lambda z: MISSING_SENTINEL if z is MISSING_SENTINEL else str(z).strip())
    return pd.Series(out, index=s.index, dtype=object)


def cat_frame(src: pd.DataFrame, cols: list[str], rows: np.ndarray) -> pd.DataFrame:
    """Build the categorical twin block for a specific set of GLOBAL row indices."""
    data = {}
    for c in cols:
        if c not in src.columns:
            raise KeyError(f"categorical source column {c!r} is absent from the frame")
        data[cat_name(c)] = to_cat_series(src[c]).to_numpy()[rows]
    return pd.DataFrame(data, index=pd.RangeIndex(len(rows)))


def cat_cardinality(src: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Distinct levels per categorical twin, computed on the source frame.

    Reported rather than assumed. A twin whose cardinality is 1 carries no information and would
    silently waste a CTR budget slot.
    """
    return {c: int(to_cat_series(src[c]).nunique()) for c in cols}


def attach(X: np.ndarray, names: list[str], cats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Combine the numeric matrix with the categorical block into one CatBoost frame.

    Numeric columns keep their float32 dtype and their original names. Categorical columns are
    appended as object dtype and their names returned separately so the caller can pass them to
    `cat_features` by name.
    """
    num = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=list(names))
    num.index = pd.RangeIndex(len(num))
    frame = pd.concat([num, cats], axis=1)
    return frame, list(cats.columns)


def verify_no_target(frame_cols: list[str], target: str = "satisfaction") -> None:
    """Hard guard: no categorical twin may be derived from, or named like, the target."""
    bad = [c for c in frame_cols if target.lower() in c.lower()]
    if bad:
        raise AssertionError(f"target-derived column(s) present in frame: {bad}")


def default_cat_cols(service: bool = True) -> list[str]:
    """META4 always; SERVICE13 when requested."""
    return list(META4) + (list(SURVEY13) if service else [])


def main() -> None:
    """Print the audit this module exists to record, on the real data."""
    from src.common import TARGET, load_cached_parquet
    tr, te = load_cached_parquet()
    cols = default_cat_cols(True)
    print("NATIVE CATBOOST CATEGORICAL TWIN AUDIT")
    print(f"  catboost categorical source columns: {len(cols)}  (META4={len(META4)}, "
          f"SERVICE13={len(SURVEY13)})")
    print(f"  missing sentinel: {MISSING_SENTINEL!r}")
    card = cat_cardinality(tr, cols)
    print(f"  {'column':<26}{'cardinality':>12}{'expected':>10}  {'ok':>4}")
    for c in cols:
        exp = EXPECTED_CARDINALITY.get(c, DEFAULT_SERVICE_CARDINALITY)
        ok = card[c] == exp
        print(f"  {c:<26}{card[c]:>12}{exp:>10}  {'yes' if ok else 'NO':>4}")
    bad = [c for c in cols
           if cat_cardinality(tr, [c])[c] != EXPECTED_CARDINALITY.get(c, DEFAULT_SERVICE_CARDINALITY)]
    print(f"  cardinality mismatches: {bad if bad else 'none'}")
    n_missing = int((to_cat_series(tr["Inflight wifi service"]) == MISSING_SENTINEL).sum())
    print(f"  rows with a missing rating sentinel: {n_missing:,}")
    collide = [c for c in cols if MISSING_SENTINEL in set(to_cat_series(tr[c]).unique())]
    print(f"  sentinel colliding with a real level: {collide if collide else 'none'}")
    print(f"\n  Existing champion view already carries '*__cat' float twins (label-free, ordinal);")
    print(f"  these are LEFT UNTOUCHED so arm C0 reproduces the numeric control exactly.")


if __name__ == "__main__":
    main()
