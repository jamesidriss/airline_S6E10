"""`te_all21` -- exact-value target encoding of ALL 21 raw columns. Phase 8A.

Why this is a NEW hypothesis and not a rerun of a rejected branch
---------------------------------------------------------------
`src/features/s6e10.py::TE_KEYS_DEFAULT` target-encodes **12 keys**, and only **4 of the 21 raw
columns get a direct single-column exact-value TE** (`te_fd`, `te_age`, `te_online`, `te_ent`).
Three more (`Class`, `Type of Travel`, `Customer Type`) appear only as components of crosses. The
remaining **14 columns have no exact-value target statistic at all**, and 13 of those 14 have only
**<=6 distinct values**:

    Inflight wifi service 6   Departure/Arrival time convenient 6   Ease of Online booking 6
    Gate location 6           Food and drink 6                       Seat comfort 6
    On-board service 6        Leg room service 6                     Baggage handling 5
    Checkin service 6         Cleanliness 6                          Gender 2
    Departure Delay in Minutes 167                                   Arrival Delay in Minutes 168

So the difference from the public mechanism (a `TargetEncoder` over every raw value converted to a
string) is not cosmetic: for the service ratings it is the difference between *no* target statistic
and a direct estimate of p(y | rating). A tree can eventually carve out those six levels with five
threshold splits, but it has to spend depth doing it, and under `extra_trees` plus
`colsample_bytree=0.8` every tree only sees a random 80% of the columns anyway.

Therefore "TE already failed here" does NOT apply to this block, and the earlier negative results
for `core3_te` / `full` say nothing about it.

Fold-safety contract (the thing that must be provable, not asserted)
--------------------------------------------------------------------
For each OUTER fold:

    outer-FIT rows
      -> TargetEncoder.fit_transform() with a y-INDEPENDENT inner KFold, so no row's own label ever
         enters the counts used to encode that row
      -> TE features for outer-FIT

    TargetEncoder.fit(outer-FIT)   -> transform(outer-validation), transform(test)

The outer validation target must never influence any encoded value it is scored on. The seven
regression tests in `tests/test_te_all21.py` check this directly, including the one that randomising
outer-validation labels changes nothing.

WHY THE INNER SPLIT MUST NOT DEPEND ON y -- this is the subtle part
--------------------------------------------------------------------
sklearn 1.9.1 marks `TargetEncoder(shuffle=..., random_state=...)` as DEPRECATED and recommends
passing a cross-validation generator as `cv`. Passing `cv=5` gets you
`StratifiedKFold(shuffle=True, random_state=None)` -- and `StratifiedKFold.split` stratifies ON `y`.

That is not a leak in sklearn's encoding, which is genuinely leave-out either way. But it makes the
leave-out property **unverifiable**: you cannot perturb one label and watch what happens to that row's
own encoding, because perturbing the label can also move the row into a different fold. Measured on a
200-row toy problem:

    KFold(shuffle=True, random_state=0)           splits y-independent   max own-change 0.000e+00
    StratifiedKFold(shuffle=True, random_state=0)  splits y-DEPENDENT    max own-change 2.217e-02

So this block uses `KFold(shuffle=True, random_state=seed)`: deterministic, seeded, y-independent, and
free of the deprecated arguments. An inner split for a target encoder should not be a function of the
target, because the property that makes target encoding safe is exactly that it is not.

An honesty note the tests encode: with a y-independent split, flipping one fit row's target changes
that row's OWN cross-fitted encoding by **exactly zero**. With `smooth="auto"` there is no residual
through a global smoothing prior either, because auto-smoothing is computed per fold from that fold's
training labels only (`lambda_` in `_fit_encoding_fast_auto_smooth`). `te_all21` is therefore
verifiably leak-free and the test asserts exact equality rather than a loose bound.

An earlier draft of this file pre-shuffled the fit rows before calling the encoder, reasoning that
contiguous inner folds were the risk. That treated a symptom: the real issue was the split's
dependence on y, which pre-shuffling does nothing about.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.view import RAW21

# Stable sentinel for missing Arrival Delay. An explicit marker rather than the string "nan" or a
# float sentinel, so that it cannot collide with a real value and cannot be produced by a dtype cast.
NA_SENTINEL = "__NA__"

# Columns whose exact values are ordinal 0-5 ratings or 2-level factors; treated as exact categories,
# never binned.
ORDINAL6 = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
META_FACTORS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMERIC_EXACT = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def exact_value_labels(comb: pd.DataFrame, col: str) -> np.ndarray:
    """Canonical exact-value string labels for one raw column, as an object array.

    Exactness rules, in order of importance:

    * no float32 round trip anywhere -- a float32 cast of 761.0 can alias two distinct float64
      values, which would silently merge categories;
    * integer-valued floats get an INTEGER string, so 761 and 761.0 are the same category, while
      761 and 761.5 stay distinct;
    * missing values become the explicit ``__NA__`` sentinel;
    * object columns keep their original spelling, so "Female" and "female" remain different, as they
      are in the source data.
    """
    s = comb[col]
    if col in ORDINAL6 or col in META_FACTORS:
        # read as its own dtype; these are small integers or strings in the source
        return s.astype("object").to_numpy()
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    out = np.empty(len(v), dtype=object)
    finite = np.isfinite(v)
    integral = finite & (np.abs(v - np.rint(v)) < 1e-12)
    # integers as int64 strings, non-integers with 17 significant digits so float64 values round-trip
    out[finite & integral] = [str(int(t)) for t in v[finite & integral]]
    nf = finite & ~integral
    if nf.any():
        out[nf] = [format(t, ".17g") for t in v[nf]]
    out[~finite] = NA_SENTINEL
    return out


def build_all21_codes(comb: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Integer category codes for all 21 raw columns, from a LABEL-FREE global value->index map.

    Codes are derived from the covariates of train+test only. No target is involved, so this is the
    same fold-safety class as the transductive count blocks: the mapping cannot leak, and using the
    pooled vocabulary means a value seen only in test still receives a stable code instead of
    silently becoming a category of its own.

    Returning codes rather than strings is deliberate: sklearn ordinal-encodes internally anyway, and
    int32 input is both faster and removes any chance of the encoder re-parsing a string and
    reintroducing float aliasing.
    """
    cols, names = [], []
    for c in RAW21:
        lab = exact_value_labels(comb, c)
        uniq, inv = np.unique(lab.astype("str"), return_inverse=True)
        cols.append(inv.astype("int32"))
        names.append(f"te_all21:{c}")
        if len(uniq) > 200_000:
            raise ValueError(f"{c} produced {len(uniq)} exact values; exact-value encoding is "
                             f"not appropriate for it")
    return np.column_stack(cols), names


class All21TargetEncoder:
    """Cross-fitted exact-value target encoding over all 21 raw columns.

    ``fit_rows``   : outer-FIT row positions. Cross-fitted internally; returns TE for those rows.
    ``apply_rows`` : outer-validation / test rows. Encoded by a table fitted on ALL outer-FIT rows.

    WHY KFold AND NOT StratifiedKFold -- this is not a style choice
    -------------------------------------------------------------
    sklearn's internal cross-fitting is genuinely leave-out: `fit_transform` fits each fold's table on
    the complement of that fold and writes only that fold's own rows. Verified directly on a toy
    problem -- crossfit Z[j] equals TargetEncoder(...).fit(X[other folds]).transform(X[j]) exactly.

    But `check_cv(cv=5, y, classifier=True, shuffle=True)` returns **StratifiedKFold**, and
    `StratifiedKFold.split` stratifies ON `y`. So flipping one row's label can move that row into a
    DIFFERENT fold, where it is encoded by a different table -- one that excludes a different set of
    rows. Measured on a 200-row toy problem:

        KFold(shuffle=True, random_state=0)           splits y-independent   max own-change 0.000e+00
        StratifiedKFold(shuffle=True, random_state=0)  splits y-DEPENDENT    max own-change 2.217e-02

    Neither is a leak in sklearn's encoding -- the encoding is leave-out either way -- but the
    StratifiedKFold version makes the leave-out property **unverifiable**, because you cannot perturb
    one label without also moving the folds. An inner split for a target encoder must be
    y-independent for its own safety property to be testable, so `KFold(shuffle=True,
    random_state=seed)` is used. It is deterministic, seeded, y-independent, and it avoids sklearn
    1.9's DEPRECATED `shuffle`/`random_state` arguments entirely -- the library's own deprecation
    message recommends passing a CV generator, which is exactly what this does.

    The cost is inner folds that are not class-balanced. At 560k rows with ~24% positives across 5
    folds that is a fold-size deviation of order 1/5, immaterial for a smoothed mean.
    """

    def __init__(self, smooth="auto", cv: int = 5, seed: int = 0):
        self.smooth = smooth
        self.cv = int(cv)
        self.seed = int(seed)
        self._encoder: TargetEncoder | None = None

    def _new_encoder(self) -> TargetEncoder:
        # y-independent inner split; see the class docstring for why StratifiedKFold is wrong here
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=int(self.seed))
        return TargetEncoder(target_type="binary", smooth=self.smooth, cv=splitter)

    def fit_rows(self, codes: np.ndarray, y: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
        """TE for the outer-FIT rows, cross-fitted so no row's own label enters its own counts."""
        enc = self._new_encoder()
        Z = np.asarray(enc.fit_transform(codes[fit_idx], y[fit_idx].astype("float64")),
                       dtype="float64")
        self._encoder = enc
        return Z

    def apply(self, codes: np.ndarray, y: np.ndarray, fit_idx: np.ndarray,
              apply_idx: np.ndarray) -> np.ndarray:
        """TE for rows outside the fit set, from a table fitted on ALL outer-FIT rows.

        ``y`` is read only at ``fit_idx`` positions; ``apply_idx`` labels are never touched.
        """
        enc = self._new_encoder()
        enc.fit(codes[fit_idx], y[fit_idx].astype("float64"))
        Z = np.asarray(enc.transform(codes[apply_idx]), dtype="float64")
        self._encoder = enc
        return Z
        return Z
