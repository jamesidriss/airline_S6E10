"""`te_conditional` -- service-rating x traveller-segment target encoding. Phase 8B.

Why this block and not more `te_all21`
-------------------------------------
Phase 8A measured `te_all21` -- exact-value TE of all 21 raw columns -- at -0.3e-5 stacked and
-13.3e-5 replacing, under both extra_trees and deterministic trees. The informative part was that our
EXISTING selective `te` block beats it, and that the existing block wins on four axes while `te_all21`
varied only one:

  1. crosses                                   <-- te_all21 has none
  2. a shrinkage spectrum (smooth 10/20/100)   <-- te_all21 has one setting
  3. log counts per key                        <-- te_all21 has none
  4. binned / modulo Flight Distance variants  <-- te_all21 has none

So the value of TE here is INTERACTIONS plus a shrinkage spectrum. This block is therefore built to
those two axes specifically, using the same `FoldSafeTE` machinery (which already supplies smooth
10/20/100 and a count column per key) rather than sklearn's single-setting encoder.

The keys, predeclared and NOT combinatorially extended beyond them
---------------------------------------------------------------
Segment definitions, exactly as specified:

    traveller_segment   = Type of Travel | Customer Type | Class
    trip_customer       = Type of Travel | Customer Type

  * 13 conditional keys: `service_value | traveller_segment` for each of the 13 survey ratings.
  * 5 service-pair conditional keys, each crossed with `trip_customer`:

        Inflight wifi service        x Ease of Online booking
        Inflight wifi service        x Online boarding
        Online boarding              x Seat comfort
        Seat comfort                 x Inflight entertainment
        Inflight entertainment       x Cleanliness

All 18 keys are exact-value, so no binning is introduced. Missing values keep the existing convention
that `np.nan` forms its own group inside `FoldSafeTE._table`.

Why the pairs are the five that matter, mechanically: the survey ratings are strongly correlated with
each other, and the pairs chosen are adjacent links in that correlation chain. A pair's conditional
target mean isolates a combination a single rating cannot express -- and Phase 8A says combinations are
what this block family is good at.

Fold-safety is inherited from `FoldSafeTE`, which is the implementation already used by the `te` block
and already covered by the campaign's leakage tests: fit rows are encoded by an inner cross-fit that
excludes each row's own label, and every apply set is encoded by a table fitted on all fit rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.view import META4, RAW21, SURVEY13  # noqa: E402

# A separator that cannot occur in any raw value, so key concatenation is unambiguous.
SEP = "\x1f"

TRAVELLER_SEGMENT = ["Type of Travel", "Customer Type", "Class"]
TRIP_CUSTOMER = ["Type of Travel", "Customer Type"]

# The five service pairs, predeclared. Order is fixed so feature names are stable.
SERVICE_PAIRS = [
    ("Inflight wifi service", "Ease of Online booking"),
    ("Inflight wifi service", "Online boarding"),
    ("Online boarding", "Seat comfort"),
    ("Seat comfort", "Inflight entertainment"),
    ("Inflight entertainment", "Cleanliness"),
]

SMOOTHS = (10.0, 20.0, 100.0)


def _as_str(s: pd.Series) -> pd.Series:
    """Canonical string form: numeric columns keep integer spelling, missing stays a distinct value."""
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
        finite = np.isfinite(v)
        integral = finite & (np.abs(v - np.rint(v)) < 1e-12)
        out = np.where(finite & integral, [str(int(t)) for t in v],
                       np.where(finite, [format(t, ".17g") for t in v], "__NA__"))
        return pd.Series(out, index=s.index)
    return s.astype(str)


def build_te_conditional_keys(comb: pd.DataFrame) -> dict[str, pd.Series]:
    """18 exact-value keys: 13 (rating x traveller_segment) + 5 (service pair x trip_customer)."""
    seg = None
    for c in TRAVELLER_SEGMENT:
        part = _as_str(comb[c])
        seg = part if seg is None else seg + SEP + part
    tc = None
    for c in TRIP_CUSTOMER:
        part = _as_str(comb[c])
        tc = part if tc is None else tc + SEP + part

    keys: dict[str, pd.Series] = {}
    for r in SURVEY13:
        keys[f"tec_{r}"] = _as_str(comb[r]) + SEP + seg
    for a, b in SERVICE_PAIRS:
        keys[f"tecp_{a}_x_{b}"] = (_as_str(comb[a]) + SEP + _as_str(comb[b]) + SEP + tc)
    return keys


def conditional_key_cardinalities(comb: pd.DataFrame) -> dict[str, dict]:
    """Per-key counts and value coverage -- the diagnostic that says whether a key can be estimated.

    A key with many distinct values and a small median group size produces encodings that are mostly
    prior, which is not leakage but is also not information. Worth knowing before spending compute on
    it.
    """
    out = {}
    for name, k in build_te_conditional_keys(comb).items():
        vc = k.value_counts()
        out[name] = {"n_distinct": int(len(vc)), "median_group": int(vc.median()),
                     "p90_group": int(np.percentile(vc.to_numpy(), 90)),
                     "frac_rows_in_groups_le_5": float((vc <= 5).sum() / len(k))}
    return out
