"""Feature-set factory.

All feature builders are declared as `FeatureSet` objects with an id, so every
experiment records exactly which representation it used.

Design rules enforced here
--------------------------
* Exact-value categorical copies are computed from the RAW dtype, never float32.
* Any target-dependent statistic is computed inside a fold (see `src.features.te`).
* Transductive (train+test) statistics are named with the `TRANS_` prefix so they are
  never confused with in-fold features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SURVEY13 = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
            "Gate location", "Food and drink", "Online boarding", "Seat comfort",
            "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
            "Checkin service", "Cleanliness"]
META4 = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMCOLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


@dataclass
class FeatureSet:
    fs_id: str
    kind: str                      # "numeric" | "numeric+cat"
    cats: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    cat_as_int_code: bool = True   # encode categories as int codes for GBDTs


def encode_categorical(df: pd.DataFrame, cols: list[str], as_int: bool = True) -> pd.DataFrame:
    """Stable, sorted, joint train+test factorisation (target-free => transductive-safe)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype("object")
        if as_int:
            out[c] = pd.factorize(s, sort=True)[0].astype("float32")
        else:
            out[c] = pd.Categorical(s)
    return out


def numeric_frame(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = df[feats].copy()
    for c in out.columns:
        if out[c].dtype.kind in "OUS":
            out[c] = pd.factorize(out[c], sort=True)[0].astype("float32")
        else:
            out[c] = out[c].astype("float32")
    return out


def exact_cat_copies(df: pd.DataFrame, cols: list[str], suffix: str = "__cat") -> pd.DataFrame:
    """Exact-value categorical duplicates.

    Values are preserved from the raw dtype (float64/int64) so that the mapping is exact.
    """
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        if s.dtype.kind == "f":
            out[c + suffix] = s.round(6)
        else:
            out[c + suffix] = s
    return out


def value_counts_both(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], suffix: str = "__cnt") -> tuple:
    """Transductive exact-value occurrence counts over combined train+test."""
    comb = pd.concat([tr[cols], te[cols]], axis=0, ignore_index=True)
    out = {}
    for c in cols:
        out[c + suffix] = comb[c].map(comb[c].value_counts())
    return pd.DataFrame(out, index=tr.index), pd.DataFrame(
        {k: v.iloc[len(tr):].to_numpy() for k, v in out.items()}, index=te.index)