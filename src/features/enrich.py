"""Feature block `enrich`: extra label-free candidate columns for random-split GBDTs.

Motivated by a measured result, not by guesswork. `extra_trees=True` picks a random feature *and*
a random threshold at every split, so its quality depends directly on how many plausible candidate
columns it can draw from. It is worth +2.3e-4 on the 285-column `full` view and -0.0056 on the
21-column `raw` view. That is an interaction with candidate-column count, so the natural
follow-up experiment is: *does adding more plausible label-free columns extend the gain?*

Everything in this block is label-free:
  * per-`Flight Distance` spread statistics and categorical shares (route profile, unsupervised)
  * per-FD means of every other column **in the original survey** (original *features* only --
    no original label is touched here; labels are handled by the `external` block)
  * digit / modulo / magnitude decomposition of the numeric columns
  * rating *differences* (the digital-vs-service gap, etc.) and zero-aware aggregates
  * quantile / rank transforms of the numeric columns

An ablation measures whether any of it survives next to what we already have.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from src.features.s6e10 import AGE, ARR, DEP, FD, META4, NUMS, SURVEY13, load_original

# rating pairs whose difference carries a specific domain meaning
GAP_PAIRS = [
    ("Online boarding", "Inflight wifi service"),
    ("Online boarding", "Seat comfort"),
    ("Seat comfort", "Leg room service"),
    ("Inflight entertainment", "Inflight wifi service"),
    ("Seat comfort", "On-board service"),
    ("Cleanliness", "Baggage handling"),
    ("Checkin service", "Gate location"),
    ("Ease of Online booking", "Online boarding"),
    ("Food and drink", "Seat comfort"),
    ("Departure/Arrival time convenient", "Gate location"),
]


def _numeric_view(comb: pd.DataFrame) -> pd.DataFrame:
    N = comb.copy()
    for c in N.columns:
        if N[c].dtype.kind in "OUS":
            N[c] = pd.factorize(N[c].astype("object"), sort=True)[0].astype("float64")
        else:
            N[c] = pd.to_numeric(N[c], errors="coerce").astype("float64")
    return N


def build_enrich(comb: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (matrix, names) of additional label-free columns."""
    N = _numeric_view(comb)
    names: list[str] = []
    cols: list[np.ndarray] = []

    def add(name: str, arr) -> None:
        a = np.asarray(arr, dtype="float64").ravel()
        if len(a) != len(N):
            return
        names.append(name)
        cols.append(a)

    # ---------------------------------------------------------------- route spread profile
    g = N.groupby(FD, observed=True)
    spread = {}
    for c in SURVEY13 + ["Age", DEP, ARR]:
        spread[c] = g[c].std()
        add(f"enr_fdsd_{c}", spread[c].to_numpy())
    add("enr_fdmed_Age", g["Age"].median().to_numpy())
    add("enr_fdmed_OB", g["Online boarding"].median().to_numpy())
    add("enr_fdrange_OB", (g["Online boarding"].max() - g["Online boarding"].min()).to_numpy())
    # how many of the 13 ratings are 5 on this route
    R = N[SURVEY13]
    gR = R.groupby(N[FD], observed=True)
    add("enr_fd_n5", gR.apply(lambda d: (d == 5).sum(axis=1)).to_numpy())
    add("enr_fd_n0", gR.apply(lambda d: (d == 0).sum(axis=1)).to_numpy())
    add("enr_fd_meanall", R.groupby(N[FD], observed=True).mean().mean(axis=1).to_numpy())

    # route-level categorical shares (unsupervised)
    for c in ("Class", "Type of Travel", "Customer Type"):
        ct = pd.crosstab(N[FD], N[c], normalize="index")
        for lvl in ct.columns:
            add(f"enr_fd_{c}_p{int(lvl)}", N[FD].map(ct[lvl]).to_numpy())

    # ---------------------------------------------------------------- digit / magnitude
    for c in NUMS:
        v = N[c]
        for m in (10, 100, 1000, 10000):
            add(f"enr_{c}_m{m}", (v % m).to_numpy())
        for d in (10, 100, 1000):
            add(f"enr_{c}_d{d}", (v // d).to_numpy())
        add(f"enr_{c}_log", np.log1p(v.clip(lower=0)).to_numpy())
        add(f"enr_{c}_rank", v.rank(pct=True).to_numpy())
        add(f"enr_{c}_nbits", np.log2(v.clip(lower=1) + 1).to_numpy())

    # GPT-2 style token count of the numeric string (label-free, generator fingerprint)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        for c in NUMS:
            v = N[c]
            cache: dict[float, int] = {}

            def ntok(x: float) -> float:
                k = float(x)
                if k not in cache:
                    cache[k] = float(len(enc.encode(" " + str(int(k))))) if np.isfinite(k) else 0.0
                return cache[k]

            uniq = v.map(ntok).to_numpy()
            add(f"enr_{c}_ntok", uniq)
            add(f"enr_{c}_ntok_x_m100", uniq * (v.to_numpy() % 100))
    except Exception:  # noqa: BLE001
        pass

    # ---------------------------------------------------------------- rating differences
    for a, b in GAP_PAIRS:
        add(f"enr_dif_{a}__{b}", (N[a] - N[b]).to_numpy())
        add(f"enr_sum_{a}__{b}", (N[a] + N[b]).to_numpy())

    # ---------------------------------------------------------------- rating aggregates
    Rm = N[SURVEY13]
    pos = Rm > 0
    cnt = pos.sum(axis=1)
    add("enr_rate_std_all", Rm.std(axis=1).to_numpy())
    add("enr_rate_iqr", (Rm.quantile(0.75, axis=1) - Rm.quantile(0.25, axis=1)).to_numpy())
    add("enr_rate_top2mean", Rm.apply(lambda r: np.sort(r.values)[-2:].mean(), axis=1).to_numpy())
    add("enr_rate_bottom2mean", Rm.apply(lambda r: np.sort(r.values)[:2].mean(), axis=1).to_numpy())
    add("enr_rate_digital", Rm[["Online boarding", "Inflight wifi service",
                                "Inflight entertainment", "Ease of Online booking"]].mean(axis=1).to_numpy())
    add("enr_rate_cabin", Rm[["Seat comfort", "Leg room service", "On-board service",
                              "Cleanliness", "Food and drink", "Inflight entertainment"]].mean(axis=1).to_numpy())
    add("enr_rate_ground", Rm[["Checkin service", "Gate location",
                               "Departure/Arrival time convenient"]].mean(axis=1).to_numpy())
    add("enr_rate_nbad", (Rm <= 2).sum(axis=1).to_numpy())
    add("enr_rate_gap_maxmin", (Rm.max(axis=1) - Rm.min(axis=1)).to_numpy())

    # ---------------------------------------------------------------- original-survey profile
    # NOTE: original *features* only. No original label is read in this function.
    og = load_original()
    Num = _numeric_view(og)
    for c in NUMS:
        m = Num.groupby(Num[FD], observed=True)[c].mean()
        add(f"enr_ogfd_{c}", N[FD].map(m).to_numpy())
    for c in ("Online boarding", "Seat comfort", "Inflight entertainment", "Age"):
        m = Num.groupby(Num["Class"], observed=True)[c].mean()
        add(f"enr_ogclass_{c}", N["Class"].map(m).to_numpy())

    X = np.column_stack(cols).astype("float32")
    X = np.nan_to_num(X, nan=-999.0, posinf=1e9, neginf=-1e9)
    return X, names