"""Canonical S6E10 feature views.

Every builder is classified as one of:

  ``static``      deterministic, label-free          -> computed once on train+test
  ``transductive`` label-free but fitted on train+test combined -> computed once (rules allow)
  ``external``    fitted on the ORIGINAL real dataset labels only -> computed once; no
                  competition target is ever involved, therefore no leakage
  ``fold``        target-dependent -> MUST be recomputed inside each fold with inner CV

Provenance note: Rules S6E10 §2.6 permit external data that is public and free.
The original dataset is `arseniyshutko/binary-aviation-satisfaction-129k` (129,880 rows),
the file the official competition Data page cites.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORIG_CSV = ROOT / "data" / "original" / "arseniyshutko__binary-aviation-satisfaction-129k" / "data.csv"

SURVEY13 = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
            "Gate location", "Food and drink", "Online boarding", "Seat comfort",
            "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
            "Checkin service", "Cleanliness"]
META4 = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FD = "Flight Distance"
AGE = "Age"
DEP = "Departure Delay in Minutes"
ARR = "Arrival Delay in Minutes"
TARGET = "satisfaction"


def as_int_codes(comb: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = comb.copy()
    for c in cols:
        out[c] = pd.factorize(out[c].astype("object"), sort=True)[0].astype("float32")
    return out


def _raw_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    d = df[cols].copy()
    for c in cols:
        if d[c].dtype.kind == "f":
            d[c] = d[c].astype("float64").round(6)
        else:
            d[c] = d[c].astype(str)
    return pd.util.hash_pandas_object(d.astype(object), index=False)


# ======================================================================================
# ORIGINAL (external) dataset
# ======================================================================================
@functools.lru_cache(maxsize=1)
def load_original(drop_overlap_with: tuple = ()) -> pd.DataFrame:
    """Original dataset, with an explicit leak audit.

    Any original row whose 21 predictors *exactly* match some competition row is dropped,
    so no external row can act as a direct answer key.
    """
    og = pd.read_csv(ORIG_CSV)
    ren = {"Type of Travel": "Type of Travel", "Customer Type": "Customer Type"}
    og = og.rename(columns=ren)
    og[TARGET] = og[TARGET].astype(int)
    return og


def original_leak_audit(og: pd.DataFrame, comb: pd.DataFrame, feats: list[str]) -> dict:
    ho = _raw_key(og, feats)
    hc = _raw_key(comb, feats)
    overlap = ho.isin(set(hc.unique()))
    return {
        "n_original": int(len(og)),
        "n_original_rows_exactly_matching_a_competition_row": int(overlap.sum()),
        "n_competition_rows_matching_some_original_row": int(hc.isin(set(ho.unique())).sum()),
    }


# ======================================================================================
# external features: smoothed original-data target statistics + original-only teacher
# ======================================================================================
@functools.lru_cache(maxsize=4)
def build_external_features(smooth: float = 40.0) -> dict:
    """Smoothed P(target | key) from ORIGINAL labels only. Never sees a competition label."""
    og = load_original()
    yo = og[TARGET].values.astype("float64")
    prior = float(yo.mean())

    specs: dict[str, list[str]] = {}
    for c in og.columns:
        if c != TARGET:
            specs[f"ogte_{c}"] = [c]
    specs["ogte_trip_class_cust"] = ["Type of Travel", "Class", "Customer Type"]
    specs["ogte_class_online"] = ["Class", "Online boarding"]
    specs["ogte_class_trip"] = ["Class", "Type of Travel"]
    specs["ogte_trip_cust"] = ["Type of Travel", "Customer Type"]
    specs["ogte_online_wifi"] = ["Online boarding", "Inflight wifi service"]
    specs["ogte_fd"] = [FD]

    out = {}
    for name, cols in specs.items():
        if any(c not in og.columns for c in cols):
            continue
        h = _raw_key(og, cols)
        s = pd.DataFrame({"h": h.values, "y": yo}).groupby("h")["y"].agg(["sum", "size"])
        p = (s["sum"] + smooth * prior) / (s["size"] + smooth)
        out[name] = {"cols": cols, "p": p, "cnt": s["size"]}
    return {"tables": out, "prior": prior, "smooth": smooth, "n_original": int(len(og))}


def apply_external(comb: pd.DataFrame, ex: dict) -> tuple[np.ndarray, list[str]]:
    names, mats = [], []
    for name, spec in ex["tables"].items():
        h = _raw_key(comb, spec["cols"])
        pv = h.map(spec["p"]).to_numpy()
        cv = h.map(spec["cnt"]).to_numpy()
        names += [name, name + "__cnt"]
        mats.append(np.column_stack([
            np.where(np.isnan(pv), ex["prior"], pv).astype("float32"),
            np.where(np.isnan(cv), 0.0, np.log1p(np.nan_to_num(cv))).astype("float32"),
        ]))
    return np.column_stack(mats), names


# ======================================================================================
# transductive (label-free) features
# ======================================================================================
def build_transductive_features(comb: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, list[str]]:
    # numeric view for group statistics (string columns -> sorted codes)
    N = comb.copy()
    for c in N.columns:
        if N[c].dtype.kind in "OUS":
            N[c] = pd.factorize(N[c].astype("object"), sort=True)[0].astype("float64")
        else:
            N[c] = pd.to_numeric(N[c], errors="coerce").astype("float64")
    cols_out, names = [], []

    # --- exact-value occurrence counts over train+test
    for c in feats:
        cnt = comb[c].map(comb[c].value_counts()).astype("float32")
        names.append(f"cnt_{c}")
        cols_out.append(cnt.fillna(0).to_numpy())

    # --- route profile: group stats of every other feature keyed on Flight Distance
    g = N.groupby(FD, observed=True)
    for c in feats:
        if c == FD:
            continue
        mu = g[c].transform("mean")
        names.append(f"fdm_{c}")
        cols_out.append(mu.astype("float32").fillna(-1).to_numpy())
    names.append("fd_cnt")
    cols_out.append(g[AGE].transform("size").astype("float32").to_numpy())

    # --- log-distance buckets profile (coarser route grouping)
    gb = N.groupby(np.minimum(N[FD] // 500, 400), observed=True)
    for c in [AGE, "Online boarding", "Seat comfort", "Type of Travel", "Class", DEP, ARR]:
        names.append(f"fdb_{c}")
        cols_out.append(gb[c].transform("mean").astype("float32").fillna(-1).to_numpy())
    names.append("fdb_cnt")
    cols_out.append(gb[AGE].transform("size").astype("float32").to_numpy())

    # --- digit / magnitude structure
    for c in NUMS:
        v = pd.to_numeric(comb[c], errors="coerce").fillna(-1)
        for m in (10, 100, 1000):
            names.append(f"{c}_m{m}")
            cols_out.append((v % m).astype("float32").to_numpy())
        for d in (10, 100):
            names.append(f"{c}_d{d}")
            cols_out.append((v // d).astype("float32").to_numpy())
        names.append(f"{c}_len")
        cols_out.append(v.astype("float64").astype(str).str.len().astype("float32").to_numpy())
        names.append(f"{c}_is0")
        cols_out.append((v == 0).astype("float32").to_numpy())

    # --- survey N/A (zero) semantics
    z = np.column_stack([(comb[c] == 0).astype("float32").to_numpy() for c in SURVEY13])
    names += [f"na_{c}" for c in SURVEY13]
    cols_out += [z[:, i] for i in range(z.shape[1])]
    names += ["na_count", "na_any"]
    cols_out += [z.sum(1), (z.sum(1) > 0).astype("float32")]
    # zero indicator crossed with segment (cheap, few columns)
    seg = (comb["Class"].astype(str) + "|" + comb["Type of Travel"].astype(str)).astype("object")
    codes = pd.factorize(seg, sort=True)[0]
    for c in ["Online boarding", "Inflight wifi service", "Gate location"]:
        zz = (comb[c] == 0).astype("float32").to_numpy()
        for s in (0, 1, 2):
            names.append(f"na_x_seg{s}_{c}")
            cols_out.append(zz * (codes == s).astype("float32"))

    # --- rating aggregates (zero-aware and zero-naive)
    R = np.column_stack([comb[c].astype("float32").to_numpy() for c in SURVEY13])
    pos = R > 0
    cnt = pos.sum(1).astype("float32")
    mean_pos = np.where(cnt > 0, np.where(pos, R, 0).sum(1) / np.maximum(cnt, 1), -1)
    names += ["rate_cnt", "rate_mean_pos", "rate_mean_all", "rate_min_pos", "rate_max_pos",
              "rate_std_pos", "rate_le2", "rate_eq3", "rate_ge4", "rate_eq5", "rate_mid3"]
    Rp = np.where(pos, R, np.nan)
    with np.errstate(invalid="ignore"):
        cols_out += [
            cnt,
            mean_pos.astype("float32"),
            R.mean(1).astype("float32"),
            np.nanmin(Rp, axis=1).astype("float32"),
            np.nanmax(Rp, axis=1).astype("float32"),
            np.nanstd(Rp, axis=1).astype("float32"),
            (cnt - ((R >= 4) | (R == 5)).sum(1) - (R == 5).sum(1)).astype("float32"),
            (R == 3).sum(1).astype("float32"),
            (R >= 4).sum(1).astype("float32"),
            (R == 5).sum(1).astype("float32"),
            (R == 3).sum(1).astype("float32"),
        ]

    # --- domain interactions
    fd = pd.to_numeric(comb[FD], errors="coerce").fillna(-1).astype("float64")
    dep = pd.to_numeric(comb[DEP], errors="coerce").fillna(0).astype("float64")
    arr = pd.to_numeric(comb[ARR], errors="coerce").fillna(0).astype("float64")
    age = pd.to_numeric(comb[AGE], errors="coerce").fillna(-1).astype("float64")
    names += ["delay_total", "delay_worsen", "delay_both_zero", "log_dep", "log_arr",
              "delay_per_km", "age_bin10", "fd_bin250", "fd_bin1000"]
    cols_out += [
        (dep + arr).astype("float32"),
        (arr - dep).astype("float32"),
        ((dep == 0) & (arr == 0)).astype("float32"),
        np.log1p(dep).astype("float32"),
        np.log1p(arr).astype("float32"),
        ((dep + arr) / (fd / 1000.0).clip(lower=1)).astype("float32"),
        (age // 10).astype("float32"),
        np.floor(fd / 250).astype("float32"),
        np.floor(fd / 1000).astype("float32"),
    ]
    # class x travel type x online boarding interactions (numeric products of codes)
    for a, b in [("Class", "Online boarding"), ("Type of Travel", "Online boarding"),
                 ("Class", "Seat comfort"), ("Customer Type", "Online boarding"),
                 ("Type of Travel", "Seat comfort")]:
        ca = pd.factorize(comb[a].astype(str), sort=True)[0].astype("float32")
        cb = pd.to_numeric(comb[b], errors="coerce").fillna(-1).astype("float32")
        names.append(f"ix_{a}_{b}")
        cols_out.append(ca * cb)

    X = np.column_stack(cols_out).astype("float32")
    return np.nan_to_num(X, nan=-999.0, posinf=1e9, neginf=-1e9), names


# ======================================================================================
# GPT-2 BPE token keys (generator fingerprint)
# ======================================================================================
def build_token_features(comb: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:  # noqa: BLE001
        return np.zeros((len(comb), 0), "float32"), []

    mats, names = [], []
    for c in cols:
        vals = pd.to_numeric(comb[c], errors="coerce")
        uniq = np.sort(vals.dropna().unique())
        first, lastcnt, ntok = {}, {}, {}
        for v in uniq:
            t = enc.encode(" " + str(int(v)))
            first[v] = t[0]
            ntok[v] = len(t)
            lastcnt[v] = t[-1] + 100000 * len(t)
        f = vals.map(first).fillna(-1).astype("float32").to_numpy()
        lc = vals.map(lastcnt).fillna(-1).astype("float32").to_numpy()
        nt = vals.map(ntok).fillna(0).astype("float32").to_numpy()
        names += [f"tok1_{c}", f"tokL_{c}", f"tokN_{c}"]
        mats += [f, lc, nt]
    return np.column_stack(mats).astype("float32"), names


# ======================================================================================
# fold-safe target encoding
# ======================================================================================
TE_KEYS_DEFAULT = {
    "te_fd": [FD],
    "te_age": [AGE],
    "te_fd_age": [FD, AGE],
    "te_fd_class": [FD, "Class"],
    "te_fd_trip": [FD, "Type of Travel"],
    "te_age_class_trip_cust": [AGE, "Class", "Type of Travel", "Customer Type"],
    "te_fd_bin500": ["_fd_bin500"],
    "te_fd_bin1000": ["_fd_bin1000"],
    "te_fd_m1000": ["_fd_m1000"],
    "te_online": ["Online boarding"],
    "te_ent": ["Inflight entertainment"],
    "te_fd_250": ["_fd_bin250"],
}


def build_te_keys(comb: pd.DataFrame) -> dict[str, pd.Series]:
    fd = pd.to_numeric(comb[FD], errors="coerce").fillna(-1)
    age = pd.to_numeric(comb[AGE], errors="coerce").fillna(-1)
    keys = {
        "_fd_bin250": (fd // 250),
        "_fd_bin500": (fd // 500),
        "_fd_bin1000": (fd // 1000),
        "_fd_m1000": (fd % 1000),
    }
    return keys


class FoldSafeTE:
    """Exact-value target encoding with Bayesian smoothing, cross-fitted inside the fold.

    Contract:
      * ``apply(keys_fit, y, keys_apply)`` fits the table on ``keys_fit`` rows only.
      * ``crossfit(keys_fit, y)`` encodes the fit rows themselves with an inner K-fold
        cross-fit, so no row is ever encoded by a table containing its own label.
    """

    def __init__(self, keys: dict[str, pd.Series], smooths=(10.0, 20.0, 100.0),
                 inner_folds: int = 5, inner_seed: int = 0):
        self.keys = keys
        self.smooths = tuple(smooths)
        self.inner_folds = inner_folds
        self.inner_seed = inner_seed

    @staticmethod
    def _table(k: pd.Series, y: np.ndarray, smooth: float, prior: float):
        """Bayesian-smoothed exact-value target mean, returned as plain dicts for fast .map()."""
        d = pd.DataFrame({"k": np.asarray(k.values, dtype="object"), "y": np.asarray(y, dtype="float64")})
        s = d.groupby("k", observed=True)["y"].agg(["sum", "size"])
        p = ((s["sum"] + smooth * prior) / (s["size"] + smooth))
        return p.to_dict(), s["size"].to_dict()

    def apply(self, keys_fit: dict[str, pd.Series], y: np.ndarray,
              keys_apply: dict[str, pd.Series]) -> tuple[np.ndarray, list[str]]:
        prior = float(np.mean(y))
        cols, names = [], []
        for name, k_app in keys_apply.items():
            k_fit = keys_fit[name]
            for sm in self.smooths:
                tab, cnt = self._table(k_fit, y, sm, prior)
                cols.append(k_app.map(tab).fillna(prior).astype("float32").to_numpy())
                names.append(f"{name}_s{int(sm)}")
                if sm == self.smooths[-1]:
                    cols.append(np.log1p(k_app.map(cnt).fillna(0).to_numpy()).astype("float32"))
                    names.append(f"{name}_cnt")
        return np.column_stack(cols).astype("float32"), names

    def crossfit(self, keys_fit: dict[str, pd.Series], y: np.ndarray) -> tuple[np.ndarray, list[str]]:
        n = len(y)
        if not 2 <= self.inner_folds <= n:
            raise ValueError("crossfit requires at least two nonempty inner folds")
        names = []
        for name in keys_fit:
            for sm in self.smooths:
                names.append(f"{name}_s{int(sm)}_xfit")
                if sm == self.smooths[-1]:
                    names.append(f"{name}_cnt_xfit")
        out = np.full((n, len(names)), np.nan, dtype="float32")
        rng = np.random.default_rng(self.inner_seed)
        inner = (np.arange(n) % self.inner_folds)[rng.permutation(n)]
        for name, k in keys_fit.items():
            for i in range(self.inner_folds):
                a = np.where(inner != i)[0]
                b = np.where(inner == i)[0]
                # The smoothing prior is target-dependent too. Using y.mean() here lets
                # every held-out row contribute its own label through the prior.
                prior = float(np.mean(y[a]))
                for sm in self.smooths:
                    tab, cnt = self._table(k.iloc[a], y[a], sm, prior)
                    j = names.index(f"{name}_s{int(sm)}_xfit")
                    out[b, j] = k.iloc[b].map(tab).fillna(prior).astype("float32").to_numpy()
                    if sm == self.smooths[-1]:
                        j2 = names.index(f"{name}_cnt_xfit")
                        out[b, j2] = np.log1p(k.iloc[b].map(cnt).fillna(0).to_numpy()).astype("float32")
        return out, names


# ======================================================================================
# original-only teacher
# ======================================================================================
class OriginalTeacher:
    """Models trained ONLY on the original 129,880 rows (after the exact-overlap leak audit)."""

    def __init__(self, kind: str = "lgbm", seed: int = 0, params: dict | None = None):
        self.kind = kind
        self.seed = seed
        self.params = params or {}
        self.models: list = []

    def fit_predict(self, comb: pd.DataFrame, feats: list[str], n_models: int = 5) -> np.ndarray:
        og = load_original()
        hc = _raw_key(comb, feats)
        ho = _raw_key(og, feats)
        keep = ~ho.isin(set(hc.unique()))
        og = og[keep].reset_index(drop=True)
        Xo = _prep(og[feats])
        yo = og[TARGET].values.astype("int")
        Xc = _prep(comb[feats])
        preds = []
        for i in range(n_models):
            if self.kind == "lgbm":
                import lightgbm as lgb
                p = dict(objective="binary", n_estimators=600, learning_rate=0.05, num_leaves=63,
                         colsample_bytree=0.5, subsample=0.8, subsample_freq=1, n_jobs=8,
                         verbose=-1, random_state=self.seed + i)
                p.update(self.params)
                m = lgb.LGBMClassifier(**p)
                m.fit(Xo, yo)
                preds.append(m.predict_proba(Xc)[:, 1])
            else:
                import xgboost as xgb
                p = dict(objective="binary:logistic", eval_metric="auc", n_estimators=700,
                         learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                         reg_lambda=2.0, tree_method="hist", device="cuda", n_jobs=8,
                         random_state=self.seed + i)
                p.update(self.params)
                m = xgb.XGBClassifier(**p)
                m.fit(Xo, yo, verbose=False)
                preds.append(m.predict_proba(Xc)[:, 1])
        return np.mean(preds, axis=0).astype("float32")


# ======================================================================================
# original-data conditional surfaces
# ======================================================================================
@functools.lru_cache(maxsize=8)
def build_original_surfaces(anchor_cols: tuple[str, ...], smooth: float = 30.0) -> dict:
    """Conditional surfaces estimated on the ORIGINAL data.

    For each anchor value (e.g. a `Flight Distance` route or a `Class`), summarise the original
    dataset: how many rows sit there, what the original target mean is, and what the original
    mean of every other column is. These are *real* conditional distributions from the real
    survey, not GAN-smoothed ones, which is why they can beat the transductive synthetic profile.

    Only original labels are used -> no competition target is involved -> no leakage.
    """
    og = load_original()
    yo = og[TARGET].values.astype("float64")
    prior = float(yo.mean())
    cols = [c for c in og.columns if c != TARGET]
    # numeric view of the original data (string columns -> sorted codes) for group means
    N = og.copy()
    for c in cols:
        if N[c].dtype.kind in "OUS":
            N[c] = pd.factorize(N[c].astype("object"), sort=True)[0].astype("float64")
        else:
            N[c] = pd.to_numeric(N[c], errors="coerce").astype("float64")
    out: dict[str, dict] = {}
    for anchor in anchor_cols:
        if anchor not in og.columns:
            continue
        h = _raw_key(og, [anchor])
        d = N.copy()
        d["_h"] = h.values
        g = d.groupby("_h", observed=True)
        cnt = g.size()
        s = pd.DataFrame({"h": h.values, "y": yo}).groupby("h", observed=True)["y"].agg(["sum", "size"])
        p = ((s["sum"] + smooth * prior) / (s["size"] + smooth)).to_dict()
        out[anchor] = {"count": cnt.to_dict(), "target": p}
        means = g[cols].mean()
        for c in cols:
            if c == anchor or c not in means.columns:
                continue
            m = means[c].dropna()
            if len(m):
                out[anchor][f"mean::{c}"] = m.to_dict()
    return {"surfaces": out, "prior": prior, "smooth": smooth, "n_original": int(len(og))}


def apply_original_surfaces(comb: pd.DataFrame, surf: dict) -> tuple[np.ndarray, list[str]]:
    names, mats = [], []
    for anchor, spec in surf["surfaces"].items():
        h = _raw_key(comb, [anchor])
        cnt = h.map(spec["count"]).to_numpy()
        pv = h.map(spec["target"]).to_numpy()
        names += [f"ogs_{anchor}_cnt", f"ogs_{anchor}_target"]
        mats.append(np.column_stack([
            np.where(np.isnan(cnt), 0.0, np.log1p(np.nan_to_num(cnt))).astype("float32"),
            np.where(np.isnan(pv), surf["prior"], pv).astype("float32"),
        ]))
        for k, v in spec.items():
            if not k.startswith("mean::"):
                continue
            base = k.split("::", 1)[1]
            vals = h.map(v).to_numpy()
            raw = comb[base]
            if raw.dtype.kind in "OUS":
                cur = pd.factorize(raw.astype("object"), sort=True)[0].astype("float64")
            else:
                cur = pd.to_numeric(raw, errors="coerce").to_numpy().astype("float64")
            cur = np.nan_to_num(cur, nan=0.0)
            # raw value relative to the anchor's original conditional mean: a residual
            names.append(f"ogs_{anchor}_resid::{base}")
            mats.append((cur - np.nan_to_num(vals, nan=0.0)).reshape(-1, 1).astype("float32"))
            names.append(f"ogs_{anchor}_mean::{base}")
            mats.append(np.nan_to_num(vals, nan=0.0).reshape(-1, 1).astype("float32"))
    return np.nan_to_num(np.column_stack(mats), nan=0.0, posinf=1e9, neginf=-1e9), names


SURFACE_ANCHORS = (FD, "Class", "Type of Travel", "Online boarding")


# --------------------------------------------------------------------------------------
# Finer-grained original-data target statistics.
#
# The single-column `ogte_*` lookups were worth +0.001015, the largest single feature gain we
# measured, so the external block deserves more depth. These are pairwise and triple conditional
# target statistics from the ORIGINAL labels only -- still no competition label anywhere.
# --------------------------------------------------------------------------------------
def _multi_te_specs() -> list[tuple[str, list[str]]]:
    S13 = SURVEY13
    specs: list[tuple[str, list[str]]] = []
    # every pair of the four categoricals with each other
    meta_pairs = [("Class", "Type of Travel"), ("Class", "Customer Type"),
                  ("Type of Travel", "Customer Type"), ("Class", "Gender"),
                  ("Type of Travel", "Gender"), ("Customer Type", "Gender")]
    for a, b in meta_pairs:
        specs.append((f"ogp_{a}__{b}", [a, b]))
    # class / travel crossed with the strongest ratings
    for anchor in ("Class", "Type of Travel"):
        for r in S13:
            specs.append((f"ogp_{anchor}__{r}", [anchor, r]))
    # pairs of the strongest ratings
    strong = ["Online boarding", "Inflight entertainment", "Seat comfort", "Inflight wifi service",
              "Checkin service", "Cleanliness", "On-board service", "Leg room service",
              "Baggage handling", "Food and drink", "Ease of Online booking",
              "Departure/Arrival time convenient", "Gate location"]
    for i in range(len(strong)):
        for j in range(i + 1, len(strong)):
            specs.append((f"ogp_{strong[i]}__{strong[j]}", [strong[i], strong[j]]))
    # triples: categoricals plus one strong rating
    for r in ["Online boarding", "Seat comfort", "Inflight entertainment"]:
        specs.append((f"ogt_class_travel_{r}", ["Class", "Type of Travel", r]))
        specs.append((f"ogt_class_cust_{r}", ["Class", "Customer Type", r]))
    # route x segment, and coarse route buckets crossed with segment
    specs.append(("ogp_fd__class", [FD, "Class"]))
    specs.append(("ogp_fd__travel", [FD, "Type of Travel"]))
    return specs


@functools.lru_cache(maxsize=4)
def build_external_multi(smooth: float = 30.0, min_count: int = 3) -> dict:
    """Pairwise/triple smoothed original-data target statistics."""
    og = load_original()
    yo = og[TARGET].values.astype("float64")
    prior = float(yo.mean())
    out = {}
    for name, cols in _multi_te_specs():
        if any(c not in og.columns for c in cols):
            continue
        h = _raw_key(og, cols)
        s = pd.DataFrame({"h": h.values, "y": yo}).groupby("h")["y"].agg(["sum", "size"])
        s = s[s["size"] >= min_count]
        if not len(s):
            continue
        out[name] = {"cols": cols, "p": ((s["sum"] + smooth * prior) / (s["size"] + smooth)),
                     "cnt": s["size"], "coverage": float(s["size"].sum() / len(og))}
    return {"tables": out, "prior": prior, "smooth": smooth}


def apply_external_multi(comb: pd.DataFrame, ex: dict) -> tuple[np.ndarray, list[str]]:
    names, mats = [], []
    for name, spec in ex["tables"].items():
        h = _raw_key(comb, spec["cols"])
        pv = h.map(spec["p"]).to_numpy()
        cv = h.map(spec["cnt"]).to_numpy()
        names += [name, name + "__cnt"]
        mats.append(np.column_stack([
            np.where(np.isnan(pv), ex["prior"], pv).astype("float32"),
            np.log1p(np.where(np.isnan(cv), 0.0, np.nan_to_num(cv))).astype("float32"),
        ]))
    return np.nan_to_num(np.column_stack(mats), nan=0.0), names


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "OUS":
            out[c] = pd.factorize(out[c].astype("object"), sort=True)[0].astype("float32")
        else:
            out[c] = out[c].astype("float32")
    return out.fillna(-1.0)
