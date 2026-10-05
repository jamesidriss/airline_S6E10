"""Section 14 -- a class-conditional GENERATIVE score instead of a discriminative one.

Why this is worth an hour
-------------------------
Everything in the pool estimates p(y | x) directly. The generator behind this synthetic dataset may
be better described by p(x | y), and if the two directions disagree that disagreement is information
the discriminative models cannot see by construction. It is the cheapest genuinely orthogonal probe
available, and it is falsifiable: if the standalone score is very weak AND its marginal blend gain
is ~0, the synthetic generator leaves no class-conditional density structure worth exploiting and
the axis dies immediately.

Method (deliberately low-cost, no deep generative model)
-------------------------------------------------------
Quantile-binned class-conditional likelihood ratios, fold-safe:

    score(x) = sum_j  w_j * [ log( (c_{1,jb} + a) / (N_1 + a*B_j) )
                            - log( (c_{0,jb} + a) / (N_0 + a*B_j) ) ]

where b is the bin of x_j, c_{y,jb} counts class-y rows in bin b, B_j is the number of bins, and
`a` is Laplace smoothing. Bins come from QUANTILES OF THE FIT ROWS ONLY, so no validation-fold
information -- and none of any label -- is used to place a bin edge. The target enters only through
the counts, which are estimated on fit rows and applied to validation rows: standard cross-fitting.

Three arms, because naive Bayes over-counts correlated evidence
--------------------------------------------------------------
  gen_plain     w_j = 1 for every feature
  gen_weighted  w_j proportional to |univariate cross-fitted AUC - 0.5|
  gen_topk      only the K features with the strongest univariate cross-fitted AUC

The survey columns are strongly redundant (`class_business` and `type_of_travel_business` carry much
the same evidence), which is exactly why the plain arm over-counts. `gen_topk` tests whether pruning
redundancy recovers a usable score.

Nested protocol
---------------
Meta folds are the immutable primary folds. The entire score -- bin edges, class-conditional counts,
per-feature weights, feature selection -- is fitted on the other four folds and applied to the fifth.
No row is ever scored by statistics that saw its own label.

Usage: python scripts/run_generative_probe.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import RAW21  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

N_BINS = 32
ALPHA = 1.0          # Laplace smoothing
TOPK = 8
# META4: Gender, Customer Type, Type of Travel, Class. Everything else in RAW21 is numeric.
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_matrices(df):
    """Numeric + categorical codes over the 21 raw columns only, by design.

    The probe is meant to ask whether the GENERATIVE direction carries information, so it is
    deliberately restricted to the raw columns. Feeding it the 285-column engineered view would
    confound "generative vs discriminative" with "more features".
    """
    num_cols = [c for c in RAW21 if c not in CAT_COLS]
    Xn = np.column_stack([df[c].to_numpy(dtype="float64") for c in num_cols])
    cat_cols = [c for c in CAT_COLS if c in RAW21]
    cats = []
    for c in cat_cols:
        v = df[c].to_numpy()
        u = {uu: i for i, uu in enumerate(sorted(set(v.tolist())))}
        cats.append(np.array([u[vv] for vv in v], dtype="int64"))
    names = [f"num:{c}" for c in num_cols] + [f"cat:{c}" for c in cat_cols]
    return Xn, cats, names


def quantile_edges(x, nbins):
    qs = np.linspace(0, 1, nbins + 1)[1:-1]
    e = np.unique(np.quantile(x, qs))
    return e


class GenerativeScorer:
    """Class-conditional binned log-likelihood ratio. Fitted on fit rows only."""

    def __init__(self, n_bins=N_BINS, alpha=ALPHA):
        self.nb, self.a = n_bins, alpha
        self.edges, self.tables, self.kind, self.nlev = [], [], [], []

    def fit(self, Xn, cats, y):
        self.edges, self.tables, self.kind, self.nlev = [], [], [], []
        for j in range(Xn.shape[1]):
            e = quantile_edges(Xn[:, j], self.nb)
            self.edges.append(e)
            self.kind.append("num")
            self.tables.append(self._counts(np.digitize(Xn[:, j], e), y, len(e) + 1))
        for c in cats:
            self.edges.append(None)
            self.kind.append("cat")
            u = int(c.max()) + 1
            self.nlev.append(u)
            self.tables.append(self._counts(c, y, u))
        return self

    def _counts(self, codes, y, nbins):
        T = np.zeros((2, nbins))
        np.add.at(T[0], codes[y == 0], 1.0)
        np.add.at(T[1], codes[y == 1], 1.0)
        return T

    def component_llr(self, Xn, cats):
        """Per-feature log-likelihood-ratio contribution, shape (n_features, n_rows)."""
        out = []
        nj = 0
        for j in range(Xn.shape[1]):
            T = self.tables[nj]
            b = np.digitize(Xn[:, j], self.edges[nj])
            N = T.sum(axis=1)
            p1 = (T[1] + self.a) / (N[1] + self.a * T.shape[1])
            p0 = (T[0] + self.a) / (N[0] + self.a * T.shape[1])
            out.append(np.log(p1[b]) - np.log(p0[b]))
            nj += 1
        for ci, c in enumerate(cats):
            T = self.tables[nj]
            N = T.sum(axis=1)
            p1 = (T[1] + self.a) / (N[1] + self.a * T.shape[1])
            p0 = (T[0] + self.a) / (N[0] + self.a * T.shape[1])
            out.append(np.log(p1[c]) - np.log(p0[c]))
            nj += 1
        return np.array(out)


def univariate_auc(Xn, cats, y):
    """|AUC - 0.5| per feature, from an inner 4-way split of the FIT rows only.

    The arrays passed in are ALREADY subset to the fit rows, so everything here is positional.
    An earlier version took the global index array as well and then used it to index the subset
    columns, which silently mixed two index spaces.
    """
    n = len(y)
    pos = np.arange(n)
    inner = pos[pos % 4 == 0]
    outer = np.setdiff1d(pos, inner)
    cols = [Xn[:, j] for j in range(Xn.shape[1])] + [c.astype("float64") for c in cats]
    aucs = []
    for v in cols:
        try:
            a = roc_auc_score(y[outer], v[outer])
            a2 = roc_auc_score(y[inner], v[inner])
            aucs.append(abs(0.5 * (a + a2) - 0.5))
        except Exception:  # noqa: BLE001
            aucs.append(0.0)
    return np.asarray(aucs)


def main() -> None:
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("int8")
    folds = get_scheme("primary", y, tr["id"]).folds
    Xn, cats, names = build_matrices(tr)
    nf = Xn.shape[1] + len(cats)
    print(f"features: {nf} ({Xn.shape[1]} numeric + {len(cats)} categorical)   rows={len(y):,}\n")

    fin = store.load_oof("blend_v3_final").astype("float64")
    fin_auc = float(roc_auc_score(y, fin))
    fin_logit = logit(fin)

    arms = {"gen_plain": ("plain", None), "gen_weighted": ("weighted", None),
            "gen_top8": ("topk", TOPK)}
    out = {"base": "blend_v3_final", "base_oof_auc": fin_auc, "n_features": int(nf),
           "feature_names": names, "n_bins": N_BINS, "alpha": ALPHA, "arms": []}

    print("=" * 100)
    print(f"{'arm':<14}{'standalone AUC':>16}{'vs 0.5':>11}{'best w':>9}{'best gain':>12}"
          f"{'logit corr':>13}")
    print("=" * 100)

    for arm, (mode, k) in arms.items():
        t0 = time.time()
        score = np.zeros(len(y))
        for fo in sorted(set(folds.tolist())):
            fit = np.where(folds != fo)[0]
            va = np.where(folds == fo)[0]
            gs = GenerativeScorer().fit(Xn[fit], [c[fit] for c in cats], y[fit])
            LL = gs.component_llr(Xn[va], [c[va] for c in cats])          # (nf, n_va)
            w = np.ones(nf)
            if mode == "weighted":
                w = univariate_auc(Xn[fit], [c[fit] for c in cats], y[fit])
                s = w.sum()
                w = w / s if s > 0 else np.ones(nf) / nf
            elif mode == "topk":
                w = np.zeros(nf)
                sel = np.argsort(-univariate_auc(Xn[fit], [c[fit] for c in cats], y[fit]))[:k]
                w[sel] = 1.0 / k
            score[va] = (w[:, None] * LL).sum(axis=0)
        auc = float(roc_auc_score(y, score))
        # the generative score is on an arbitrary scale, so blend it as a standardised additive term
        # on the finalist's logit: logit(fin) + w * z(score)
        z = (score - score.mean()) / (score.std() + 1e-12)
        gains = {}
        for w in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
            gains[str(w)] = float(roc_auc_score(y, fin_logit + w * z)) - fin_auc
        bw = max(gains, key=lambda a: gains[a])
        print(f"{arm:<14}{auc:>16.6f}{(auc-0.5)*1e5:>+10.1f}e{bw:>9}{gains[bw]*1e5:>+11.1f}e"
              f"{corr(fin_logit, score):>13.5f}")
        print(f"               per-weight gains: "
              f"{', '.join(f'w={a}:{b*1e5:+.2f}' for a, b in gains.items())}"
              f"   ({time.time()-t0:.0f}s)")
        out["arms"].append({
            "arm": arm, "mode": mode, "topk": k, "standalone_auc": auc,
            "standalone_minus_half": auc - 0.5,
            "blend_gains": gains, "best_weight": float(bw), "best_gain": gains[bw],
            "logit_corr_with_finalist": corr(fin_logit, score),
            "spearman_with_finalist": spearman(score, fin),
        })

    print("=" * 104)
    best = max(out["arms"], key=lambda a: a["best_gain"])
    print(f"\n  best arm: {best['arm']}  standalone {best['standalone_auc']:.6f}  "
          f"best blend gain {best['best_gain']*1e5:+.1f}e-5 at w={best['best_weight']}")
    print(f"  predeclared kill rule: standalone very weak AND marginal gain ~0 -> kill immediately.")
    weak = best["standalone_auc"] < 0.60
    if weak and best["best_gain"] < 1.5e-5:
        print("  VERDICT: KILLED. Both conditions met -- the class-conditional generative direction "
              "carries no\n           usable structure here. The synthetic generator leaves no "
              "density signal that a\n           discriminative GBDT is failing to exploit. Do not "
              "spend more on this axis.")
    elif best["best_gain"] >= 1.5e-5:
        print("  VERDICT: candidate -- marginal gain meets the +1.5e-5 admission gate. Worth a "
              "blend test at more\n           capacity before any submission decision.")
    else:
        print("  VERDICT: no arm meets the admission gate. Axis closed.")

    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()[:12]
    save_json(out, REPORTS / "generative_probe.json")
    print("\nwrote", REPORTS / "generative_probe.json")


if __name__ == "__main__":
    main()
