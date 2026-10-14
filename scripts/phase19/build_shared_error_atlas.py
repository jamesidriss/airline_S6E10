"""Phase 19 Axis 19-A — shared-error atlas: which errors do BOTH v7 halves miss?

The v7 portfolio = sigmoid((logit(route2) + aux_logits) / 2), where route2 is the
2-estimator TabPFN route and aux_logits is the float64 mean of the certified
aux10 member logits (scripts.evaluate_sol_foundation.classical). Because logit
and sigmoid are monotone (and float32 rounding is monotone non-decreasing),
the following holds EXACTLY for every positive-negative pair (n, p):

    route_n >= route_p  AND  aux_n >= aux_p   ==>   port_n >= port_p
    ("shared fail" set S is a SUBSET of "portfolio fail" set W)

so the portfolio residual decomposes exactly as:
    W = S  +  (W \\ S)      -- shared approximation error vs aggregation loss

and each base's strict errors decompose exactly as:
    route_strict = shared_strict + route_only     (asserted, integer-exact)
    aux_strict   = shared_strict + aux_only       (asserted, integer-exact)

Counting convention (tie-aware, verified exact in tmp_sanity on all 5 folds):
    AUC = 1 - (strict_wrong + 0.5 * ties) / total_pairs

Everything is computed rank-based in O(n log n) per fold; the pair matrix is
never materialized. Exact pair identities are asserted with integer equality.

Deliverables (reports/phase19/):
  shared_error_atlas.json / shared_error_atlas.md   -- pooled + per-fold counts
  shared_error_segments.parquet                     -- segment burden + AUCs
  per_row_burden.parquet                            -- per-row shared burden,
      confidence flags, teacher disagreement (label-noise proxy)
Stage-0 question answered: is residual difficulty predictable from covariates
AND from inference-time model confidences WITHOUT validation labels?
(cross-fitted LightGBM, inner ES carved from FIT rows only, full rich view).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, arr_sha256, file_sha256, git_commit, save_json  # noqa: E402
from src.validation.compare import logit  # noqa: E402
from scripts.phase16_common import bank  # noqa: E402
from scripts.evaluate_sol_foundation import classical  # noqa: E402
from scripts.replay_sol_classical import frozen_roles  # noqa: E402

OUT = ROOT / "reports" / "phase19"
CHAMPION_PORTFOLIO_SHA256 = "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552"
CHAMPION_ROUTE_SHA256 = "bcb2811329d3686882450e93435647f8f7ab65abfc48c1a1b51db1e9d1ef95e0"


# --------------------------------------------------------------------- helpers
def strict_mis(y: np.ndarray, r: np.ndarray, pos_sorted: np.ndarray) -> int:
    """#pairs with value(neg) > value(pos) STRICTLY (rank-based, average ranks)."""
    neg = r[y == 0]
    return int(np.searchsorted(pos_sorted, neg, side="left").sum())


def fail_ge(y: np.ndarray, r: np.ndarray, pos_sorted: np.ndarray) -> int:
    """#pairs with value(neg) >= value(pos) (strict + ties)."""
    neg = r[y == 0]
    return int(np.searchsorted(pos_sorted, neg, side="right").sum())


def tie_pairs(y: np.ndarray, p: np.ndarray) -> int:
    """#cross-class pairs with exactly equal values (each costs 0.5 credit)."""
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    pc = np.zeros(len(cnt), dtype=np.int64)
    nc = np.zeros(len(cnt), dtype=np.int64)
    np.add.at(pc, inv[y == 1], 1)
    np.add.at(nc, inv[y == 0], 1)
    return int((pc * nc).sum())


def _compress(v: np.ndarray) -> tuple[np.ndarray, int]:
    u = np.unique(v)
    return np.searchsorted(u, v).astype(np.int64) + 1, len(u)


class _Fenwick:
    __slots__ = ("n", "bit")

    def __init__(self, n: int):
        self.n = n
        self.bit = np.zeros(n + 1, dtype=np.int64)

    def add(self, i: int) -> None:
        bit = self.bit
        n = self.n
        while i <= n:
            bit[i] += 1
            i += i & (-i)

    def query(self, i: int) -> int:
        s = 0
        bit = self.bit
        while i > 0:
            s += int(bit[i])
            i -= i & (-i)
        return s


def two_class_counts(y: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> dict:
    """Exact pair counts over the 2-D rank plane of two models A and B.

    Conventions (all cross-class, average ranks, ties = equal ranks):
      shared_ge     : ra_p <= ra_n AND rb_p <= rb_n   (neither ranks the pair right)
      shared_strict : ra_p <  ra_n AND rb_p <  rb_n
      route_only    : ra_p <  ra_n AND rb_p >= rb_n   (A wrong, B not-strict)
      aux_only      : ra_p >= ra_n AND rb_p <  rb_n
      burden_neg[n] : #shared_ge pairs of this negative row
      burden_pos[p] : #shared_ge pairs of this positive row

    Identities (asserted by caller): route_strict = shared_strict + route_only,
    aux_strict = shared_strict + aux_only, sum(burden_neg) == sum(burden_pos).
    Each of the four counts uses one Fenwick sweep: O(n log n), no pair matrix.
    """
    is_pos = y == 1
    n = len(y)
    ca, na = _compress(ra)
    cb, nb = _compress(rb)
    out: dict = {}

    # ---- shared_ge + burden_neg: sweep A ascending, add same-A group first,
    #      query negatives with <= on B (count of added positives with cb <= cb_n)
    order = np.argsort(ra, kind="mergesort")
    starts = np.flatnonzero(np.r_[True, ra[order][1:] != ra[order][:-1]])
    bounds = list(starts) + [n]
    fw = _Fenwick(nb)
    burden_neg = np.zeros(n, dtype=np.int64)
    shared_ge = 0
    for gi in range(len(starts)):
        grp = order[bounds[gi]:bounds[gi + 1]]
        for i in grp[is_pos[grp]]:
            fw.add(int(cb[i]))
        for i in grp[~is_pos[grp]]:
            c = fw.query(int(cb[i]))
            burden_neg[i] = c
            shared_ge += c
    del fw

    # ---- route_strict-style helper generalized below via 4 sweeps.
    # shared_strict: sweep A ascending, query BEFORE adding same-A group (ra_p < ra_n),
    #   B strictly less: query(cb_n - 1)
    fw = _Fenwick(nb)
    shared_strict = 0
    for gi in range(len(starts)):
        grp = order[bounds[gi]:bounds[gi + 1]]
        for i in grp[~is_pos[grp]]:
            shared_strict += fw.query(int(cb[i]) - 1)
        for i in grp[is_pos[grp]]:
            fw.add(int(cb[i]))
    del fw

    # route_only: ra_p < ra_n AND rb_p >= rb_n. Same A sweep (query-before-add),
    #   count added positives with cb >= cb_n = added_total - query(cb_n - 1)
    fw = _Fenwick(nb)
    added = 0
    route_only = 0
    for gi in range(len(starts)):
        grp = order[bounds[gi]:bounds[gi + 1]]
        for i in grp[~is_pos[grp]]:
            route_only += added - fw.query(int(cb[i]) - 1)
        for i in grp[is_pos[grp]]:
            fw.add(int(cb[i]))
            added += 1
    del fw

    # aux_only: ra_p >= ra_n AND rb_p < rb_n. Sweep B ascending (groups on rb),
    #   query-before-add on B (rb_p < rb_n), A condition >= ra_n:
    #   added_total - query(ca_n - 1)
    order_b = np.argsort(rb, kind="mergesort")
    starts_b = np.flatnonzero(np.r_[True, rb[order_b][1:] != rb[order_b][:-1]])
    bounds_b = list(starts_b) + [n]
    fw = _Fenwick(na)
    added = 0
    aux_only = 0
    for gi in range(len(starts_b)):
        grp = order_b[bounds_b[gi]:bounds_b[gi + 1]]
        for i in grp[~is_pos[grp]]:
            aux_only += added - fw.query(int(ca[i]) - 1)
        for i in grp[is_pos[grp]]:
            fw.add(int(ca[i]))
            added += 1
    del fw

    # burden_pos: #negatives with ra_n >= ra_p AND rb_n >= rb_p. Sweep A
    #   DESCENDING (negatives with larger A added first), group-add before query
    #   (ra_n >= ra_p), B >= : added_total - query(cb_p - 1) computed per group.
    order_d = order[::-1]
    starts_d = np.flatnonzero(np.r_[True, ra[order_d][1:] != ra[order_d][:-1]])
    bounds_d = list(starts_d) + [n]
    fw = _Fenwick(nb)
    added = 0
    burden_pos = np.zeros(n, dtype=np.int64)
    for gi in range(len(starts_d)):
        grp = order_d[bounds_d[gi]:bounds_d[gi + 1]]
        for i in grp[~is_pos[grp]]:
            fw.add(int(cb[i]))
            added += 1
        for i in grp[is_pos[grp]]:
            burden_pos[i] = added - fw.query(int(cb[i]) - 1)
    del fw

    out.update(shared_ge=shared_ge, shared_strict=shared_strict,
               route_only=route_only, aux_only=aux_only,
               burden_neg=burden_neg, burden_pos=burden_pos)
    return out


# ------------------------------------------------------------------------ main
def main() -> int:
    if (OUT / "shared_error_atlas.json").exists():
        raise FileExistsError("Preserve measured atlas")
    OUT.mkdir(parents=True, exist_ok=True)

    tr, te, y, folds, vectors, _, proof = bank()
    ids = tr.id.to_numpy()
    roles = [r for r in frozen_roles() if r["aux_arm"]]

    n = len(y)
    route2 = np.full(n, np.nan, dtype="float64")
    port = np.full(n, np.nan, dtype="float64")
    aux = np.full(n, np.nan, dtype="float64")
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        route2[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "prediction.npy")
        port[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "portfolio.npy")
        aux_logits, _ = classical(k, "sol_clean_aux10_primary", roles, ids, folds, y,
                                  proof["data_sha256"])
        assert aux_logits.shape == (len(fl),), f"fold {k}: classical shape"
        aux[fl] = aux_logits  # store the FLOAT64 mean logits (composition input)
    assert np.isfinite(route2).all() and np.isfinite(port).all() and np.isfinite(aux).all()
    assert arr_sha256(port.astype("float32")) == CHAMPION_PORTFOLIO_SHA256, "portfolio drift"
    assert arr_sha256(route2.astype("float32")) == CHAMPION_ROUTE_SHA256, "route drift"
    # composition identity: stored portfolio == f32(sigmoid((logit(route) + aux_logits)/2))
    recomposed = (1.0 / (1.0 + np.exp(-(logit(route2) + aux) / 2.0))).astype("float32")
    assert np.array_equal(recomposed, port.astype("float32")), "composition identity drift"
    # cross-check stored float32 aux10 against sigmoid(aux_logits) (<= float32 eps)
    aux_gap = float(np.abs(1.0 / (1.0 + np.exp(-aux)) - vectors["aux10"].astype(np.float64)).max())
    assert aux_gap < 6e-8, f"aux10 cross-check failed: {aux_gap}"

    aux_prob = 1.0 / (1.0 + np.exp(-aux))  # monotone-equivalent to aux for ranks

    # ---------------------------------------------------------------- per fold
    per_fold: dict = {}
    burden = np.zeros(n, dtype=np.int64)
    shared_flags = np.zeros(n, dtype=bool)
    rows = []
    for k in range(5):
        m = folds == k
        yy = y[m]
        rr, aa, pp = route2[m], aux_prob[m], port[m]
        ra, rb, rp = rankdata(rr), rankdata(aa), rankdata(pp)
        pos_sorted = np.sort(rp[yy == 1])
        total = int((yy == 1).sum() * (yy == 0).sum())

        # tie-aware AUC identity (must be exact)
        ties_port = tie_pairs(yy, pp)
        strict_port = strict_mis(yy, rp, pos_sorted)
        auc_direct = float(roc_auc_score(yy, pp))
        auc_from = 1.0 - (strict_port + 0.5 * ties_port) / total
        assert abs(auc_direct - auc_from) < 1e-15, f"fold {k}: AUC identity"

        route_strict = strict_mis(yy, ra, np.sort(ra[yy == 1]))
        aux_strict = strict_mis(yy, rb, np.sort(rb[yy == 1]))
        fail_port = fail_ge(yy, rp, pos_sorted)  # strict_port + ties_port

        c = two_class_counts(yy, ra, rb)
        sg, ss, ro, ao = c["shared_ge"], c["shared_strict"], c["route_only"], c["aux_only"]
        # integer-exact decomposition identities
        assert route_strict == ss + ro, f"fold {k}: route decomp {route_strict} != {ss}+{ro}"
        assert aux_strict == ss + ao, f"fold {k}: aux decomp {aux_strict} != {ss}+{ao}"
        assert int(c["burden_neg"].sum()) == sg == int(c["burden_pos"].sum()), \
            f"fold {k}: burden sums != shared_ge"
        # monotone theorem: shared_ge pairs all fail the portfolio (>= convention).
        # port ties among shared: enumerate the (tiny, ~130/fold) cross-class tie
        # pairs of the portfolio and count how many are shared -> exact split.
        _, inv, cnt = np.unique(pp, return_inverse=True, return_counts=True)
        pc = np.zeros(len(cnt), dtype=np.int64)
        nc = np.zeros(len(cnt), dtype=np.int64)
        pos_rows, neg_rows = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
        np.add.at(pc, inv[pos_rows], 1)
        np.add.at(nc, inv[neg_rows], 1)
        shared_port_tie = 0
        for v in np.flatnonzero((pc > 0) & (nc > 0)):
            pr = pos_rows[inv[pos_rows] == v]
            for pn in neg_rows[inv[neg_rows] == v]:
                shared_port_tie += int(((ra[pr] <= ra[pn]) &
                                        (rb[pr] <= rb[pn])).sum())
        shared_and_port_strict = sg - shared_port_tie
        assert 0 <= shared_and_port_strict <= strict_port, f"fold {k}: S subset W violated"
        aggregation_loss = fail_port - sg  # portfolio fails although >=1 base ranks right
        assert aggregation_loss >= 0, f"fold {k}: negative aggregation loss"

        rec = {
            "rows": int(m.sum()), "pos": int((yy == 1).sum()), "neg": int((yy == 0).sum()),
            "pairs": total,
            "port_auc": auc_direct,
            "port_strict_wrong": strict_port, "port_ties": ties_port,
            "port_fail_ge": fail_port,
            "route_strict_wrong": route_strict, "aux_strict_wrong": aux_strict,
            "route_ties": tie_pairs(yy, rr), "aux_ties": tie_pairs(yy, aa),
            "shared_ge": sg, "shared_strict": ss,
            "shared_port_tie": shared_port_tie,
            "shared_and_port_strict": shared_and_port_strict,
            "route_only": ro, "aux_only": ao,
            "aggregation_loss_ge": aggregation_loss,
            "shared_share_of_port_fail": sg / fail_port,
            "shared_auc_mass": sg / total,
            "route_only_auc_mass": ro / total, "aux_only_auc_mass": ao / total,
            "aggregation_loss_auc_mass": aggregation_loss / total,
        }
        per_fold[str(k)] = rec
        rows.append({"fold": k, **rec})
        fl = np.flatnonzero(m)
        burden[fl] = c["burden_neg"] + c["burden_pos"]
        shared_flags[fl] = (c["burden_neg"] > 0) | (c["burden_pos"] > 0)
        print(f"fold {k}: pairs {total:.3e} shared_ge {sg} ({sg/fail_port:.1%} of port fail) "
              f"route_only {ro} aux_only {ao} agg_loss {aggregation_loss} "
              f"auc_err {abs(auc_direct-auc_from):.1e}", flush=True)

    # --------------------------------------------------------------- pooled
    # Two different aggregations, both meaningful, kept distinct:
    #  - `within_fold_sums`: sums of the per-fold (within-fold pair) counts.
    #    These are what the per-fold decomposition identities verify.
    #  - `pooled_global`: ONE rank decomposition over the full 699,635-row OOF
    #    vector (cross-fold pairs included). This is the decomposition whose
    #    residual reproduces the certified global OOF AUC exactly.
    df = pd.DataFrame(rows)
    within_fold_sums = {
        "pairs": int(df.pairs.sum()),
        "port_strict_wrong": int(df.port_strict_wrong.sum()),
        "port_ties": int(df.port_ties.sum()),
        "port_fail_ge": int(df.port_fail_ge.sum()),
        "route_strict_wrong": int(df.route_strict_wrong.sum()),
        "aux_strict_wrong": int(df.aux_strict_wrong.sum()),
        "shared_ge": int(df.shared_ge.sum()),
        "shared_strict": int(df.shared_strict.sum()),
        "shared_and_port_strict": int(df.shared_and_port_strict.sum()),
        "shared_port_tie": int(df.shared_port_tie.sum()),
        "route_only": int(df.route_only.sum()),
        "aux_only": int(df.aux_only.sum()),
        "aggregation_loss_ge": int(df.aggregation_loss_ge.sum()),
    }

    ra_g, rb_g = rankdata(route2), rankdata(aux_prob)
    rp_g = rankdata(port)
    pos_sorted_g = np.sort(rp_g[y == 1])
    total_g = int((y == 1).sum() * (y == 0).sum())
    ties_g = tie_pairs(y, port)
    strict_g = strict_mis(y, rp_g, pos_sorted_g)
    fail_port_g = fail_ge(y, rp_g, pos_sorted_g)
    route_strict_g = strict_mis(y, ra_g, np.sort(ra_g[y == 1]))
    aux_strict_g = strict_mis(y, rb_g, np.sort(rb_g[y == 1]))
    cg = two_class_counts(y, ra_g, rb_g)
    sg, ss = cg["shared_ge"], cg["shared_strict"]
    ro_g, ao_g = cg["route_only"], cg["aux_only"]
    assert route_strict_g == ss + ro_g, "global route decomp"
    assert aux_strict_g == ss + ao_g, "global aux decomp"
    assert int(cg["burden_neg"].sum()) == sg == int(cg["burden_pos"].sum()), "global burden sums"
    auc_pooled = float(roc_auc_score(y, port))
    assert abs((1 - (strict_g + 0.5 * ties_g) / total_g) - auc_pooled) < 1e-15, \
        "pooled global AUC identity"
    pooled = {
        "pairs": total_g,
        "port_strict_wrong": strict_g, "port_ties": ties_g, "port_fail_ge": fail_port_g,
        "route_strict_wrong": route_strict_g, "aux_strict_wrong": aux_strict_g,
        "shared_ge": sg, "shared_strict": ss, "route_only": ro_g, "aux_only": ao_g,
        "aggregation_loss_ge": fail_port_g - sg,
    }
    assert pooled["aggregation_loss_ge"] >= 0, "global aggregation loss sign"
    pooled["shared_share_of_port_fail"] = pooled["shared_ge"] / pooled["port_fail_ge"]
    pooled["residual_auc_mass"] = (strict_g + 0.5 * ties_g) / total_g
    pooled["shared_auc_mass"] = pooled["shared_ge"] / total_g
    pooled["route_only_auc_mass"] = ro_g / total_g
    pooled["aux_only_auc_mass"] = ao_g / total_g
    pooled["aggregation_loss_auc_mass"] = pooled["aggregation_loss_ge"] / total_g

    # ------------------------------------------------- row-level diagnostics
    teacher = np.load(ROOT / "artifacts/predictions/view_lgbm_raw_teacher_primary_oof.npy")
    assert teacher.shape == (n,), "teacher shape"
    # per-fold opposite-class normalisation: fraction of the fold's opposite-class
    # rows that this row is shared-misranked with.
    neg_per_fold = {k: int((y[folds == k] == 0).sum()) for k in range(5)}
    pos_per_fold = {k: int((y[folds == k] == 1).sum()) for k in range(5)}
    rate_fold = np.zeros(n, dtype=np.float64)
    for k in range(5):
        m = folds == k
        rate_fold[m & (y == 1)] = burden[m & (y == 1)] / neg_per_fold[k]
        rate_fold[m & (y == 0)] = burden[m & (y == 0)] / pos_per_fold[k]

    confident_wrong = ((y == 1) & (route2 < 0.4) & (aux_prob < 0.4)) | \
                      ((y == 0) & (route2 > 0.6) & (aux_prob > 0.6))
    near_tie_both = (np.abs(route2 - 0.5) <= 0.01) & (np.abs(aux_prob - 0.5) <= 0.01)
    teacher_wrong = ((y == 1) & (teacher <= 0.3)) | ((y == 0) & (teacher >= 0.7))
    hi = rate_fold >= np.quantile(rate_fold, 0.8)
    enrichment = {
        "top_quintile_burden_rows": int(hi.sum()),
        "teacher_confident_wrong_rate_top_burden": float(teacher_wrong[hi].mean()),
        "teacher_confident_wrong_rate_all": float(teacher_wrong.mean()),
        "teacher_confident_wrong_enrichment": float(teacher_wrong[hi].mean()
                                                    / max(teacher_wrong.mean(), 1e-12)),
        "confident_both_wrong_rows": int(confident_wrong.sum()),
        "confident_both_wrong_burden_rate": float(rate_fold[confident_wrong].mean())
        if confident_wrong.any() else None,
        "near_tie_both_rows": int(near_tie_both.sum()),
        "near_tie_both_burden_rate": float(rate_fold[near_tie_both].mean())
        if near_tie_both.any() else None,
        "burden_rate_label_mean_pos": float(rate_fold[y == 1].mean()),
        "burden_rate_label_mean_neg": float(rate_fold[y == 0].mean()),
    }

    # ------------------------------------------------------------- segments
    ratings = ["Inflight wifi service", "Departure/Arrival time convenient",
               "Ease of Online booking", "Gate location", "Food and drink",
               "Online boarding", "Seat comfort", "Inflight entertainment",
               "On-board service", "Leg room service", "Baggage handling",
               "Checkin service", "Cleanliness"]
    key = None
    for c in ratings:
        s = tr[c].astype(str) + "|"
        key = s if key is None else key + s
    cnt = pd.concat([key, te[[c for c in ratings]].astype(str)
                     .agg("|".join, axis=1)], ignore_index=True).value_counts()
    is_dup = key.map(cnt).to_numpy() > 1

    seg_defs: list[tuple[str, np.ndarray]] = [("dup_rating_key", is_dup),
                                              ("unique_rating_key", ~is_dup)]
    for col in ("Class", "Type of Travel", "Customer Type", "Gender"):
        colv = tr[col].astype(str).to_numpy()
        for val in sorted(np.unique(colv)):
            seg_defs.append((f"{col}=={val}", colv == val))
    dep = tr["Departure Delay in Minutes"].to_numpy()
    arrd = tr["Arrival Delay in Minutes"].to_numpy()
    seg_defs += [("delay_dep>30", dep > 30), ("delay_arr>30", arrd > 30),
                 ("delay_none", (dep == 0) & (arrd == 0)),
                 ("age<30", tr["Age"].to_numpy() < 30),
                 ("age>=55", tr["Age"].to_numpy() >= 55)]
    seg_rows = []
    for name, mask in seg_defs:
        yy = y[mask]
        if len(np.unique(yy)) < 2 or mask.sum() < 100:
            continue
        seg_rows.append({
            "segment": name, "rows": int(mask.sum()),
            "participant_rate": float(shared_flags[mask].mean()),
            "mean_burden_rate": float(rate_fold[mask].mean()),
            "port_auc": float(roc_auc_score(yy, port[mask])),
            "route_auc": float(roc_auc_score(yy, route2[mask])),
            "aux_auc": float(roc_auc_score(yy, aux_prob[mask])),
            "high_burden_share": float(hi[mask].mean()),
        })
    seg_df = pd.DataFrame(seg_rows).sort_values("mean_burden_rate", ascending=False)
    seg_df.to_parquet(OUT / "shared_error_segments.parquet", index=False)

    pd.DataFrame({
        "id": ids, "fold": folds, "satisfaction": y, "route": route2,
        "aux": aux_prob, "portfolio": port, "teacher": teacher,
        "burden": burden, "burden_rate": rate_fold,
        "shared_participant": shared_flags, "confident_both_wrong": confident_wrong,
        "near_tie_both": near_tie_both, "teacher_confident_wrong": teacher_wrong,
        "dup_key": is_dup,
    }).to_parquet(OUT / "per_row_burden.parquet", index=False)

    # ------------------------------------------- difficulty predictability
    # Target: top-quintile burden (non-degenerate). Two feature sets, both
    # fold-honest: (A) the full rich view; (B) rich view + inference-time model
    # confidences (available on the test set without labels).
    from scripts.run_views import _fit_lgbm_es, _inner_es_split  # noqa: E402
    from src.features.view import ViewBuilder  # noqa: E402
    import gc  # noqa: E402

    # fold-wise quintile avoids cross-fold burden-scale artifacts
    flag = np.zeros(n, dtype=np.int8)
    for k in range(5):
        m = folds == k
        flag[m] = (rate_fold[m] >= np.quantile(rate_fold[m], 0.8)).astype(np.int8)
    assert 0.15 < flag.mean() < 0.25, f"flag balance {flag.mean()}"

    sig = np.column_stack([
        route2, aux_prob, np.abs(route2 - aux_prob),
        -(route2 * np.log(np.clip(route2, 1e-9, 1))
          + (1 - route2) * np.log(np.clip(1 - route2, 1e-9, 1))),
        -(aux_prob * np.log(np.clip(aux_prob, 1e-9, 1))
          + (1 - aux_prob) * np.log(np.clip(1 - aux_prob, 1e-9, 1))),
        np.minimum(np.abs(route2 - 0.5), np.abs(aux_prob - 0.5)),
    ]).astype(np.float32)

    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    pred_a = np.full(n, np.nan, dtype=np.float32)
    pred_b = np.full(n, np.nan, dtype=np.float32)
    params = {"n_estimators": 3000, "learning_rate": 0.05,
              "num_leaves": 96, "min_child_samples": 40}
    for k in range(5):
        fi = np.flatnonzero(folds != k)
        va = np.flatnonzero(folds == k)
        itr, es = _inner_es_split(fi, flag.astype(np.int8), 1201 + k)
        xf, xa, _names = vb.assemble(fi, y, va, None, inner_seed=k)
        tl = np.searchsorted(fi, itr)
        el = np.searchsorted(fi, es)
        Xtr, Xes, Xva = xf[tl], xf[el], xa["val"]
        for tag, X_extra, pred in (("A", None, pred_a), ("B", sig, pred_b)):
            Xt = Xtr if X_extra is None else np.column_stack([Xtr, X_extra[itr]])
            Xe = Xes if X_extra is None else np.column_stack([Xes, X_extra[es]])
            Xv = Xva if X_extra is None else np.column_stack([Xva, X_extra[va]])
            p, _it = _fit_lgbm_es(Xt, flag[itr], Xv, params, 1201 + k,
                                  es_X=Xe, es_y=flag[es])
            pred[va] = p
        del xf, xa, Xtr, Xes, Xva
        gc.collect()
        print(f"  difficulty model fold {k} done", flush=True)
    del vb
    gc.collect()
    assert np.isfinite(pred_a).all() and np.isfinite(pred_b).all()

    def diff_metrics(pred: np.ndarray) -> dict:
        per_fold_aucs = [float(roc_auc_score(flag[folds == k], pred[folds == k]))
                         for k in range(5)]
        return {
            "pooled_auc": float(roc_auc_score(flag, pred)),
            "per_fold_auc": per_fold_aucs,
            "fold_mean": float(np.mean(per_fold_aucs)),
            "fold_se": float(np.std(per_fold_aucs, ddof=1) / np.sqrt(5)),
            "spearman_vs_burden_rate": float(spearmanr(pred, rate_fold).statistic),
            "satisfaction_auc_context": float(roc_auc_score(y, pred)),
        }

    trivial_sig = np.minimum(np.abs(route2 - 0.5), np.abs(aux_prob - 0.5))
    difficulty = {
        "target": "top-quintile shared-burden row (fold-wise), positive rate "
                  f"{float(flag.mean()):.4f}",
        "protocol": "cross-fitted 5-fold LightGBM, inner 10% ES from FIT rows only, "
                    "outer-fold labels never used in training or selection",
        "set_A_rich_view_only": diff_metrics(pred_a),
        "set_B_rich_plus_inference_confidence": diff_metrics(pred_b),
        "trivial_min_margin_auc": float(roc_auc_score(flag, trivial_sig)),
        "interpretation": "pooled_auc ~0.5 => residual difficulty NOT predictable; "
                          ">0.55 => structure usable at inference time",
    }

    # ------------------------------------------------------------------ save
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": "AXIS-19-A",
        "git": git_commit(),
        "champion": "v7_sol_tabpfn_internal2_aux10",
        "champion_portfolio_sha256": arr_sha256(port.astype("float32")),
        "champion_route_sha256": arr_sha256(route2.astype("float32")),
        "aux_logits_source": "classical() float64 mean member logits, per fold",
        "aux_stored_cross_check_max_gap": aux_gap,
        "composition_identity": "bit-exact (asserted)",
        "counting_convention": "AUC = 1 - (strict_wrong + 0.5*ties)/pairs, "
                               "identity asserted < 1e-15 per fold and pooled",
        "monotone_theorem": "shared_ge subseteq port_fail_ge (proof: logit+sigmoid "
                            "monotone, f32 rounding monotone; verified per fold)",
        "per_fold": per_fold,
        "pooled": pooled,
        "within_fold_sums": within_fold_sums,
        "pooled_auc": auc_pooled,
        "row_level": enrichment,
        "difficulty_predictability": difficulty,
        "segments_top10_burden": json.loads(seg_df.head(10).to_json(orient="records")),
        "source_sha256": {q: file_sha256(q) for q in (
            "scripts/phase19/build_shared_error_atlas.py",)},
        "verdict_inputs": "shared_error_atlas.md",
    }
    save_json(result, OUT / "shared_error_atlas.json")

    md = [
        "# Phase 19 shared-error atlas — BOTH v7 halves miss what?",
        "",
        f"- champion portfolio AUC **{auc_pooled:.9f}** (hash verified, composition "
        "bit-exact)",
        f"- residual fail mass (pairs with port_n >= port_p): "
        f"**{pooled['port_fail_ge'] / pooled['pairs']:.6f}** of all "
        f"{pooled['pairs']:.4e} pairs",
        f"- **shared fail (neither route nor aux ranks the pair right): "
        f"{pooled['shared_ge']:,} = {pooled['shared_share_of_port_fail']:.1%} of "
        "portfolio fail mass**",
        f"- shared strict: {pooled['shared_strict']:,}; within-fold shared pairs "
        f"that the portfolio ties: {within_fold_sums['shared_port_tie']:,}",
        f"- aggregation loss (portfolio fails while >=1 base ranks it right): "
        f"{pooled['aggregation_loss_ge']:,} "
        f"({pooled['aggregation_loss_ge'] / pooled['pairs']:.6f} AUC mass)",
        f"- route-only errors: {pooled['route_only']:,} | aux-only errors: "
        f"{pooled['aux_only']:,} (one family right, the other wrong)",
        "",
        "## Per-fold",
        "",
        df[["fold", "shared_ge", "shared_share_of_port_fail", "route_only",
            "aux_only", "aggregation_loss_ge", "port_auc"]].to_string(index=False),
        "",
        "## Are shared errors predictable WITHOUT validation labels?",
        "",
        f"- set A (rich view only): pooled AUC "
        f"**{difficulty['set_A_rich_view_only']['pooled_auc']:.4f}** "
        f"(folds {['%.4f' % x for x in difficulty['set_A_rich_view_only']['per_fold_auc']]})",
        f"- set B (+ inference-time confidences): pooled AUC "
        f"**{difficulty['set_B_rich_plus_inference_confidence']['pooled_auc']:.4f}**",
        f"- trivial margin baseline: {difficulty['trivial_min_margin_auc']:.4f}",
        "",
        "## Row-level structure",
        "",
        f"- rows participating in >=1 shared pair: {int(shared_flags.sum()):,} "
        f"({shared_flags.mean():.1%})",
        f"- confident-both-wrong rows: {enrichment['confident_both_wrong_rows']:,}",
        f"- near-tie-in-both rows: {enrichment['near_tie_both_rows']:,}",
        f"- teacher-confident-wrong enrichment in top-burden quintile: "
        f"{enrichment['teacher_confident_wrong_enrichment']:.3f}x",
        "",
        "## Highest-burden segments",
        "",
        seg_df.head(8).to_string(index=False),
        "",
        "## Reading guide",
        "",
        "- `shared_ge / port_fail` is the ceiling for what a *third model added to "
        "the average* could ever touch: pairs already failed by BOTH halves are "
        "invisible to reweighting.",
        "- `aggregation_loss` is the headroom of aggregation changes (closed axis).",
        "- `route_only` vs `aux_only` asymmetry shows which family carries the "
        "correcting signal on shared-adjacent pairs.",
        "- Predictability AUCs answer whether a difficulty-aware mechanism can "
        "operate at test time (covariates/confidences only, no labels).",
    ]
    (OUT / "shared_error_atlas.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"pooled": pooled, "enrichment": enrichment,
                      "difficulty": {k: (v.get("pooled_auc") if isinstance(v, dict) else v)
                                     for k, v in difficulty.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
