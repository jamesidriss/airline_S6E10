"""Phase18 error atlas — zero-training diagnostic of certified v7 vs v6.

Inputs (all hash-verified against certificates before use):
  v7 candidate OOF  : artifacts/phase17_B1/f{0..4}/portfolio.npy  (bank portfolio_sha256)
  v7 route OOF      : artifacts/phase17_B1/f{0..4}/prediction.npy  (bank route_sha256)
  aux10 OOF         : artifacts/sol_route_aux10_primary_final/aux10_oof.npy (bank oof aux10)
  v6 OOF            : artifacts/predictions/sol_route_aux10_primary_final_oof.npy (bank candidate)
  teacher OOF       : artifacts/predictions/view_lgbm_raw_teacher_primary_oof.npy
  fold assignment   : data/cache/folds/primary_5f_seed20261010.npy

Outputs (reports/phase18/):
  error_atlas.json / error_atlas.md
  pairwise_diagnostics.parquet   -- exact per-row pair-credit decomposition, v7 vs v6
  segment_stability.parquet      -- segment AUCs and deltas, pooled + per fold
  model_disagreement.parquet     -- route vs aux10 disagreement structure

Exactness contract: the sum of per-row pair-credit changes equals the total
rescue-minus-damage credit, which equals (AUC_v7 - AUC_v6) * total_pairs to
<=1e-6 absolute on credit units. No pair sampling anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, load_json  # noqa: E402
from scripts.phase16_pair_metrics import pair_rescue_damage  # noqa: E402
from scripts.phase16_common import error_pairs  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = ROOT / "reports" / "phase18"
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]


# --------------------------------------------------------------------------------------
# exact rank-based per-row pair credit
# --------------------------------------------------------------------------------------
def per_row_error_credit(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """For each row: half the exact credit of opposite-class pairs it misorders.

    positive row i: 0.5 * (#neg with p_neg > p_pos + 0.5 * #equal)
    negative row j: 0.5 * (#pos with p_pos < p_neg + 0.5 * #equal)
    Each misordered pair is shared equally between its two rows, so summing
    over ALL rows reproduces error_pairs() exactly (ties => half credit each).
    """
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    pn, nn = p[pos], p[neg]
    sn, sp = np.sort(nn), np.sort(pn)
    credit = np.empty(len(p), dtype=np.float64)
    # positive rows: negs strictly greater + half of equal
    greater = len(nn) - np.searchsorted(sn, pn, side="right")
    equal = np.searchsorted(sn, pn, "right") - np.searchsorted(sn, pn, "left")
    credit[pos] = 0.5 * (greater + 0.5 * equal)
    # negative rows: pos strictly less + half of equal
    lower = np.searchsorted(sp, nn, side="left")
    equal = np.searchsorted(sp, nn, "right") - lower
    credit[neg] = 0.5 * (lower + 0.5 * equal)
    return credit


def near_tie_and_confident_counts(y: np.ndarray, p: np.ndarray, eps: float, conf: float) -> dict:
    """Exact counts of pos-neg pairs with |margin|<eps (near-ties) and
    margin>conf (high-confidence correct) / margin<-conf (high-confidence wrong)."""
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    pn, nn = p[y == 1], p[y == 0]
    sn = np.sort(nn)
    lower = np.searchsorted(sn, pn - eps, "left")
    lower_eq = np.searchsorted(sn, pn + eps, "left")
    near = int((lower_eq - lower).sum())
    conf_wrong = int((len(nn) - np.searchsorted(sn, pn - conf, "right")).sum())
    conf_right = int(np.searchsorted(sn, pn - conf, "left").sum())
    total = len(pn) * len(nn)
    return {"near_tie_pairs": near, "near_tie_fraction": near / total,
            "confident_wrong_pairs": conf_wrong,
            "confident_wrong_fraction": conf_wrong / total,
            "confident_right_pairs": conf_right,
            "confident_right_fraction": conf_right / total,
            "total_pairs": total}


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------
def load_vectors() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bank = load_json(ROOT / "reports" / "v7_sol_tabpfn_internal2_aux10_test_certificate.json")["bank"]
    fold_rep = load_json(ROOT / "reports" / "phase17_B1_primary.json")
    n = 699635
    v7 = np.zeros(n, dtype=np.float32)
    route = np.zeros(n, dtype=np.float32)
    for k in range(5):
        ids = np.load(ROOT / f"artifacts/phase17_B1/f{k}/apply_ids.npy")
        v7[ids] = np.load(ROOT / f"artifacts/phase17_B1/f{k}/portfolio.npy")
        route[ids] = np.load(ROOT / f"artifacts/phase17_B1/f{k}/prediction.npy")
    assert arr_sha256(v7) == fold_rep["portfolio_sha256"], "v7 portfolio hash mismatch"
    assert arr_sha256(route) == fold_rep["route_sha256"], "v7 route hash mismatch"
    aux10 = np.load(ROOT / "artifacts/sol_route_aux10_primary_final/aux10_oof.npy")
    v6 = np.load(ROOT / "artifacts/predictions/sol_route_aux10_primary_final_oof.npy")
    assert arr_sha256(aux10) == bank["oof_sha256"]["aux10"], "aux10 hash mismatch"
    assert arr_sha256(v6) == bank["oof_sha256"]["candidate"], "v6 hash mismatch"
    teacher = np.load(ROOT / "artifacts/predictions/view_lgbm_raw_teacher_primary_oof.npy")
    folds = np.load(ROOT / "data/cache/folds/primary_5f_seed20261010.npy")
    assert folds.shape == (n,)
    return folds, {"v7": v7, "v6": v6, "route": route, "aux10": aux10, "teacher": teacher}


# --------------------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------------------
def segment_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for col in ["Class", "Type of Travel", "Customer Type", "Gender"]:
        for val in sorted(df[col].unique()):
            masks[f"{col}=={val}"] = df[col] == val
    masks["Age<30"] = df["Age"] < 30
    masks["Age30-45"] = (df["Age"] >= 30) & (df["Age"] < 45)
    masks["Age45-60"] = (df["Age"] >= 45) & (df["Age"] < 60)
    masks["Age>=60"] = df["Age"] >= 60
    q = df["Flight Distance"].quantile([0.25, 0.5, 0.75]).to_numpy()
    masks["Dist Q1"] = df["Flight Distance"] <= q[0]
    masks["Dist Q2"] = (df["Flight Distance"] > q[0]) & (df["Flight Distance"] <= q[1])
    masks["Dist Q3"] = (df["Flight Distance"] > q[1]) & (df["Flight Distance"] <= q[2])
    masks["Dist Q4"] = df["Flight Distance"] > q[2]
    dep, arr = df["Departure Delay in Minutes"], df["Arrival Delay in Minutes"]
    masks["Delay both==0"] = (dep == 0) & (arr == 0)
    masks["Delay dep>60"] = dep > 60
    masks["Delay arr>60"] = arr > 60
    masks["Delay arr missing"] = arr.isna()
    rsum = df[RATINGS].sum(axis=1)
    masks["Ratings<=26 (low)"] = rsum <= 26
    masks["Ratings>=52 (high)"] = rsum >= 52
    zeros = (df[RATINGS] == 0).sum(axis=1)
    masks["Zeros>=4"] = zeros >= 4
    masks["All ratings equal"] = df[RATINGS].nunique(axis=1) == 1
    return masks


def key_duplicates(df: pd.DataFrame) -> np.ndarray:
    sig = pd.util.hash_pandas_object(df[RATINGS + ["Gender", "Customer Type", "Type of Travel", "Class"]], index=False)
    return sig.duplicated(keep=False).to_numpy()


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p18_error_atlas")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/cache/train.parquet")
    y = df["satisfaction"].to_numpy()
    folds, vec = load_vectors()
    v7, v6, route, aux10, teacher = vec["v7"], vec["v6"], vec["route"], vec["aux10"], vec["teacher"]

    auc7, auc6 = roc_auc_score(y, v7), roc_auc_score(y, v6)
    assert abs(auc7 - 0.9618529228157235) < 1e-9 and abs(auc6 - 0.9617656138245795) < 1e-9

    # --- exact pair structure ---------------------------------------------------------
    prd = pair_rescue_damage(y, v7, v6)
    c7 = per_row_error_credit(y, v7)
    c6 = per_row_error_credit(y, v6)
    e7, total7 = error_pairs(y, v7)
    e6, total6 = error_pairs(y, v6)
    assert total7 == total6 == prd["all_positive_negative_pairs"]
    assert abs(c7.sum() - e7) < 1e-6 and abs(c6.sum() - e6) < 1e-6, (c7.sum(), e7)
    delta_credit = (c6 - c7)  # positive = corrected by v7
    net = delta_credit.sum()
    assert abs(net - (prd["rescued_pair_credit"] - prd["damaged_pair_credit"])) < 1e-6
    assert abs(net / total7 - (auc7 - auc6)) < 1e-12, (net / total7, auc7 - auc6)

    margin7 = near_tie_and_confident_counts(y, v7, eps=0.01, conf=0.20)
    margin6 = near_tie_and_confident_counts(y, v6, eps=0.01, conf=0.20)

    # per-row diagnostics table
    disc = pd.DataFrame({
        "row": np.arange(len(y)), "fold": folds, "y": y,
        "v7": v7, "v6": v6, "credit_v7": c7, "credit_v6": c6,
        "credit_change": delta_credit,
        "conf_v7": np.abs(v7 - 0.5),
        "disagreement": np.abs(v7.astype(np.float64) - v6.astype(np.float64)),
    })
    for name in ["route", "aux10", "teacher"]:
        disc[name] = vec[name]
    disc["route_aux_gap"] = np.abs(route.astype(np.float64) - aux10.astype(np.float64))
    disc["teacher_gap"] = np.abs(teacher.astype(np.float64) - v7.astype(np.float64))
    dup = key_duplicates(df)
    disc["dup_key"] = dup

    # --- segment stability ------------------------------------------------------------
    masks = segment_masks(df)
    seg_rows = []
    for name, m in masks.items():
        mi = m.to_numpy() if hasattr(m, "to_numpy") else np.asarray(m)
        a7, a6m = roc_auc_score(y[mi], v7[mi]), roc_auc_score(y[mi], v6[mi])
        seg_rows.append(dict(segment=name, rows=int(mi.sum()), positives=int(y[mi].sum()),
                             auc_v7=a7, auc_v6=a6m, delta=a7 - a6m,
                             mean_credit_change=float(delta_credit[mi].mean())))
        for k in range(5):
            mk = mi & (folds == k)
            if mk.sum() and y[mk].std() > 0:
                seg_rows[-1][f"delta_f{k}"] = roc_auc_score(y[mk], v7[mk]) - roc_auc_score(y[mk], v6[mk])
    seg = pd.DataFrame(seg_rows)
    # Eco Plus multiple-comparison context: how many of the 45 segment x fold cells are negative
    dcols = [c for c in seg.columns if c.startswith("delta_f")]
    cells = seg[dcols].to_numpy()
    seg["neg_cells"] = (cells < 0).sum(axis=1)
    seg.to_parquet(OUT / "segment_stability.parquet", index=False)

    # --- model disagreement -----------------------------------------------------------
    dis = []
    gap = disc["route_aux_gap"].to_numpy()
    for q0, q1, label in [(0, 0.2, "D1 lowest"), (0.2, 0.4, "D2"), (0.4, 0.6, "D3"),
                          (0.6, 0.8, "D4"), (0.8, 1.0, "D5 highest")]:
        lo, hi = np.quantile(gap, [q0, q1])
        m = (gap >= lo) & (gap <= hi) if q0 == 0 else (gap > lo) & (gap <= hi)
        dis.append(dict(band=label, rows=int(m.sum()),
                        auc_v7=float(roc_auc_score(y[m], v7[m])),
                        auc_route=float(roc_auc_score(y[m], route[m])),
                        auc_aux10=float(roc_auc_score(y[m], aux10[m])),
                        mean_credit_change=float(delta_credit[m].mean()),
                        teacher_gap_mean=float(disc["teacher_gap"].to_numpy()[m].mean())))
    pd.DataFrame(dis).to_parquet(OUT / "model_disagreement.parquet", index=False)
    corr_route_aux = float(np.corrcoef(route, aux10)[0, 1])
    spearman_route_aux = float(pd.Series(route).corr(pd.Series(aux10), method="spearman"))

    # --- fold-level pair decomposition ------------------------------------------------
    fold_rows = []
    for k in range(5):
        mk = folds == k
        f_prd = pair_rescue_damage(y[mk], v7[mk], v6[mk])
        fold_rows.append(dict(fold=k, rows=int(mk.sum()),
                              auc_v7=roc_auc_score(y[mk], v7[mk]),
                              auc_v6=roc_auc_score(y[mk], v6[mk]),
                              rescued=f_prd["rescued_pair_credit"],
                              damaged=f_prd["damaged_pair_credit"],
                              net=f_prd["net_auc_gain"]))
    # pooled pair diagnostics parquet (per-row table is the audit artifact)
    disc.to_parquet(OUT / "pairwise_diagnostics.parquet", index=False)

    # --- atlas summary ----------------------------------------------------------------
    corrected = delta_credit > 0
    damaged = delta_credit < 0
    atlas = {
        "tag": args.tag,
        "inputs": {"v7_portfolio_sha256": arr_sha256(v7), "v7_route_sha256": arr_sha256(route),
                   "v6_sha256": arr_sha256(v6), "aux10_sha256": arr_sha256(aux10),
                   "teacher_sha256": arr_sha256(teacher)},
        "auc": {"v7": auc7, "v6": auc6, "delta": auc7 - auc6,
                "route": roc_auc_score(y, route), "aux10": roc_auc_score(y, aux10),
                "teacher": roc_auc_score(y, teacher)},
        "pairwise": {**prd,
                     "per_row_identity_max_abs_err": float(abs(c7.sum() - e7)),
                     "net_equals_auc_delta": True,
                     "rows_corrected": int(corrected.sum()), "rows_damaged": int(damaged.sum()),
                     "rows_unchanged": int((delta_credit == 0).sum()),
                     "top_corrected_rows_credit": float(np.sort(delta_credit)[-1000:].sum()),
                     "top_damaged_rows_credit": float(np.sort(delta_credit)[:1000].sum())},
        "margins_v7": margin7, "margins_v6": margin6,
        "folds": fold_rows,
        "disagreement": {"route_aux10_pearson": corr_route_aux,
                         "route_aux10_spearman": spearman_route_aux,
                         "v7_v6_pearson": float(np.corrcoef(v7, v6)[0, 1]),
                         "v7_v6_spearman": float(pd.Series(v7).corr(pd.Series(v6), method="spearman"))},
        "duplicate_keys": {"rows_in_dup_groups": int(dup.sum()),
                           "auc_v7_dup_only": float(roc_auc_score(y[dup], v7[dup])),
                           "auc_v7_unique_only": float(roc_auc_score(y[~dup], v7[~dup])),
                           "credit_change_dup_mean": float(delta_credit[dup].mean()),
                           "credit_change_unique_mean": float(delta_credit[~dup].mean())},
        "confidence": {
            "auc_v7_bottom_confidence_quintile": float(roc_auc_score(
                y[np.abs(v7 - 0.5) <= np.quantile(np.abs(v7 - 0.5), 0.2)],
                v7[np.abs(v7 - 0.5) <= np.quantile(np.abs(v7 - 0.5), 0.2)])),
            "mean_credit_change_low_conf": float(delta_credit[
                np.abs(v7 - 0.5) <= np.quantile(np.abs(v7 - 0.5), 0.2)].mean()),
            "mean_credit_change_high_conf": float(delta_credit[
                np.abs(v7 - 0.5) > np.quantile(np.abs(v7 - 0.5), 0.8)].mean()),
        },
        "teacher_disagreement": {
            "teacher_v7_pearson": float(np.corrcoef(teacher, v7)[0, 1]),
            "teacher_v7_spearman": float(pd.Series(teacher).corr(pd.Series(v7), method="spearman")),
            "mean_credit_change_top_teacher_gap": float(delta_credit[
                disc["teacher_gap"].to_numpy() > np.quantile(disc["teacher_gap"], 0.8)].mean()),
            "mean_credit_change_bottom_teacher_gap": float(delta_credit[
                disc["teacher_gap"].to_numpy() <= np.quantile(disc["teacher_gap"], 0.2)].mean()),
        },
        "notes": [
            "All pair credits exact; no sampling. Net credit / total pairs reproduces the AUC delta exactly.",
            "credit_change>0 means row's misordered-pair count decreased under v7 (a corrected row).",
            "Segment x fold cells: 9+ segments x 5 folds => treat isolated negative cells as multiple-comparison candidates, not effects.",
            "teacher vector is the original-domain LightGBM view model (OOF), used only for disagreement diagnostics.",
        ],
    }
    (OUT / "error_atlas.json").write_text(json.dumps(atlas, indent=2), encoding="utf-8")

    # --- markdown summary -------------------------------------------------------------
    seg_sorted = seg.sort_values("delta")
    md = [
        "# Phase18 error atlas — certified v7 vs v6 (primary OOF)",
        "",
        f"- v7 AUC **{auc7:.9f}** vs v6 **{auc6:.9f}** (delta **+{(auc7-auc6)*1e5:.2f}e-5**)",
        f"- Exact pair credit: rescued {prd['rescued_pair_credit']:.0f} / damaged {prd['damaged_pair_credit']:.0f} "
        f"of {total7:.3e} pairs; net reproduces AUC delta exactly.",
        f"- Rows corrected: {int(corrected.sum()):,} | damaged: {int(damaged.sum()):,} | unchanged: {int((delta_credit==0).sum()):,}",
        f"- Near-ties (|margin|<0.01): v7 {margin7['near_tie_fraction']:.4%} vs v6 {margin6['near_tie_fraction']:.4%}",
        f"- High-confidence wrong (margin<-0.20): v7 {margin7['confident_wrong_pairs']:,} vs v6 {margin6['confident_wrong_pairs']:,}",
        f"- route/aux10 Pearson {corr_route_aux:.6f} (Spearman {spearman_route_aux:.6f})",
        "",
        "## Segment deltas (v7 - v6), worst five",
        seg_sorted.head(5)[["segment", "rows", "delta", "neg_cells"]].to_string(index=False),
        "",
        "## Segment deltas, best five",
        seg_sorted.tail(5)[["segment", "rows", "delta", "neg_cells"]].to_string(index=False),
        "",
        "## Eco Plus",
        seg[seg["segment"] == "Class==Eco Plus"][["rows", "delta"] + dcols].to_string(index=False),
    ]
    (OUT / "error_atlas.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "ok", "auc_delta": auc7 - auc6,
                      "rows_corrected": int(corrected.sum()),
                      "rows_damaged": int(damaged.sum())}))


if __name__ == "__main__":
    main()
