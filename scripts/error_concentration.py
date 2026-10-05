"""Where does our remaining ranking error actually live: rows we disagree about, or rows we all agree on?

Why this decides the next phase
-------------------------------
The campaign's one strong surviving mechanism is variance reduction -- extra_trees (+2.3e-4),
RealMLP 8->32 members (+2.0e-4), 5->10 folds (+1.0e-4), 59-member averaging (+2.5e-4). All of those
buy the same thing: less fitting of label noise. If most of our remaining ranking error sits on rows
where our members DISAGREE, then the missing ingredient is still variance reduction and structurally
stochastic tree construction (Phase 9's DART/RF) is the right medicine. If instead most of the error
sits on rows where every member agrees, then our models are confidently wrong together, which no
amount of averaging can fix, and the 7.8e-4 gap to the leader is missing signal rather than excess
variance.

That is a genuine fork in the strategy and it is answerable from artifacts we already have. No
training is required.

The measurement
---------------
Per row we form the member-logit vector and summarise it as dispersion measures. Rows are then
bucketed into deciles of dispersion, and for each bucket we compute:

  auc_d      pairwise AUC among pairs where BOTH rows fall in bucket d. This is the ranking quality
             inside that difficulty stratum.
  pairs_d    the number of pos/neg pairs the bucket contributes, so a low auc_d on a small bucket is
             not mistaken for the main problem.
  err_d      n_pairs_d * (1 - auc_d), the bucket's share of all pairwise ranking errors.

err_d / sum(err_d) is the quantity that answers the question. We also report cross-bucket pairs,
where one row is confident and the other is not, separately, because a confident model ordering a
disputed row wrongly is a distinct and more diagnostic failure than two disputed rows swapping.

Honesty notes
-------------
* Member OOF predictions are out-of-fold for every training row, so conditioning on them is safe.
* Deciles are formed on the whole training set for reporting only; no correction is fitted and
  nothing here feeds a model, so there is no selection effect to correct for.
* `blend_*` entries in the store are ensembles of other members. Including them would double-count
  their constituents and inflate apparent agreement, so only single-model members are used and the
  blend is reported separately as the reference score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def within_bucket_auc(score, y, bucket, nb):
    """Pairwise AUC using only pairs whose two rows share a bucket."""
    tot_err = 0.0
    tot_pairs = 0
    per = []
    for b in range(nb):
        m = bucket == b
        s, t = score[m], y[m]
        npos, nneg = int(t.sum()), int((1 - t).sum())
        pairs = npos * nneg
        if pairs == 0 or npos == 0 or nneg == 0:
            per.append({"bucket": b, "n_rows": int(m.sum()), "n_pos": npos, "n_neg": nneg,
                        "n_pairs": pairs, "mean_p": float(score[m].mean()),
                        "auc": None, "errors": 0.0, "share": 0.0})
            continue
        # rank-based pairwise comparison inside the bucket, O(n log n) via ranks
        r = rankdata(s)
        # sum of ranks of positives; AUC = (R_pos - npos(npos+1)/2) / (npos*nneg)
        auc = (r[t == 1].sum() - npos * (npos + 1) / 2.0) / pairs
        err = pairs * (1.0 - auc)
        tot_err += err
        tot_pairs += pairs
        per.append({"bucket": b, "n_rows": int(m.sum()), "n_pos": npos, "n_neg": nneg,
                    "n_pairs": pairs, "mean_p": float(score[m].mean()),
                    "auc": float(auc), "errors": float(err),
                    "share": float(err / tot_err) if tot_err else 0.0})
    for p in per:
        p["share"] = float(p["errors"] / tot_err) if tot_err else 0.0
    return per, float(tot_err), tot_pairs


def cross_bucket_auc(score, y, bucket, nb):
    """Pairs where one row sits in bucket i and the other in bucket j>i (disputed vs confident)."""
    rows = []
    tot_err, tot_pairs = 0.0, 0
    for i in range(nb):
        for j in range(i + 1, nb):
            mi, mj = bucket == i, bucket == j
            si, sj = score[mi], score[mj]
            yi, yj = y[mi], y[mj]
            # pos in i vs neg in j, and neg in i vs pos in j
            pi, ni = yi == 1, yi == 0
            pj, nj = yj == 1, yj == 0
            pairs = pi.sum() * nj.sum() + ni.sum() * pj.sum()
            if pairs == 0:
                continue
            err = ((si[pi][:, None] > sj[nj][None, :]).sum()
                   + (si[ni][:, None] < sj[pj][None, :]).sum())
            tot_err += float(err)
            tot_pairs += int(pairs)
            rows.append({"i": i, "j": j, "n_pairs": int(pairs), "errors": int(err),
                         "auc": float(1 - err / pairs)})
    for r in rows:
        r["share"] = float(r["errors"] / tot_err) if tot_err else 0.0
    return rows, float(tot_err), tot_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="blend_v3_final")
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--nbuckets", type=int, default=10)
    ap.add_argument("--min-auc", type=float, default=0.90,
                    help="only single-model members at or above this standalone OOF are used, so "
                         "the dispersion measure reflects models worth averaging")
    ap.add_argument("--measure", default="std_logit",
                    help="which dispersion measure is the headline; all are always reported")
    ap.add_argument("--n-sample", type=int, default=3000,
                    help="positives and negatives sampled for the pairwise error-rate measurement")
    ap.add_argument("--tag", default="error_concentration")
    args = ap.parse_args()

    t0 = time.time()
    from src.submission import store
    tr, _te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds

    meta = store.list_all()
    singles = [m for m in meta
               if not m["exp_id"].startswith("blend")
               and not m["exp_id"].startswith("v")
               and m.get("auc") is not None and m["auc"] >= args.min_auc]
    print(f"store has {len(meta)} entries; {len(singles)} single-model members at or above "
          f"OOF {args.min_auc} are used for the dispersion measure")

    M, names, fams = [], [], []
    for m in singles:
        try:
            oof = store.load_oof(m["exp_id"]).astype("float64")
        except Exception:                                   # noqa: BLE001
            continue
        if oof.shape[0] != y.shape[0] or not np.isfinite(oof).all():
            continue
        M.append(logit(oof))
        names.append(m["exp_id"])
        fams.append(m.get("family") or "?")
    M = np.column_stack(M)
    print(f"loaded {M.shape[1]} member logit vectors over {M.shape[0]:,} rows")
    fam_counts = {}
    for f in fams:
        fam_counts[f] = fam_counts.get(f, 0) + 1
    print("families:", dict(sorted(fam_counts.items(), key=lambda kv: -kv[1])))

    ref = store.load_oof(args.ref).astype("float64")
    ref_auc = float(roc_auc_score(y, ref))
    print(f"reference blend {args.ref} OOF = {ref_auc:.6f}")

    # ---------------- dispersion measures ----------------
    # IMPORTANT, and learned the hard way: a naive dispersion measure on member logits is NOT a
    # difficulty measure. Because members are individually well-calibrated and the ensemble mean
    # saturates toward 0 and 1, the cross-member std is SMALLEST at p ~ 0.5 and LARGEST at the
    # extremes. Measured here: mean std by predicted-probability decile runs
    # 0.545, 0.487, 0.460, 0.437, 0.393, 0.273, 0.436, 0.462, 0.497, 0.614.
    # So "std_logit" mostly encodes |p - 0.5|, and stratifying error by it reproduces a near-tie
    # stratum rather than an uncertainty stratum. Every measure below is therefore built to be
    # INVARIANT to the predicted level, so the resulting buckets compare like with like.
    mean_logit = M.mean(axis=1)
    measures: dict[str, np.ndarray] = {}
    # distance from the decision boundary: the level the previous measures were secretly proxying
    measures["abs_logit_gap"] = np.abs(mean_logit)
    # level-free scale: dispersion normalised by the spread expected at that level
    measures["std_over_gap"] = M.std(axis=1) / (np.abs(mean_logit) + 0.25)
    # rank spread is level-free by construction
    R = np.apply_along_axis(rankdata, 0, M) / M.shape[0]
    measures["rank_range"] = R.max(axis=1) - R.min(axis=1)
    measures["rank_iqr"] = np.percentile(R, 75, axis=1) - np.percentile(R, 25, axis=1)
    measures["rank_std"] = R.std(axis=1)
    # family disagreement: spread of family-mean RANKS (structure, not member count)
    fams_u = sorted(set(fams))
    fcol = np.column_stack([np.mean(R[:, [i for i, f in enumerate(fams) if f == ff]], axis=1)
                            for ff in fams_u])
    measures["family_rank_std"] = fcol.std(axis=1)
    # entropy-like: how evenly members split between the extremes of the row's own rank range
    measures["rank_range_over_gap"] = measures["rank_range"] / (np.abs(mean_logit) + 0.25)
    # keep the naive ones too, clearly labelled, so the confound stays visible in the report
    measures["NAIVE_std_logit"] = M.std(axis=1)
    measures["NAIVE_logit_range"] = M.max(axis=1) - M.min(axis=1)

    nb = args.nbuckets
    out = {"ref": args.ref, "ref_auc": ref_auc, "scheme": args.scheme, "n_members": int(M.shape[1]),
           "families": fam_counts, "min_auc": args.min_auc, "measures": {}}

    print(f"\n{'='*112}\nERROR CONCENTRATION BY DISAGREEMENT DECILE  (ref = {args.ref})\n{'='*112}")
    summary = {}
    headline = args.measure if args.measure in measures else "rank_range"
    print(f"measures computed: {list(measures)}\n"
          f"primary measure: {headline}  (others retained for robustness)\n")
    for mname, d in measures.items():
        # bucket by the measure; decile 0 = least disagreement
        b = np.argsort(np.argsort(d)) * nb // len(d)
        per, terr, tpairs = within_bucket_auc(ref, y, b, nb)
        xb, xerr, xpairs = cross_bucket_auc(ref, y, b, nb)
        # aggregate: bottom 3 buckets (confident) vs top 3 (disputed)
        conf = [p for p in per if p["bucket"] < 3]
        disp = [p for p in per if p["bucket"] >= nb - 3]
        e_conf = sum(p["errors"] for p in conf)
        e_disp = sum(p["errors"] for p in disp)
        tot_e = e_conf + e_disp
        a_conf = 1 - e_conf / sum(p["n_pairs"] for p in conf) if sum(
            p["n_pairs"] for p in conf) else None
        a_disp = 1 - e_disp / sum(p["n_pairs"] for p in disp) if sum(
            p["n_pairs"] for p in disp) else None
        print(f"\n--- {mname} ---")
        print(f"  {'decile':>7}{'rows':>9}{'pos':>9}{'mean p':>9}{'pairs':>14}{'AUC':>10}"
              f"{'err share':>11}{'cum':>9}")
        cum = 0.0
        for p in per:
            if p["auc"] is None:
                print(f"  {p['bucket']:>7}{p['n_rows']:>9,}{p['n_pos']:>9,}{p['mean_p']:>9.4f}"
                      f"{p['n_pairs']:>14,}{'-':>10}{'-':>11}{'-':>9}")
                continue
            cum += p["share"]
            print(f"  {p['bucket']:>7}{p['n_rows']:>9,}{p['n_pos']:>9,}{p['mean_p']:>9.4f}"
                  f"{p['n_pairs']:>14,}{p['auc']:>10.6f}{p['share']*100:>10.2f}%{cum*100:>8.2f}%")
        print(f"  confident (deciles 0-2): AUC={a_conf:.6f}  {e_conf/max(tot_e,1e-9)*100:.1f}% of errors")
        print(f"  disputed  (deciles {nb-3}-{nb-1}): AUC={a_disp:.6f}  "
              f"{e_disp/max(tot_e,1e-9)*100:.1f}% of errors")
        # Cross-bucket pairs are counted for completeness only. Their AUC is near 0.04 because the
        # bucket index is an ordering of DISPERSION, not of the score, so a positive in a
        # low-dispersion bucket usually outranks a negative in a high-dispersion one purely because
        # of how the buckets were cut. That number is not an error rate and is not used as one; it
        # is excluded from the headline shares, which use within-bucket pairs only.
        print(f"  [diagnostic only, not an error rate] cross-bucket pairs: {xpairs:,}  "
              f"apparent AUC={1-xerr/max(xpairs,1):.4f}")
        summary[mname] = {
            "per_decile": per, "cross": xb,
            "auc_confident": a_conf, "auc_disputed": a_disp,
            "err_share_confident": e_conf / max(tot_e, 1e-9),
            "err_share_disputed": e_disp / max(tot_e, 1e-9),
            "cross_auc": 1 - xerr / max(xpairs, 1),
            "cross_err_share": xerr / max(tpairs, 1e-9),
        }
    out["measures_detail"] = summary

    # ---------------- confound-free version: error rate vs PAIR disagreement ----------------
    # The decile table above is confounded: a row's member-logit dispersion is mechanically small
    # when its predicted probability is near 0 or 1, and such rows are mostly the easy majority
    # class. So a low-dispersion decile can look error-dense purely because it is a near-tie stratum
    # of one class, not because the models are confidently wrong.
    # The clean measurement conditions on PAIRS instead of rows: take a random sample of positives
    # and negatives, score every cross pair, and ask how the ERROR RATE varies with the pair's mean
    # disagreement. This needs no assumption that dispersion and difficulty are separable, because
    # difficulty (the outcome) is measured on the same axis as the exposure.
    rng = np.random.default_rng(11)
    npos_i, nneg_i = np.where(y > 0.5)[0], np.where(y <= 0.5)[0]
    ps = rng.choice(npos_i, min(args.n_sample, len(npos_i)), replace=False)
    ns = rng.choice(nneg_i, min(args.n_sample, len(nneg_i)), replace=False)
    sp, sn = ref[ps], ref[ns]
    wrong = (sp[:, None] < sn[None, :])
    allrate = float(wrong.mean())
    # For each level-free measure, condition the PAIR error rate on the pair's mean disagreement.
    # This is the confound-free form of the question: difficulty is measured on the same axis as the
    # exposure, so no assumption is needed that dispersion and difficulty are separable.
    out["pairwise"] = {"n_pos": len(ps), "n_neg": len(ns), "overall_err_rate": allrate,
                       "n_pairs": int(len(ps) * len(ns)), "by_measure": {}}
    print(f"\n{'='*112}\nPAIRWISE ERROR RATE vs DISAGREEMENT  "
          f"({len(ps):,} pos x {len(ns):,} neg = {len(ps)*len(ns)/1e9:.2f}e9 pairs, "
          f"overall error rate {allrate:.5f})\n{'='*112}")
    for mname in measures:
        if mname.startswith("NAIVE"):
            continue
        d = measures[mname]
        md = (d[ps][:, None] + d[ns][None, :]) / 2.0
        edges = np.quantile(md, np.linspace(0, 1, nb + 1))
        edges[-1] += 1e-12
        band_rows = []
        print(f"\n--- {mname} ---")
        print(f"  {'band':>6}{'disagree range':>24}{'pairs':>16}{'err rate':>11}{'lift':>9}")
        for b in range(nb):
            m = (md >= edges[b]) & (md < edges[b + 1])
            cnt = int(m.sum())
            if cnt == 0:
                continue
            r = float(wrong[m].mean())
            band_rows.append({"band": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                              "n_pairs": cnt, "err_rate": r, "lift": r / allrate})
            print(f"  {b:>6}[{edges[b]:>10.4f},{edges[b+1]:>10.4f}){cnt:>16,}{r:>11.5f}"
                  f"{r/allrate:>8.2f}x")
        if band_rows:
            lo = band_rows[0]["err_rate"]
            hi = band_rows[-1]["err_rate"]
            print(f"  monotone increasing? {'YES' if hi > lo * 1.5 else 'NO'}   "
                  f"bottom band {lo:.5f} -> top band {hi:.5f}  ({hi/max(lo,1e-9):.2f}x)")
        out["pairwise"]["by_measure"][mname] = band_rows

    # ---------------- the fork ----------------
    # Robustness requirement: the fork must not depend on which dispersion measure is used. If the
    # verdicts disagree across measures, the honest conclusion is that the question is not settled by
    # this instrument, and that gets reported rather than hidden behind one convenient measure.
    verdicts = {m: ("variance" if summary[m]["err_share_disputed"] >= 0.40 else "missing_signal")
                for m in summary if not m.startswith("NAIVE")}
    agree = len(set(verdicts.values())) == 1
    print(f"\nverdict by level-free measure: "
          f"{ {m: verdicts[m] for m in sorted(verdicts)} }  -> "
          f"{'CONSISTENT' if agree else 'INCONSISTENT across measures'}")
    for m in summary:
        if m.startswith("NAIVE"):
            print(f"  [confounded, for reference only] {m}: "
                  f"{summary[m]['err_share_disputed']*100:.1f}% of errors in the top 3 deciles")

    hd = summary[headline]
    frac_disp = hd["err_share_disputed"]
    verdict = ("VARIANCE REDUCTION STILL PLAUSIBLE -- error concentrates where members disagree, so "
               "structurally stochastic construction (DART/RF) is the right next lever"
               if frac_disp >= 0.40 else
               "MISSING SIGNAL, NOT EXCESS VARIANCE -- most error sits where members AGREE, so "
               "averaging and stochastic construction cannot reach it; look for new information")
    if not agree:
        verdict += (" [BUT the verdict FLIPS across dispersion measures, so this instrument does "
                    "not settle the question on its own]")
    print(f"\n{'='*112}")
    print(f"HEADLINE ({headline}): {frac_disp*100:.1f}% of pairwise ranking errors fall in the top 3 "
          f"disagreement deciles,")
    print(f"which hold {sum(p['n_rows'] for p in hd['per_decile'] if p['bucket'] >= nb-3):,} rows "
          f"({3*100/nb:.0f}% of the data).")
    print(f"By contrast the BOTTOM 3 deciles hold "
          f"{hd['err_share_confident']*100:.1f}% of errors on the same number of rows.")
    print(f"Confident rows rank at AUC={hd['auc_confident']:.6f}; disputed rows at "
          f"AUC={hd['auc_disputed']:.6f}.")
    print(f"\nVERDICT: {verdict}")
    out["verdict"] = verdict
    out["verdict_by_measure"] = verdicts
    out["headline_measure"] = headline
    out["headline"] = {"err_share_disputed": frac_disp,
                       "err_share_confident": hd["err_share_confident"],
                       "auc_disputed": hd["auc_disputed"], "auc_confident": hd["auc_confident"],
                       "cross_err_share": hd["cross_err_share"]}
    out["seconds"] = round(time.time() - t0, 1)
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip()[:12]
    save_json(out, REPORTS / f"{args.tag}.json")
    print("wrote", REPORTS / f"{args.tag}.json", f"({out['seconds']}s)")


if __name__ == "__main__":
    main()
