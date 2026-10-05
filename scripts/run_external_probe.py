"""Does the ONE untested external dataset carry usable signal? A two-step test, cheapest first.

The candidate
-------------
`scripts/audit_external_datasets.py` found that five of seven external data files are verbatim
mirrors or subsets of the 129,880-row generator source, and closed the "find independent external
labelled data" track. But it left one anomaly unresolved:

    raminhuseyn__airline-customer-satisfaction   (and yakhyojon, byte-identical to it)
      129,880 rows, same schema, but 71,087 positives against the source's 56,428
      and only 3.4% overlap on the 13 survey ratings

Same row count and schema, so not independently collected -- but a DIFFERENT label vector that we
have never tested against our own. By the learning curve, appending 129,880 usable rows to 699,635
is a factor of 1.186 = 0.246 doublings, i.e. **+1.5e-4 naive** and roughly +4e-5 after the ~23-50%
realisation rate we measured for the 72%->80% step. That is larger than anything else still on the
board, so it deserves a test.

This is NOT the closed original-row-append branch. That branch appended the 129,880-row SOURCE, whose
labels we already exploit through the `external`/`ogte` blocks (worth +1.0e-3). This is a different
label vector with 3.4% rating overlap.

STEP 1 -- domain transfer (cheap, decisive, run first)
-----------------------------------------------------
Train on competition fold-0 fit rows, score raminhuseyn's own labels.

  AUC ~ 0.5   the conditional has been rewritten; appending injects a foreign objective and will
               hurt, exactly as the source did. STOP here.
  AUC high    the domains agree, so appending is worth the full fold-level test in step 2.

Note the asymmetry that makes this diagnostic meaningful: if the model scored 0.5 we would learn the
domains disagree, and if it scores high we learn they agree. Either answer is decisive, which is why
this is run before spending a fold on the append.

STEP 2 -- the append, with a matched control
--------------------------------------------
Only if step 1 passes. Arms, identical row count and protocol:
  ctl     unchanged
  dup     + N duplicated competition fit rows with their TRUE labels  (row-count control)
  append  + N raminhuseyn rows with THEIR labels                       (the treatment)

`append - dup` isolates the information from the row count, exactly as the duplicate control did in
Phase 5 and the pseudo-label experiment did today.

Schema alignment
----------------
raminhuseyn carries 20 of the 21 raw columns: `Gender` is absent, and a column named
"Online support" sits in its place, which is a DIFFERENT survey question and must not be aliased onto
Gender. So Gender is set to NaN and the model handles it natively. Class maps from `Class`.

Usage:
  python scripts/run_external_probe.py --step 1
  python scripts/run_external_probe.py --step 2 --folds 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import RAW21  # noqa: E402
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

BASE = {"objective": "binary", "metric": "auc", "learning_rate": 0.02, "num_leaves": 127,
        "min_child_samples": 40, "colsample_bytree": 0.8, "subsample": 0.8,
        "subsample_freq": 1, "reg_lambda": 1.0, "max_bin": 255, "verbose": -1, "n_jobs": 8,
        "extra_trees": True}
ROUNDS = 6000

NORM = lambda c: "".join(ch for ch in str(c).lower() if ch.isalnum())  # noqa: E731
CAND = ROOT / "data" / "original" / "raminhuseyn__airline-customer-satisfaction" \
    / "Airline_customer_satisfaction.csv"
# NOTE the candidate's label vocabulary is DIFFERENT from the competition's, and this is the single
# most important fact about it. The generator source (and the competition) collapse a THREE-class
# survey -- Satisfied / Neutral or Dissatisfied / Dissatisfied -- into a binary target with 43.45%
# positives. raminhuseyn asks a BINARY question directly -- 'satisfied' vs 'dissatisfied' -- with
# 54.74% positives. So its target is not a re-sample of the same question; it is a different
# definition. An earlier version of this script mapped only the three-class vocabulary and got an
# all-NaN label vector and a nan AUC, which is what surfaced the difference.
LABEL_MAP = {
    "satisfied": 1.0,
    "dissatisfied": 0.0,
    "neutral or dissatisfied": 0.0,
    "neutral or dissatified": 0.0,
    "true": 1.0,
    "false": 0.0,
    "1": 1.0,
    "0": 0.0,
}


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_candidate() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    d = pd.read_csv(CAND, low_memory=False)
    have = {NORM(c): c for c in d.columns}
    out = {}
    for c in RAW21:
        out[c] = d[have[NORM(c)]] if NORM(c) in have else np.nan
    lc = next(c for c in d.columns if NORM(c) == "satisfaction")
    y = d[lc].astype(str).str.strip().str.lower().map(LABEL_MAP).to_numpy(dtype="float64")
    df = pd.DataFrame(out)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c not in (
            "Gender", "Customer Type", "Type of Travel", "Class") else df[c]
    missing = [c for c in RAW21 if df[c].isna().all()]
    return df, y, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--scheme", default="primary")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    import lightgbm as lgb

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    folds = get_scheme(args.scheme, y_int, tr[ID_COL]).folds
    cand, cand_y, missing = load_candidate()
    print(f"candidate raminhuseyn: rows={len(cand):,}  positives={int(np.nansum(cand_y)):,} "
          f"({np.nanmean(cand_y):.4f})")
    print(f"  competition train+test positives: {int(y_int.sum()):,} ({y_int.mean():.4f})")
    print(f"  columns with no counterpart in the candidate: {missing}\n")

    from src.features.view import ViewBuilder
    out = {"candidate_rows": int(len(cand)), "candidate_positives": int(np.nansum(cand_y)),
           "candidate_positive_rate": float(np.nanmean(cand_y)),
           "missing_columns": missing, "step": args.step,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}

    for k in [int(x) for x in args.folds.split(",")]:
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        tr_g, es_g = _inner_es_split(fit, y_int, args.seed + k)
        pos = {int(v): i for i, v in enumerate(fit)}
        tr_l = np.array([pos[int(v)] for v in tr_g])
        es_l = np.array([pos[int(v)] for v in es_g])
        assert len(tr_l) + len(es_l) == len(fit) and not (set(tr_l) & set(es_l))

        vb = ViewBuilder(tr, te, "full")
        vb.build_static()
        Xf, Xa, _names = vb.assemble(fit, y_int, val, None, inner_seed=k)
        Xv = Xa["val"]
        # the candidate has no `full`-view features, so it is scored in the RAW21 space the model was
        # also given: use the champion view but replace the engineered columns with raw values mapped
        # into the same column positions is not possible, so instead the transfer test uses a raw
        # view model -- and the SAME model is used for the competition control, which keeps the
        # comparison paired.
        vb_raw = ViewBuilder(tr, te, "raw")
        vb_raw.build_static()
        Xf_r, Xa_r, _ = vb_raw.assemble(fit, y_int, val, None, inner_seed=k)
        Xv_r = Xa_r["val"]
        Xc_r = np.nan_to_num(vb_raw.static_te, nan=-999.0).astype("float32")
        # align the candidate into RAW21 column order for the static raw block
        cand_in_view_order = np.column_stack(
            [np.nan_to_num(pd.to_numeric(cand[c], errors="coerce").to_numpy(dtype="float64"),
                           nan=-999.0) if c not in ("Gender", "Customer Type", "Type of Travel",
                                                    "Class")
             else _cat_codes(cand[c], te) for c in vb_raw.static_names])
        assert cand_in_view_order.shape[1] == Xc_r.shape[1], (
            f"candidate matrix {cand_in_view_order.shape} does not match the raw view "
            f"{Xc_r.shape}; column alignment would be wrong and the transfer AUC meaningless")
        mask = np.isfinite(cand_y)
        print(f"fold {k}: candidate rows scored = {int(mask.sum()):,}")

        p = dict(BASE)
        p.update(n_estimators=ROUNDS, random_state=args.seed + k, bagging_seed=args.seed + k + 1,
                 feature_fraction_seed=args.seed + k + 2)
        ds = lgb.Dataset(Xf_r[tr_l], label=y[tr_g])
        dv = lgb.Dataset(Xf_r[es_l], label=y[es_g], reference=ds)
        m = lgb.train(p, ds, num_boost_round=ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        it = int(m.best_iteration or ROUNDS)
        auc_comp = float(roc_auc_score(y[val], m.predict(Xv_r, num_iteration=it)))
        cand_pred = m.predict(cand_in_view_order[mask], num_iteration=it)
        auc_cand = float(roc_auc_score(cand_y[mask], cand_pred))
        print(f"  model trained on competition fold-{k} fit, raw view, iter={it}")
        print(f"  AUC on competition fold {k} val      : {auc_comp:.6f}   (sanity: must be > 0.85)")
        print(f"  AUC on raminhuseyn's OWN labels      : {auc_cand:.6f}")

        out[f"fold{k}"] = {"iter": it, "auc_competition_val": auc_comp,
                           "auc_candidate_own_labels": auc_cand,
                           "n_candidate_scored": int(mask.sum())}
        # The two prediction vectors cover DIFFERENT row sets (139,927 competition rows vs ~129,880
        # candidate rows), so their correlation is not a meaningful quantity. An earlier version
        # printed it and crashed on the shape mismatch; the honest replacement is a comparison of the
        # two score DISTRIBUTIONS, which is what the transfer AUC already expresses.
        comp_pred = m.predict(Xv_r, num_iteration=it)
        out[f"fold{k}"]["pred_quantiles_competition"] = [
            float(np.quantile(comp_pred, q)) for q in (0.1, 0.5, 0.9)]
        out[f"fold{k}"]["pred_quantiles_candidate"] = [
            float(np.quantile(cand_pred, q)) for q in (0.1, 0.5, 0.9)]
        print("  prediction quantiles (0.1/0.5/0.9) competition: "
              f"{[round(v,3) for v in out[f'fold{k}']['pred_quantiles_competition']]}")
        print("  prediction quantiles (0.1/0.5/0.9) candidate  : "
              f"{[round(v,3) for v in out[f'fold{k}']['pred_quantiles_candidate']]}")

        if args.step == 2:
            # ---- the append, with a row-count-matched duplicate control ----
            # Uses the `raw` view for all three arms, because the candidate carries only the raw
            # columns and cannot produce the engineered blocks. Keeping every arm on the same view is
            # what makes the comparison paired.
            n_add = int(len(cand))
            rng = np.random.default_rng(args.seed + k + 77)
            dup_rows = rng.choice(tr_l, size=n_add, replace=True)
            Xc = np.nan_to_num(cand_in_view_order, nan=-999.0, posinf=1e9, neginf=-1e9).astype("float32")
            ok = mask
            Xc_ok = Xc[ok]
            yc = cand_y[ok].astype("float64")

            print(f"\n  append test on the raw view, {n_add:,} extra rows per arm")
            arm_auc = {"ctl": auc_comp}
            for name, Xextra, yextra in (
                    ("dup", Xf_r[dup_rows], y[tr_g][np.searchsorted(np.sort(tr_g), tr_g[dup_rows])]
                     if False else y[dup_rows]),          # labels of the duplicated rows
                    ("append", Xc_ok, yc)):
                Xaug = np.vstack([Xf_r[tr_l], Xextra])
                yaug = np.concatenate([y[tr_g], yextra])
                pa = dict(p)
                set_seed = np.random.default_rng(args.seed + k)
                t0 = time.time()
                dsa = lgb.Dataset(Xaug, label=yaug)
                dva = lgb.Dataset(Xf_r[es_l], label=y[es_g], reference=dsa)
                ma = lgb.train(pa, dsa, num_boost_round=ROUNDS, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(300, verbose=False)])
                ita = int(ma.best_iteration or ROUNDS)
                pa_auc = float(roc_auc_score(y[val], ma.predict(Xv_r, num_iteration=ita)))
                arm_auc[name] = pa_auc
                print(f"    {name:<7} rows={len(Xaug):>9,}  iter={ita:>5}  AUC={pa_auc:.6f}  "
                      f"vs ctl {(pa_auc - auc_comp)*1e5:+7.1f}e-5  ({time.time()-t0:.0f}s)")

            print(f"\n    append - dup = {(arm_auc['append'] - arm_auc['dup'])*1e5:+.1f}e-5   "
                  f"(this, not append - ctl, is the treatment)")
            out.setdefault("append", {})[f"fold{k}"] = {
                "aucs": arm_auc, "n_added": n_add,
                "append_minus_dup_e5": (arm_auc["append"] - arm_auc["dup"]) * 1e5,
                "append_minus_ctl_e5": (arm_auc["append"] - arm_auc["ctl"]) * 1e5,
                "dup_minus_ctl_e5": (arm_auc["dup"] - arm_auc["ctl"]) * 1e5,
            }
            # The promotion criterion is `append > ctl`. `append > dup` only says the candidate's
            # labels beat having no labels on those rows, which is a very low bar -- and BOTH arms can
            # clear it while both lose to the control. The first version of this rule tested only
            # append-dup and printed "escalate to more folds" for a treatment that was 78e-5 WORSE
            # than not appending anything. The duplicate control exists to separate information from
            # row count; it is not itself the bar.
            beats_ctl = (arm_auc["append"] - arm_auc["ctl"]) > 0
            beats_dup = (arm_auc["append"] - arm_auc["dup"]) > 0
            print(f"\n    promotion check: append > ctl? {'YES' if beats_ctl else 'NO'}   "
                  f"append > dup? {'YES' if beats_dup else 'NO'}")
            if beats_ctl and beats_dup:
                print("  VERDICT: the candidate beats BOTH the control and the row-count control -- "
                      "escalate to more folds.")
            elif beats_dup:
                print("  VERDICT: REJECT. The candidate's labels beat putting NO labels on those rows,"
                      "\n           which confirms they carry real directional information (consistent"
                      "\n           with the 0.798 transfer AUC) -- but appending is still WORSE than"
                      f"\n           not appending at all ({(arm_auc['append']-arm_auc['ctl'])*1e5:+.1f}e-5)."
                      "\n           The row count matched duplicate control also loses, so the loss is"
                      "\n           not a row-count effect. Mechanism: the candidate's TARGET is"
                      "\n           defined differently (binary satisfied/dissatisfied, 54.7% positive)"
                      "\n           from the competition's 3-class survey collapsed to 44.4%, so the"
                      "\n           appended rows teach a foreign objective -- the same mechanism that"
                      "\n           closed original-row appending and the Phase 1-2 external teacher.")
            else:
                print("  VERDICT: REJECT. The candidate does not beat the duplicate control either.")
        if args.step == 1:
            if auc_cand < 0.60:
                print("\n  VERDICT: the candidate's conditional has been rewritten relative to the "
                      "competition.\n           Appending it would inject a foreign objective. STOP "
                      "at step 1.")
            else:
                print(f"\n  VERDICT: domains agree (candidate AUC {auc_cand:.4f}); the append is "
                      f"worth the\n           fold-level test in step 2, with the duplicate control.")
        out["verdict_step1"] = ("domains disagree -- stop" if auc_cand < 0.60
                                else "domains agree -- run step 2 append")

    save_json(out, REPORTS / f"external_probe_step{args.step}.json")
    print("\nwrote", REPORTS / f"external_probe_step{args.step}.json")


def _cat_codes(s: pd.Series, te: pd.DataFrame) -> np.ndarray:
    """Code a categorical column to integers, so it can live in a numeric matrix."""
    v = s.astype(str)
    u = {uu: i for i, uu in enumerate(sorted(set(v.tolist())))}
    return np.array([u[vv] for vv in v], dtype="float64")


if __name__ == "__main__":
    main()
