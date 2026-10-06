"""Phase 11 -- generic native-categorical counterpart runner, driven by the slot inventory.

One code path for all seven slots
---------------------------------
Seven hand-written configs would be seven chances to drift from the inventory. This runner reads
reports/native_cat_slot_inventory.json and for each slot builds the counterpart by changing ONLY the
representation mechanism:

  KEEP   the slot's numeric engineered view, fold scheme, seed, learning_rate, depth, l2_leaf_reg,
         boosting_type="Plain"
  ADD    the 17 native string categorical twins, declared to CatBoost via real `cat_features`

Not changed: depth, seed, view, fold scheme, CTR complexity, one-hot threshold, iteration policy.
Ordered boosting is banned outright (Phase 10: -71.7e-5). Age and Flight Distance twins are NOT
added here -- those are separate arms (C4/C5) and adding them now would confound the block test.

Round selection: the Phase 10 fixed-round protocol
--------------------------------------------------
    outer fold k
      -> inner train / inner ES on a 10% carve OF OUTER-FIT
      -> iteration count chosen from inner ES only
      -> refit from scratch on 100% of outer-fit at that fixed count
      -> score outer validation ONCE

Outer-validation labels choose nothing. Every inner curve, selected iteration, row count, schema
hash, cat-feature hash and prediction hash is persisted.

The confound, stated not hidden
-------------------------------
The originals were trained under the OLD protocol, which discarded the 10% carve permanently. So
`native - original` mixes the representation change with a 10% larger training fraction. This runner
therefore ALSO trains a NUMERIC counterpart per slot under the identical new protocol, which costs
little relative to the information it buys: it separates the protocol effect from the mechanism and
turns the operational swap delta into a decomposable quantity.

    native_slot - numeric_counterpart = the mechanism, protocol-matched (this is the causal number)
    numeric_counterpart - original    = the protocol effect alone
    native_slot - original            = the operational replacement delta (the two added)

Usage:
  python scripts/run_native_counterparts.py --slots 0-6 --scheme primary --tag p11_f0
  python scripts/run_native_counterparts.py --slots 0 --scheme block10 --tag p11_blk10
  python scripts/run_native_counterparts.py --slots 3 --timing-probe 40 --tag p11_timing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.native_cat import (attach, cat_frame, cat_indices, default_cat_cols,  # noqa: E402
                                verify_no_target)
from scripts.run_views import _inner_es_split  # noqa: E402

ES_ROUNDS = 6000
ES_PATIENCE = 300
INV_PATH = REPORTS / "native_cat_slot_inventory.json"


def sha(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def parse_slots(spec: str, n: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [s for s in out if s < 0 or s >= n]
    if bad:
        raise SystemExit(f"slot indices out of range 0..{n-1}: {bad}")
    return sorted(set(out))


def build_frames(vb, tr, y_int, fit, val, cats_on):
    """Numeric frame from the slot's view, optionally with the 17 native twins appended."""
    Xf, Xa, names = vb.assemble(fit, y_int, val, None, inner_seed=0)
    num = pd.DataFrame(np.asarray(Xf, dtype=np.float32), columns=list(names))
    numv = pd.DataFrame(np.asarray(Xa["val"], dtype=np.float32), columns=list(names))
    if not cats_on:
        return num, numv, []
    src = default_cat_cols(True)
    verify_no_target([f"ncat__{c}" for c in src])
    missing = [c for c in src if c not in tr.columns]
    if missing:
        raise KeyError(f"categorical source columns absent: {missing}")
    f, cn = attach(Xf, names, cat_frame(tr, src, fit))
    v, _ = attach(Xa["val"], names, cat_frame(tr, src, val))
    return f, v, cn


def fit_catboost(frame, y, params, cat_names, seed, n_rounds, es=None):
    from catboost import CatBoostClassifier
    p = dict(params)
    p.pop("iterations", None)
    p.pop("cat_features", None)
    p["random_seed"] = int(seed)
    p["thread_count"] = 8
    p["verbose"] = 0
    p["allow_writing_files"] = False
    p["boosting_type"] = "Plain"          # Ordered is banned in Phase 11
    kw = {"cat_features": cat_indices(frame, cat_names)} if cat_names else {}
    if es is None:
        p["iterations"] = int(n_rounds)
        p.pop("eval_metric", None)
        m = CatBoostClassifier(**p)
        m.fit(frame, y, **kw)
        return m, int(n_rounds)
    p["iterations"] = int(ES_ROUNDS)
    p["eval_metric"] = "AUC"
    m = CatBoostClassifier(**p)
    m.fit(frame, y, eval_set=(es[0], es[1]), early_stopping_rounds=ES_PATIENCE, verbose=0, **kw)
    return m, int(m.get_best_iteration() or ES_ROUNDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="0-6")
    ap.add_argument("--scheme", default="",
                    help="override the fold scheme; by default each slot uses its OWN scheme from "
                         "the inventory, which is required because slot 0 is a block10 member")
    ap.add_argument("--tag", default="p11")
    ap.add_argument("--variants", default="native,numeric",
                    help="native = with the 17 twins; numeric = same protocol without them. Both "
                         "are needed to decompose the operational delta.")
    ap.add_argument("--folds", default="",
                    help="comma list or a-b range of fold indices to train; default = all folds of "
                         "the slot's scheme. Stage 1 screens fold 0 only.")
    ap.add_argument("--timing-probe", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="retrain even if a prediction file exists")
    args = ap.parse_args()

    inv = json.loads(INV_PATH.read_text(encoding="utf-8"))
    slots = {s["slot"]: s for s in inv["slots"]}
    want = parse_slots(args.slots, len(slots))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in ("native", "numeric")]
    if bad:
        raise SystemExit(f"unknown variants {bad}")

    tr, te = load_cached_parquet()
    y = tr[TARGET].values.astype("float64")
    y_int = tr[TARGET].values.astype("int8")

    print("PHASE 11 -- NATIVE CATEGORICAL COUNTERPARTS")
    print(f"  inventory_hash = {inv['inventory_hash'][:32]}")
    print(f"  slots {want}   variants {variants}")
    print(f"  protocol: inner ES on a 10% carve of outer-fit -> fixed count -> refit on 100% of "
          f"outer-fit -> score outer fold once")
    print(f"  boosting_type is forced to Plain (Ordered banned)\n")

    out = {"tag": args.tag, "inventory_hash": inv["inventory_hash"],
           "protocol": "phase10 fixed-round: inner ES on a 10% outer-fit carve selects the "
                       "iteration count; refit on 100% of outer-fit; outer fold scored once",
           "variants": variants, "runs": {},
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip()[:12]}

    by_scheme: dict[str, list[int]] = {}
    for i in want:
        sc = args.scheme or slots[i]["fold_scheme"]
        by_scheme.setdefault(sc, []).append(i)

    for sc, idxs in by_scheme.items():
        folds = get_scheme(sc, y_int, tr[ID_COL]).folds
        nfold = len(set(folds.tolist()))
        klist = (parse_folds(args.folds, nfold) if args.folds
                 else list(range(nfold)))
        print(f"{'='*104}\nfold scheme {sc} ({nfold} folds)  slots {idxs}  folds {klist}\n"
              f"{'='*104}")
        for i in idxs:
            s = slots[i]
            vb = ViewBuilder(tr, te, s["view"])
            vb.build_static()
            params = {k: v for k, v in s["params"].items()
                      if k in ("learning_rate", "depth", "l2_leaf_reg", "random_strength",
                               "rsm", "border_count")}
            for k in klist:
                fit = np.where(folds != k)[0]
                val = np.where(folds == k)[0]
                assert not (set(fit.tolist()) & set(val.tolist()))
                for var in variants:
                    cats_on = (var == "native")
                    tagname = f"{args.tag}_s{i}_{var}"
                    pred_path = REPORTS / f"{tagname}_{sc}_f{k}.npy"
                    if pred_path.exists() and not args.force:
                        print(f"  slot {i} {var:<7} fold {k}: cached")
                        continue
                    f, v, cn = build_frames(vb, tr, y_int, fit, val, cats_on)
                    if args.timing_probe:
                        t0 = time.time()
                        fit_catboost(f, y[fit], params, cn, s["seed"], args.timing_probe)
                        dt = time.time() - t0
                        print(f"  slot {i} {var:<7} {f.shape[1]:>4} feat ({len(cn)} cat)  "
                              f"{args.timing_probe} rounds in {dt:.1f}s = "
                              f"{dt/args.timing_probe:.4f}s/round -> 1200 rounds ~ "
                              f"{dt/args.timing_probe*1200/3600:.2f} h", flush=True)
                        continue
                    # inner ES on a 10% carve of outer-fit
                    itr_g, es_g = _inner_es_split(fit, y_int, int(s["seed"]) + k)
                    pos = {int(vv): j for j, vv in enumerate(fit)}
                    itr_l = np.array([pos[int(vv)] for vv in itr_g])
                    es_l = np.array([pos[int(vv)] for vv in es_g])
                    t0 = time.time()
                    _, n_iter = fit_catboost(f, y[fit], params, cn, int(s["seed"]) + k, 0,
                                             es=(f.iloc[es_l], y[es_g]))
                    t_es = time.time() - t0
                    # refit from scratch on 100% of outer-fit at that fixed count
                    t0 = time.time()
                    m, used = fit_catboost(f, y[fit], params, cn, int(s["seed"]) + k, n_iter)
                    pred = m.predict_proba(v)[:, 1]
                    t_refit = time.time() - t0
                    np.save(pred_path, pred.astype("float32"))
                    auc = float(roc_auc_score(y[val], pred))
                    rec = {
                        "slot": i, "exp_id": s["exp_id"], "variant": var, "fold_scheme": sc,
                        "fold": k, "n_folds": nfold, "view": s["view"], "seed": s["seed"],
                        "params": params, "n_features": int(f.shape[1]), "n_cat": len(cn),
                        "cat_cols": cn, "cat_hash": sha(cn),
                        "schema_hash": sha({"cols": list(f.columns)}),
                        "inner_best_iter": int(n_iter), "refit_iter": int(used),
                        "n_rows_es_fit": int(len(itr_l)), "n_rows_es": int(len(es_g)),
                        "n_rows_refit": int(len(fit)), "n_rows_eval": int(len(val)),
                        "auc": auc, "seconds_es": round(t_es, 1),
                        "seconds_refit": round(t_refit, 1),
                        "pred_sha": hashlib.sha256(pred.astype("float32").tobytes()).hexdigest()[:16],
                        "config_hash": s["config_hash"],
                    }
                    out["runs"].setdefault(f"{sc}|{i}|{var}", {})[str(k)] = rec
                    print(f"  slot {i} {var:<7} fold {k}  iters={used:<5} "
                          f"es_rows={len(itr_l):,} refit_rows={len(fit):,}  AUC={auc:.6f}  "
                          f"(ES {t_es:.0f}s + refit {t_refit:.0f}s)", flush=True)

    save_json(out, REPORTS / f"{args.tag}_runs.json")
    print("\nwrote", REPORTS / f"{args.tag}_runs.json")


def parse_folds(spec: str, nfold: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [f for f in out if f < 0 or f >= nfold]
    if bad:
        raise SystemExit(f"fold indices out of range 0..{nfold-1}: {bad}")
    return sorted(set(out))


if __name__ == "__main__":
    main()
