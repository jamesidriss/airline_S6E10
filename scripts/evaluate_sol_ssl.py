"""Frozen B0/B1/B2/B3 controls for the transductive masked representation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, REPORTS, arr_sha256, file_sha256, git_commit, load_cached_parquet, save_json
from src.features.view import ViewBuilder
from src.validation.folds import get_scheme
from src.validation.compare import logit
from scripts.run_views import _inner_es_split, _fit_xgb_es
from scripts.run_phase14 import XGB_ARMS, XGB_BASE, fit_xgb
from scripts.run_phase14c import counter_path
from scripts.audit_sol_state import reconstruct_v5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--representation", default="sol_ssl12")
    ap.add_argument("--tag", default="sol_b")
    ap.add_argument("--variant", default="B0,B1,B2,B3")
    args = ap.parse_args()
    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = np.r_[tr["id"].to_numpy(), te["id"].to_numpy()]
    folds = get_scheme("primary", y, tr["id"]).folds
    rep = ARTIFACTS / args.representation
    manifest = json.loads((rep / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert np.array_equal(np.load(rep / "ids.npy"), ids), "SSL row alignment failure"
    embedding, summary = np.load(rep / "embedding.npy"), np.load(rep / "summary.npy")
    assert arr_sha256(embedding) == manifest["embedding_sha256"]
    assert arr_sha256(summary) == manifest["summary_sha256"]
    vb = ViewBuilder(tr, te, "full")
    vb.build_static()
    _, champ, _ = reconstruct_v5(y, folds)
    member, seed, params = XGB_ARMS["X2"]
    records = REPORTS / args.tag
    preds = ARTIFACTS / args.tag
    records.mkdir(exist_ok=True)
    preds.mkdir(exist_ok=True)
    for k in map(int, args.folds.split(",")):
        fi, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        Xf, Xa, names = vb.assemble(fi, y, va, None, inner_seed=k)
        itr, es = _inner_es_split(fi, y, 1)
        tl, el = np.searchsorted(fi, itr), np.searchsorted(fi, es)
        Xv = Xa["val"]
        # Independent canonical runner control, on the corrected TE implementation.
        control, control_it = _fit_xgb_es(Xf[tl].astype(float), y[itr], Xv.astype(float),
                     params, seed, Xf[el].astype(float), y[es])
        b0 = None
        old_slot = np.load(counter_path(member, k))
        for variant in args.variant.split(","):
            extra = {"B0": [], "B1": [embedding], "B2": [summary], "B3": [embedding, summary]}[variant]
            F = np.column_stack([Xf] + [m[fi] for m in extra]).astype(float)
            V = np.column_stack([Xv] + [m[va] for m in extra]).astype(float)
            pth = preds / f"{variant}_f{k}.npy"
            rpth = records / f"{variant}_f{k}.json"
            if rpth.exists():
                raise ValueError(f"result exists; choose another tag or omit completed folds: {rpth}")
            t0 = time.monotonic()
            pred, it = fit_xgb(F[tl], y[itr], V, seed, params, F[el], y[es])
            pred = pred.astype("float32")
            if variant == "B0":
                assert np.array_equal(pred, control.astype("float32")), "STOP: canonical B0 control fails"
                assert it == control_it
                b0 = pred
            if b0 is None:
                b0 = np.load(preds / f"B0_f{k}.npy")
            auc = float(roc_auc_score(y[va], pred))
            delta = auc - float(roc_auc_score(y[va], b0))
            slot_champ = champ[va] + (logit(pred) - logit(old_slot)) / 59
            marginal = float(roc_auc_score(y[va], slot_champ) - roc_auc_score(y[va], champ[va]))
            np.save(pth, pred)
            full_params = dict(XGB_BASE, **params, random_state=seed)
            rec = {"git": git_commit(), "variant": variant, "fold": k, "model_family": "xgb",
                   "member": member, "params": full_params, "seed": seed, "train_rows": len(itr),
                   "es_rows": len(es), "validation_rows": len(va), "n_features": F.shape[1],
                   "fit_ids_sha256": arr_sha256(tr["id"].to_numpy()[itr]),
                   "validation_ids_sha256": arr_sha256(tr["id"].to_numpy()[va]),
                   "fold_sha256": arr_sha256(folds), "feature_fit_sha256": arr_sha256(F),
                   "feature_val_sha256": arr_sha256(V), "representation_fingerprint": manifest["fingerprint"],
                   "auc": auc, "delta_vs_B0": delta, "v5_slot_delta": marginal, "n_trees": it + 1,
                   "canonical_control_bit_identical": True, "prediction_sha256": arr_sha256(pred),
                   "seconds": time.monotonic() - t0, "test_policy": "same fixed transductive encoder; full-label tree refit at primary median tree count",
                   "logit_corr_vs_B0": float(np.corrcoef(logit(pred), logit(b0))[0, 1]),
                   "logit_corr_vs_v5": float(np.corrcoef(logit(pred), champ[va])[0, 1]),
                   "status": "BASELINE" if variant == "B0" else "POSITIVE_UNCONFIRMED" if delta > 0 else "NEGATIVE",
                   "source_sha256": {p: file_sha256(p) for p in ("scripts/evaluate_sol_ssl.py", "scripts/run_phase14.py", "scripts/run_views.py")}}
            save_json(rec, rpth)
            print(f"{variant} fold{k}: {auc:.9f}; vs B0 {delta:+.9f}; v5 slot {marginal:+.9f}; corr {rec['logit_corr_vs_v5']:.6f}; {rec['seconds']:.1f}s", flush=True)
            del F, V
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
