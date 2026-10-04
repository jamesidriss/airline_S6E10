"""Phase 4: a genuinely orthogonal model family -- TabR (retrieval-based).

Why TabR
--------
Every model already in the pool is LightGBM / XGBoost / CatBoost / RealMLP / TabM, and their
logit-space correlations sit at 0.995-0.999. Adding more of them buys +1e-6..+5e-6 each. The
success criterion here is **marginal ensemble gain**, not standalone AUC.

TabR is architecturally unlike anything else in the pool: it retrieves the k nearest training rows
in a *learned* embedding space and classifies from a parameter-efficient ensemble of heads
conditioned on that retrieved context. There is no axis-aligned partition of the feature space
(GBDTs) and no single global parametric function (MLPs) -- the decision rule is explicitly local
and retrieval-based. That is the strongest available candidate for decorrelated errors.

Configuration
-------------
Defaults come from `pytabkit.models.sklearn.default_params.RealTABR_D_CLASS` (a TabZilla-derived
tuned preset). `num_embeddings` there is a *module spec dict*, not the list of per-head embedding
widths used by the original TabR paper code -- passing a list raises a bare ValueError deep inside
`tabr_lib.make_module`, so we use the library's own preset shape.

Fold safety
-----------
Early stopping / patience is driven by `X_val`. Passing the outer validation rows there would leak
evaluation labels into model selection, so we always carve the inner holdout out of the FIT rows
with the same `_inner_es_split` used by every other runner in this repository.

Usage:
  python scripts/run_tabr.py --view full --context 96 --epochs 30
  python scripts/run_tabr.py --view full --only-fold 0          # smoke test, one fold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ID_COL, REPORTS, TARGET, load_cached_parquet, save_json, set_seed  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.models.realmlp import _twin_frame  # noqa: E402
from src.models.tabr_retrieval import (find_torch_index, install_faiss_gpu_shim,  # noqa: E402
                                  verify_matches_reference)
from src.submission import store  # noqa: E402
from src.validation.compare import corr, spearman  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

# pytabkit's tuned TabR preset (TabZilla-derived). Keys and shapes taken verbatim from
# pytabkit.models.sklearn.default_params.RealTABR_D_CLASS.
TABR_PRESET = dict(
    d_main=265,
    d_multiplier=2.0,
    encoder_n_blocks=0,
    predictor_n_blocks=1,
    mixer_normalization="auto",
    context_dropout=0.38920071545944357,
    dropout0=0.38852797479169876,
    dropout1=0.0,
    normalization="LayerNorm",
    activation="ReLU",
    eval_batch_size=4096,
    patience=16,
    n_epochs=100_000,
    context_size=96,
    freeze_contexts_after_n_epochs=None,
    num_embeddings={"type": "PBLDEmbeddings", "n_frequencies": 8, "d_embedding": 4,
                    "frequency_scale": 0.1},
    tfms=["median_center", "robust_scale", "smooth_clip"],
    optimizer={"type": "AdamW", "lr": 0.0003121273641315169, "weight_decay": 1.2260352006404615e-06,
               "betas": (0.9, 0.95)},
    add_scaling_layer=True,
    scale_lr_factor=96,
    ls_eps=0.1,
)


def build_params(args):
    p = dict(TABR_PRESET)
    p.update(batch_size=args.batch, n_epochs=args.epochs, context_size=args.context,
             patience=args.patience, random_state=args.seed, n_cv=1, n_refit=0, verbosity=0)
    if args.memory_efficient:
        p["memory_efficient"] = True
        p["candidate_encoding_batch_size"] = args.cand_bs
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="full")
    ap.add_argument("--folds", default="primary")
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--cand-bs", type=int, default=128)
    ap.add_argument("--query-chunk", type=int, default=256)
    ap.add_argument("--subsample", type=int, default=0,
                    help="subsample the FIT rows to this many (smoke tests only); 0 = use all")
    ap.add_argument("--memory-efficient", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only-fold", type=int, default=-1, help="smoke test a single fold")
    ap.add_argument("--tag", default="tabr")
    ap.add_argument("--save-test", action="store_true")
    args = ap.parse_args()

    tr, te = load_cached_parquet()
    y_int = tr[TARGET].values.astype("int8")
    y = y_int.astype("float64")
    ntr, nte = len(tr), len(te)
    folds = get_scheme(args.folds, y_int, tr[ID_COL]).folds

    vb = ViewBuilder(tr, te, args.view)
    vb.build_static()
    print(f"[cache] view={args.view}: {len(vb.static_names)} features", flush=True)

    oof = np.zeros(ntr)
    testp = np.zeros(nte)
    ks = sorted(set(folds.tolist()))
    smoke = args.only_fold >= 0
    if smoke:
        ks = [args.only_fold]

    params = build_params(args)
    from pytabkit import RealTabR_D_Classifier

    # hard gate: never train on a retrieval implementation we have not verified
    rc = verify_matches_reference(device="cuda")
    if not rc["passes"]:
        raise SystemExit("retrieval verification FAILED -- refusing to train TabR")
    faiss_mode = install_faiss_gpu_shim(query_chunk=args.query_chunk)
    print(f"  faiss GPU index: {faiss_mode}", flush=True)

    diag = []
    for k in ks:
        t0 = time.time()
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        set_seed(args.seed + k)

        # inner early-stopping holdout carved from the FIT rows only -- never the eval fold
        inner_tr, inner_es = _inner_es_split(fit, y_int, args.seed + k)

        Xa, Xout, names = vb.assemble(inner_tr, y_int, inner_es, None, inner_seed=k)
        tr_d = _twin_frame(Xa, names)
        es_d = _twin_frame(Xout["val"], names)

        y_sub = y_int[inner_tr]
        if args.subsample and len(inner_tr) > args.subsample:
            sel = np.random.default_rng(args.seed).choice(len(inner_tr), args.subsample,
                                                          replace=False)
            tr_d = tr_d.iloc[sel].reset_index(drop=True)
            y_sub = y_sub[sel]

        # score the outer validation block with the same fitted model
        # Exact k-NN retrieval on the GPU. faiss-cpu (the only wheel for CPython 3.11 on Windows) has no
        # GPU index symbols, and a CPU IndexFlatL2 over 560k x 265 per batch step is not viable, so
        # we make pytabkit's own faiss-GPU branch resolve to a torch exact-L2 index. Verified
        # against a float64 brute-force reference (see src/models/tabr_retrieval.py).
        m = RealTabR_D_Classifier(device="cuda", **params)
        m.fit(tr_d, y_sub, X_val=es_d, y_val=y_int[inner_es])
        sidx = find_torch_index(m)
        print(f"          retrieval index: {sidx.stats() if sidx else 'NOT FOUND'}",
              flush=True)

        p = m.predict_proba(_twin_frame(vb.static_tr[val], names))[:, 1]
        if not smoke:
            oof[val] = p
            if args.save_test:
                testp += m.predict_proba(_twin_frame(vb.static_te, names))[:, 1] / len(ks)
        a = float(roc_auc_score(y[val], p))
        diag.append({"fold": int(k), "auc": a, "seconds": round(time.time() - t0, 1)})
        print(f"  fold{k}: AUC={a:.6f}  ({time.time()-t0:.0f}s)", flush=True)

    if smoke:
        print("\nsmoke test complete (single fold, subsampled train).")
        return

    auc = float(roc_auc_score(y, oof))
    fa = [float(roc_auc_score(y[folds == k], oof[folds == k])) for k in ks]
    v3 = store.load_oof("blend_v3_final").astype("float64")
    print("\n" + "=" * 92)
    print(f"TabR  view={args.view}  context={args.context}  epochs={args.epochs}  "
          f"OOF AUC = {auc:.6f}")
    print(f"  folds = {[round(x, 6) for x in fa]}")
    print(f"  vs finalist (OOF {float(roc_auc_score(y, v3)):.6f}): "
          f"logit corr {corr(oof, v3):.5f}  spearman {spearman(oof, v3):.5f}")
    print("=" * 92)

    eid = f"{args.tag}_{args.view}_c{args.context}_e{args.epochs}"
    store.save(eid, oof, testp if args.save_test else None, fold_scheme=args.folds,
               meta={"family": "tabr", "featureset": args.view, "auc": round(auc, 6),
                     "context": args.context, "epochs": args.epochs, "params": params})
    save_json({"exp_id": eid, "auc": auc, "fold_aucs": fa, "per_fold": diag,
               "corr_with_finalist": corr(oof, v3), "spearman_with_finalist": spearman(oof, v3),
               "params": params, "view": args.view, "scheme": args.folds},
              REPORTS / f"{eid}.json")
    print("\nwrote", REPORTS / f"{eid}.json")


if __name__ == "__main__":
    main()
