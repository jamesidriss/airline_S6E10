"""Phase 20 -- FLM9 Xiaomi-TabLDM fold-0 screen (F2 discovery).

Screens the Apache-2.0 Xiaomi-TabLDM classifier (occams/Xiaomi-TabLDM,
revision-pinned clf_default.ckpt, weights sha256-asserted) as a candidate 11th
ensemble member under the phase-19/20 fold-0 protocol: the champion's
raw21+coarse10 route representation (frames()), context drawn from FIT ONLY
(stratified 20k or 50k), predict ALL fold-0 rows, score against the
hash-asserted champion portfolio/route and the frozen-rule route_ref.

Xiaomi-TabLDM is in-context learning: fit() updates NO weights, it only
preprocesses the labeled context (fitted on FIT rows only -> fold-safe).

Library workarounds (documented in prereg):
  - kv_cache=False (conflicts with enhance_candidates=True default)
  - enhance_candidates=False (gaussian-rank perturbation broadcast bug)
  - device='cuda:0' (torch.cuda.set_device rejects indexless 'cuda')

Frozen rule: PROMISING iff port_delta >= +5e-5 AND
Spearman(candidate, route_ref) < 0.995, where
port_delta = AUC(equal-logit blend(candidate, route_ref)) - AUC(route_ref).

Run:
  .\\.venv\\Scripts\\python.exe scripts/phase20/flm9_tabldm_screen.py --arm A_ctx20k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import CONTEXT_SEED, screen_metrics, stratified_context  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402

OUT = REPORTS / "phase20" / "flm9_tabldm"
ART = ARTIFACTS / "phase20" / "flm9_tabldm"
PREREG = "configs/phase20/flm9_tabldm_prereg.json"
PREREG_SHA = "4118475a6a894e32a1ffe76ae9b19c3b9ebb5330c835ed6054fd58091d00fae3"  # pinned before the first run (preregister-before-result)

HF_REPO = "occams/Xiaomi-TabLDM"
HF_REVISION = "712e5635be9fd8728084cd5ae105ddecb31dc03c"
CKPT_REL = "checkpoints/clf_default.ckpt"
CKPT_SHA256 = "7f8ec8b949c13e7187c0f11b189e03560eb28e02d286aa2cdad2d2a56b62bc9e"
ROUTE_REF = "sol_route_aux10_primary_final"

ARMS = {"A_ctx20k": 20000, "B_ctx50k": 50000}

ESTIMATOR = dict(
    n_estimators=8, batch_size=4, kv_cache=False, enhance_candidates=False,
    device="cuda:0", random_state=1201, norm_methods=None,
    feat_shuffle_method="latin", class_shuffle_method="shift",
)


def log(msg: str) -> None:
    print(f"[flm9] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = ap.parse_args()
    ctx_rows = ARMS[args.arm]

    sha = hashlib.sha256((ROOT / PREREG).read_bytes()).hexdigest()
    assert sha == PREREG_SHA, f"prereg sha mismatch {sha}"
    log(f"prereg sha256 {sha}")

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    # ---- checkpoint integrity (revision-pinned, weights sha-asserted) ----
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(HF_REPO, CKPT_REL, revision=HF_REVISION)
    got = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    assert got == CKPT_SHA256, f"tabldm checkpoint hash drift: {got}"
    log(f"checkpoint sha256 OK ({got[:16]}...) rev={HF_REVISION[:8]}")

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)

    from scripts.run_sol_tabpfn import frames

    Xall, _Xte, names, cats, maps = frames(tr, te, route=True)
    assert Xall.shape[0] == len(y), "feature/row misalignment"

    ctx = stratified_context(FIT, y, ctx_rows, seed=CONTEXT_SEED)
    assert np.isin(ctx, FIT).all() and not np.intersect1d(ctx, EVAL).size
    X_fit, y_fit = Xall[ctx], y[ctx]
    X_eval = Xall[EVAL]
    route0, port0 = route_all[EVAL], port_all[EVAL]
    y_eval = y[EVAL]

    # frozen-rule reference: champion-v7 proxy OOF (fold-0 slice)
    from src.submission.store import load_oof

    ref = load_oof(ROUTE_REF).astype(np.float64)
    assert np.isfinite(ref).all() and ref.min() >= 0.0 and ref.max() <= 1.0
    ref0 = ref[EVAL]

    log(f"arm={args.arm} ctx={len(ctx)} eval={len(EVAL)} features={Xall.shape[1]} "
        f"cats={[int(c) for c in cats]}")

    from tabldm import TabLDMClassifier

    clf = TabLDMClassifier(categorical_indices=[int(c) for c in cats], **ESTIMATOR)
    t1 = time.time()
    clf.fit(X_fit, y_fit)  # ICL: no weight updates; preprocessing fitted on FIT context only
    fit_s = time.time() - t1
    log(f"fit done in {fit_s:.1f}s")

    t2 = time.time()
    pr = np.asarray(clf.predict_proba(X_eval))
    predict_s = time.time() - t2
    assert pr.shape == (len(EVAL), 2), f"bad proba shape {pr.shape}"
    assert np.isfinite(pr).all() and (pr >= 0).all() and (pr <= 1).all()
    assert np.allclose(pr.sum(axis=1), 1.0, atol=1e-4), "rows must sum to 1"
    pred = pr[:, 1].astype(np.float64)
    log(f"predict done in {predict_s:.1f}s ({len(EVAL)/max(predict_s,1e-9):.1f} rows/s)")

    # standard screen table (vs champion portfolio) for cross-screen comparability
    m = screen_metrics(y_eval, pred, port0, route0)

    # frozen-rule quantities vs route_ref
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from src.phase19.foundation_screen import _cl, logit

    auc_ref = float(roc_auc_score(y_eval, ref0))
    blend = 1.0 / (1.0 + np.exp(-(logit(_cl(pred)) + logit(_cl(ref0))) / 2.0))
    auc_blend_ref = float(roc_auc_score(y_eval, blend))
    port_delta = auc_blend_ref - auc_ref
    spearman = float(spearmanr(pred, ref0).statistic)
    prom = bool(port_delta >= 5e-5 and spearman < 0.995)

    pred_path = ART / f"tabldm_{args.arm}_fold0.npy"
    np.save(pred_path, pred.astype("float32"))

    import torch

    meta = {
        "model": "xiaomi-tabldm", "arm": args.arm, "status": "DONE",
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "prereg": PREREG, "prereg_sha256": sha,
        "hf_repo": HF_REPO, "hf_revision": HF_REVISION,
        "checkpoint_sha256": got,
        "install": "pip install git+https://github.com/xiaomi-research/xiaomi-tabldm (0.1.0)",
        "license_note": "LICENSE file = Apache-2.0 (Xiaomi 2026); README News bullet "
                        "says 'Non-Commercial License' but links to the same Apache "
                        "file; contradiction recorded, LICENSE treated as operative",
        "recipe": "ICL fit on stratified FIT context (seed %d); no weight updates; "
                  "n_estimators=8, batch_size=4, kv_cache=False, enhance_candidates=False, "
                  "device=cuda:0, random_state=1201" % CONTEXT_SEED,
        "context_rows": int(len(ctx)), "context_seed": CONTEXT_SEED,
        "context_sha256": arr_sha256(ctx.astype("int64")),
        "eval_fold": 0, "eval_rows": int(len(EVAL)), "fit_folds": [1, 2, 3, 4],
        "features": list(names), "categorical_indices": [int(c) for c in cats],
        "label_free_category_maps": maps,
        "route_ref": ROUTE_REF, "route_ref_auc_fold0": auc_ref,
        "prediction_sha256": arr_sha256(np.load(pred_path)),
        "metrics": m,
        "frozen_rule": {
            "port_delta": port_delta, "route_ref_auc": auc_ref,
            "blend_auc": auc_blend_ref, "spearman_vs_route_ref": spearman,
            "promising": prom,
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "fit_seconds": fit_s, "predict_seconds": predict_s,
        "total_seconds": time.time() - t0,
    }
    save_json(meta, OUT / f"tabldm_{args.arm}_fold0.json")

    fr = meta["frozen_rule"]
    log(f"standalone={m['standalone_auc']:.6f} (ref {auc_ref:.6f}) "
        f"blend={fr['blend_auc']:.6f} port_delta={port_delta:+.3e} "
        f"spearman={spearman:.4f} -> PROMISING={prom}")
    log(f"screen table: port_delta={m['portfolio_delta']:+.3e} "
        f"pearson_port={m['logit_pearson_vs_portfolio']:.4f}")
    log(f"peak vram {meta['peak_vram_gb']}GB; total {meta['total_seconds']:.1f}s")
    log(f"report -> {(OUT / f'tabldm_{args.arm}_fold0.json').relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
