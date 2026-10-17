"""Phase 19 -- foundation-model screen (fold-0 discovery).

Screens installable tabular foundation models (TabICL-v2, TabDPT, Causilo,
TabFM, xRFM) as candidate 11th ensemble members. Every candidate sees the SAME
raw-21+coarse10 representation the champion TabPFN route arm uses (built by the
champion's own `frames()` builder) and a fold-0-safe stratified context drawn
only from FIT (folds 1-4). Evaluation is fold 0 only.

This is a SCREEN, not a promotion: it cannot pass the OOF admission gate. It
decides whether a model merits a full 5-fold OOF + admission-gate run. Frozen
thresholds live in configs/phase19/foundation_screen_prereg.json.

Anti-leakage: no target encoding (raw view); context is a subset of FIT; the
champion reference is hash-asserted from phase17_B1 artifacts.

Run:  .\\.venv\\Scripts\\python.exe scripts/phase19/foundation_screen_fold0.py --models tabicl,tabdpt,causilo,xrfm
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import (  # noqa: E402
    CONTEXT_SEED, build_model, screen_metrics, stratified_context,
)

# Champion fold-safe hashes (over FULL OOF arrays) -- reused from H19-A.
CHAMPION_PORTFOLIO_SHA256 = "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552"
CHAMPION_ROUTE_SHA256 = "bcb2811329d3686882450e93435647f8f7ab65abfc48c1a1b51db1e9d1ef95e0"

OUT = REPORTS / "phase19" / "foundation_screen"
ART = ARTIFACTS / "phase19" / "foundation_screen"


def log(msg: str) -> None:
    print(f"[fmscreen] {msg}", flush=True)


def load_champion_fold0(folds: np.ndarray):
    """Assemble hash-asserted champion route/portfolio; return fold-0 slices."""
    n = folds.shape[0]
    route = np.full(n, np.nan, dtype=np.float64)
    port = np.full(n, np.nan, dtype=np.float64)
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        route[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "prediction.npy")
        port[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "portfolio.npy")
    assert np.isfinite(route).all() and np.isfinite(port).all()
    assert arr_sha256(port.astype("float32")) == CHAMPION_PORTFOLIO_SHA256, "portfolio hash drift"
    assert arr_sha256(route.astype("float32")) == CHAMPION_ROUTE_SHA256, "route hash drift"
    return route, port


def run_model(name: str, args, Xtr: np.ndarray, y: np.ndarray, ctx: np.ndarray,
              eval_idx: np.ndarray, pred_path: Path) -> np.ndarray:
    """Fit a model on context rows, predict eval rows in batches, save + return."""
    model = build_model(name, args.device, seed=args.seed,
                        disk_offload_dir=getattr(args, "tabicl_offload_dir", None))
    model.fit(Xtr[ctx], y[ctx])
    gc.collect()
    chunks = []
    bs = 8192
    for b in range(0, len(eval_idx), bs):
        sl = eval_idx[b:b + bs]
        pr = model.predict_proba(Xtr[sl])
        if getattr(pr, "ndim", 0) == 2:
            pr = pr[:, 1]
        chunks.append(np.asarray(pr, dtype=np.float64))
    pred = np.concatenate(chunks)
    assert pred.shape == (len(eval_idx),) and np.isfinite(pred).all(), f"{name}: bad prediction"
    np.save(pred_path, pred.astype("float32"))
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tabicl,tabdpt,causilo,xrfm")
    ap.add_argument("--context", type=int, default=20000)
    ap.add_argument("--eval-rows", type=int, default=0, help="0 = full fold-0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="fmscreen_fold0_ctx20k")
    ap.add_argument("--seed", type=int, default=1201)
    ap.add_argument("--view", default="route", choices=["route", "full"],
                    help="route = phase-19 raw22 view (frames); full = the certified 285-feature "
                         "rich view (ViewBuilder.assemble, fold-safe TE) that lifted neural "
                         "standalone AUC by ~+2.3e-3 in phase 20")
    ap.add_argument("--tabicl-offload-dir", default="",
                    help="H20-FM-RICH fallback: enable TabICL disk offload (CUDA OOM at 285 feat)")
    args = ap.parse_args()
    if args.tabicl_offload_dir:
        Path(args.tabicl_offload_dir).mkdir(parents=True, exist_ok=True)
        args.tabicl_offload_dir = str(Path(args.tabicl_offload_dir).resolve())

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    from scripts.run_sol_tabpfn import frames  # champion feature builder (bit-exact)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    # champion raw representation: raw21 + Flight Distance coarse10, meta as codes
    if args.view == "route":
        Xtr, _Xte, names, cats, maps = frames(tr, te, route=True)
        assert Xtr.shape[0] == len(y), "feature/row misalignment"
    else:
        # certified 285-feature rich view. Fold-safe: TE for FIT rows is an inner
        # cross-fit excluding each row's own label; EVAL rows are encoded by a table
        # fitted on FIT only; EVAL labels are never read.
        from src.features.view import ViewBuilder
        vb = ViewBuilder(tr, te, "full")
        vb.build_static()
        X_fit, Xa, names = vb.assemble(FIT, y, EVAL, None, inner_seed=args.seed)
        X_eval = Xa["val"]
        assert X_fit.shape[0] == len(FIT) and X_eval.shape[0] == len(EVAL)
        Xtr = np.empty((len(y), X_fit.shape[1]), dtype="float32")
        Xtr[FIT] = X_fit
        Xtr[EVAL] = X_eval
        cats, maps = [], {}

    route_all, port_all = load_champion_fold0(folds)
    eval_idx = EVAL if (not args.eval_rows or args.eval_rows >= len(EVAL)) else EVAL[:args.eval_rows]
    route0, port0 = route_all[eval_idx], port_all[eval_idx]
    y_eval = y[eval_idx]

    log(f"rows={len(y)} view={args.view} FIT={len(FIT)} EVAL={len(eval_idx)} context={args.context} "
        f"features={Xtr.shape[1]} device={args.device}")

    ctx = stratified_context(FIT, y, args.context, seed=CONTEXT_SEED)
    log(f"context rows={len(ctx)} pos_rate={y[ctx].mean():.4f}")

    results = []
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        pred_path = ART / f"{name}_{args.tag}.npy"
        meta_path = OUT / f"{name}_{args.tag}.json"
        if pred_path.exists() and meta_path.exists():
            saved = np.load(pred_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") == "DONE" and arr_sha256(saved) == meta.get("prediction_sha256") \
                    and meta.get("eval_rows") == int(len(eval_idx)):
                log(f"  skip {name} (verified prediction exists)")
                results.append(meta)
                continue
        log(f"  {name}: fit context {len(ctx)} -> predict {len(eval_idx)}")
        t1 = time.time()
        try:
            pred = run_model(name, args, Xtr, y, ctx, eval_idx, pred_path)
            m = screen_metrics(y_eval, pred, port0, route0)
            status = "DONE"
            err = None
        except Exception as exc:  # resource/impl limits are recorded, not fatal
            pred = None
            m = None
            status = "FAILED"
            err = repr(exc)
        meta = {
            "model": name, "status": status, "error": err,
            "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
            "context_rows": int(len(ctx)), "context_seed": CONTEXT_SEED,
            "eval_rows": int(len(eval_idx)), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
            "view": args.view,
            "device": args.device, "features": list(names), "categorical_indices": list(cats),
            "label_free_category_maps": maps,
            "prediction_sha256": arr_sha256(np.load(pred_path)) if pred is not None else None,
            "metrics": m, "seed": args.seed, "tag": args.tag,
            "tabicl_offload_dir": args.tabicl_offload_dir or None,
            "prereg": "configs/phase19/foundation_screen_prereg.json",
            "fit_predict_seconds": time.time() - t1,
        }
        save_json(meta, meta_path)
        if m is not None:
            results.append(meta)
            log(f"    AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
                f"corr_port={m['logit_pearson_vs_portfolio']:.4f} "
                f"({meta['fit_predict_seconds']:.1f}s)")
        else:
            log(f"    FAILED: {err}")

    results.sort(key=lambda r: -r["metrics"]["portfolio_delta"])
    log("=== screen summary (fold 0) ===")
    for r in results:
        m = r["metrics"]
        log(f"  {r['model']:10s} AUC={m['standalone_auc']:.6f} "
            f"port_delta={m['portfolio_delta']:+.3e} corr={m['logit_pearson_vs_portfolio']:.4f}")

    summary = {
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "context": args.context, "context_seed": CONTEXT_SEED, "eval_fold": 0,
        "eval_rows": int(len(eval_idx)), "tag": args.tag, "device": args.device,
        "elapsed_seconds": time.time() - t0,
        "results": [{"model": r["model"], **r["metrics"]} for r in results],
    }
    save_json(summary, OUT / f"summary_{args.tag}.json")
    log(f"done in {time.time()-t0:.1f}s -> {OUT / f'summary_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
