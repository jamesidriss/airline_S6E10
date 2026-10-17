"""Phase 20 -- multi-task neural fold-0 screen (catalogue items 35+36, OBJ).

PLR trunk with a shared representation feeding (a) the satisfaction head and
(b) an ordinal CORAL auxiliary head over the 13 service ratings, trained
end-to-end on FIT with BERT-style rating INPUT MASKING (the aux head must infer
masked ratings from the other features -- the rating<->satisfaction coupling
hypothesis; without masking the aux task is copyable and shapes nothing).

Protocol matches the accepted neural full-view screen exactly: certified 285-
feature rich view (fold-safe TE), scaler fit on FIT-train only, inner 10% ES
split carved from FIT (judged on satisfaction AUC under inference conditions,
i.e. NO masking), predict full fold-0, score vs the hash-asserted champion.

Run (certified .venv, GPU):
  .\\.venv\\Scripts\\python.exe scripts/phase20/mtl_screen_fold0.py --p-mask 0.3 --lam 1.0
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

from src.common import ARTIFACTS, REPORTS, arr_sha256, git_commit, load_cached_parquet, save_json  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from src.phase19.foundation_screen import screen_metrics  # noqa: E402
from scripts.phase19.foundation_screen_fold0 import load_champion_fold0  # noqa: E402
from scripts.phase20.neural_screen_fold0 import inner_es_split, _fit_scaler  # noqa: E402

OUT = REPORTS / "phase20" / "mtl_screen"
ART = ARTIFACTS / "phase20" / "mtl_screen"
PREREG = ROOT / "configs" / "phase20" / "mtl_screen_prereg.json"

RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]


def log(msg: str) -> None:
    print(f"[mtl] {msg}", flush=True)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_and_predict(model, Xtr, ytr, Rtr, Xval, yval, Xeval, cfg, device, seed):
    """AdamW + warmup/cosine; loss = satisfaction BCE + lam * masked CORAL aux."""
    import torch
    import torch.nn as nn

    from src.phase20.mtl_arch import aux_loss

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.0))
    crit = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Rtr_t = torch.tensor(Rtr, dtype=torch.long, device=dev)
    Xval_t = torch.tensor(Xval, dtype=torch.float32, device=dev)
    yval_np = np.asarray(yval, dtype=np.float64)
    Xeval_t = torch.tensor(Xeval, dtype=torch.float32, device=dev)

    n = Xtr_t.shape[0]
    bs = int(cfg["batch_size"])
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["es_patience"])
    warmup = 5
    p_mask = float(cfg["p_mask"])
    lam = float(cfg["lam"])

    from sklearn.metrics import roc_auc_score

    best_val = -1.0
    best_state = None
    since_best = 0
    for epoch in range(max_epochs):
        if epoch < warmup:
            lr_scale = (epoch + 1) / warmup
        else:
            t = (epoch - warmup) / max(1, max_epochs - warmup)
            lr_scale = 0.5 * (1.0 + np.cos(np.pi * t))
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * lr_scale
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            mask = torch.rand(idx.shape[0], 13, device=dev) < p_mask
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                main_logits, aux_logits = model(Xtr_t[idx], mask_ratings=mask)
                loss = crit(main_logits, ytr_t[idx]) + lam * aux_loss(
                    aux_logits.float(), Rtr_t[idx], mask)
            loss.backward()
            opt.step()
        # inner-val satisfaction AUC under INFERENCE conditions (no masking)
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                vlog, _ = model(Xval_t)
                vlog = vlog.float().cpu().numpy()
        v_auc = roc_auc_score(yval_np, vlog)
        if v_auc > best_val + 1e-6:
            best_val = v_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = []
        for s in range(0, Xeval_t.shape[0], 65536):
            chunk = Xeval_t[s:s + 65536]
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                logit, _aux = model(chunk)
                out.append(logit.float().cpu().numpy())
    return np.concatenate(out), best_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-mask", type=float, default=0.3,
                    help="per-(sample,rating) input-mask probability during training")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight of the masked CORAL aux loss")
    ap.add_argument("--view", default="full", choices=["route", "full"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    # Frozen training config, identical to the accepted plrmlp_full_fold0 reference
    # (arch family + optimizer schedule); only the MTL factors differ. Validated
    # against the preregistered arms so no unregistered (p_mask, lam) can run.
    cfg = dict(prereg["protocol"]["train_kwargs"])
    cfg["p_mask"] = args.p_mask
    cfg["lam"] = args.lam
    arch_cfg = dict(prereg["protocol"]["arch_kwargs"])
    allowed = {(float(a["p_mask"]), float(a["lam"]), int(a["seed"]))
               for a in prereg["protocol"]["arms"]}
    if (float(args.p_mask), float(args.lam), int(args.seed)) not in allowed:
        raise SystemExit(f"arm (p_mask={args.p_mask}, lam={args.lam}, seed={args.seed}) "
                         f"is not in the preregistered arms {sorted(allowed)}")
    tag = args.tag or f"mtl_{args.view}_pm{args.p_mask:g}_lam{args.lam:g}_s{args.seed}"

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    import torch

    from src.phase20 import mtl_arch

    set_seed(args.seed)

    tr, te = load_cached_parquet()
    y = tr["satisfaction"].to_numpy(dtype="int8")
    ids = tr["id"].to_numpy()
    folds = get_scheme("primary", y, ids).folds
    R_all = tr[RATINGS].to_numpy(dtype=np.int64)
    assert R_all.min() >= 0 and R_all.max() == 5, "unexpected rating levels"

    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert not np.intersect1d(FIT, EVAL).size

    route_all, port_all = load_champion_fold0(folds)

    if args.view == "route":
        from scripts.run_sol_tabpfn import frames
        Xall, _Xte, names, cats, _maps = frames(tr, te, route=True)
        X_fit, X_eval_all = Xall[FIT], Xall[EVAL]
    else:
        from src.features.view import ViewBuilder
        vb = ViewBuilder(tr, te, "full")
        vb.build_static()
        X_fit, Xa, names = vb.assemble(FIT, y, EVAL, None, inner_seed=args.seed)
        X_eval_all = Xa["val"]
    assert names[:13] == RATINGS, "rating columns must be view columns 0..12"
    n_feat = X_fit.shape[1]

    route0, port0 = route_all[EVAL], port_all[EVAL]
    y_eval = y[EVAL]

    fit_pos = np.arange(len(FIT))
    train_pos, val_pos = inner_es_split(fit_pos, y[FIT], frac=0.1, seed=args.seed)
    y_fit = y[FIT].astype(np.float64)
    mu, sd = _fit_scaler(X_fit[train_pos])
    Xtr_s = (X_fit[train_pos] - mu) / sd
    Xval_s = (X_fit[val_pos] - mu) / sd
    Xeval_s = (X_eval_all - mu) / sd
    Rtr = R_all[FIT][train_pos]

    log(f"rows={len(y)} view={args.view} FIT={len(FIT)} train={len(train_pos)} "
        f"val={len(val_pos)} EVAL={len(EVAL)} features={n_feat} "
        f"p_mask={args.p_mask} lam={args.lam} device={args.device}")

    pred_path = ART / f"{tag}.npy"
    meta_path = OUT / f"{tag}.json"
    if pred_path.exists() and meta_path.exists():
        saved = np.load(pred_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "DONE" and arr_sha256(saved) == meta.get("prediction_sha256") \
                and meta.get("eval_rows") == int(len(EVAL)):
            m = meta["metrics"]
            log(f"verified prediction exists; reusing AUC={m['standalone_auc']:.6f} "
                f"port_delta={m['portfolio_delta']:+.3e}")
            return 0

    model = mtl_arch.MultiTaskPLR(n_feat, **arch_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"training MultiTaskPLR ({n_params:,} params) -> predict {len(EVAL)}")
    t1 = time.time()
    try:
        logits, best_val = train_and_predict(model, Xtr_s, y_fit[train_pos], Rtr,
                                             Xval_s, y_fit[val_pos], Xeval_s,
                                             cfg, args.device, args.seed)
        pred = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        np.save(pred_path, pred.astype("float32"))
        m = screen_metrics(y_eval, pred, port0, route0)
        status = "DONE"
        err = None
    except Exception as exc:
        m = None
        best_val = None
        status = "FAILED"
        err = repr(exc)
    secs = time.time() - t1

    meta = {
        "model": "multitask_plr", "status": status, "error": err,
        "arch_cfg": arch_cfg, "train_cfg": cfg,
        "utc": datetime.now(timezone.utc).isoformat(), "git": git_commit(),
        "eval_rows": int(len(EVAL)), "eval_fold": 0, "fit_folds": [1, 2, 3, 4],
        "view": args.view,
        "train_rows": int(len(train_pos)), "val_rows": int(len(val_pos)),
        "inner_es_seed": args.seed, "scaler": "zero-mean/unit-var fit on FIT-train only",
        "device": args.device, "seed": args.seed, "tag": tag, "n_params": int(n_params),
        "best_inner_val_auc": None if best_val is None else float(best_val),
        "features": list(names),
        "protocol": "masked-rating multi-task (CORAL aux) on FIT, inner 10% ES from FIT "
                    "under inference conditions, predict fold-0; champion hash-asserted",
        "prediction_sha256": arr_sha256(np.load(pred_path)) if status == "DONE" else None,
        "metrics": m, "fit_predict_seconds": secs,
        "prereg": {"path": PREREG.relative_to(ROOT).as_posix()},
    }
    save_json(meta, meta_path)
    if m is not None:
        log(f"AUC={m['standalone_auc']:.6f} port_delta={m['portfolio_delta']:+.3e} "
            f"corr_port={m['logit_pearson_vs_portfolio']:.4f} best_val={best_val:.4f} "
            f"({secs:.1f}s)")
    else:
        log(f"FAILED: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
