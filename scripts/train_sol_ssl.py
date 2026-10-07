"""Transductive masked encoder: covariates only, fixed reconstruction objective.

Run a timing probe first. Model selection sees reconstruction loss only; no
satisfaction targets or Kaggle scores are ever loaded by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS, RAW, REPORTS, arr_sha256, file_sha256, git_commit, save_json
from src.features.view import RAW21
from src.models.masked import MaskedEncoder, prepare_covariates, masked_loss

SEED = 1201


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--tag", default="sol_ssl12")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tr = pd.read_csv(RAW / "train.csv", usecols=["id"] + RAW21)
    te = pd.read_csv(RAW / "test.csv", usecols=["id"] + RAW21)
    frame = pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)
    cat, num, missing, prep = prepare_covariates(frame)
    ids = np.r_[tr["id"].to_numpy(), te["id"].to_numpy()]
    n = len(ids)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    valid, fit = order[:int(.02 * n)], order[int(.02 * n):]
    contract = {"seed": SEED, "epochs": args.epochs, "batch": args.batch, "latent": 32, "width": 256,
                "mask_probability": .25, "validation_fraction": .02,
                "objective": "masked covariate CE + smooth L1; no satisfaction targets",
                "data_sha256": {f: file_sha256(RAW / f) for f in ("train.csv", "test.csv")},
                "ids_sha256": arr_sha256(ids), "fit_ids_sha256": arr_sha256(ids[fit]),
                "validation_ids_sha256": arr_sha256(ids[valid]), "preprocessing": prep,
                "source_sha256": {p: file_sha256(p) for p in ("scripts/train_sol_ssl.py", "src/models/masked.py")},
                "rules_source": "research/raw/sol_rules_20261007.json"}
    from hashlib import sha256
    fp = sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    dest = ARTIFACTS / args.tag
    dest.mkdir(exist_ok=True)
    model = MaskedEncoder(prep["cardinalities"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    C = torch.tensor(cat, device="cuda")
    N = torch.tensor(num, device="cuda")
    M = torch.tensor(missing, device="cuda")
    val_mask = torch.rand((len(valid), 21), device="cuda") < .25
    vi = torch.tensor(valid, device="cuda")
    start = time.monotonic()
    if args.timing:
        batches = 60
        for step in range(batches + 5):
            ii = torch.randint(n, (args.batch,), device="cuda")
            mask = torch.rand((args.batch, 21), device="cuda") < .25
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, C[ii], N[ii], M[ii], mask)
            loss.backward()
            optimizer.step()
            if step == 4:
                torch.cuda.synchronize()
                start = time.monotonic()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        rec = {"contract": contract, "fingerprint": fp, "steps": batches, "seconds": elapsed,
               "seconds_per_epoch_estimate": elapsed / batches * np.ceil(len(fit) / args.batch),
               "peak_vram_bytes": torch.cuda.max_memory_allocated(), "status": "TIMING_ONLY"}
        save_json(rec, REPORTS / f"{args.tag}_timing.json")
        print(json.dumps({k: rec[k] for k in ("seconds", "seconds_per_epoch_estimate", "peak_vram_bytes")}))
        return 0
    manifest_path = dest / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["fingerprint"] != fp:
            raise ValueError("existing SSL artifact has another contract; choose a new tag")
        if existing.get("status") == "COMPLETE":
            print("reuse complete SSL artifact", flush=True)
            return 0
    history, best_loss, first_epoch = [], float("inf"), 0
    cp = dest / "checkpoint.pt"
    if cp.exists():
        state = torch.load(cp, weights_only=False)
        assert state["fingerprint"] == fp
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"])
        history, best_loss, first_epoch = state["history"], state["best_loss"], state["epoch"] + 1
    for epoch in range(first_epoch, args.epochs):
        model.train()
        perm = torch.tensor(fit, device="cuda")[torch.randperm(len(fit), device="cuda")]
        total = []
        epoch_start = time.monotonic()
        for ii in perm.split(args.batch):
            mask = torch.rand((len(ii), 21), device="cuda") < .25
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, C[ii], N[ii], M[ii], mask)
            loss.backward()
            optimizer.step()
            total.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = []
            for offset in range(0, len(vi), args.batch):
                ii = vi[offset:offset + args.batch]
                validation_loss.append(float(masked_loss(model, C[ii], N[ii], M[ii], val_mask[offset:offset + len(ii)])) * len(ii))
        vl = sum(validation_loss) / len(vi)
        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), dest / "best.pt")
        scheduler.step()
        row = {"epoch": epoch, "fit_loss": float(np.mean(total)), "validation_loss": vl,
               "seconds": time.monotonic() - epoch_start}
        history.append(row)
        torch.save({"fingerprint": fp, "epoch": epoch, "history": history, "best_loss": best_loss,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state()}, cp)
        save_json({"contract": contract, "fingerprint": fp, "history": history, "status": "TRAINING"}, manifest_path)
        print(f"epoch {epoch + 1}/{args.epochs}: fit {row['fit_loss']:.5f}, valid {vl:.5f}, {row['seconds']:.1f}s", flush=True)
        if time.monotonic() - start > 2 * 3600:
            raise RuntimeError("SSL first training budget exhausted; checkpoint saved")
    model.load_state_dict(torch.load(dest / "best.pt", weights_only=True))
    model.eval()
    embedding = np.zeros((n, 32), dtype="float32")
    surprise = np.zeros((n, 13), dtype="float32")
    residual = np.zeros((n, 13), dtype="float32")
    with torch.no_grad():
        for offset in range(0, n, args.batch):
            ii = torch.arange(offset, min(n, offset + args.batch), device="cuda")
            mask = torch.zeros((len(ii), 21), device="cuda", dtype=torch.bool)
            embedding[offset:offset + len(ii)] = model.encode(C[ii], N[ii], M[ii], mask).cpu().numpy()
            for j in range(13):
                mask[:, j] = True
                _, heads, _ = model(C[ii], N[ii], M[ii], mask)
                logits = heads[j]
                probs = torch.softmax(logits, dim=-1)
                actual = C[ii, j]
                observed = probs.gather(1, actual[:, None])[:, 0]
                surprise[offset:offset + len(ii), j] = (-observed.clamp_min(1e-12).log()).cpu().numpy()
                # Rating category code may start at 1 for baggage: use real levels.
                levels = torch.tensor(prep["maps"][RAW21[j]], device="cuda", dtype=probs.dtype)
                residual[offset:offset + len(ii), j] = (levels[actual] - probs @ levels).cpu().numpy()
                mask[:, j] = False
            if offset % (args.batch * 25) == 0:
                print(f"features {offset}/{n}", flush=True)
    summary = np.column_stack([surprise, surprise.mean(axis=1), surprise.max(axis=1), surprise.std(axis=1),
                               np.abs(residual).mean(axis=1), np.abs(residual).max(axis=1), np.abs(residual).std(axis=1)]).astype("float32")
    assert embedding.shape == (n, 32) and summary.shape == (n, 19)
    assert np.isfinite(embedding).all() and np.isfinite(summary).all()
    np.save(dest / "embedding.npy", embedding)
    np.save(dest / "summary.npy", summary)
    np.save(dest / "ids.npy", ids)
    rec = {"contract": contract, "fingerprint": fp, "git": git_commit(), "history": history,
           "seconds": time.monotonic() - start, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
           "embedding_sha256": arr_sha256(embedding), "summary_sha256": arr_sha256(summary),
           "status": "COMPLETE", "provenance": "ssl_*: transductive label-free covariate reconstruction; targets never loaded"}
    save_json(rec, manifest_path)
    save_json(rec, REPORTS / f"{args.tag}_training.json")
    print("SSL artifact complete", dest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
