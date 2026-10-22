"""Phase 24 -- probe downloadable external OOF/test pairs against the certified champion.

Every external OOF vector is a per-row out-of-sample prediction produced by its author on
their own 5-fold split.  For any row i the value never saw row i's label, so its AUC on our
train labels, and its paired per-fold delta against v7 on OUR immutable primary folds, are
honest measurements.  What is NOT honest is fitting blend weights on the rows they are
scored on -- that is done separately, inside the FIT boundary, by blend_probe.py.

    .\\.venv\\Scripts\\python.exe scripts\\phase24\\external_oof_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import arr_sha256, load_cached_parquet  # noqa: E402
from src.validation.compare import fold_auc, logit, paired_fold_deltas  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

EXT = ROOT / "data/external/s6e10_public"
OUT = ROOT / "reports/phase24/external_oof_probe.json"
V7_OOF = ROOT / "artifacts/predictions/v7_sol_tabpfn_internal2_aux10_oof.npy"


def csv_vec(path: Path, ids: np.ndarray) -> np.ndarray:
    d = pd.read_csv(path)
    lower = {c.lower(): c for c in d.columns}
    idc = lower.get("id", d.columns[0])
    valc = lower.get("satisfaction")
    if valc is None:
        cand = [c for c in d.columns if c != idc]
        valc = cand[-1]
    s = d.set_index(idc)[valc]
    if s.index.has_duplicates:
        raise ValueError(f"{path.name}: duplicate ids")
    v = s.reindex(ids)
    if v.isna().any():
        raise ValueError(f"{path.name}: {int(v.isna().sum())} train ids missing")
    return v.to_numpy(np.float64)


def npy_vec(path: Path, n: int) -> np.ndarray:
    a = np.load(path).astype(np.float64)
    if a.shape != (n,):
        raise ValueError(f"{path.name}: shape {a.shape} != ({n},)")
    return a


def summarise(name: str, vec: np.ndarray, y: np.ndarray, folds: np.ndarray,
              v7: np.ndarray, v7_logit: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, vec))
    per_fold = fold_auc(y, vec, folds)
    mean_delta, deltas = paired_fold_deltas(y, vec, v7, folds)
    d = np.asarray(deltas, dtype=float)
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else float("nan")
    lg = logit(vec)
    r = float(np.corrcoef(lg, v7_logit)[0, 1])
    return {
        "name": name,
        "pooled_auc": auc,
        "fold_mean_auc": float(np.mean(per_fold)),
        "fold_auc": per_fold,
        "delta_vs_v7_foldmean": mean_delta,
        "per_fold_delta": deltas,
        "t": t,
        "folds_positive": int((d > 0).sum()),
        "logit_corr_v7": r,
        "sha256": arr_sha256(vec.astype(np.float32)),
    }


def main() -> None:
    train, _ = load_cached_parquet()
    y = train["satisfaction"].to_numpy(np.int8)
    ids = train["id"].to_numpy()
    folds = get_scheme("primary", y, train["id"]).folds
    v7 = np.load(V7_OOF).astype(np.float64)
    v7_logit = logit(v7)

    rows: list[dict] = []
    failed: list[dict] = []

    # Deotte TFM library: 11 members, npy, row order = train id order.
    tfm_dir = EXT / "s6e10-tfm-oof-and-test-predictions"
    for p in sorted(tfm_dir.glob("oof_*.npy")):
        name = "tfm__" + p.stem.replace("oof_", "")
        try:
            rows.append(summarise(name, npy_vec(p, len(y)), y, folds, v7, v7_logit))
        except Exception as e:  # noqa: BLE001
            failed.append({"name": name, "error": str(e)})

    # goodpjw2008 TabPFN/TabICL member library: csv with id.
    gp = EXT / "s6e10-tabpfn-member"
    for p in sorted(gp.glob("oof_*.csv")):
        name = "gp__" + p.stem.replace("oof_", "")
        try:
            rows.append(summarise(name, csv_vec(p, ids), y, folds, v7, v7_logit))
        except Exception as e:  # noqa: BLE001
            failed.append({"name": name, "error": str(e)})

    # najiama blends: csv with id.
    nj = EXT / "s6e10-oof"
    for p in sorted(nj.glob("*_oof.csv")):
        name = "nj__" + p.stem.replace("_oof", "")
        try:
            rows.append(summarise(name, csv_vec(p, ids), y, folds, v7, v7_logit))
        except Exception as e:  # noqa: BLE001
            failed.append({"name": name, "error": str(e)})

    rows.sort(key=lambda r: -r["pooled_auc"])
    OUT.write_text(json.dumps({
        "champion": {"name": "v7", "pooled_auc": float(roc_auc_score(y, v7))},
        "n_vectors": len(rows),
        "n_failed": len(failed),
        "failed": failed,
        "vectors": rows,
    }, indent=1), encoding="utf-8")

    print(f"{'vector':34} {'AUC':>10} {'vs v7':>10} {'t':>7} {'+folds':>6} {'corr':>8}")
    for r in rows:
        print(f"{r['name']:34} {r['pooled_auc']:10.6f} {r['delta_vs_v7_foldmean']:+10.2e} "
              f"{r['t']:7.2f} {r['folds_positive']:4d}/5 {r['logit_corr_v7']:8.5f}")
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    main()
