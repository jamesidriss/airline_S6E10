"""Assemble every staged public S6E10 OOF/test vector into one aligned pool.

Alignment is proven, not assumed: a vector is kept only if, on the registered primary
folds, it scores an AUC consistent with a real model (0.90 <= AUC <= 0.995). A mis-ordered
vector scores ~0.5, and a leaked label column scores 1.0; both are dropped with a reason.
Everything is stored as a probability on the competition row order.

Output (staging, never committed):
    pool.npz    arrays ``oof__<member>`` (699 635,) and ``test__<member>`` (299 844,) float32
    pool_meta.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import TEST_CSV, TRAIN_CSV  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402

STAGE = Path(r"C:\Users\James\AppData\Local\Temp\opencode\s6e10_oof_pool")
OUT = STAGE / "_pool"
N_TRAIN, N_TEST = 699_635, 299_844
LABELISH = {"satisfaction", "target", "targets", "label", "labels", "y", "truth", "true", "actual"}
SKIP_COLS = {"id", "row_id", "index", "fold", "kfold", "row"}
SIG = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))  # noqa: E731


def as_prob(a) -> np.ndarray | None:
    a = np.asarray(a, dtype="float64").ravel()
    if a.size == 0 or not np.isfinite(a).all():
        return None
    if a.min() >= 0.0 and a.max() <= 1.0:
        return np.clip(a, 1e-7, 1 - 1e-7)
    if a.min() > -40 and a.max() < 40:
        return np.clip(SIG(a), 1e-7, 1 - 1e-7)
    return None


def canonical(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r"^(oof|test|pred|preds|val)[_\-]+", "", s)
    s = re.sub(r"[_\-]+(oof|test|pred|preds|val)$", "", s)
    return re.sub(r"[^0-9a-z]+", "_", s).strip("_")


def main() -> None:
    train = pd.read_csv(TRAIN_CSV, usecols=["id", "satisfaction"])
    test = pd.read_csv(TEST_CSV, usecols=["id"])
    y = train["satisfaction"].to_numpy(dtype="int8")
    ids_tr = train["id"].to_numpy()
    ids_te = test["id"].to_numpy()
    folds = get_scheme("primary", y, train["id"]).folds

    pool: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    rejected: list[dict] = []

    def score_auc(p: np.ndarray) -> float:
        aucs = []
        for k in range(int(folds.max()) + 1):
            m = folds == k
            if m.sum() and len(np.unique(y[m])) == 2:
                aucs.append(float(roc_auc_score(y[m], p[m])))
        return float(np.mean(aucs))

    def consider(member: str, vec, side: str, src: str, explicit_id: np.ndarray | None = None) -> None:
        n = N_TRAIN if side == "oof" else N_TEST
        if vec is None:
            rejected.append({"member": member, "side": side, "src": src, "why": "unreadable"})
            return
        vec = np.asarray(vec, dtype="float64").ravel()
        if explicit_id is not None:
            ids = ids_tr if side == "oof" else ids_te
            s = pd.Series(vec, index=pd.Index(explicit_id))
            if not s.index.is_unique:
                rejected.append({"member": member, "side": side, "src": src, "why": "dup ids"})
                return
            vec = s.reindex(ids).to_numpy(dtype="float64")
        if vec.shape != (n,):
            rejected.append({"member": member, "side": side, "src": src,
                             "why": f"len={vec.shape}"})
            return
        p = as_prob(vec)
        if p is None:
            rejected.append({"member": member, "side": side, "src": src, "why": "nonfinite"})
            return
        if side == "oof":
            a = score_auc(p)
            if not (0.90 <= a <= 0.995):
                rejected.append({"member": member, "side": side, "src": src,
                                 "why": f"auc={a:.4f}"})
                return
            pool[f"oof__{member}"] = p.astype("float32")
            meta.setdefault(member, {})["oof_auc_primary"] = a
            meta[member]["oof_src"] = src
        else:
            pool[f"test__{member}"] = p.astype("float32")
            meta.setdefault(member, {})["test_src"] = src

    def frame_members(df: pd.DataFrame, n_expected: int, slope: str) -> None:
        side = "oof" if len(df) == N_TRAIN else "test" if len(df) == N_TEST else None
        if side is None:
            return
        ids = ids_tr if side == "oof" else ids_te
        has_id = "id" in df.columns
        numeric = [c for c in df.columns
                   if str(c).lower() not in SKIP_COLS and pd.api.types.is_numeric_dtype(df[c])]
        # A two-column frame whose only numeric column is a float in [0, 1] is a prediction
        # file, even when the column happens to be named "satisfaction".
        labelish = [c for c in numeric if str(c).lower() in LABELISH]
        keep_labelish = (len(numeric) == 1 and labelish
                         and not set(np.unique(df[labelish[0]].to_numpy())) <= {0.0, 1.0})
        for col in numeric:
            c = str(col).lower()
            if c in LABELISH and not keep_labelish:
                continue
            member = f"{slope}_{canonical(str(col))}"
            consider(member, df[col].to_numpy(), side, f"frame:{col}",
                     df["id"].to_numpy() if has_id else None)

    for path in sorted(STAGE.rglob("*")):
        if path.is_dir() or OUT in path.parents:
            continue
        rel = path.relative_to(STAGE)
        slug = rel.parts[0].split("__")[0][:10]
        stem = path.stem
        low = path.name.lower()
        if low.endswith((".md", ".png", ".json")):
            continue
        base = f"{slug}_{canonical(stem)}"
        try:
            if low.endswith(".npy"):
                a = np.load(path, allow_pickle=False)
                if a.ndim != 1:
                    continue
                side = "oof" if a.size == N_TRAIN else "test" if a.size == N_TEST else None
                if side:
                    consider(base, a, side, str(rel))
            elif low.endswith(".npz"):
                z = np.load(path, allow_pickle=False)
                for key in z.files:
                    a = np.asarray(z[key])
                    if a.ndim != 1 or a.size not in (N_TRAIN, N_TEST):
                        continue
                    side = "oof" if a.size == N_TRAIN else "test"
                    member = base if len(z.files) == 1 else f"{base}_{canonical(key)}"
                    consider(member, a, side, str(rel))
            elif low.endswith(".csv"):
                df = pd.read_csv(path, low_memory=False)
                if df.shape[1] >= 2:
                    frame_members(df, len(df), base)
            elif low.endswith(".parquet"):
                df = pd.read_parquet(path)
                frame_members(df, len(df), base)
        except Exception as exc:
            rejected.append({"member": base, "src": str(rel), "why": f"{type(exc).__name__}: {exc}"[:160]})

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "pool.npz", **pool)
    paired = sorted({k[5:] for k in pool if k.startswith("oof__")}
                    & {k[6:] for k in pool if k.startswith("test__")})
    report = {"n_oof": sum(k.startswith("oof__") for k in pool),
              "n_test": sum(k.startswith("test__") for k in pool),
              "n_members_with_test": len(paired),
              "members": meta, "rejected": rejected[:300]}
    (OUT / "pool_meta.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("n_oof", "n_test", "n_members_with_test")}))
    top = sorted(((v["oof_auc_primary"], k) for k, v in meta.items() if "oof_auc_primary" in v),
                 reverse=True)[:20]
    for a, k in top:
        print(f"{a:.6f}  {k}")


if __name__ == "__main__":
    main()
