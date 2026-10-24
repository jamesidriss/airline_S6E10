"""Two closing structural checks that random-fold OOF cannot answer.

1. ADVERSARIAL VALIDATION -- can a model tell a train row from a test row using the 21
   covariates alone? If yes (AUC above ~0.52), the test set is drawn from a different part of
   the covariate distribution, and train rows closer to it should be weighted up. If no
   (AUC ~ 0.50), every train row is equally relevant and reweighting cannot help.

2. ID LEAK -- the ids are one contiguous stream (train 0..699634, test 699635..999478), so the
   split is a block split of a generated sequence. If the generator emitted correlated labels in
   id runs, `id` (or `id mod k`) would carry label information that a random-fold OOF would
   happily report as a gain and the private LB would not reproduce. Checked with the
   *blocked* fold scheme, where contiguous id ranges stay inside one fold.

Both checks are label-free on the test side, so neither can leak.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORTS, save_json  # noqa: E402


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """All columns numeric: strings become integer codes on the union of categories."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype("float32")
        else:
            out[c] = s.astype("string").fillna("<NA>").astype("category").cat.codes.astype("float32")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--tag", default="adversarial")
    args = ap.parse_args()

    tr = pd.read_csv(ROOT / "data/raw/train.csv")
    te = pd.read_csv(ROOT / "data/raw/test.csv")
    feat = [c for c in te.columns if c != "id"]
    y = tr["satisfaction"].to_numpy(dtype="int8")

    # ---------------- 1. adversarial validation ----------------
    X = encode(pd.concat([tr[feat], te[feat]], ignore_index=True))
    z = np.r_[np.zeros(len(tr), dtype="int8"), np.ones(len(te), dtype="int8")]
    import lightgbm as lgb

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    oof = np.zeros(len(z))
    for a, b in skf.split(X, z):
        m = lgb.LGBMClassifier(n_estimators=args.trees, learning_rate=0.05, num_leaves=63,
                               n_jobs=8, verbose=-1)
        m.fit(X.iloc[a], z[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    adv = roc_auc_score(z, oof)
    print(f"adversarial train-vs-test AUC = {adv:.6f}", flush=True)

    # ---------------- 2. id leak ----------------
    idf = pd.DataFrame({
        "id": tr["id"].to_numpy(dtype="float64"),
        "id_mod_2": tr["id"] % 2, "id_mod_3": tr["id"] % 3, "id_mod_5": tr["id"] % 5,
        "id_mod_7": tr["id"] % 7, "id_mod_10": tr["id"] % 10, "id_mod_100": tr["id"] % 100,
        "id_mod_1000": tr["id"] % 1000, "id_div_1000": tr["id"] // 1000,
    }).astype("float32")
    auc_id = roc_auc_score(y, idf["id"].to_numpy())
    print(f"raw id alone as a score: AUC = {auc_id:.6f}", flush=True)

    from src.validation.folds import blocked_id_scheme, get_scheme
    out = {"adversarial_auc": float(adv), "id_alone_auc": float(auc_id), "schemes": {}}
    for scheme in ("primary", "blocked"):
        if scheme == "primary":
            f = np.asarray(get_scheme("primary", y, tr["id"]).folds)
        else:
            f = np.asarray(blocked_id_scheme(y, n_blocks=5))
        p = np.zeros(len(y))
        for k in sorted(set(f.tolist())):
            a, b = np.where(f != k)[0], np.where(f == k)[0]
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                   n_jobs=8, verbose=-1)
            m.fit(idf.iloc[a], y[a])
            p[b] = m.predict_proba(idf.iloc[b])[:, 1]
        auc = float(roc_auc_score(y, p))
        out["schemes"][scheme] = auc
        print(f"id-derived GBDT, {scheme} folds: AUC = {auc:.6f}", flush=True)

    out["verdict"] = ("NO_COVARIATE_SHIFT_NO_ID_LEAK"
                      if adv < 0.52 and all(v < 0.501 for v in out["schemes"].values())
                      else "STRUCTURE_DETECTED")
    save_json(out, REPORTS / f"{args.tag}.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
