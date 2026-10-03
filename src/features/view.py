"""Feature-view assembler.

A *view* is a named bundle of feature families. Views are declared in `VIEWS` and each has a
fixed list of blocks so that ablation is a one-word change.

Blocks
------
raw          : the 21 raw columns
trans        : label-free statistics fitted on train+test combined (transductive)
token        : GPT-2 BPE token keys (deterministic)
external     : smoothed original-dataset target statistics (external labels only)
teacher      : original-dataset-only model prediction (external labels only)
te           : fold-safe target encoding (recomputed per fold, inner cross-fitted)
cat          : exact-value categorical twin copies
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import FEATURES, ROOT
from src.features import s6e10 as S

SURVEY13 = S.SURVEY13
META4 = S.META4
NUMS = S.NUMS
RAW21 = SURVEY13 + META4 + NUMS

VIEWS: dict[str, list[str]] = {
    "raw": ["raw"],
    "raw_cat": ["raw", "cat"],
    "raw_trans": ["raw", "trans"],
    "raw_ext": ["raw", "external"],
    "raw_teacher": ["raw", "teacher"],
    "raw_ogsurf": ["raw", "ogsurf"],
    "core3": ["raw", "cat", "trans", "external", "teacher"],
    "core3_ogsurf": ["raw", "cat", "trans", "external", "teacher", "ogsurf"],
    "core3_te": ["raw", "cat", "trans", "external", "teacher", "te"],
    "full": ["raw", "cat", "trans", "token", "external", "teacher", "te"],
    "full_ogsurf": ["raw", "cat", "trans", "token", "external", "teacher", "ogsurf"],
    "full_ogsurf_te": ["raw", "cat", "trans", "token", "external", "teacher", "ogsurf", "te"],
    "core3_ogm": ["raw", "cat", "trans", "external", "teacher", "ogmulti"],
    "full_ogm": ["raw", "cat", "trans", "token", "external", "teacher", "ogmulti"],
    "full_ogm_ogs": ["raw", "cat", "trans", "token", "external", "teacher", "ogmulti", "ogsurf"],
    "full_enrich": ["raw", "cat", "trans", "token", "external", "teacher", "te", "enrich"],
    "full_enrich_note": ["raw", "cat", "trans", "token", "external", "teacher", "te"],
    "core3_enrich": ["raw", "cat", "trans", "external", "teacher", "enrich"],
}

_CACHE = ROOT / "artifacts" / "cache"


def _prep_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df[RAW21].copy()
    for c in out.columns:
        if out[c].dtype.kind in "OUS":
            out[c] = pd.factorize(out[c].astype("object"), sort=True)[0].astype("float32")
        else:
            out[c] = out[c].astype("float32")
    return out.fillna(-1.0)


def _cat_twins(comb: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    names, mats = [], []
    for c in cols:
        names.append(c + "__cat")
        if comb[c].dtype.kind in "OUS":
            mats.append(pd.factorize(comb[c].astype("object"), sort=True)[0].astype("float32"))
        else:
            v = comb[c]
            v = v.round(6) if v.dtype.kind == "f" else v
            mats.append(pd.factorize(v.astype("object"), sort=True)[0].astype("float32"))
    return np.column_stack(mats).astype("float32"), names


class ViewBuilder:
    """Builds the static (fold-independent) part of a view once, and the fold-safe part per fold."""

    def __init__(self, tr: pd.DataFrame, te: pd.DataFrame, view: str,
                 teacher_kinds=("lgbm",), teacher_models: int = 5):
        self.tr, self.te, self.view = tr, te, view
        self.ntr = len(tr)
        self.blocks = VIEWS[view]
        self.teacher_kinds = tuple(teacher_kinds)
        self.teacher_models = teacher_models
        self.notes: dict = {}

    # ---------------------------------------------------------------- static
    def build_static(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        tr, te = self.tr, self.te
        # `comb` is needed by build_te even on a cache hit, so always materialise it.
        self.comb = pd.concat([tr[RAW21], te[RAW21]], ignore_index=True)
        cache = FEATURES / f"static_{self.view}.npz"
        meta = FEATURES / f"static_{self.view}.json"
        if cache.exists() and meta.exists():
            try:
                z = np.load(cache)
                self.static_tr = z["tr"]
                self.static_te = z["te"]
                self.static_names = json.loads(meta.read_text())
                print(f"    [cache] view {self.view}: {self.static_tr.shape}")
                return self.static_tr, self.static_te, self.static_names
            except Exception as exc:  # noqa: BLE001
                print(f"    [cache] unusable ({exc}); rebuilding")

        comb = self.comb
        n = len(comb)
        mats, names = [], []
        feats = RAW21

        if "raw" in self.blocks:
            mats.append(_prep_raw(comb).to_numpy().astype("float32"))
            names += list(RAW21)
        if "cat" in self.blocks:
            m, nm = _cat_twins(comb, RAW21)
            mats.append(m)
            names += nm
        if "trans" in self.blocks:
            m, nm = S.build_transductive_features(comb, feats)
            mats.append(m)
            names += nm
        if "token" in self.blocks:
            m, nm = S.build_token_features(comb, [S.FD, S.AGE, S.DEP, S.ARR])
            mats.append(m)
            names += nm
        if "external" in self.blocks:
            ex = S.build_external_features()
            m, nm = S.apply_external(comb, ex)
            mats.append(m)
            names += nm
            self.notes["n_original"] = ex["n_original"]
        if "teacher" in self.blocks:
            for kind in self.teacher_kinds:
                t = S.OriginalTeacher(kind=kind)
                p = t.fit_predict(comb, feats, n_models=self.teacher_models)
                mats.append(np.column_stack([
                    p, np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                ]).astype("float32"))
                names += [f"teach_{kind}", f"teachlogit_{kind}"]
                self.notes[f"teacher_{kind}_auc_vs_train"] = None
            self._teacher_p = p[: self.ntr]
            self._teacher_y = tr[S.TARGET].values
        if "ogsurf" in self.blocks:
            surf = S.build_original_surfaces(S.SURFACE_ANCHORS)
            m, nm = S.apply_original_surfaces(comb, surf)
            mats.append(m)
            names += nm
            self.notes["ogsurf_n_original"] = surf["n_original"]
        if "ogmulti" in self.blocks:
            exm = S.build_external_multi()
            m, nm = S.apply_external_multi(comb, exm)
            mats.append(m)
            names += nm
            self.notes["ogmulti_n_tables"] = len(exm["tables"])
        if "enrich" in self.blocks:
            from src.features.enrich import build_enrich

            m, nm = build_enrich(comb)
            mats.append(m)
            names += nm
            self.notes["enrich_n"] = len(nm)

        X = np.column_stack(mats).astype("float32")
        X = np.nan_to_num(X, nan=-999.0, posinf=1e9, neginf=-1e9)
        self.static_names = names
        self.static_tr = X[: self.ntr]
        self.static_te = X[self.ntr:]
        try:
            FEATURES.mkdir(parents=True, exist_ok=True)
            np.savez(cache, tr=self.static_tr, te=self.static_te)
            meta.write_text(json.dumps(names), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"    [cache] write failed ({exc})")
        return self.static_tr, self.static_te, names

    def teacher_auc(self) -> float | None:
        from sklearn.metrics import roc_auc_score

        if getattr(self, "_teacher_p", None) is None:
            return None
        return float(roc_auc_score(self._teacher_y, self._teacher_p))

    # ---------------------------------------------------------------- fold-safe
    def build_te(self, fit_idx: np.ndarray, apply_sets: dict[str, np.ndarray], y: np.ndarray,
                 inner_seed: int = 0):
        """Fold-safe target encoding.

        ``apply_sets`` maps a name (e.g. 'val', 'test') to the row indices to encode.
        Returns (X_fit, {name: X_apply}, fit_names, apply_names).
        """
        derived = S.build_te_keys(self.comb)
        src = self.comb[RAW21].copy()
        for k, v in derived.items():
            src[k] = v.to_numpy()

        keys_all: dict[str, pd.Series] = {}
        for name, cols in S.TE_KEYS_DEFAULT.items():
            if not all(c in src.columns for c in cols):
                continue
            if len(cols) == 1:
                keys_all[name] = src[cols[0]].reset_index(drop=True)
            else:
                keys_all[name] = pd.util.hash_pandas_object(
                    src[cols].astype(object).round(6), index=False
                ).reset_index(drop=True)

        te = S.FoldSafeTE(keys_all, inner_seed=inner_seed)
        yfit = y[fit_idx]
        fit_keys = {k: v.iloc[fit_idx].reset_index(drop=True) for k, v in keys_all.items()}
        Xfit, fnames = te.crossfit(fit_keys, yfit)
        outs = {}
        for nm, idx in apply_sets.items():
            ak = {k: v.iloc[idx].reset_index(drop=True) for k, v in keys_all.items()}
            Xa, anames = te.apply(fit_keys, yfit, ak)
            outs[nm] = Xa
        return Xfit, outs, fnames, anames

    def assemble(self, fit_idx: np.ndarray, y: np.ndarray, val_idx: np.ndarray,
                 test_idx: np.ndarray | None = None, inner_seed: int = 0):
        Xf = self.static_tr[fit_idx]
        pieces_f, pieces_a, names_f, names_a = [], {}, [], []
        apply_sets = {"val": val_idx}
        if test_idx is not None:
            apply_sets["test"] = test_idx
        if "te" in self.blocks:
            Xf_te, outs, fn, an = self.build_te(fit_idx, apply_sets, y, inner_seed)
            Xf = np.column_stack([Xf, Xf_te])
            names_f = list(self.static_names) + list(fn)
            for k, v in outs.items():
                pieces_a[k] = v
            names_a = list(an)
        else:
            names_f = list(self.static_names)
            names_a = []
        Xa = {"val": self.static_tr[val_idx]}
        for k, v in pieces_a.items():
            Xa[k] = np.column_stack([self.static_tr[val_idx] if k == "val" else self.static_te, v])
        if test_idx is not None:
            if "te" in self.blocks:
                Xa["test"] = np.column_stack([self.static_te, pieces_a["test"]])
            else:
                Xa["test"] = self.static_te
        Xf = np.nan_to_num(Xf, nan=-999.0, posinf=1e9, neginf=-1e9).astype("float32")
        Xa = {k: np.nan_to_num(v, nan=-999.0, posinf=1e9, neginf=-1e9).astype("float32")
              for k, v in Xa.items()}
        return Xf, Xa, names_f