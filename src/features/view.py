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
    # ---- Phase 8A: te_all21, and the swap view that replaces the old `te` rather than stacking ----
    # `full_te21`  = full + te_all21  (old `te` KEPT)
    # `full_all21te` = full with the old `te` REMOVED and te_all21 in its place
    # Both are needed. Two redundant TE systems can inflate variance and, under
    # extra_trees + colsample_bytree=0.8, consume the same random 80% of split candidates, which can
    # make a genuinely useful block look harmful. The swap view is what separates "more TE columns"
    # from "TE columns competing with each other".
    "full_te21": ["raw", "cat", "trans", "token", "external", "teacher", "te", "te_all21"],
    "full_all21te": ["raw", "cat", "trans", "token", "external", "teacher", "te_all21"],
    "core3_te21": ["raw", "cat", "trans", "external", "teacher", "te", "te_all21"],
    "core3_all21te": ["raw", "cat", "trans", "external", "teacher", "te_all21"],
    # ---- Phase 8B: conditional TE (service rating x traveller segment, service pair x trip/customer) ----
    # Built to the two axes te_all21 lacked and which the existing `te` block demonstrably exploits:
    # crosses, and a 10/20/100 shrinkage spectrum with a count column per key.
    # 18 keys x 4 columns = 72 new columns. All keys are well-estimated: 53-133 distinct values,
    # median group 674-4,495 rows, and 0% of rows fall in a group of size <= 5, so nothing here is a
    # prior-dominated encoding.
    "full_tec": ["raw", "cat", "trans", "token", "external", "teacher", "te", "te_cond"],
    "full_tec_swap": ["raw", "cat", "trans", "token", "external", "teacher", "te_cond"],
    "core3_tec": ["raw", "cat", "trans", "external", "teacher", "te", "te_cond"],
    "core3_tec_swap": ["raw", "cat", "trans", "external", "teacher", "te_cond"],
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
    def build_te_generic(self, fit_idx: np.ndarray, apply_sets: dict[str, np.ndarray],
                         y: np.ndarray, keys_all: dict[str, pd.Series], inner_seed: int = 0):
        """Fold-safe TE for an ARBITRARY key dict, so `te` and `te_conditional` share one code path.

        Fold-safety: fit rows are encoded by an inner cross-fit that excludes each row's own label, and
        every apply set is encoded by a table fitted on ALL fit rows. Apply-row labels are never read.
        """
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

        return self.build_te_generic(fit_idx, apply_sets, y, keys_all, inner_seed)

    def build_te_conditional(self, fit_idx: np.ndarray, apply_sets: dict[str, np.ndarray],
                             y: np.ndarray, inner_seed: int = 0):
        """Phase 8B: 13 (service rating x traveller segment) + 5 (service pair x trip/customer) keys."""
        from src.features.te_conditional import SMOOTHS, build_te_conditional_keys
        keys_all = build_te_conditional_keys(self.comb)
        return self.build_te_generic(fit_idx, apply_sets, y, keys_all, inner_seed)

    def build_te_all21(self, fit_idx: np.ndarray, apply_sets: dict[str, np.ndarray],
                       y: np.ndarray, inner_seed: int = 0, smooth="auto"):
        """Phase 8A: exact-value target encoding of all 21 raw columns.

        Kept as a SEPARATE block from `te` so the two can be stacked or swapped without touching the
        existing implementation -- `core3_te` / `full` keep using `te` exactly as before, which is
        what makes the ablation clean.

        Fold-safety mirrors `build_te`: the fit rows are encoded by an inner cross-fit that excludes
        each row's own label (exactly, verified in tests/test_te_all21.py), and every apply set is
        encoded by a table fitted on ALL fit rows. The apply rows' labels are never read.
        """
        from src.features.te_all21 import All21TargetEncoder, build_all21_codes

        codes, _names = build_all21_codes(self.comb)

        yfit = y[fit_idx].astype("float64")
        Xfit = All21TargetEncoder(smooth=smooth, cv=5, seed=inner_seed).fit_rows(codes, y, fit_idx)

        outs = {}
        for nm, idx in apply_sets.items():
            outs[nm] = All21TargetEncoder(smooth=smooth, cv=5,
                                          seed=inner_seed).apply(codes, y, fit_idx, idx)
        return Xfit, outs, [f"te_all21[{i}]" for i in range(codes.shape[1])], None

    def assemble(self, fit_idx: np.ndarray, y: np.ndarray, val_idx: np.ndarray,
                 test_idx: np.ndarray | None = None, inner_seed: int = 0,
                 te_all21_smooth="auto"):
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
        if "te_all21" in self.blocks:
            # stacked ON TOP of any existing `te` block. The swap view `full_all21te` omits `te`
            # entirely, which is the comparison that separates "more TE columns" from "redundant TE
            # systems competing for the same colsample_bytree slots". Measured: te_all21 is REJECTED
            # (-0.3e-5 stacked, -13.3e-5 replacing, under both extra_trees and deterministic), because
            # our existing selective `te` block already wins on crosses, the 10/20/100 shrinkage
            # spectrum, counts, and binned Flight Distance variants.
            Xf_21, outs21, fn21, _ = self.build_te_all21(fit_idx, apply_sets, y, inner_seed,
                                                          smooth=te_all21_smooth)
            Xf = np.column_stack([Xf, Xf_21])
            names_f = list(names_f) + list(fn21)
            for k, v in outs21.items():
                pieces_a[k] = (np.column_stack([pieces_a[k], v]) if k in pieces_a else v)
            names_a = list(names_a) + list(fn21)
        if "te_cond" in self.blocks:
            # Phase 8B. Built to the two axes te_all21 lacked: CROSSES and a shrinkage spectrum
            # (smooth 10/20/100 plus a count column, all from FoldSafeTE).
            Xf_tc, outs_tc, fn_tc, _ = self.build_te_conditional(fit_idx, apply_sets, y, inner_seed)
            Xf = np.column_stack([Xf, Xf_tc])
            names_f = list(names_f) + list(fn_tc)
            for k, v in outs_tc.items():
                pieces_a[k] = (np.column_stack([pieces_a[k], v]) if k in pieces_a else v)
            names_a = list(names_a) + list(fn_tc)
        Xa = {"val": self.static_tr[val_idx]}
        for k, v in pieces_a.items():
            Xa[k] = np.column_stack([self.static_tr[val_idx] if k == "val" else self.static_te, v])
        if test_idx is not None:
            if pieces_a:
                Xa["test"] = np.column_stack([self.static_te, pieces_a["test"]])
            else:
                Xa["test"] = self.static_te
        Xf = np.nan_to_num(Xf, nan=-999.0, posinf=1e9, neginf=-1e9).astype("float32")
        Xa = {k: np.nan_to_num(v, nan=-999.0, posinf=1e9, neginf=-1e9).astype("float32")
              for k, v in Xa.items()}
        return Xf, Xa, names_f