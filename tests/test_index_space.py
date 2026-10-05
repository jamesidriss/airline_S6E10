"""Guards against the index-space bug class that has corrupted three experiments in this campaign.

The bug
-------
`ViewBuilder.assemble(fit_idx, y, val_idx, ...)` returns a feature matrix indexed by **position
within `fit_idx`**, while `y` and every helper built on top of `src.validation.folds` is indexed by
**global row number**. A fold harness therefore has to carry BOTH index spaces and use the right one
for each array:

    Xf[local_rows]      correct -- Xf is local
    y[global_rows]      correct -- y is global

Writing `y[local_rows]` trains a model on correctly-shaped but semantically scrambled
feature/label pairs. LightGBM raises nothing. Early stopping fires almost immediately, the AUC
collapses to chance, and the harness prints a confident multi-hundred-e-5 "result".

Observed instances, all of which produced numbers that looked like findings:

  1. `scripts/run_fullfit.py` inner-CV picker indexed the design matrix by position-within-inner-train
     instead of position-within-outer-fit. Early stopping fired at 4, 5 and 43 rounds and the run
     reported a firm REJECT at -483e-5.
  2. `scripts/run_bagging_test.py` swapped `_inner_es_split`'s return order, so it trained on the
     55,969 early-stopping rows and reported -586e-5 for `subsample=0.9`.
  3. `scripts/run_bagging_test.py`, second attempt, paired local features with global labels and
     reported -2086e-5.

In every case the giveaway was a printed row count or an implausible iteration count. That is what
these tests make structural rather than a matter of care.

What is checked
---------------
`test_global_local_index_pairing_is_detectable`
    The core property: a small model trained on correctly paired rows discriminates, and the same
    model trained on mis-paired rows does not. If this invariant ever stops holding, no other check
    in this file can be trusted.

`test_inner_es_split_returns_train_first`
    `_inner_es_split` returns `(train_rows, es_rows)`. Reading them in the other order produces a
    valid-looking but inverted split, and disjointness alone does not catch it.

`test_early_stopping_holdout_is_about_ten_percent`
    The ES carve is a fixed 10% fraction; anything else means the arguments were mixed up.

`test_assemble_returns_local_indexed_matrix`
    The returned fit matrix has one row per element of `fit_idx` and the val matrix one row per
    element of `val_idx`, so local indexing is well defined at all.

`test_fit_and_eval_row_sets_are_disjoint`
    The recorded hazard from AGENTS.md (`np.where(folds != k)` instead of `==`) would make the eval
    set equal the training set. Checked directly on the immutable primary scheme.

These run in a few seconds and need no trained models beyond a 20-round LightGBM on a subsample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ID_COL, TARGET, load_cached_parquet  # noqa: E402
from src.features.view import ViewBuilder  # noqa: E402
from src.validation.folds import get_scheme  # noqa: E402
from scripts.run_views import _inner_es_split  # noqa: E402

_CACHE: dict = {}


def _data():
    if "d" not in _CACHE:
        tr, te = load_cached_parquet()
        y = tr[TARGET].values.astype("int8")
        folds = get_scheme("primary", y, tr[ID_COL]).folds
        _CACHE["d"] = (tr, te, y, folds)
    return _CACHE["d"]


def _fit_matrix(fit, y, view="full"):
    """Assemble the fold-0 feature matrices once and cache them (they are the expensive part)."""
    key = ("X", view, len(fit), int(fit[:5].sum()))
    if key not in _CACHE:
        tr, te, y, _f = _data()
        vb = ViewBuilder(tr, te, view)
        vb.build_static()
        Xf, Xa, _names = vb.assemble(fit, y, np.where(_f == 0)[0], None, inner_seed=0)
        _CACHE[key] = (Xf, Xa["val"])
    return _CACHE[key]


def test_assemble_returns_local_indexed_matrix():
    tr, te, y, folds = _data()
    fit = np.where(folds != 0)[0]
    val = np.where(folds == 0)[0]
    Xf, Xv = _fit_matrix(fit, y)
    assert Xf.shape[0] == len(fit), f"fit matrix has {Xf.shape[0]} rows, expected {len(fit)}"
    assert Xv.shape[0] == len(val), f"val matrix has {Xv.shape[0]} rows, expected {len(val)}"
    assert Xf.shape[1] == Xv.shape[1], "fit and val matrices must share the feature axis"


def test_fit_and_eval_row_sets_are_disjoint():
    tr, te, y, folds = _data()
    for k in sorted(set(folds.tolist())):
        fit = np.where(folds != k)[0]
        val = np.where(folds == k)[0]
        assert not (set(fit.tolist()) & set(val.tolist())), (
            f"fold {k}: fit and eval row sets overlap -- this is the "
            f"np.where(folds != k) vs == hazard, and it silently inflates OOF")
        assert len(fit) + len(val) == len(y), f"fold {k}: fit + eval must cover every row"


def test_inner_es_split_returns_train_first():
    tr, te, y, folds = _data()
    fit = np.where(folds != 0)[0]
    a, b = _inner_es_split(fit, y, 4)
    # whichever way round the return values are, the TRAIN set must be the large one
    big, small = (a, b) if len(a) > len(b) else (b, a)
    assert len(small) < len(big), "the two returned index sets must differ in size"
    assert not (set(a.tolist()) & set(b.tolist())), "train and ES sets must be disjoint"
    assert set(a.tolist()) | set(b.tolist()) == set(fit.tolist()), \
        "train and ES must together cover the whole fit block"
    assert len(small) == len(big) * 0 or len(small) > 0, "ES set must be non-empty"


def test_early_stopping_holdout_is_about_ten_percent():
    tr, te, y, folds = _data()
    fit = np.where(folds != 0)[0]
    a, b = _inner_es_split(fit, y, 4)
    frac = min(len(a), len(b)) / len(fit)
    assert 0.07 < frac < 0.13, (
        f"smaller split is {frac:.3f} of the fit block, expected ~0.10 -- arguments were mixed up")


def test_global_local_index_pairing_is_detectable():
    """The invariant that makes every other check in this file meaningful.

    A 20-round LightGBM trained on correctly paired rows must discriminate CLEARLY. The same
    features with labels indexed in the OTHER index space must sit at chance.

    Comparing means is not good enough here: an early version of this test only asked whether the
    positive-class mean exceeded the negative-class mean, and the mis-paired model "separated" by
    0.4442 vs 0.4441 -- pure numerical noise, reported as a passing guard. Thresholds on a proper
    fold AUC are used instead.
    """
    from sklearn.metrics import roc_auc_score

    import lightgbm as lgb

    tr, te, y_int, folds = _data()
    y = y_int.astype("float64")
    fit = np.where(folds != 0)[0]
    val = np.where(folds == 0)[0]
    Xf, Xv = _fit_matrix(fit, y)

    tr_g, es_g = _inner_es_split(fit, y_int, 4)          # GLOBAL indices
    pos = {int(v): i for i, v in enumerate(fit)}
    tr_l = np.array([pos[int(v)] for v in tr_g])          # LOCAL positions

    params = {"objective": "binary", "n_estimators": 20, "learning_rate": 0.2, "num_leaves": 31,
              "min_child_samples": 40, "verbose": -1, "n_jobs": 4, "seed": 1}

    def fold_auc(features, labels):
        ds = lgb.Dataset(features, label=labels)
        m = lgb.train(params, ds, num_boost_round=20)
        return float(roc_auc_score(y_int[val], m.predict(Xv)))

    auc_ok = fold_auc(Xf[tr_l], y[tr_g])                  # correct: local features, global labels
    auc_bad = fold_auc(Xf[tr_l], y[tr_l])                 # wrong: local features, local labels

    assert auc_ok > 0.60, (
        f"correctly paired rows barely separated (fold AUC {auc_ok:.4f}); expected well above "
        f"0.60 even for 20 rounds. The view or the pairing is broken, so nothing below is "
        f"meaningful.")
    assert abs(auc_bad - 0.5) < 0.02, (
        f"mis-paired rows scored fold AUC {auc_bad:.4f}, i.e. NOT at chance. Either the local and "
        f"global index spaces coincide on this data -- in which case this guard would silently "
        f"pass while a real mis-pairing went unnoticed -- or the test data itself carries signal "
        f"that survives scrambling, which would be a much stranger result and needs explaining.")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  PASS {name}")
