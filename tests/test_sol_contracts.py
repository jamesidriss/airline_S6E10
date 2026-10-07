import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features.aux_distribution import signatures


def test_aux_fit_and_apply_rows_are_actually_unseen():
    from scripts.run_phase13 import RATINGS, build_aux
    names = RATINGS + ["Gender", "Customer Type", "Type of Travel", "Class", "Age", "Flight Distance",
                      "Departure Delay in Minutes", "Arrival Delay in Minutes", "te_poison"]
    rng = np.random.default_rng(3)
    X = rng.integers(0, 6, (120, len(names))).astype(float)
    X[:, names.index("Age")] = np.arange(120) + 10000
    X[:, -1] = 1e9  # target-dependent column MUST never reach the aux models
    calls = []

    class Dataset:
        def __init__(self, data, label):
            assert data.shape[1] == 20, "own rating or extra feature reached the design"
            assert data.max() < 1e9, "target-encoded poison reached aux design"
            self.ids = set(data.max(axis=1).tolist())

    def train(params, dataset, num_boost_round):
        def predict(data):
            assert not dataset.ids.intersection(data.max(axis=1).tolist()), "prediction used a training row"
            calls.append(len(data))
            return np.full((len(data), 6), 1 / 6)
        return SimpleNamespace(predict=predict)

    with patch.dict(sys.modules, {"lightgbm": SimpleNamespace(Dataset=Dataset, train=train)}):
        f, a, _ = build_aux(X[:90], X[90:], names, 1, rounds=2, inner_folds=3,
                            return_probabilities=True)
    assert f.shape == (90, 13, 6) and a.shape == (30, 13, 6)
    assert len(calls) == 13 * 4


def test_aux_distribution_arithmetic_and_invalid_inputs():
    p = np.zeros((2, 13, 6))
    p[..., 2], p[..., 4] = .25, .75
    ratings = np.full((2, 13), 4)
    a, names = signatures(p, ratings, 3)
    for key, expected in (("ev", 3.5), ("residual", .5), ("abs_residual", .5),
                           ("p_observed", .75), ("surprisal", -np.log(.75)), ("variance", .75)):
        assert np.allclose(a[:, names.index(f"aux_{key}_0")], expected)
    try:
        signatures(p * 2, ratings, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("bad probability simplex accepted")


def test_fold_registry_rejects_changed_bytes_ids_and_force(tmp_path=None):
    import tempfile
    from pathlib import Path
    from src.validation import folds as F
    with tempfile.TemporaryDirectory() as d, patch.object(F, "REGISTRY_PATH", Path(d)):
        y, ids = np.tile([0, 1], 50).astype("int8"), pd.Series(np.arange(100))
        f = F.get_scheme("unit", y, ids, n_splits=5, seed=3)
        for mutation in ("ids", "bytes", "force"):
            try:
                if mutation == "ids":
                    F.get_scheme("unit", y, ids.iloc[::-1])
                elif mutation == "force":
                    F.get_scheme("unit", y, ids, n_splits=5, seed=3, force=True)
                else:
                    registry = json.loads((Path(d) / "registry.json").read_text())
                    changed = f.folds.copy()
                    changed[0] = (changed[0] + 1) % 5
                    np.save(Path(d) / registry["unit"]["file"], changed)
                    F.get_scheme("unit", y, ids)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError(f"fold mutation accepted: {mutation}")


def test_v5_refit_preserves_all_six_xt_configs_and_seeds():
    from scripts.build_v5_aux_cross import XT_SLOTS, xt_refit_params
    from scripts.run_phase13b import SLOTS, SLOT_SEEDS
    for member in XT_SLOTS:
        p = xt_refit_params(member, 123)
        assert p["random_state"] == SLOT_SEEDS[member]
        assert p["n_estimators"] == 123
        for key, value in SLOTS[member].items():
            assert p[key] == value
    assert len({xt_refit_params(m, 123)["random_state"] for m in XT_SLOTS}) == 6


def test_v5_iterations_require_exact_five_treatment_folds():
    import tempfile
    from pathlib import Path
    from scripts.build_v5_aux_cross import XT_SLOTS, XGB_ARM, CAT_MEMBER, median_ints
    arms = {s: s for s in XT_SLOTS} | XGB_ARM | {CAT_MEMBER: "C1"}
    with tempfile.TemporaryDirectory() as d:
        for member, arm in arms.items():
            for fold in range(5):
                rec = {"n_trees": 100 + fold, "contract": {"fold": fold, "stage": 0, "spec": {"member": member}}}
                (Path(d) / f"{arm}_A0_f{fold}.json").write_text(json.dumps(rec))
        assert set(median_ints(d).values()) == {102}
        (Path(d) / "X0_A0_f0.json").unlink()
        try:
            median_ints(d)
        except SystemExit:
            pass
        else:
            raise AssertionError("missing fold 0 was silently replaced by duplicate fold counts")
