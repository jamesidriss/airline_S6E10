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


def test_neural_twins_preserve_values_across_split_vocabularies():
    from src.models.realmlp import _twin_frame, build_twin_frame
    train = np.array([[10., 100.], [20., 200.], [30., 300.]])
    apply = np.array([[20., 200.], [30., 900.]])
    names = ["Age", "Flight Distance"]
    a, b = _twin_frame(train, names), _twin_frame(apply, names)
    assert a.loc[1, "Age__tw"] == b.loc[0, "Age__tw"] == 20
    assert a.loc[1, "Flight Distance__tw"] == b.loc[0, "Flight Distance__tw"] == 200
    assert b.loc[1, "Flight Distance__tw"] == 900
    built, cats = build_twin_frame(apply, names)
    assert built.equals(b) and cats == [2, 3]


def test_all_neural_entry_points_isolate_outer_labels_and_respect_no_twin():
    from src.models import realmlp as RM
    folds = np.repeat(np.arange(5), 40)
    y = np.tile([0, 1], 100)
    X = np.column_stack([1000 + np.arange(200), np.zeros(200)])
    Xte = np.array([[2000., 0.], [2001., 0.]])
    names = ["Flight Distance", "signal"]
    calls = []

    class Model:
        def __init__(self, **params):
            self.params = params

        def fit(self, frame, labels, X_val, y_val):
            calls.append((frame.copy(), np.array(labels), X_val.copy(), np.array(y_val), self.params))

        def predict_proba(self, frame):
            return np.full((len(frame), 2), .5)

    class Builder:
        def assemble(self, fit, labels, val, test, inner_seed):
            return X[fit], {"val": X[val], "test": Xte}, names

    module = SimpleNamespace(RealMLP_TD_Classifier=Model, TabM_D_Classifier=Model)
    with patch.dict(sys.modules, {"pytabkit": module}):
        for family in ("array", "realmlp", "tabm"):
            for twin in (True, False) if family != "array" else (False,):
                def run(labels):
                    calls.clear()
                    if family == "array":
                        RM.realmlp(X, labels, Xte, folds, names, params={"random_seed": 17})
                    else:
                        fn = RM.realmlp_view if family == "realmlp" else RM.tabm_view
                        fn(Builder(), folds, labels, 200, 2, twin=twin, params={"random_seed": 17})
                    return [(a.copy(), b.copy(), c.copy(), d.copy(), p) for a, b, c, d, p in calls]

                original = run(y)
                changed = y.copy()
                changed[folds == 0] = 1 - changed[folds == 0]
                flipped = run(changed)
                # Flipping only evaluation labels cannot alter fold0 fit or ES inputs.
                for idx in (0, 1, 2, 3):
                    assert np.array_equal(original[0][idx], flipped[0][idx])
                for k, (train, labels, es, es_labels, params) in enumerate(original):
                    fit_ids = set(X[folds != k, 0])
                    train_ids, es_ids = set(train.iloc[:, 0]), set(es.iloc[:, 0])
                    assert not train_ids.intersection(es_ids)
                    assert train_ids | es_ids == fit_ids
                    assert len(train) == 144 and len(es) == 16
                    assert params["random_state"] == 17
                    assert ("Flight Distance__tw" in train) == twin


def test_generic_gbdt_validation_never_receives_outer_rows():
    from src.models import gbdt
    X, y, folds = np.arange(200.)[:, None], np.tile([0, 1], 100), np.repeat(np.arange(5), 40)
    calls = []

    class Pool:
        def __init__(self, data, label=None, **kwargs):
            self.data, self.label = data, label

    class Model:
        best_iteration = 3

        def __init__(self, **kwargs):
            pass

        def fit(self, data, label=None, eval_set=None, **kwargs):
            if isinstance(data, Pool):
                calls.append((data.data, data.label, eval_set.data, eval_set.label))
            else:
                calls.append((data, label, eval_set[0][0], eval_set[0][1]))

        def predict_proba(self, data):
            return np.full((len(data.data if isinstance(data, Pool) else data), 2), .5)

        def predict(self, data, **kwargs):
            return np.full(len(data), .5)

        def get_best_iteration(self):
            return 3

    def train(params, dataset, **kwargs):
        validation = kwargs['valid_sets'][0]
        calls.append((dataset.data, dataset.label, validation.data, validation.label))
        return Model()

    modules = {'lightgbm': SimpleNamespace(Dataset=Pool, train=train, early_stopping=lambda *a, **k: None),
               'xgboost': SimpleNamespace(XGBClassifier=Model),
               'catboost': SimpleNamespace(CatBoostClassifier=Model, Pool=Pool)}
    with patch.dict(sys.modules, modules):
        for fn in (gbdt.lgbm, gbdt.xgboost, gbdt.catboost):
            calls.clear()
            fn(X, y, X[:2], folds)
            for k, (fit, fit_y, es, es_y) in enumerate(calls):
                assert not set(fit[:, 0]).intersection(es[:, 0])
                assert set(fit[:, 0]) | set(es[:, 0]) == set(X[folds != k, 0])
                assert np.array_equal(fit_y, y[fit[:, 0].astype(int)])
                assert np.array_equal(es_y, y[es[:, 0].astype(int)])


def test_static_cache_aliases_only_identical_ordered_label_free_blocks():
    from src.features.view import VIEWS, static_cache_view
    dynamic = {'te', 'te_all21', 'te_cond'}
    for view, blocks in VIEWS.items():
        canonical = static_cache_view(view)
        assert [b for b in blocks if b not in dynamic] == [b for b in VIEWS[canonical] if b not in dynamic]
    for view in ('full_te21', 'full_all21te', 'full_tec', 'full_tec_swap'):
        assert static_cache_view(view) == 'full'
    for view in ('core3_te', 'core3_tec', 'core3_tec_swap'):
        assert static_cache_view(view) == 'core3'


def test_array_hash_buffer_path_preserves_historical_digests():
    import hashlib
    from src.common import arr_sha256
    examples = [np.arange(30, dtype='float32').reshape(5, 6),
                np.arange(30, dtype='int64').reshape(5, 6)[:, ::2],
                np.asfortranarray(np.arange(30, dtype='float64').reshape(5, 6)),
                np.array([1, 2, 3], dtype='>i4'), np.empty((0, 4), dtype='float32')]
    for example in examples:
        canonical = np.ascontiguousarray(example)
        old = hashlib.sha256(str(canonical.dtype).encode() + str(canonical.shape).encode()
                             + canonical.tobytes()).hexdigest()
        assert arr_sha256(example) == old


def test_banked_predictions_cannot_be_replaced_or_lose_test_sidecar():
    import tempfile
    from pathlib import Path
    from src.submission import store
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / 'artifacts' / 'predictions'
        with patch.object(store, 'PREDICTIONS', root), patch.object(store, 'INDEX', root / 'index.json'):
            oof, test = np.array([.1, .2, .3]), np.array([.4, .5])
            saved = store.save('banked', oof, test)
            assert store.save('banked', oof, None) == saved
            for a, b in ((oof + .1, test), (oof, test + .1)):
                try:
                    store.save('banked', a, b)
                except FileExistsError:
                    pass
                else:
                    raise AssertionError('Banked prediction was replaced')
            assert np.array_equal(store.load_oof('banked'), oof.astype('float32'))
            assert np.array_equal(store.load_test('banked'), test.astype('float32'))


def test_banked_submission_cannot_be_overwritten():
    import tempfile
    from pathlib import Path
    from src.submission import make
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sample, test = root / 'sample.csv', root / 'test.csv'
        pd.DataFrame({'id': [10, 11, 12], 'satisfaction': [0., 0., 0.]}).to_csv(sample, index=False)
        pd.DataFrame({'id': [10, 11, 12]}).to_csv(test, index=False)
        with patch.object(make, 'SAMPLE_CSV', sample), patch.object(make, 'TEST_CSV', test), \
             patch.object(make, 'SUBMISSIONS', root), patch.object(make, 'MANIFEST', root / 'manifest.csv'):
            path = make.build(np.array([.1, .2, .3]), 'banked')
            original = path.read_bytes()
            try:
                make.build(np.array([.7, .8, .9]), 'banked')
            except FileExistsError:
                pass
            else:
                raise AssertionError('Banked submission was overwritten')
            assert path.read_bytes() == original and len(pd.read_csv(make.MANIFEST)) == 1
def test_full_refit_preserves_early_stop_estimator_configuration():
    """Changing label population/tree count must preserve the model recipe."""
    import numpy as np
    from unittest.mock import patch
    import scripts.run_views as runner
    seen = {}

    class Model:
        best_iteration = 123
        def get_best_iteration(self):
            return self.best_iteration
        def __init__(self, **params):
            seen['refit'] = params
        def fit(self, *args, **kwargs):
            return self
        def predict_proba(self, x):
            return np.tile([.4, .6], (len(x), 1))
        def predict(self, x, **kwargs):
            return np.full(len(x), .6)

    def train(params, *args, **kwargs):
        seen['selection'] = params
        return Model()

    x, y = np.zeros((10, 2)), np.arange(10) % 2
    for overrides in ({}, {'extra_trees': True, 'min_child_samples': 80, 'reg_lambda': 3.0}):
        with patch('lightgbm.Dataset'), patch('lightgbm.train', side_effect=train), patch('lightgbm.LGBMClassifier', Model):
            runner._fit_lgbm_es(x, y, x, overrides, 7, x, y)
            runner._fit_full_predict_lgbm(x, y, x, overrides, 7, 123)
        a, b = dict(seen['selection']), dict(seen['refit'])
        a.pop('n_estimators'); b.pop('n_estimators')
        assert a == b, 'Selection and full refit changed the model configuration'
    with patch('catboost.CatBoostClassifier', Model):
        runner._fit_cat_es(x, y, x, {}, 7, x, y)
        selected = dict(seen['refit'])
        runner._fit_full_predict_cat(x, y, x, {}, 7, 124)
        refit = dict(seen['refit'])
    selected.pop('iterations'); refit.pop('iterations')
    assert selected == refit, 'CatBoost refit changed defaults'


def test_zero_based_best_iteration_zero_refits_one_tree():
    import numpy as np
    from unittest.mock import patch
    import scripts.run_views as runner
    import scripts.run_zoo as zoo
    class Model:
        best_iteration = 0
        def __init__(self, **params):
            pass
        def fit(self, *args, **kwargs):
            return self
        def predict_proba(self, x):
            return np.tile([.4, .6], (len(x), 1))
    x, y = np.zeros((10, 2)), np.arange(10) % 2
    with patch('xgboost.XGBClassifier', Model):
        _, best = runner._fit_xgb_es(x, y, x, {}, 1, x, y)
    assert best == 0, 'First selected tree was replaced by the maximum training budget'
    for family, function in [('xgb', '_fit_full_predict_xgb'), ('cat', '_fit_full_predict_cat')]:
        with patch.object(zoo, function, return_value=np.full(10, .6)) as fit:
            zoo.predict_test(family, x, x, y, np.arange(10), {}, 1, best)
        assert fit.call_args.args[-1] == 1, 'A zero-based index was used as the tree count'


def test_native_cat_zero_based_first_tree_is_preserved():
    import numpy as np
    from unittest.mock import patch
    from scripts.run_phase14 import fit_cat
    class Model:
        def __init__(self, **params):
            pass
        def fit(self, *args, **kwargs):
            return self
        def get_best_iteration(self):
            return 0
        def predict_proba(self, x):
            return np.tile([.4, .6], (len(x), 1))
    x, y = np.zeros((10, 2)), np.arange(10) % 2
    with patch('catboost.CatBoostClassifier', Model):
        _, best = fit_cat(x, y, x, 1, {}, [], x, y)
    assert best == 0, 'Native CatBoost first selected tree was replaced by the budget'


def test_clean_classical_role_plan_is_frozen_and_retains_original_schemes():
    from scripts.replay_sol_classical import frozen_roles,resolved_params
    from src.submission import store
    roles=frozen_roles()
    index=store._load_index()
    assert len(roles)==47 and sum(r['aux_arm'] is not None for r in roles)==10
    assert len({r['member'] for r in roles})==47
    for role in roles:
        assert role['scheme']==index[role['member']]['fold_scheme']
        p=resolved_params(role)
        if p.get('extra_trees'):
            assert role['view'] in ('full','full_ogsurf','full_ogm','core3')
        if role['family']=='lgbm':
            assert p['bagging_seed']==role['seed']+1 and p['feature_fraction_seed']==role['seed']+2
        if role['aux_arm']:
            assert role['scheme']=='primary' and role['es_seed_policy']=='1'


def test_clean_portfolio_refuses_missing_test_predictions_before_loading_data():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import scripts.assemble_sol_clean as assembler
    with tempfile.TemporaryDirectory() as temporary:
        with patch.object(assembler, 'REPORTS', Path(temporary)), \
             patch.object(assembler, 'frozen_plan', return_value={'members': [{'store_id': 'new'}]}), \
             patch.object(assembler.store, '_load_index', return_value={'new': {'oof': 'anything'}}), \
             patch.object(assembler, 'load_cached_parquet') as load, \
             patch.object(assembler.store, 'save') as save, \
             patch('sys.argv', ['assemble_sol_clean.py']):
            try:
                assembler.main()
            except RuntimeError as error:
                assert 'complete OOF/test members are missing' in str(error)
            else:
                raise AssertionError('Partial portfolio was accepted')
            load.assert_not_called()
            save.assert_not_called()


def test_prediction_store_preserves_unindexed_test_only_artifact():
    import tempfile
    from pathlib import Path
    import numpy as np
    from unittest.mock import patch
    from src.submission import store
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        test_path = folder/'new_test.npy'
        np.save(test_path, np.array([.1, .2], dtype='float32'))
        before = test_path.read_bytes()
        with patch.object(store, 'PREDICTIONS', folder), patch.object(store, 'INDEX', folder/'index.json'):
            try:
                store.save('new', np.array([.3, .4]), np.array([.5, .6]))
            except FileExistsError:
                pass
            else:
                raise AssertionError('Unindexed test-only artifact was overwritten')
        assert test_path.read_bytes() == before
        assert not (folder/'new_oof.npy').exists()


def test_shadow_auxiliary_features_never_reuse_primary_fold_arrays():
    import numpy as np
    from unittest.mock import patch
    import scripts.replay_sol_classical as runner
    names=list(runner.RATINGS)
    fit=np.zeros((8,13)); val=np.zeros((2,13))
    pf=np.zeros((8,13,6)); pv=np.zeros((2,13,6))
    pf[:,:,3]=1; pv[:,:,3]=1
    cache={'fingerprint':'shadow','contract':{}}
    with patch.object(runner,'probability_cache',return_value=(pf,pv,cache)), \
         patch.object(runner.np,'load',side_effect=AssertionError('Primary fold data was accessed')):
        a,b,contract=runner.add_expected_values(fit,val,names,np.arange(8),np.arange(8,10),'shadow_hash',0,'shadow')
    assert a.shape==(8,26) and b.shape==(2,26)
    assert np.all(a[:,13:]==3) and np.all(b[:,13:]==3)
    assert contract['independent_ev_max_gap'] is None and contract['upstream_fingerprint']=='shadow'

