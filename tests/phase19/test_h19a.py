"""Phase19 H19-A tests: fold safety, difficulty provenance, weight contract,
pair rescue/damage + tie-aware counting, champion identity, prereg-before-results,
and the documentary stretch-gap arithmetic fix.

Run via tests/run_tests.py (registered) or directly:
    python tests/phase19/test_h19a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------- fold safety
def test_fold_registry_immutable_and_hash_checked():
    """Registered fold arrays match their on-disk hashes; labels are 0..n-1."""
    from src.common import CACHE, arr_sha256

    reg = CACHE / "folds" / "registry.json"
    assert reg.exists(), "fold registry missing"
    meta = json.loads(reg.read_text())
    assert "primary" in meta, "primary scheme not registered"
    for name, rec in meta.items():
        arr = np.load(CACHE / "folds" / rec["file"])
        assert set(np.unique(arr)).issubset(set(range(rec["n_splits"]))), name
        if "fold_sha256" in rec:
            assert arr_sha256(arr) == rec["fold_sha256"], f"{name}: fold array tampered"
    # primary must be 5-fold with the frozen seed
    assert meta["primary"]["n_splits"] == 5 and meta["primary"]["seed"] == 20261010


def test_fit_partition_excludes_eval_fold():
    """FIT (folds 1-4) and eval (fold 0) are disjoint; validators catch leakage."""
    from src.phase19.h19a_difficulty import assert_no_fold0, assert_partition

    folds = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    FIT = np.flatnonzero(folds != 0)
    EVAL = np.flatnonzero(folds == 0)
    assert_partition(FIT, EVAL, folds, eval_fold=0)
    assert_no_fold0(FIT, folds, "synthetic FIT")
    # negative: fold-0 row smuggled into FIT must be caught
    bad = np.append(FIT, 0)
    try:
        assert_no_fold0(bad, folds, "leaky FIT")
        raise AssertionError("assert_no_fold0 failed to catch fold-0 row")
    except AssertionError as e:
        assert "leaky" not in str(e) or "fold-0" in str(e)
    # negative: overlapping partition must be caught
    try:
        assert_partition(np.array([0, 2, 3]), np.array([0, 1]), folds, 0)
        raise AssertionError("assert_partition failed to catch overlap")
    except AssertionError:
        pass


def test_difficulty_index_provenance_fold0_safe():
    """The inner-cross-fit index arithmetic never includes fold 0 or the eval fold j."""
    from src.phase19.h19a_difficulty import assert_no_fold0
    from scripts.run_views import _inner_es_split

    n = 500
    rng = np.random.default_rng(0)
    folds = rng.integers(0, 5, size=n)  # synthetic 5 folds
    y = rng.integers(0, 2, size=n)
    FIT = np.flatnonzero(folds != 0)
    for j in (1, 2, 3, 4):
        fi = np.flatnonzero((folds != j) & (folds != 0))  # FIT \ {j}
        va = np.flatnonzero(folds == j)
        assert_no_fold0(fi, folds, f"difficulty fi j={j}")       # no fold 0
        assert not np.any(folds[fi] == j), "self fold leaked into its own fit"
        assert len(np.intersect1d(fi, va)) == 0, "fit/eval overlap in inner split"
        # ES carved from fi only -> never fold 0, never va
        itr, es = _inner_es_split(fi, y, 1201 + j)
        assert_no_fold0(itr, folds, f"difficulty trn j={j}")
        assert_no_fold0(es, folds, f"difficulty es j={j}")
        assert len(np.intersect1d(itr, va)) == 0, "ES used the eval fold"
        assert set(np.concatenate([itr, es])) == set(fi), "ES split must partition fi"
    # FIT/fold-0 partition still holds
    assert np.all(folds[FIT] != 0) and np.all(folds[folds == 0] == 0)


def test_inner_es_split_carves_from_fit_only():
    """_inner_es_split returns a partition of the passed fit rows, no eval rows."""
    from scripts.run_views import _inner_es_split

    folds = np.array([0, 0, 1, 2, 3, 4, 1, 2])
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    FIT = np.flatnonzero(folds != 0)
    trn, es = _inner_es_split(FIT, y, 1)
    assert set(np.concatenate([trn, es])) == set(FIT.tolist()), "not a partition"
    assert len(np.intersect1d(trn, es)) == 0, "trn/es overlap"
    assert not np.any(folds[np.concatenate([trn, es])] == 0), "eval fold in ES split"


# ---------------------------------------------------------- weight contract
def test_weight_function_frozen_contract():
    """weights_from_score: mean 1, bounded, finite, positive, deterministic, monotone."""
    from src.phase19.h19a_difficulty import assert_weight_valid, weights_from_score

    rng = np.random.default_rng(7)
    d = rng.random(1000)
    w, info = weights_from_score(d, lam=0.5, lo=0.25, hi=2.0)
    assert_weight_valid(w, lo=0.25, hi=2.0)
    assert abs(info["mean"] - 1.0) < 1e-9, "mean weight must be exactly 1"
    assert info["lambda"] == 0.5, "prereg lambda changed"
    # deterministic
    w2, _ = weights_from_score(d, lam=0.5, lo=0.25, hi=2.0)
    assert np.array_equal(w, w2), "weight not reproducible"
    # monotone in difficulty: higher score -> higher weight (rank-based)
    order = np.argsort(d)
    assert np.all(np.diff(w[order]) >= -1e-12), "weight not monotone in difficulty"
    # bounds actually enforced even for extreme lambda
    w3, _ = weights_from_score(d, lam=50.0, lo=0.5, hi=1.5)
    assert w3.min() >= 0.5 - 1e-9 and w3.max() <= 1.5 + 1e-9, "clip not enforced"
    # negative cases
    for bad in ([1.0, np.nan], [1.0, -0.1], [2.0, 2.0]):
        try:
            assert_weight_valid(np.array(bad, dtype=float))
            raise AssertionError(f"assert_weight_valid accepted {bad}")
        except AssertionError:
            pass


# ------------------------------------------------- pair rescue/damage + ties
def test_sample_rescue_damage_directional():
    """Treatment-perfect/control-wrong => net>0, corrected>degraded; reversed => net<0."""
    from src.phase19.h19a_difficulty import sample_pair_stats

    # 2 pos (idx 0,1), 2 neg (idx 2,3); champion fails ALL pairs (shared);
    # treatment ranks every pos above every neg; control ranks them inverted.
    y = np.array([1, 1, 0, 0])
    route = np.array([0.1, 0.2, 0.9, 0.8])   # champion wrong on all pairs
    aux = np.array([0.1, 0.2, 0.9, 0.8])
    treat = np.array([0.9, 0.8, 0.2, 0.1])   # perfect
    ctrl = np.array([0.1, 0.2, 0.9, 0.8])    # wrong
    r = sample_pair_stats(y, route, aux, treat, ctrl, n_pairs=20000, seed=5)
    sh = r["shared"]
    assert r["sampled_shared_rate"] > 0.999, "all pairs should be shared-fail"
    assert sh["acc_treat"] > 0.999 and sh["acc_ctrl"] < 0.001, "credits wrong"
    assert sh["net_rate"] > 0.999 and sh["corrected"] > 0 and sh["degraded"] == 0

    # reversed roles: treatment wrong, control perfect => net<0, degraded>corrected
    r2 = sample_pair_stats(y, route, aux, ctrl, treat, n_pairs=20000, seed=5)
    sh2 = r2["shared"]
    assert sh2["net_rate"] < -0.999 and sh2["degraded"] > 0 and sh2["corrected"] == 0


def test_sample_tie_aware_credit_half():
    """An exact tie earns 0.5 credit (AUC convention), and is counted as a tie."""
    from src.phase19.h19a_difficulty import sample_pair_stats

    y = np.array([1, 0])
    route = np.array([0.4, 0.6])   # champion wrong -> shared
    aux = np.array([0.4, 0.6])
    treat = np.array([0.5, 0.5])   # exact tie
    ctrl = np.array([0.8, 0.2])    # control right (pos scores above neg)
    r = sample_pair_stats(y, route, aux, treat, ctrl, n_pairs=500, seed=1)
    sh = r["shared"]
    assert sh["n"] == 500, "single pair sampled every time"
    assert abs(sh["acc_treat"] - 0.5) < 1e-12, "tie must earn 0.5"
    assert sh["ties_treat"] == 500, "tie not counted"
    assert abs(sh["acc_ctrl"] - 1.0) < 1e-12, "control should be perfect"
    assert abs(sh["net_rate"] - (0.5 - 1.0)) < 1e-12


def test_tie_aware_auc_identity_exact():
    """AUC = 1 - (strict_wrong + 0.5*ties)/pairs holds exactly (with ties present)."""
    from sklearn.metrics import roc_auc_score

    from scripts.phase19.build_shared_error_atlas import strict_mis, tie_pairs
    from scipy.stats import rankdata

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=400)
    if y.sum() == 0 or y.sum() == 400:
        y[0], y[1] = 1, 0
    p = rng.integers(0, 5, size=400).astype(float)  # heavy ties
    r = rankdata(p)
    total = int((y == 1).sum() * (y == 0).sum())
    strict = strict_mis(y, r, np.sort(r[y == 1]))
    ties = tie_pairs(y, r)
    auc = roc_auc_score(y, p)
    assert abs(auc - (1.0 - (strict + 0.5 * ties) / total)) < 1e-12, "AUC identity broken"


# ----------------------------------------------------- champion identity
def test_champion_artifacts_unchanged():
    """Stored champion OOF portfolio + route (all 5 folds) match frozen hashes."""
    from src.common import ARTIFACTS, CACHE, arr_sha256

    # the frozen hashes cover the FULL OOF arrays in ORIGINAL ROW ORDER, so
    # scatter each fold's slice into its flatnonzero(folds==k) positions.
    meta = json.loads((CACHE / "folds" / "registry.json").read_text())
    folds = np.load(CACHE / "folds" / meta["primary"]["file"])
    n = len(folds)
    route = np.full(n, np.nan, dtype=np.float64)
    port = np.full(n, np.nan, dtype=np.float64)
    for k in range(5):
        fl = np.flatnonzero(folds == k)
        route[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "prediction.npy")
        port[fl] = np.load(ARTIFACTS / "phase17_B1" / f"f{k}" / "portfolio.npy")
    assert np.isfinite(route).all() and np.isfinite(port).all()
    assert arr_sha256(port.astype("float32")) == \
        "1981745c3d4f5b743c3f02c23f92a1e8975692ca473bc13e3156165bda1a7552", "portfolio drift"
    assert arr_sha256(route.astype("float32")) == \
        "bcb2811329d3686882450e93435647f8f7ab65abfc48c1a1b51db1e9d1ef95e0", "route drift"


# ----------------------------------------------------- prereg-before-results
def test_prereg_committed_before_results_and_hash_frozen():
    """The frozen prereg matches its committed hash (committed ffcb28b before any result)."""
    from src.common import file_sha256
    from scripts.phase19.h19a_fold0_falsifier import PREREG_SHA256

    p = ROOT / "configs" / "phase19" / "h19a_prereg.json"
    assert p.exists(), "prereg missing"
    assert file_sha256(p) == PREREG_SHA256, "prereg changed after freeze"
    # prereg declares it froze before results and names the parent registry commit
    rec = json.loads(p.read_text())
    assert rec["frozen_before_any_result"] is True
    assert rec["parent_registry_commit"] == "b5b83a4a05338228c86c021071a84e08fab5d7dc"
    assert rec["scope"]["only_fold0"] is True and rec["scope"]["no_submission"] is True
    assert rec["scope"]["fit_folds"] == [1, 2, 3, 4] and rec["scope"]["outer_eval_fold"] == 0
    assert "FIT partition" in rec["scope"]["no_folds_1_4"], "no_folds_1_4 note missing"
    # the anti-leakage conclusion + documented deviation are present (not silent)
    assert "per_row_burden.parquet" in json.dumps(rec["anti_leakage_audit"]["forbidden_training_inputs"])
    dev = json.dumps(rec["documented_deviations"])
    assert rec["documented_deviations"][0]["is_silent"] is False, "deviation must be explicit"


# --------------------------------------------------- documentary arithmetic fix
def test_arithmetic_consistency_stretch_gap():
    """Exact stretch gap = 0.962100000 - 0.9618529228157235 = 2.470771842765e-4
    (2.47e-4, NOT 2.47e-5); the 4 corrected docs no longer carry the 10x error."""
    target = 0.962100000
    champion = 0.9618529228157235
    gap = target - champion
    assert abs(gap - 0.0002470771842765) < 1e-15, f"gap {gap!r} != 0.0002470771842765"
    assert abs(gap - 2.470771842765e-4) < 1e-15, "gap != 2.470771842765e-4"
    assert 2.46e-4 < gap < 2.48e-4, "gap must be ~2.47e-4 (2.47e-5 is 10x too small)"

    docs = [
        ROOT / "research" / "phase19_research_queue.md",
        ROOT / "research" / "phase19_compute_budget.json",
        ROOT / "reports" / "phase19" / "experiment_scoreboard.json",
        ROOT / "reports" / "phase19" / "experiment_scoreboard.md",
    ]
    for d in docs:
        txt = d.read_text(encoding="utf-8")
        assert "2.47e-5" not in txt, f"{d.name} still has the 10x-too-small value"
        assert ("2.47e-4" in txt or "2.470771842765e-04" in txt
                or "0.0002470771842765" in txt), f"{d.name} lacks the corrected gap"


# --------------------------------------------------------- feature non-regression
def test_member_uses_certified_rich_view_and_champion_params():
    """The falsifier must train on the certified 'full' view with CHAMPION_PARAMS
    (extra_trees lever) so control/treatment are a clean weight-only A/B."""
    import importlib
    mod = importlib.import_module("scripts.phase19.h19a_fold0_falsifier")
    # certified rich view
    src = (ROOT / "scripts" / "phase19" / "h19a_fold0_falsifier.py").read_text(encoding="utf-8")
    assert 'ViewBuilder(tr, te, "full")' in src, "must use the certified 'full' rich view"
    # champion LightGBM params (the extra_trees lever) present and identical arms
    p = mod.MEMBER_PARAMS
    assert p["extra_trees"] is True, "extra_trees lever missing on rich view"
    assert p["n_estimators"] == 6000 and p["learning_rate"] == 0.02, "champion params changed"
    assert p["num_leaves"] == 127 and p["min_child_samples"] == 40, "champion params changed"
    assert mod.MEMBER_SEED == 20261010 and mod.INNER_ES_SEED == 1
    # control vs treatment differ ONLY in sample_weight (same params/seed/ES)
    assert mod.JUDGE_A["extra_trees"] is True and mod.JUDGE_B["extra_trees"] is False, \
        "judges must be decorrelated"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)
