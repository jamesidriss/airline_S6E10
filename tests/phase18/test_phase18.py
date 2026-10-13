"""Phase18 tests: fingerprint determinism, registry duplicate protection,
atomic writes, and error-atlas pair-credit arithmetic (non-vacuous).

Run via tests/run_tests.py (auto-discovered) or directly:
    python tests/phase18/test_phase18.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_fingerprint_deterministic_and_sensitive():
    from src.phase18.fingerprint import canonical_json, fingerprint

    kwargs = dict(
        data_hash="d" * 64, fold_hash="f" * 64,
        feature_config={"blocks": ["raw", "aux10"]},
        model_config={"family": "lgbm", "n_estimators": 512},
        seed=20261010,
        source_entry=ROOT / "src" / "phase18" / "fingerprint.py",
        training_policy={"es": "inner_10pct_fit_only"},
        inference_policy={"aggregation": "equal_logit"},
    )
    a = fingerprint(**kwargs)
    b = fingerprint(**kwargs)
    assert a["fingerprint"] == b["fingerprint"], "identical inputs must fingerprint identically"
    assert len(a["fingerprint"]) == 64

    # sensitivity: each mutated field must change the fingerprint
    for field, value in [
        ("data_hash", "e" * 64), ("fold_hash", "a" * 64), ("seed", 777001),
    ]:
        mutated = dict(kwargs); mutated[field] = value
        assert fingerprint(**mutated)["fingerprint"] != a["fingerprint"], field
    for field, value in [
        ("feature_config", {"blocks": ["raw"]}),
        ("model_config", {"family": "lgbm", "n_estimators": 513}),
        ("training_policy", {"es": "eval_fold"}),
        ("inference_policy", {"aggregation": "mean_prob"}),
    ]:
        mutated = dict(kwargs); mutated[field] = value
        assert fingerprint(**mutated)["fingerprint"] != a["fingerprint"], field

    # canonical json is key-order invariant
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_registry_duplicate_protection_and_statuses():
    from src.phase18 import registry

    fp = "t" * 64
    original = registry.REGISTRY
    try:
        with tempfile.TemporaryDirectory() as td:
            registry.REGISTRY = Path(td) / "reg.jsonl"
            assert registry.check_reuse_allowed(fp) == (True, "no prior record")
            registry.append_record({"exp_id": "x1", "fingerprint": fp, "status": "COMPLETE"})
            allowed, reason = registry.check_reuse_allowed(fp)
            assert not allowed and "COMPLETE" in reason, "valid completed jobs must not retrain"
            registry.append_record({"exp_id": "x2", "fingerprint": fp, "status": "RESOURCE_BLOCKED"})
            allowed, reason = registry.check_reuse_allowed(fp)
            assert allowed and "resume" in reason, "resource-blocked jobs may resume"
            registry.append_record({"exp_id": "x3", "fingerprint": fp, "status": "INVALID"})
            allowed, reason = registry.check_reuse_allowed(fp)
            assert not allowed and "INVALID" in reason, "invalid jobs must not be reused"
            try:
                registry.append_record({"exp_id": "x4", "fingerprint": fp, "status": "MADE_UP"})
                raised = False
            except ValueError:
                raised = True
            assert raised, "unknown statuses must be rejected"
            # every written record round-trips
            assert len(registry.load_all()) == 3
    finally:
        registry.REGISTRY = original


def test_atomic_write_no_partial_state():
    from src.phase18.registry import atomic_write_text

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "out.json"
        atomic_write_text(target, '{"ok": true}')
        assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
        # no temp litter remains
        leftovers = [p for p in target.parent.iterdir() if p.name != "out.json"]
        assert not leftovers, leftovers


def _toy_pair_case():
    """Toy labels/predictions with deliberate ties so tie-handling is exercised."""
    y = np.array([1, 1, 0, 0, 1, 0], dtype=int)
    baseline = np.array([0.9, 0.5, 0.5, 0.1, 0.2, 0.8], dtype=float)
    candidate = np.array([0.9, 0.7, 0.4, 0.1, 0.2, 0.6], dtype=float)
    return y, candidate, baseline


def test_atlas_pair_credit_arithmetic_exact():
    """The per-row credit identity must hold exactly, including ties."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "atlas_mod", ROOT / "scripts" / "phase18" / "build_error_atlas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from sklearn.metrics import roc_auc_score
    from scripts.phase16_common import error_pairs
    from scripts.phase16_pair_metrics import pair_rescue_damage

    rng = np.random.default_rng(4242)
    y = (rng.random(5000) > 0.55).astype(int)
    base = rng.integers(0, 20, size=5000).astype(float) / 20.0  # coarse => many ties
    cand = np.clip(base + rng.normal(0, 0.06, 5000), 0, 1)

    credit = mod.per_row_error_credit(y, cand)
    errors, total = error_pairs(y, cand)
    assert abs(credit.sum() - errors) < 1e-6, (credit.sum(), errors)
    assert total == int(y.sum()) * int((1 - y).sum())

    credit_b = mod.per_row_error_credit(y, base)
    net = (credit_b - credit).sum()
    prd = pair_rescue_damage(y, cand, base)
    assert abs(net - (prd["rescued_pair_credit"] - prd["damaged_pair_credit"])) < 1e-6
    auc_delta = roc_auc_score(y, cand) - roc_auc_score(y, base)
    assert abs(net / total - auc_delta) < 1e-12, (net / total, auc_delta)

    # non-vacuity: swapped roles must mirror rescue/damage
    swapped = pair_rescue_damage(y, base, cand)
    assert swapped["rescued_pair_credit"] == prd["damaged_pair_credit"]

    counts = mod.near_tie_and_confident_counts(y, cand, eps=0.02, conf=0.3)
    assert counts["total_pairs"] == total
    assert 0 <= counts["near_tie_pairs"] <= total
    assert counts["confident_right_pairs"] + counts["confident_wrong_pairs"] <= total


def test_probe_contract_values_match_prereg():
    """The S0 probe must use the frozen certified config, not paraphrases."""
    prereg = json.loads((ROOT / "configs/phase18/h18a_coverage_prereg.json").read_text(encoding="utf-8"))
    frozen = prereg["matched_design"]["frozen_params"]
    assert "random_state=1201" in frozen or "1201" in frozen
    src = (ROOT / "scripts/phase18/probe_h18a_s0.py").read_text(encoding="utf-8")
    assert "SEED = 1201" in src, "probe must use the certified v7 random_state"
    assert "CONTEXT_ROWS = 629_671" in src
    assert "n_estimators\": 1" in src
    import numpy as np
    b = np.load(ROOT / "data/cache/folds/block10_10f_seed20261010.npy")
    assert b.shape[0] - int((b == 0).sum()) == 629_671


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            import traceback
            traceback.print_exc()
    print(f"{len(tests) - failed} PASS / {failed} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
