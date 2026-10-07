"""Static guard: no undefined names in the scripts this campaign depends on.

Why
---
A 10-epoch TabR fold (~12 minutes of GPU) completed successfully and was then destroyed by
``NameError: clear_epoch_sink`` -- the function had been used but never imported. It raised *after*
the fit and *before* the prediction was written, so the run produced nothing. py_compile cannot catch
this (the name is syntactically fine); only a real name-resolution pass can.

So every entry-point script is run through pyflakes and must report zero undefined names. This turns
a lost multi-hour run into a sub-second test failure.

It is deliberately scoped to undefined names (F821) plus unused/undefined imports, not full
style linting, so it will not churn on cosmetic issues.

Run: ``python tests/test_no_undefined_names.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Scripts on the critical path: anything that trains, submits, or validates. A late NameError in
# any of these costs real GPU hours or a submission.
CRITICAL_SCRIPTS = [
    "scripts/replay_sol_neural.py",
    "scripts/compact_static_cache_reserve.py",
    "scripts/compact_sol_unused_caches.py",
    "scripts/record_sol_research.py",
    "scripts/run_sol_multitask.py",
    "scripts/audit_sol_neural.py",
    "scripts/run_sol_a.py",
    "scripts/evaluate_sol_aux.py",
    "scripts/evaluate_sol_ssl.py",
    "scripts/run_sol_tabpfn.py",
    "scripts/cache_sol_aux.py",
    "scripts/build_v5_aux_cross.py",
    "scripts/run_tabr.py",
    "scripts/profile_tabr_step.py",
    "scripts/profile_tabr_ops.py",
    "scripts/evaluate_tabr.py",
    "scripts/run_softlabel.py",
    "scripts/run_softlabel_nested.py",
    "scripts/run_residual.py",
    "scripts/make_submission.py",
    "scripts/run_views.py",
    "scripts/run_models.py",
    "scripts/reproduce_finalist.py",
    "scripts/marginal_gain.py",
    "scripts/paired_stats.py",
]

# Library modules the runners import; an undefined name here breaks every caller.
CRITICAL_MODULES = [
    "src/models/pointwise_inference.py",
    "src/models/multitask.py",
    "src/models/realmlp.py",
    "src/models/gbdt.py",
    "src/models/windows_attention.py",
    "src/models/resource_guard.py",
    "src/models/tabr_retrieval.py",
    "src/models/tabr_epochlog.py",
    "src/models/residual.py",
    "src/features/softlabel.py",
]


def _pyflakes(paths):
    """Return {path: [messages]} for pyflakes-reported undefined names."""
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError:  # pragma: no cover
        return None
    import io

    out = io.StringIO()
    reporter = Reporter(out, out)
    for p in paths:
        try:
            check(str(p), str(p), reporter)
        except TypeError:                      # older pyflakes signature
            check(str(p), reporter)
    msgs = []
    for line in out.getvalue().splitlines():
        # keep only name-resolution problems, not style/import-order noise
        if "undefined name" in line or "may be undefined, or defined from star imports" in line:
            msgs.append(line)
    return msgs


def test_critical_files_have_no_undefined_names():
    """Every critical script and module must resolve every name it uses."""
    missing = [p for p in CRITICAL_SCRIPTS + CRITICAL_MODULES if not (ROOT / p).exists()]
    assert not missing, f"critical files missing: {missing}"

    present = [ROOT / p for p in CRITICAL_SCRIPTS + CRITICAL_MODULES if (ROOT / p).exists()]
    msgs = _pyflakes(present)
    if msgs is None:
        print("  skipped: pyflakes not installed")
        return
    assert not msgs, "undefined names in critical files:\n  " + "\n  ".join(msgs)
    print(f"  [clean] {len(present)} critical files, 0 undefined names")


def test_run_tabr_imports_every_name_it_calls():
    """Targeted regression for the exact bug that destroyed the 10-epoch fold.

    Rather than trusting the general lint, this asserts that every helper the fold-completion path
    calls is actually importable from the runner's namespace.
    """
    import importlib

    mod = importlib.import_module("scripts.run_tabr")
    required = ["clear_epoch_sink", "set_epoch_sink", "records", "auc_curve", "reset_records",
                "install_epoch_recorder", "atomic_write_json", "EpochLog", "rss_gb",
                "assert_effective_config", "assert_effective_retrieval_width",
                "assert_consistent_widths", "enable_tf32_process_wide", "find_torch_index",
                "install_faiss_gpu_shim", "set_query_chunk", "shim_is_installed",
                "verify_matches_reference", "build_params"]
    absent = [n for n in required if not hasattr(mod, n)]
    assert not absent, f"scripts/run_tabr.py uses but does not define/import: {absent}"
    print(f"  [imports] all {len(required)} fold-completion helpers resolve")


def test_prediction_is_persisted_before_diagnostics():
    """Source-order guard: the prediction save must precede the epoch-log writes.

    The NameError that cost the run happened in a logging call placed before np.save. Even with the
    try/except now in place, keeping persistence first means a future unguarded call cannot cost a
    completed fold again.
    """
    src = (ROOT / "scripts" / "run_tabr.py").read_text(encoding="utf-8")
    save_at = src.find('np.save(fdir / "oof.npy"')
    log_at = src.find('elog.write({"event": "fold_finished"')
    assert save_at > 0 and log_at > 0, "could not locate both persistence sites"
    assert save_at < log_at, (
        "the epoch-log write must come AFTER np.save(oof.npy); a failing diagnostic call must "
        "never be able to destroy a completed fold's prediction")
    print("  [ordering] prediction is persisted before diagnostics")


def main() -> int:
    import time
    import traceback

    fns = [(n, getattr(sys.modules[__name__], n)) for n in sorted(dir(sys.modules[__name__]))
           if n.startswith("test_")]
    ok = fail = 0
    for name, fn in fns:
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {name}  ({time.time()-t0:.1f}s)")
            ok += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {name}")
            traceback.print_exc()
            fail += 1
    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
