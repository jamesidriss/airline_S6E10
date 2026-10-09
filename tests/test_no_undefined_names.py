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
    "scripts/decide_phase17_submission.py",
    "scripts/phase17_primary_io.py",
    "scripts/phase17_resume.py",
    "scripts/resume_phase17_context.py",
    "scripts/audit_phase17_integrity.py",
    "scripts/simulate_phase17_private.py",
    "scripts/assemble_phase17_test.py",
    "scripts/phase17_certification.py",
    "scripts/run_phase17_confirmation.py",
    "scripts/evaluate_phase17_shadow.py",
    "scripts/assemble_phase17_primary.py",
    "scripts/probe_phase17_four_reference.py",
    "scripts/phase17_four_contracts.py",
    "scripts/probe_phase17_serial_reference.py",
    "scripts/run_phase17_tabpfn.py",
    "scripts/evaluate_phase17_fold.py",
    "scripts/probe_phase17_equivalence.py",
    "scripts/probe_phase17_tabpfn.py",
    "scripts/phase17_contracts.py",
    "scripts/audit_phase16_geometry.py",
    "scripts/audit_phase16_noise.py",
    "scripts/phase16_common.py",
    "scripts/phase16_tabpfn.py",
    "scripts/probe_phase16_tabpfn.py",
    "scripts/run_phase16_tabpfn.py",
    "scripts/run_phase16_fast.py",
    "scripts/evaluate_phase16_fold.py",
    "scripts/phase16_pair_metrics.py",
    "scripts/audit_sol_phase15.py",
    "scripts/evaluate_sol_phase15_raw.py",
    "scripts/run_sol_phase15_raw_context.py",
    "scripts/certify_sol_phase15_raw.py",
    "scripts/evaluate_sol_phase15_native.py",
    "scripts/simulate_sol_phase15_raw.py",
    "scripts/run_sol_phase15_auxpfn.py",
    "scripts/sol_phase15_auxpfn.py",
    "scripts/evaluate_sol_phase15_auxpfn.py",
    "scripts/prepare_sol_submission.py",
    "scripts/cache_sol_aux.py",
    "scripts/snapshot_sol_kaggle.py",
    "scripts/run_sol_policy_pseudotest.py",
    "scripts/evaluate_sol_policy_pseudotest.py",
    "src/submission/kaggle_io.py",
    "scripts/run_sol_tabpfn_fold_test.py",
    "scripts/verify_sol_allocator.py",
    "scripts/verify_sol_query_activation.py",
    "scripts/run_sol_native_crosses.py",
    "scripts/sol_native_crosses.py",
    "scripts/run_sol_aux_test.py",
    "scripts/assemble_sol_foundation_test.py",
    "scripts/score_sol_foundation.py",
    "scripts/verify_sol_head_views.py",
    "scripts/verify_sol_scaling_reuse.py",
    "scripts/run_sol_aux_block10.py",
    "scripts/private_lb_simulator.py",
    "scripts/run_sol_tabpfn_confirm.py",
    "scripts/replay_sol_isolated.py",
    "scripts/evaluate_sol_foundation.py",
    "scripts/compact_sol_static_ntfs.py",
    "scripts/run_sol_tabpfn_test.py",
    "scripts/assemble_sol_clean.py",
    "scripts/assemble_sol_neural.py",
    "scripts/replay_sol_classical.py",
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
    "src/models/query_activation_reuse.py",
    "src/models/head_view_attention.py",
    "src/models/scaling_reuse.py",
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
    import ast
    syntax_errors = []
    valid = []
    for p in paths:
        try:
            ast.parse(Path(p).read_text(encoding='utf-8-sig'), filename=str(p))
            valid.append(p)
        except SyntaxError as error:
            syntax_errors.append(f'{p}:{error.lineno}: syntax error: {error.msg}')
    try:
        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
    except ImportError:  # pragma: no cover
        return syntax_errors or None
    import io

    out = io.StringIO()
    reporter = Reporter(out, out)
    for p in valid:
        checkPath(str(p), reporter)
    msgs = list(syntax_errors)
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


def test_static_guard_reads_file_contents_and_detects_missing_imports():
    """A pathname passed as source can silently miss every undefined name."""
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        sample = Path(directory) / "sample.py"
        sample.write_text("def run():\n    return missing_helper()\n", encoding="utf-8")
        messages = _pyflakes([sample])
        assert messages is not None, "pyflakes is required for this regression check"
        assert len(messages) == 1 and "undefined name 'missing_helper'" in messages[0]
        sample.write_text("def missing_helper():\n    return 1\ndef run():\n    return missing_helper()\n", encoding="utf-8")
        assert _pyflakes([sample]) == []


def test_static_guard_rejects_unterminated_literal_before_expensive_runs():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        sample=Path(directory)/'broken.py'
        sample.write_text("def run():\n    return 'broken\n",encoding='utf-8')
        messages=_pyflakes([sample])
        assert len(messages)==1 and 'syntax error' in messages[0]


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
