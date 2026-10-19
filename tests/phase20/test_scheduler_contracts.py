"""Phase 20 scheduler integration contracts (CPU-only; no Kaggle/GPU access)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from scripts.phase20 import scheduler as S
from src.common import arr_sha256


def test_fingerprint_tracks_actual_python_runner_bytes():
    with TemporaryDirectory() as tmp:
        script = Path(tmp) / "worker.py"
        script.write_text("print('first')\n", encoding="utf-8")
        job = {"cmd": [sys.executable, str(script)], "prereg": {"sha256": "dummy"}}
        first = S.fingerprint(job)
        script.write_text("print('changed')\n", encoding="utf-8")
        second = S.fingerprint(job)
        assert first and second and first != second, "fingerprint ignored runner changes"


def test_scheduler_no_timeout_means_no_timeout():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        script = root / "worker.py"
        script.write_text("print('worker successful')\n", encoding="utf-8")
        doc = {"schema": "test", "jobs": {
            "J001": {
                "id": "J001", "exp_row": 1, "family": "TEST",
                "kind": "cpu", "state": "PENDING", "state_note": None,
                "cmd": [sys.executable, str(script)], "prereg": {"sha256": "dummy"},
                "fingerprint": None, "depends_on": [], "attempts": 0,
                "expect": None, "result": {}, "history": [],
            }
        }}
        with patch.object(S, "ROOT", root), \
             patch.object(S, "SCHED", root / "scheduler.json"), \
             patch.object(S, "RUNTIME", root / "runtime.jsonl"), \
             patch.object(S, "GPU_LOCK", root / "gpu.lock"):
            doc["jobs"]["J001"]["fingerprint"] = S.fingerprint(doc["jobs"]["J001"])
            S.save(doc)
            args = argparse.Namespace(job="J001", no_prereg=False, timeout=0, why="test")
            assert S.cmd_run(args) == 0, "omitted --timeout killed a normal job"
            end = S.load()["jobs"]["J001"]
            assert end["state"] == "DONE"
            assert end["attempts"] == 1
            assert any(e.get("event") == "run-start" and e.get("pid", 0) > 0
                       for e in end["history"])
            assert (root / "runtime.jsonl").is_file()


def test_scheduler_rejects_invalid_report_manifest():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "reports").mkdir()
        (root / "preds").mkdir()
        path = root / "preds" / "predict.npy"
        vec = np.array([0.25, 0.75], dtype=np.float32)
        np.save(path, vec)
        rep = {"status": "DONE", "metrics": {"standalone_auc": 1.0},
               "eval_rows": 2, "prediction_sha256": arr_sha256(vec)}
        (root / "reports" / "result.json").write_text(json.dumps(rep), encoding="utf-8")
        job = {"expect": {"report": "reports/result.json",
                          "prediction": "preds/predict.npy", "eval_rows": 2}}
        with patch.object(S, "ROOT", root):
            assert S.verify_expect(job)[0]
            np.save(path, np.array([0.5, 0.5], dtype=np.float32))
            good, why = S.verify_expect(job)
            assert not good and "sha mismatch" in why
            rep["status"] = "FAILED"
            (root / "reports" / "result.json").write_text(json.dumps(rep), encoding="utf-8")
            assert not S.verify_expect(job)[0]


def test_scheduler_live_gpu_pid_lock_on_windows():
    if os.name != "nt":
        return
    with TemporaryDirectory() as tmp:
        lock = Path(tmp) / "gpu.lock"
        with patch.object(S, "GPU_LOCK", lock):
            S.acquire_gpu_lock("J999", os.getpid())
            try:
                busy, reason = S.gpu_busy()
                assert busy and str(os.getpid()) in reason
            finally:
                S.release_gpu_lock("J999")
            assert not lock.exists()


def _write_budget(root: Path, **over) -> None:
    doc = {"active_gpu_budget": {"cap_gpu_hours": 22, "consumed_gpu_hours_estimate": 11.8,
                                 "remaining_gpu_hours": 10.2}}
    doc["active_gpu_budget"].update(over)
    target = root / "research" / "phase20_exhaustive"
    target.mkdir(parents=True, exist_ok=True)
    (target / "campaign_budget.json").write_text(json.dumps(doc), encoding="utf-8")


def test_budget_check_blocks_exhausted_cap():
    """22h ceiling from compute_budget_extension_22h.md: exhausted balance blocks GPU runs."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_budget(root, cap_gpu_hours=12.0, consumed_gpu_hours_estimate=12.0)
        with patch.object(S, "ROOT", root):
            ok, why = S.budget_check(True)
            assert not ok and "exhausted" in why
            # CPU jobs are never budget-gated
            assert S.budget_check(False)[0]


def test_budget_check_allows_within_cap_and_missing_file():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_budget(root)  # 11.8 of 22 consumed
        with patch.object(S, "ROOT", root):
            ok, why = S.budget_check(True)
            assert ok and "10.2" in why or ok and "remaining" in why
    with TemporaryDirectory() as tmp:  # no budget file -> fail-open (sandboxed tests)
        with patch.object(S, "ROOT", Path(tmp)):
            assert S.budget_check(True)[0]


def test_budget_charge_advances_consumed_estimate():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_budget(root)
        with patch.object(S, "ROOT", root):
            S.budget_charge(1800.0)  # half an hour of GPU
            doc = json.loads((root / "research" / "phase20_exhaustive" / "campaign_budget.json").read_text())
            ext = doc["active_gpu_budget"]
            assert abs(ext["consumed_gpu_hours_estimate"] - 12.3) < 1e-6
            assert abs(ext["remaining_gpu_hours"] - 9.7) < 1e-6


def test_budget_check_blocks_unresolved_interrupted_runtime():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_budget(root, unresolved_gpu_runtime={"job": "JINT04", "unknown_tail": True})
        with patch.object(S, "ROOT", root):
            ok, why = S.budget_check(True)
            assert not ok and "unresolved" in why
            assert S.budget_check(False)[0]


def test_budget_check_blocks_corrupt_existing_budget():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_budget(root)
        (root / 'research/phase20_exhaustive/campaign_budget.json').write_text('{broken', encoding='utf-8')
        with patch.object(S, 'ROOT', root):
            ok, why = S.budget_check(True)
            assert not ok and 'unreadable' in why
