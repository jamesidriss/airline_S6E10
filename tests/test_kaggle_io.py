"""Submission identity, durable budget and score polling without network calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.submission import kaggle_io as K


def _submission(ref, file, status, score=""):
    return SimpleNamespace(ref=ref, file_name=file, status=SimpleNamespace(name=status),
                           public_score=score, private_score="")


def test_submit_preserves_numeric_ref_and_reserves_before_upload():
    with TemporaryDirectory() as directory, patch.object(K, "LEDGER", Path(directory)/"ledger.csv"):
        file = Path(directory)/"new.csv"
        file.write_text("id,satisfaction\n1,0.5\n", encoding="utf-8")
        api = Mock()
        api.competition_get_submission_limits.return_value.num_allowed_now = 10
        def upload(*args, **kwargs):
            assert K.used_today() == 1
            assert K._load().iloc[-1]["status"] == "upload_started_response_unknown"
            return SimpleNamespace(ref=56999999)
        api.competition_submit.side_effect = upload
        with patch.object(K, "_api", return_value=api):
            assert K.submit(file, "frozen candidate") == "56999999"
        row = K._load().iloc[-1]
        assert row["ref"] == "56999999" and row["status"] == "submitted"
        assert K.used_today() == 1


def test_ambiguous_upload_still_consumes_budget_and_is_never_retried():
    with TemporaryDirectory() as directory, patch.object(K, "LEDGER", Path(directory)/"ledger.csv"):
        file = Path(directory)/"new.csv"
        file.write_text("x", encoding="utf-8")
        api = Mock()
        api.competition_get_submission_limits.return_value.num_allowed_now = 10
        api.competition_submit.side_effect = TimeoutError("uncertain network response")
        with patch.object(K, "_api", return_value=api):
            try:
                K.submit(file, "frozen candidate")
            except TimeoutError:
                pass
            else:
                raise AssertionError("Uncertain upload was accepted")
        assert K.used_today() == 1
        assert K._load().iloc[-1]["status"] == "upload_started_response_unknown"
        assert api.competition_submit.call_count == 1


def test_submission_cap_refuses_before_any_api_access():
    with TemporaryDirectory() as directory, patch.object(K, "LEDGER", Path(directory)/"ledger.csv"):
        for i in range(K.CAP):
            K.record(f"{i}.csv", "reserved")
        with patch.object(K, "_api") as api:
            try:
                K.submit(Path(directory)/"new.csv", "candidate")
            except SystemExit:
                pass
            else:
                raise AssertionError("Daily cap was bypassed")
            api.assert_not_called()


def test_poll_ignores_old_complete_submission_and_persists_exact_score():
    with TemporaryDirectory() as directory, patch.object(K, "LEDGER", Path(directory)/"ledger.csv"):
        K.record("old.csv", "older", ref="100", status="complete", public=.9)
        K.record("new.csv", "candidate", ref="200")
        api = Mock()
        old = _submission(100, "old.csv", "COMPLETE", ".9")
        api.competition_submissions.side_effect = [
            [old, _submission(200, "new.csv", "PENDING")],
            [old, _submission(200, "new.csv", "COMPLETE", ".9617")],
        ]
        with patch.object(K, "_api", return_value=api):
            result = K.poll("new.csv", tries=2, wait=0)
        assert api.competition_submissions.call_count == 2
        assert result.iloc[-1]["status"] == "complete"
        assert result.iloc[-1]["publicScore"] == .9617
        assert K._load().iloc[-1]["publicScore"] == .9617
        assert result.iloc[0]["publicScore"] == .9


def test_poll_rejects_ref_filename_disagreement():
    with TemporaryDirectory() as directory, patch.object(K, "LEDGER", Path(directory)/"ledger.csv"):
        K.record("new.csv", "candidate", ref="200")
        api = Mock()
        api.competition_submissions.return_value = [_submission(200, "other.csv", "COMPLETE", ".9")]
        with patch.object(K, "_api", return_value=api):
            try:
                K.poll("new.csv", tries=1, wait=0)
            except ValueError as error:
                assert "filename disagree" in str(error)
            else:
                raise AssertionError("Mismatched submission was accepted")


def test_remote_zero_quota_does_not_upload_or_reserve():
    with TemporaryDirectory() as directory, patch.object(K, "LEDGER", Path(directory)/"ledger.csv"):
        file = Path(directory)/"new.csv"
        file.write_text("x", encoding="utf-8")
        api = Mock()
        api.competition_get_submission_limits.return_value.num_allowed_now = 0
        with patch.object(K, "_api", return_value=api):
            try:
                K.submit(file, "candidate")
            except SystemExit:
                pass
            else:
                raise AssertionError("Remote quota was bypassed")
        api.competition_submit.assert_not_called()
        assert K.used_today() == 0
