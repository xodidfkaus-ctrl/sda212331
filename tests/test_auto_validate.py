"""
tests/test_auto_validate.py

Unit and integration tests for the pre-registration order check in auto_validate.py.

These tests exist to ensure future contributors cannot bypass Rule R1
("PLAN.md must be committed to git before results.jsonl is written") by
accidentally breaking or removing the enforcement logic.

Run with:
    cd /home/elicer/sda212331
    pytest tests/test_auto_validate.py -v
"""

import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nope_analysis.analysis.auto_validate import (
    RegistrationOrderError,
    _check_preregistration_order,
    _get_plan_commit_timestamp,
    validate_experiment,
)

# ── Timestamp fixtures ────────────────────────────────────────────────────────

T_OLD = datetime(2026, 4, 24, 10, 0, 0, tzinfo=timezone.utc)   # plan committed
T_NEW = datetime(2026, 4, 25, 10, 0, 0, tzinfo=timezone.utc)   # results written
# T_OLD < T_NEW → valid order

T_REVERSED_PLAN = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)   # plan AFTER results
# T_REVERSED_PLAN > T_NEW → violation


def _make_stat(mtime: datetime) -> SimpleNamespace:
    """Return a fake os.stat_result with st_mtime set to the given datetime."""
    s = SimpleNamespace()
    s.st_mtime = mtime.timestamp()
    return s


def _make_git_output(timestamps: list[datetime]) -> SimpleNamespace:
    """
    Return a fake subprocess.CompletedProcess whose stdout contains one ISO
    timestamp per line, newest first (matching `git log --format=%aI` output).
    """
    proc = SimpleNamespace()
    proc.stdout = "\n".join(dt.isoformat() for dt in timestamps) + "\n"
    proc.returncode = 0
    return proc


# ── _get_plan_commit_timestamp ─────────────────────────────────────────────────

class TestGetPlanCommitTimestamp:
    """Tests for the git-log wrapper that extracts PLAN.md's oldest commit."""

    def test_returns_oldest_commit_when_multiple_exist(self, tmp_path):
        """git log returns newest-first; function must return the last (oldest)."""
        plan_path = tmp_path / "PLAN.md"
        plan_path.touch()

        newer = datetime(2026, 4, 25, tzinfo=timezone.utc)
        older = datetime(2026, 4, 20, tzinfo=timezone.utc)

        with patch("subprocess.run", return_value=_make_git_output([newer, older])):
            result = _get_plan_commit_timestamp(plan_path, tmp_path)

        assert result == older

    def test_returns_single_commit_timestamp(self, tmp_path):
        plan_path = tmp_path / "PLAN.md"
        plan_path.touch()

        ts = datetime(2026, 4, 24, 9, 0, 0, tzinfo=timezone.utc)
        with patch("subprocess.run", return_value=_make_git_output([ts])):
            result = _get_plan_commit_timestamp(plan_path, tmp_path)

        assert result == ts

    def test_raises_when_plan_has_no_git_history(self, tmp_path):
        """Untracked PLAN.md: git log returns empty stdout."""
        plan_path = tmp_path / "PLAN.md"
        plan_path.touch()

        empty = SimpleNamespace(stdout="", returncode=0)
        with patch("subprocess.run", return_value=empty):
            with pytest.raises(RegistrationOrderError) as exc_info:
                _get_plan_commit_timestamp(plan_path, tmp_path)

        assert "no git commit history" in str(exc_info.value)
        assert "retrofitted: true" in str(exc_info.value)

    def test_error_message_contains_plan_path(self, tmp_path):
        plan_path = tmp_path / "experiments" / "e009_test" / "PLAN.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.touch()

        empty = SimpleNamespace(stdout="", returncode=0)
        with patch("subprocess.run", return_value=empty):
            with pytest.raises(RegistrationOrderError) as exc_info:
                _get_plan_commit_timestamp(plan_path, tmp_path)

        assert str(plan_path) in str(exc_info.value)


# ── _check_preregistration_order ──────────────────────────────────────────────

class TestCheckPreregistrationOrder:
    """
    Tests for the core R1 enforcement function.

    These are the tests that guarantee the check cannot be trivially bypassed:
    a green test suite here means the guard is intact.
    """

    def _run(self, plan_committed_at, results_mtime, tmp_path):
        """Helper: set up temp files and call _check_preregistration_order."""
        plan_path = tmp_path / "experiments" / "e009_test" / "PLAN.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.touch()

        results_jsonl = tmp_path / "outputs" / "e009_test" / "results.jsonl"
        results_jsonl.parent.mkdir(parents=True)
        results_jsonl.touch()

        with patch(
            "nope_analysis.analysis.auto_validate._get_plan_commit_timestamp",
            return_value=plan_committed_at,
        ):
            with patch.object(type(results_jsonl), "stat", return_value=_make_stat(results_mtime)):
                _check_preregistration_order(plan_path, results_jsonl, tmp_path)

    def test_valid_order_does_not_raise(self, tmp_path):
        """PLAN.md committed before results.jsonl written — should pass silently."""
        self._run(
            plan_committed_at=T_OLD,
            results_mtime=T_NEW,
            tmp_path=tmp_path,
        )

    def test_violated_order_raises_registration_error(self, tmp_path):
        """PLAN.md committed AFTER results.jsonl — must raise RegistrationOrderError."""
        with pytest.raises(RegistrationOrderError):
            self._run(
                plan_committed_at=T_REVERSED_PLAN,
                results_mtime=T_NEW,
                tmp_path=tmp_path,
            )

    def test_no_results_jsonl_skips_check(self, tmp_path):
        """If results.jsonl does not exist, the experiment has not run; skip check."""
        plan_path = tmp_path / "experiments" / "e009_test" / "PLAN.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.touch()

        results_jsonl = tmp_path / "outputs" / "e009_test" / "results.jsonl"
        # Do NOT create results_jsonl — simulates pre-run state

        # Should complete without calling git at all
        with patch("subprocess.run") as mock_git:
            _check_preregistration_order(plan_path, results_jsonl, tmp_path)
            mock_git.assert_not_called()

    def test_untracked_plan_raises(self, tmp_path):
        """PLAN.md exists on disk but was never committed — must raise."""
        plan_path = tmp_path / "experiments" / "e009_test" / "PLAN.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.touch()

        results_jsonl = tmp_path / "outputs" / "e009_test" / "results.jsonl"
        results_jsonl.parent.mkdir(parents=True)
        results_jsonl.touch()

        empty = SimpleNamespace(stdout="", returncode=0)
        with patch("subprocess.run", return_value=empty):
            with patch.object(type(results_jsonl), "stat", return_value=_make_stat(T_NEW)):
                with pytest.raises(RegistrationOrderError):
                    _check_preregistration_order(plan_path, results_jsonl, tmp_path)

    def test_error_message_contains_both_timestamps(self, tmp_path):
        """Error message must include both timestamps so the researcher can debug."""
        plan_path = tmp_path / "experiments" / "e009_test" / "PLAN.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.touch()

        results_jsonl = tmp_path / "outputs" / "results.jsonl"
        results_jsonl.parent.mkdir(parents=True)
        results_jsonl.touch()

        with patch(
            "nope_analysis.analysis.auto_validate._get_plan_commit_timestamp",
            return_value=T_REVERSED_PLAN,
        ):
            with patch.object(type(results_jsonl), "stat", return_value=_make_stat(T_NEW)):
                with pytest.raises(RegistrationOrderError) as exc_info:
                    _check_preregistration_order(plan_path, results_jsonl, tmp_path)

        msg = str(exc_info.value)
        assert T_REVERSED_PLAN.isoformat() in msg, "committed-at timestamp missing from error"
        assert T_NEW.isoformat() in msg, "results-written timestamp missing from error"

    def test_error_message_instructs_retrofitted_fix(self, tmp_path):
        """Error message must tell the user how to fix the violation."""
        plan_path = tmp_path / "experiments" / "e009_test" / "PLAN.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.touch()

        results_jsonl = tmp_path / "outputs" / "results.jsonl"
        results_jsonl.parent.mkdir(parents=True)
        results_jsonl.touch()

        with patch(
            "nope_analysis.analysis.auto_validate._get_plan_commit_timestamp",
            return_value=T_REVERSED_PLAN,
        ):
            with patch.object(type(results_jsonl), "stat", return_value=_make_stat(T_NEW)):
                with pytest.raises(RegistrationOrderError) as exc_info:
                    _check_preregistration_order(plan_path, results_jsonl, tmp_path)

        assert "retrofitted: true" in str(exc_info.value)

    def test_same_timestamp_does_not_raise(self, tmp_path):
        """
        Equal timestamps (plan committed and results written in the same second)
        should not raise — this is the boundary case.
        plan_committed_at == results_written_at → not a violation (> not >=).
        """
        self._run(
            plan_committed_at=T_NEW,
            results_mtime=T_NEW,
            tmp_path=tmp_path,
        )


# ── Integration: validate_experiment calls the order check ────────────────────

class TestValidateExperimentEnforcesOrderCheck:
    """
    Integration tests confirming that validate_experiment() calls
    _check_preregistration_order and propagates RegistrationOrderError.

    These tests prove the guard cannot be bypassed by calling validate_experiment()
    directly — the check is not optional.
    """

    # Minimal valid PLAN.md content for testing
    PLAN_CONTENT = """\
# PLAN — e999 Test

## Hypothesis
Test hypothesis.

## Decision Criteria
```criteria
metric: entropy
direction: global_nope > swa
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 2
n_simultaneous_tests: 1
bootstrap_n: 100
retrofitted: false
```
"""

    def _make_repo(self, tmp_path: Path) -> Path:
        """Create a minimal repo layout for validate_experiment testing."""
        exp_dir = tmp_path / "experiments" / "e999_test"
        exp_dir.mkdir(parents=True)
        (exp_dir / "PLAN.md").write_text(self.PLAN_CONTENT, encoding="utf-8")
        (tmp_path / "outputs" / "e999").mkdir(parents=True)
        return tmp_path

    def test_order_check_is_called(self, tmp_path):
        """
        validate_experiment must call _check_preregistration_order.
        If this test fails, someone removed the call from validate_experiment.
        """
        repo = self._make_repo(tmp_path)

        with patch(
            "nope_analysis.analysis.auto_validate._check_preregistration_order"
        ) as mock_check:
            validate_experiment(
                experiment_id="e999",
                comparisons={"entropy": ([1.0, 1.1, 1.2], [0.9, 0.95, 1.0])},
                output_dir=repo / "outputs" / "e999",
                repo_root=repo,
            )
            mock_check.assert_called_once()

    def test_order_check_called_with_correct_plan_path(self, tmp_path):
        """The plan_path argument to the order check must point to PLAN.md."""
        repo = self._make_repo(tmp_path)

        with patch(
            "nope_analysis.analysis.auto_validate._check_preregistration_order"
        ) as mock_check:
            validate_experiment(
                experiment_id="e999",
                comparisons={"entropy": ([1.0, 1.1], [0.9, 0.95])},
                output_dir=repo / "outputs" / "e999",
                repo_root=repo,
            )
            call_kwargs = mock_check.call_args
            plan_arg = call_kwargs.kwargs.get("plan_path") or call_kwargs.args[0]
            assert plan_arg.name == "PLAN.md"

    def test_order_check_called_with_results_jsonl(self, tmp_path):
        """The results_jsonl argument must be output_dir/results.jsonl."""
        repo = self._make_repo(tmp_path)
        output_dir = repo / "outputs" / "e999"

        with patch(
            "nope_analysis.analysis.auto_validate._check_preregistration_order"
        ) as mock_check:
            validate_experiment(
                experiment_id="e999",
                comparisons={"entropy": ([1.0, 1.1], [0.9, 0.95])},
                output_dir=output_dir,
                repo_root=repo,
            )
            call_kwargs = mock_check.call_args
            results_arg = call_kwargs.kwargs.get("results_jsonl") or call_kwargs.args[1]
            assert results_arg == output_dir / "results.jsonl"

    def test_registration_error_propagates_to_caller(self, tmp_path):
        """
        RegistrationOrderError raised inside the order check must propagate
        out of validate_experiment unmodified — it must not be swallowed.
        """
        repo = self._make_repo(tmp_path)

        with patch(
            "nope_analysis.analysis.auto_validate._check_preregistration_order",
            side_effect=RegistrationOrderError("synthetic violation for test"),
        ):
            with pytest.raises(RegistrationOrderError, match="synthetic violation for test"):
                validate_experiment(
                    experiment_id="e999",
                    comparisons={"entropy": ([1.0, 1.1], [0.9, 0.95])},
                    output_dir=repo / "outputs" / "e999",
                    repo_root=repo,
                )

    def test_order_check_runs_before_statistical_tests(self, tmp_path):
        """
        The order check must happen before statistical tests run.
        If the check raises, compare_groups must never be called.
        This prevents a scenario where an experiment runs to completion
        and only then fails the order check.
        """
        repo = self._make_repo(tmp_path)

        with patch(
            "nope_analysis.analysis.auto_validate._check_preregistration_order",
            side_effect=RegistrationOrderError("early abort"),
        ):
            with patch(
                "nope_analysis.analysis.auto_validate.compare_groups"
            ) as mock_stats:
                with pytest.raises(RegistrationOrderError):
                    validate_experiment(
                        experiment_id="e999",
                        comparisons={"entropy": ([1.0, 1.1], [0.9, 0.95])},
                        output_dir=repo / "outputs" / "e999",
                        repo_root=repo,
                    )
                mock_stats.assert_not_called()
