"""
tests/test_auto_validate.py

Unit and integration tests for auto_validate.py enforcement logic.

Test classes:
  TestGetPlanCommitTimestamp         — git timestamp extraction
  TestCheckPreregistrationOrder      — Rule R1: PLAN.md committed before results.jsonl
  TestValidateExperimentEnforcesOrderCheck — integration: order check wired into validate_experiment
  TestForceVerdict                   — force_verdict field short-circuits _verdict()
  TestRejectNullIf                   — reject_null_if.min_layers_significant aggregation logic

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
    _aggregate_verdicts,
    _check_preregistration_order,
    _get_plan_commit_timestamp,
    _verdict,
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


# ── force_verdict field ───────────────────────────────────────────────────────

class TestForceVerdict:
    """
    Tests for the force_verdict criteria field, which short-circuits _verdict()
    regardless of observed n, d, or p values.

    Rationale: experiments like e007 (n=5, memory-constrained) must always return
    INCONCLUSIVE even if statistical tests pass, because the sample size is known
    to be inadequate before the experiment runs. force_verdict makes this explicit
    and machine-enforced rather than relying on prose notes.
    """

    CRITERIA_BASE = {
        "accept": {"min_abs_cohen_d": 0.5, "max_p_holm_corrected": 0.01},
        "min_n_per_group": 2,
    }

    def test_force_inconclusive_overrides_large_effect(self):
        """Large d + significant p, but force_verdict=INCONCLUSIVE → INCONCLUSIVE."""
        criteria = {**self.CRITERIA_BASE, "force_verdict": "INCONCLUSIVE"}
        assert _verdict(100, 100, cohens_d=2.0, p_holm=0.0001, criteria=criteria) == "INCONCLUSIVE"

    def test_force_failed_overrides_validated_stats(self):
        """force_verdict=FAILED wins even when every other check would give VALIDATED."""
        criteria = {**self.CRITERIA_BASE, "force_verdict": "FAILED"}
        assert _verdict(100, 100, cohens_d=2.0, p_holm=0.0001, criteria=criteria) == "FAILED"

    def test_force_validated_overrides_underpowered(self):
        """force_verdict=VALIDATED bypasses the n < min_n underpowered check."""
        criteria = {**self.CRITERIA_BASE, "force_verdict": "VALIDATED"}
        # n=1 << min_n=2, would normally be INCONCLUSIVE
        assert _verdict(1, 1, cohens_d=2.0, p_holm=0.0001, criteria=criteria) == "VALIDATED"

    def test_force_verdict_case_insensitive(self):
        """force_verdict: inconclusive (lowercase) must be accepted and normalised."""
        criteria = {**self.CRITERIA_BASE, "force_verdict": "inconclusive"}
        assert _verdict(100, 100, cohens_d=2.0, p_holm=0.0001, criteria=criteria) == "INCONCLUSIVE"

    def test_invalid_force_verdict_raises_value_error(self):
        """An unrecognised force_verdict value must raise ValueError, not silently pass."""
        criteria = {**self.CRITERIA_BASE, "force_verdict": "MAYBE"}
        with pytest.raises(ValueError, match="force_verdict must be one of"):
            _verdict(100, 100, cohens_d=2.0, p_holm=0.0001, criteria=criteria)

    def test_absent_force_verdict_uses_normal_logic(self):
        """Without force_verdict, the standard underpowered → INCONCLUSIVE path applies."""
        assert _verdict(1, 1, cohens_d=2.0, p_holm=0.0001, criteria=self.CRITERIA_BASE) == "INCONCLUSIVE"

    def test_force_verdict_propagates_through_validate_experiment(self, tmp_path):
        """
        Integration: when PLAN.md contains force_verdict: INCONCLUSIVE, the overall
        verdict written to stats.json must be INCONCLUSIVE even if the data is strong.
        """
        plan_content = """\
# PLAN — e998 Force Verdict Test

## Decision Criteria
```criteria
metric: entropy
direction: global_nope > swa
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 2
n_simultaneous_tests: 1
bootstrap_n: 10
retrofitted: false
force_verdict: INCONCLUSIVE
```
"""
        exp_dir = tmp_path / "experiments" / "e998_test"
        exp_dir.mkdir(parents=True)
        (exp_dir / "PLAN.md").write_text(plan_content, encoding="utf-8")
        output_dir = tmp_path / "outputs" / "e998"
        output_dir.mkdir(parents=True)

        with patch("nope_analysis.analysis.auto_validate._check_preregistration_order"):
            result = validate_experiment(
                experiment_id="e998",
                # Very different groups: would be VALIDATED without force_verdict
                comparisons={"entropy": (
                    [3.0, 3.1, 3.2, 3.3, 3.4] * 20,
                    [1.0, 1.1, 1.2, 1.3, 1.4] * 20,
                )},
                output_dir=output_dir,
                repo_root=tmp_path,
            )

        assert result["overall_verdict"] == "INCONCLUSIVE"
        stats_json = output_dir / "stats.json"
        assert stats_json.exists()
        assert json.loads(stats_json.read_text())["overall_verdict"] == "INCONCLUSIVE"


# ── reject_null_if aggregation logic ─────────────────────────────────────────

class TestRejectNullIf:
    """
    Tests for reject_null_if.min_layers_significant in _aggregate_verdicts().

    This addresses e008's criterion: "H1_alt is accepted if at least 1 of 16
    Global layer comparisons is individually significant." The default conservative
    aggregation (any FAILED → overall FAILED) would make it impossible to ever
    VALIDATE e008, because some non-significant layers are expected even under H1_alt.
    """

    CRITERIA_BASE = {
        "accept": {"min_abs_cohen_d": 0.5, "max_p_holm_corrected": 0.01},
        "min_n_per_group": 2,
    }

    def _criteria(self, min_layers: int) -> dict:
        return {**self.CRITERIA_BASE, "reject_null_if": {"min_layers_significant": min_layers}}

    # ── Core logic ────────────────────────────────────────────────────────────

    def test_min_1_met_with_exactly_1_validated(self):
        """1 of 2 comparisons VALIDATED, min_layers=1 → overall VALIDATED."""
        result = _aggregate_verdicts(["VALIDATED", "FAILED"], self._criteria(1))
        assert result == "VALIDATED"

    def test_min_1_met_with_multiple_validated(self):
        """3 of 5 VALIDATED, min_layers=1 → VALIDATED."""
        result = _aggregate_verdicts(["VALIDATED", "VALIDATED", "VALIDATED", "FAILED", "FAILED"], self._criteria(1))
        assert result == "VALIDATED"

    def test_min_1_not_met_all_failed(self):
        """0 VALIDATED, all FAILED → FAILED."""
        result = _aggregate_verdicts(["FAILED", "FAILED"], self._criteria(1))
        assert result == "FAILED"

    def test_min_1_not_met_with_inconclusive_present(self):
        """0 VALIDATED but INCONCLUSIVE present → INCONCLUSIVE (not FAILED)."""
        result = _aggregate_verdicts(["INCONCLUSIVE", "FAILED"], self._criteria(1))
        assert result == "INCONCLUSIVE"

    def test_min_2_met(self):
        """2 of 3 VALIDATED, min_layers=2 → VALIDATED."""
        result = _aggregate_verdicts(["VALIDATED", "VALIDATED", "FAILED"], self._criteria(2))
        assert result == "VALIDATED"

    def test_min_2_only_1_validated_inconclusive_present(self):
        """1 of 3 VALIDATED when need 2, INCONCLUSIVE present → INCONCLUSIVE."""
        result = _aggregate_verdicts(["VALIDATED", "INCONCLUSIVE", "FAILED"], self._criteria(2))
        assert result == "INCONCLUSIVE"

    def test_min_2_only_1_validated_rest_failed(self):
        """1 of 3 VALIDATED when need 2, no INCONCLUSIVE → FAILED."""
        result = _aggregate_verdicts(["VALIDATED", "FAILED", "FAILED"], self._criteria(2))
        assert result == "FAILED"

    def test_min_16_none_validated(self):
        """Simulates e008: 0 of 16 layers significant → FAILED."""
        verdicts = ["FAILED"] * 16
        result = _aggregate_verdicts(verdicts, self._criteria(16))
        assert result == "FAILED"

    def test_min_1_with_16_layers_1_validated(self):
        """Simulates e008: 1 of 16 layers significant → VALIDATED."""
        verdicts = ["FAILED"] * 15 + ["VALIDATED"]
        result = _aggregate_verdicts(verdicts, self._criteria(1))
        assert result == "VALIDATED"

    # ── Unknown keys warning ──────────────────────────────────────────────────

    def test_unknown_reject_null_if_key_prints_warning(self, capsys):
        """Unrecognised reject_null_if keys must warn, not raise."""
        criteria = {
            **self.CRITERIA_BASE,
            "reject_null_if": {"min_layers_significant": 1, "secret_criterion": 99},
        }
        _aggregate_verdicts(["VALIDATED"], criteria)
        captured = capsys.readouterr()
        assert "secret_criterion" in captured.out
        assert "WARNING" in captured.out

    def test_unknown_key_still_applies_known_logic(self, capsys):
        """Unknown key warns but does not prevent min_layers_significant from working."""
        criteria = {
            **self.CRITERIA_BASE,
            "reject_null_if": {"min_layers_significant": 1, "unknown_key": True},
        }
        result = _aggregate_verdicts(["VALIDATED", "FAILED"], criteria)
        assert result == "VALIDATED"

    # ── Default conservative behaviour preserved ──────────────────────────────

    def test_no_reject_null_if_any_failed_gives_failed(self):
        """Without reject_null_if, the original conservative aggregation applies."""
        result = _aggregate_verdicts(["VALIDATED", "FAILED"], self.CRITERIA_BASE)
        assert result == "FAILED"

    def test_no_reject_null_if_inconclusive_beats_validated(self):
        """Without reject_null_if, INCONCLUSIVE beats VALIDATED."""
        result = _aggregate_verdicts(["VALIDATED", "INCONCLUSIVE"], self.CRITERIA_BASE)
        assert result == "INCONCLUSIVE"

    def test_no_reject_null_if_all_validated_gives_validated(self):
        """Without reject_null_if, all VALIDATED → VALIDATED."""
        result = _aggregate_verdicts(["VALIDATED", "VALIDATED"], self.CRITERIA_BASE)
        assert result == "VALIDATED"

    # ── Integration with validate_experiment ─────────────────────────────────

    def test_validate_experiment_applies_min_layers(self, tmp_path):
        """
        Integration: PLAN.md with min_layers_significant: 1 — a single strong
        comparison among several weak ones should yield VALIDATED overall.
        """
        plan_content = """\
# PLAN — e997 Min Layers Test

## Decision Criteria
```criteria
metric: per_layer_delta
direction: global_nope > swa
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 5
n_simultaneous_tests: 3
bootstrap_n: 10
retrofitted: false
reject_null_if:
  min_layers_significant: 1
```
"""
        exp_dir = tmp_path / "experiments" / "e997_test"
        exp_dir.mkdir(parents=True)
        (exp_dir / "PLAN.md").write_text(plan_content, encoding="utf-8")
        output_dir = tmp_path / "outputs" / "e997"
        output_dir.mkdir(parents=True)

        # One strong comparison (layer_3), two null comparisons (layer_7, layer_11)
        strong = [3.0 + i * 0.1 for i in range(30)]
        null_a  = [1.5 + i * 0.01 for i in range(30)]
        null_b  = [1.5 + i * 0.01 for i in range(30)]

        with patch("nope_analysis.analysis.auto_validate._check_preregistration_order"):
            result = validate_experiment(
                experiment_id="e997",
                comparisons={
                    "layer_3_delta":  (strong, [1.0] * 30),   # large effect → VALIDATED
                    "layer_7_delta":  (null_a, null_b),        # no effect → FAILED
                    "layer_11_delta": (null_a, null_b),        # no effect → FAILED
                },
                output_dir=output_dir,
                repo_root=tmp_path,
            )

        # Conservative aggregation would give FAILED (two FAILED comparisons).
        # min_layers_significant: 1 means 1 VALIDATED is enough → VALIDATED.
        assert result["overall_verdict"] == "VALIDATED"
