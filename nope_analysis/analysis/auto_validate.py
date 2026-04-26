"""
auto_validate.py — Pre-registration enforcement and automated validation.

Every experiment script must call validate_experiment() before exit.
The function:
  1. Locates experiments/{exp_id}_*/PLAN.md — raises FileNotFoundError if missing
  2. Asserts PLAN.md was committed to git BEFORE results.jsonl was written to disk
     — raises RegistrationOrderError if ordering is violated (Rule R1)
  3. Parses the `## Decision Criteria` YAML block from PLAN.md
  4. Checks sample size sufficiency
  5. Runs Welch t-test + Cohen's d + bootstrap CI (via statistical_tests.py)
  6. Applies Holm-Bonferroni correction for multiple comparisons
  7. Tags result: VALIDATED / FAILED / INCONCLUSIVE
  8. Writes {output_dir}/stats.json
  9. Returns the full stats dict

Usage (at end of every experiment script):
    from nope_analysis.analysis.auto_validate import validate_experiment
    from pathlib import Path

    validate_experiment(
        experiment_id="e001",
        comparisons={
            "entropy": (global_entropy_list, swa_entropy_list),
            "attn_distance": (global_dist_list, swa_dist_list),
        },
        output_dir=Path("outputs/exp1_attention_entropy"),
    )
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from nope_analysis.analysis.statistical_tests import compare_groups, report_stats


# ── Custom exceptions ─────────────────────────────────────────────────────────

class RegistrationOrderError(RuntimeError):
    """
    Raised when results.jsonl was written to disk before PLAN.md was committed
    to git, violating pre-registration Rule R1.

    To fix: set 'retrofitted: true' in the ```criteria block of PLAN.md and
    mark findings as exploratory in ANALYSIS.md.
    See STATISTICAL_PROTOCOL.md §7.
    """


# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_repo_root(start: Path) -> Path:
    """Walk up from start until a directory containing experiments/ is found."""
    for parent in [start, *start.parents]:
        if (parent / "experiments").is_dir():
            return parent
    raise FileNotFoundError(
        f"Could not locate repo root (no 'experiments/' directory) from {start}. "
        "Ensure you are running from within the sda212331 repo."
    )


def _find_experiment_dir(experiment_id: str, repo_root: Path | None = None) -> Path:
    """Find experiments/{exp_id}_*/ directory."""
    if repo_root is None:
        repo_root = _find_repo_root(Path(__file__).resolve())

    exp_base = repo_root / "experiments"
    matches = sorted(exp_base.glob(f"{experiment_id}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No experiment directory found for '{experiment_id}' in {exp_base}.\n"
            f"Expected a directory named '{experiment_id}_<slug>'.\n"
            f"Create PLAN.md first: experiments/{experiment_id}_<slug>/PLAN.md\n"
            f"Pre-registration rule R1: PLAN.md must be committed BEFORE results exist."
        )
    return matches[0]


# ── Pre-registration order check ──────────────────────────────────────────────

def _get_plan_commit_timestamp(plan_path: Path, repo_root: Path) -> datetime:
    """
    Return the UTC datetime of PLAN.md's oldest git commit.
    Raises RegistrationOrderError if the file has never been committed.
    """
    try:
        rel = plan_path.relative_to(repo_root)
    except ValueError:
        rel = plan_path  # absolute path fallback for git

    result = subprocess.run(
        ["git", "log", "--format=%aI", "--", str(rel)],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RegistrationOrderError(
            f"PLAN.md has no git commit history: {plan_path}\n"
            f"Pre-registration Rule R1 violated: commit PLAN.md before running the experiment.\n"
            f"  git add {rel}\n"
            f"  git commit -m 'pre-register: {plan_path.parent.name}'\n"
            f"If results already exist, set 'retrofitted: true' in the ```criteria block."
        )
    # git log returns newest-first; last line is the oldest (creation) commit
    return datetime.fromisoformat(lines[-1])


def _check_preregistration_order(
    plan_path: Path,
    results_jsonl: Path,
    repo_root: Path,
) -> None:
    """
    Assert PLAN.md was committed to git before results.jsonl was written to disk.

    Invariant enforced: git_commit_time(PLAN.md) < fs_mtime(results.jsonl)

    - If results.jsonl does not exist, the experiment has not run yet; skip check.
    - If PLAN.md has no git commit history, raise RegistrationOrderError.
    - If PLAN.md was committed after results.jsonl was written, raise RegistrationOrderError
      with instructions to mark the experiment as retrofitted.
    """
    if not results_jsonl.exists():
        return  # experiment not yet run — nothing to compare

    plan_committed_at = _get_plan_commit_timestamp(plan_path, repo_root)
    results_written_at = datetime.fromtimestamp(
        results_jsonl.stat().st_mtime, tz=timezone.utc
    )

    if plan_committed_at > results_written_at:
        raise RegistrationOrderError(
            f"Pre-registration order violated for {plan_path.parent.name}:\n"
            f"  PLAN.md first committed : {plan_committed_at.isoformat()}\n"
            f"  results.jsonl written   : {results_written_at.isoformat()}\n"
            f"  PLAN.md was committed AFTER the experiment ran.\n"
            f"  Fix: set 'retrofitted: true' in the ```criteria block of PLAN.md\n"
            f"  and mark all findings as exploratory in ANALYSIS.md.\n"
            f"  See STATISTICAL_PROTOCOL.md §7."
        )


# ── YAML criteria parser ──────────────────────────────────────────────────────

def _parse_plan_criteria(plan_path: Path) -> dict:
    """
    Extract the ```criteria YAML block from PLAN.md ## Decision Criteria section.
    Raises ValueError if the block is missing or malformed.
    """
    text = plan_path.read_text(encoding="utf-8")

    # Find the criteria code block (language tag: criteria)
    pattern = r"```criteria\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(
            f"No ```criteria block found in {plan_path}.\n"
            f"Add a '## Decision Criteria' section with a ```criteria YAML block.\n"
            f"See STATISTICAL_PROTOCOL.md for the required format."
        )

    raw_yaml = match.group(1)
    try:
        criteria = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed YAML in criteria block of {plan_path}: {e}")

    # Validate required fields
    required = {"accept", "min_n_per_group"}
    missing = required - set(criteria.keys())
    if missing:
        raise ValueError(
            f"Missing required fields in criteria block: {missing}\n"
            f"File: {plan_path}"
        )

    return criteria


# ── Holm-Bonferroni correction ────────────────────────────────────────────────

def _holm_correct(p_values: List[float]) -> List[float]:
    """Return Holm-corrected p-values (step-down Bonferroni)."""
    k = len(p_values)
    if k == 0:
        return []
    if k == 1:
        return [min(p_values[0], 1.0)]

    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    corrected = np.zeros(k)

    for i, p in enumerate(sorted_p):
        corrected[order[i]] = min(p * (k - i), 1.0)

    # Enforce monotonicity: corrected p cannot decrease as original p increases
    for i in range(k - 2, -1, -1):
        corrected[order[i]] = max(corrected[order[i]], corrected[order[i + 1]])

    return corrected.tolist()


# ── Verdict logic ─────────────────────────────────────────────────────────────

def _verdict(
    n_a: int,
    n_b: int,
    cohens_d: float,
    p_holm: float,
    criteria: dict,
) -> str:
    min_n = criteria.get("min_n_per_group", 30)
    accept = criteria.get("accept", {})
    min_d = accept.get("min_abs_cohen_d", 0.5)
    max_p = accept.get("max_p_holm_corrected", 0.01)

    # Underpowered → always INCONCLUSIVE regardless of p
    if n_a < min_n or n_b < min_n:
        return "INCONCLUSIVE"

    abs_d = abs(cohens_d) if not np.isnan(cohens_d) else 0.0

    if p_holm <= max_p and abs_d >= min_d:
        return "VALIDATED"
    elif p_holm > 0.05 or abs_d < 0.2:
        return "FAILED"
    else:
        return "INCONCLUSIVE"


# ── Main public function ──────────────────────────────────────────────────────

def validate_experiment(
    experiment_id: str,
    comparisons: Dict[str, Tuple[List[float], List[float]]],
    output_dir: Path,
    label_a: str = "global_nope",
    label_b: str = "swa",
    repo_root: Path | None = None,
) -> dict:
    """
    Validate an experiment against its pre-registered PLAN.md criteria.

    Args:
        experiment_id: e.g. "e001"
        comparisons: dict mapping metric_name → (group_a_values, group_b_values)
        output_dir: where to write stats.json (e.g. Path("outputs/exp1_attention_entropy"))
        label_a: name for group a (default: "global_nope")
        label_b: name for group b (default: "swa")
        repo_root: override for repo root detection

    Returns:
        Full stats dict written to stats.json.

    Raises:
        FileNotFoundError: if experiments/{exp_id}_*/PLAN.md is missing.
        RegistrationOrderError: if results.jsonl predates PLAN.md's git commit (Rule R1).
        ValueError: if PLAN.md criteria block is missing or malformed.
    """
    # Detect repo root once; share across all helpers
    if repo_root is None:
        repo_root = _find_repo_root(Path(__file__).resolve())

    # 1. Locate and validate PLAN.md
    exp_dir = _find_experiment_dir(experiment_id, repo_root)
    plan_path = exp_dir / "PLAN.md"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"PLAN.md not found at {plan_path}.\n"
            f"Pre-registration Rule R1: create and commit PLAN.md before running the experiment."
        )

    # 2. Enforce pre-registration ordering (Rule R1)
    _check_preregistration_order(
        plan_path=plan_path,
        results_jsonl=Path(output_dir) / "results.jsonl",
        repo_root=repo_root,
    )

    criteria = _parse_plan_criteria(plan_path)
    retrofitted = criteria.get("retrofitted", False)

    print(f"\n{'='*60}")
    print(f"auto_validate: {experiment_id}")
    print(f"  PLAN.md: {plan_path}")
    if retrofitted:
        print(f"  WARNING: criteria are RETROFITTED (post-hoc). Results tagged accordingly.")
    print(f"{'='*60}")

    # 3. Run statistical tests for each comparison
    n_tests = len(comparisons)
    all_stats = []
    raw_p_values = []

    for metric, (a_vals, b_vals) in comparisons.items():
        stat = compare_groups(
            a_vals, b_vals,
            label_a=label_a, label_b=label_b,
            metric=metric,
        )
        all_stats.append(stat)
        raw_p_values.append(stat["p_value"])
        print(f"  {report_stats(stat)}")

    # 4. Apply Holm correction
    if n_tests > 1:
        corrected_p = _holm_correct(raw_p_values)
    else:
        corrected_p = [min(raw_p_values[0], 1.0)] if raw_p_values else []

    for i, stat in enumerate(all_stats):
        stat["p_holm"] = corrected_p[i]
        stat["n_tests_total"] = n_tests

    # 5. Determine per-comparison and overall verdicts
    verdicts = []
    for stat in all_stats:
        v = _verdict(
            stat["n_a"], stat["n_b"],
            stat["cohens_d"], stat["p_holm"],
            criteria,
        )
        stat["verdict"] = v
        verdicts.append(v)

    # Overall verdict: most conservative across all comparisons
    overall = "VALIDATED"
    if "INCONCLUSIVE" in verdicts:
        overall = "INCONCLUSIVE"
    if "FAILED" in verdicts:
        overall = "FAILED"

    # 6. Build output dict
    output = {
        "experiment_id": experiment_id,
        "plan_path": str(plan_path),
        "retrofitted": retrofitted,
        "overall_verdict": overall,
        "n_comparisons": n_tests,
        "criteria": criteria,
        "comparisons": all_stats,
    }

    # 7. Write stats.json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(output, f, indent=2)

    # 8. Print summary
    verdict_line = {
        "VALIDATED": "VALIDATED — criteria met",
        "FAILED": "FAILED — null not rejected or effect too small",
        "INCONCLUSIVE": "INCONCLUSIVE — underpowered or marginal",
    }[overall]
    print(f"\n  Overall: {verdict_line}")
    if retrofitted:
        print(f"  (retrofitted criteria — exploratory only, not confirmatory)")
    print(f"  stats.json written to: {stats_path}")
    print(f"{'='*60}\n")

    # 9. Warn if ANALYSIS.md is missing
    analysis_path = exp_dir / "ANALYSIS.md"
    if not analysis_path.exists():
        print(
            f"  REMINDER: ANALYSIS.md not found at {analysis_path}.\n"
            f"  You must commit ANALYSIS.md before marking this experiment complete.\n"
        )

    return output
