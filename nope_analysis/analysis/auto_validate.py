"""
auto_validate.py — Pre-registration enforcement and automated validation.

Every experiment script must call validate_experiment() before exit.
The function:
  1. Locates experiments/{exp_id}_*/PLAN.md — raises FileNotFoundError if missing
  2. Parses the `## Decision Criteria` YAML block from PLAN.md
  3. Checks sample size sufficiency
  4. Runs Welch t-test + Cohen's d + bootstrap CI (via statistical_tests.py)
  5. Applies Holm-Bonferroni correction for multiple comparisons
  6. Tags result: VALIDATED / FAILED / INCONCLUSIVE
  7. Writes {output_dir}/stats.json
  8. Returns the full stats dict

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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from nope_analysis.analysis.statistical_tests import compare_groups, report_stats


# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_experiment_dir(experiment_id: str, repo_root: Path | None = None) -> Path:
    """Find experiments/{exp_id}_*/ directory."""
    if repo_root is None:
        # Walk up from this file to find repo root (contains experiments/)
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "experiments").is_dir():
                repo_root = parent
                break
        else:
            raise FileNotFoundError(
                f"Could not locate 'experiments/' directory from {here}. "
                "Ensure you are running from within the sda212331 repo."
            )

    exp_base = repo_root / "experiments"
    matches = sorted(exp_base.glob(f"{experiment_id}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No experiment directory found for '{experiment_id}' in {exp_base}.\n"
            f"Expected a directory named '{experiment_id}_<slug>'.\n"
            f"Create PLAN.md first: experiments/{experiment_id}_<slug>/PLAN.md\n"
            f"Pre-registration rule: PLAN.md must be committed BEFORE results exist."
        )
    return matches[0]


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
    """
    # 1. Locate and validate PLAN.md
    exp_dir = _find_experiment_dir(experiment_id, repo_root)
    plan_path = exp_dir / "PLAN.md"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"PLAN.md not found at {plan_path}.\n"
            f"Pre-registration rule: create and commit PLAN.md before running the experiment."
        )

    criteria = _parse_plan_criteria(plan_path)
    retrofitted = criteria.get("retrofitted", False)

    print(f"\n{'='*60}")
    print(f"auto_validate: {experiment_id}")
    print(f"  PLAN.md: {plan_path}")
    if retrofitted:
        print(f"  WARNING: criteria are RETROFITTED (post-hoc). Results tagged accordingly.")
    print(f"{'='*60}")

    # 2. Run statistical tests for each comparison
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

    # 3. Apply Holm correction
    if n_tests > 1:
        corrected_p = _holm_correct(raw_p_values)
    else:
        corrected_p = [min(raw_p_values[0], 1.0)] if raw_p_values else []

    for i, stat in enumerate(all_stats):
        stat["p_holm"] = corrected_p[i]
        stat["n_tests_total"] = n_tests

    # 4. Determine per-comparison and overall verdicts
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

    # 5. Build output dict
    output = {
        "experiment_id": experiment_id,
        "plan_path": str(plan_path),
        "retrofitted": retrofitted,
        "overall_verdict": overall,
        "n_comparisons": n_tests,
        "criteria": criteria,
        "comparisons": all_stats,
    }

    # 6. Write stats.json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(output, f, indent=2)

    # 7. Print summary
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

    # 8. Warn if ANALYSIS.md is missing
    analysis_path = exp_dir / "ANALYSIS.md"
    if not analysis_path.exists():
        print(
            f"  REMINDER: ANALYSIS.md not found at {analysis_path}.\n"
            f"  You must commit ANALYSIS.md before marking this experiment complete.\n"
        )

    return output
