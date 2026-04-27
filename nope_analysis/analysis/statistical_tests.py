"""
Statistical testing utilities for NoPE analysis experiments.

Usage:
    from nope_analysis.analysis.statistical_tests import compare_groups, report_stats

All functions accept plain Python lists or numpy arrays.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats


def cohens_d(a, b):
    """Cohen's d effect size between two groups (pooled SD)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float('nan')
    pooled_std = np.sqrt(((n_a - 1) * a.std(ddof=1) ** 2 + (n_b - 1) * b.std(ddof=1) ** 2) / (n_a + n_b - 2))
    if pooled_std == 0:
        return float('nan')
    return float((a.mean() - b.mean()) / pooled_std)


def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for stat_fn applied to values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = (1 - ci) / 2
    hi = 1 - lo
    return float(np.quantile(boots, lo)), float(np.quantile(boots, hi))


def compare_groups(a, b, label_a='global_nope', label_b='swa', metric='value'):
    """
    Full statistical comparison between two groups.

    Returns a dict with:
      mean_a, mean_b, diff (a-b),
      t_stat, p_value (Welch's t-test),
      cohens_d,
      ci_a (95% bootstrap CI), ci_b,
      significant (p < 0.05)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a, b)
    ci_a = bootstrap_ci(a)
    ci_b = bootstrap_ci(b)
    return {
        'metric': metric,
        'label_a': label_a,
        'label_b': label_b,
        'n_a': int(len(a)),
        'n_b': int(len(b)),
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'std_a': float(a.std(ddof=1)) if len(a) > 1 else float('nan'),
        'std_b': float(b.std(ddof=1)) if len(b) > 1 else float('nan'),
        'diff_a_minus_b': float(a.mean() - b.mean()),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': d,
        'ci_a_95': list(ci_a),
        'ci_b_95': list(ci_b),
        'significant': bool(p_value < 0.05),
        'effect_size_label': _effect_label(d),
    }


def cohens_dz(deltas):
    """Cohen's d_z for one-sample / paired design: mean(delta) / std(delta)."""
    d = np.asarray(deltas, dtype=float)
    if len(d) < 2:
        return float('nan')
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float('nan')


def compare_one_sample(values, popmean=0.0, label_a='treatment', metric='value'):
    """One-sample t-test: tests if mean(values) != popmean. Reports Cohen's d_z."""
    a = np.asarray(values, dtype=float)
    t_stat, p_value = scipy_stats.ttest_1samp(a, popmean=popmean)
    dz = cohens_dz(a - popmean)
    ci = bootstrap_ci(a)
    return {
        'metric': metric,
        'label_a': label_a,
        'label_b': f'null_mu={popmean}',
        'n_a': int(len(a)),
        'n_b': 0,
        'mean_a': float(a.mean()),
        'mean_b': float(popmean),
        'std_a': float(a.std(ddof=1)) if len(a) > 1 else float('nan'),
        'std_b': 0.0,
        'diff_a_minus_b': float(a.mean() - popmean),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': dz,
        'ci_a_95': list(ci),
        'ci_b_95': [float(popmean), float(popmean)],
        'significant': bool(p_value < 0.05),
        'effect_size_label': _effect_label(abs(dz)) if not np.isnan(dz) else 'nan',
        'test_type': 'one_sample_t',
    }


def report_stats_one_sample(result: dict) -> str:
    """Human-readable one-liner for a compare_one_sample result."""
    sig = '✓ significant' if result['significant'] else '✗ not significant'
    return (
        f"[{result['metric']}] mean={result['mean_a']:.4f}±{result['std_a']:.4f} "
        f"vs null={result['mean_b']}  "
        f"diff={result['diff_a_minus_b']:+.4f}  "
        f"t={result['t_stat']:.3f}  p={result['p_value']:.4f}  "
        f"d_z={result['cohens_d']:.3f}({result['effect_size_label']})  {sig}"
    )


def _effect_label(d):
    if np.isnan(d):
        return 'nan'
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    if d < 0.5:
        return 'small'
    if d < 0.8:
        return 'medium'
    return 'large'


def report_stats(result: dict) -> str:
    """Human-readable one-liner for a compare_groups result."""
    sig = '✓ significant' if result['significant'] else '✗ not significant'
    return (
        f"[{result['metric']}] {result['label_a']}={result['mean_a']:.4f}±{result['std_a']:.4f} "
        f"vs {result['label_b']}={result['mean_b']:.4f}±{result['std_b']:.4f}  "
        f"diff={result['diff_a_minus_b']:+.4f}  "
        f"t={result['t_stat']:.3f}  p={result['p_value']:.4f}  "
        f"d={result['cohens_d']:.3f}({result['effect_size_label']})  {sig}"
    )


def analyze_jsonl(jsonl_path: Path, group_key='layer_type', metric_keys=('entropy', 'attn_distance')):
    """
    Load a results.jsonl from any experiment and run compare_groups for each metric
    between 'global_nope' and 'swa' groups.

    Returns list of result dicts (one per metric per seq_len if seq_len column exists).
    """
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    seq_lens = sorted(set(r.get('seq_len', None) for r in rows))
    seq_lens = [s for s in seq_lens if s is not None]

    if seq_lens:
        for seq_len in seq_lens:
            subset = [r for r in rows if r.get('seq_len') == seq_len]
            a_rows = [r for r in subset if r.get(group_key) == 'global_nope']
            b_rows = [r for r in subset if r.get(group_key) == 'swa']
            for mk in metric_keys:
                a_vals = [r[mk] for r in a_rows if mk in r]
                b_vals = [r[mk] for r in b_rows if mk in r]
                if len(a_vals) >= 2 and len(b_vals) >= 2:
                    res = compare_groups(a_vals, b_vals, metric=f'{mk}@{seq_len}tok')
                    results.append(res)
    else:
        # no seq_len grouping — compare whole file
        a_rows = [r for r in rows if r.get(group_key) == 'global_nope']
        b_rows = [r for r in rows if r.get(group_key) == 'swa']
        for mk in metric_keys:
            a_vals = [r[mk] for r in a_rows if mk in r]
            b_vals = [r[mk] for r in b_rows if mk in r]
            if len(a_vals) >= 2 and len(b_vals) >= 2:
                results.append(compare_groups(a_vals, b_vals, metric=mk))

    return results


def save_stats_report(results: list, out_path: Path):
    """Save list of compare_groups results to JSON + print summary."""
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Statistical Test Results ({len(results)} comparisons) ===")
    for r in results:
        print(report_stats(r))
    print(f"\nSaved to: {out_path}")
