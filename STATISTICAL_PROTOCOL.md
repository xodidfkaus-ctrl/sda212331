# Statistical Protocol — EXAONE 4.5 NoPE Research

This document is the single source of truth for statistical decisions.
All experiments must follow these rules. Deviations must be documented
in the experiment's ANALYSIS.md with a rationale.

---

## 1. Multiple Comparison Correction

### Policy: Holm-Bonferroni (step-down)

When an experiment tests `k` hypotheses simultaneously (e.g., entropy at 5 length
conditions, probe accuracy at 3 sequence lengths), apply Holm-Bonferroni correction.

**Algorithm**:
1. Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(k)
2. For each i from 1 to k: reject H_i if p_(i) ≤ α / (k - i + 1)
3. Stop rejecting once the first non-rejection is reached

**Implementation**: `scipy.stats.holm_bonferroni` or manual application in
`auto_validate.py`. The corrected p-values are stored as `p_holm` in stats.json.

### When correction applies

| Situation | Correction |
|-----------|-----------|
| Single comparison (Global vs SWA at one condition) | None |
| Multiple lengths (e.g., 128/256/512/1024/2048) | Holm over all k length conditions |
| Multiple metrics simultaneously (entropy + distance) | Holm over all k metrics |
| Multiple layer indices | Holm over all k layers tested |

### When correction does NOT apply

- Exploratory analysis reported as exploratory (clearly labeled, not in main conclusions)
- Confirmatory test on a pre-registered single primary metric

---

## 2. Effect Size Thresholds

Primary effect size: **Cohen's d** (standardized mean difference, pooled SD).

| Label | |d| range | Interpretation |
|-------|----------|----------------|
| negligible | < 0.2 | Not practically meaningful |
| small | 0.2 – 0.49 | Noticeable but weak |
| medium | 0.5 – 0.79 | **Minimum threshold for paper claims** |
| large | ≥ 0.8 | Strong practical significance |

**Rule**: A difference is citeable in the paper only if Cohen's d ≥ 0.5 AND the
Holm-corrected p-value meets the significance threshold below.

Secondary effect size: **Common Language Effect Size (CLES)** = P(a > b), reported
alongside Cohen's d for entropy and distance metrics where the direction claim matters.

---

## 3. Significance Threshold

| Test type | α (uncorrected) | α (Holm-corrected) |
|-----------|----------------|-------------------|
| Primary hypothesis (main paper claim) | 0.01 | 0.01 |
| Secondary / exploratory | 0.05 | 0.05 |

**Auto-validate decision rules** (implemented in `auto_validate.py`):

```
VALIDATED   : Holm-corrected p ≤ α_primary  AND  |d| ≥ 0.5  AND  n ≥ min_n
FAILED      : Holm-corrected p > 0.05  OR  |d| < 0.2  (with adequate n)
INCONCLUSIVE: Holm-corrected p in (0.01, 0.05]  OR  |d| in [0.2, 0.5)
              OR  n < min_n (underpowered regardless of p-value)
```

A FAILED or INCONCLUSIVE result is not a project failure. Both are publishable
with appropriate framing. Do not re-run experiments to fish for VALIDATED status.

---

## 4. Minimum Sample Size Rules

| Experiment type | Min n per group | Justification |
|-----------------|----------------|---------------|
| Attention entropy (per layer, per length) | 10 | Allows Welch t with df ≥ 9 |
| Positional probe (per layer, per length) | 30 prompts → ~24 test samples with 5-fold CV | Power ≈ 0.80 for d=0.5 at α=0.05 |
| Upgraded probe (e008, publication-grade) | 300 prompts | Power ≈ 0.95 for d=0.3 at α=0.01 |
| Ablation perplexity (e005, e006) | 20 sequences per condition | Practical minimum given 33B model |
| Long-context hook (e007) | 5 sequences per length | Memory-constrained; flag as exploratory |

**Underpowered experiments**: if n < min_n, auto_validate tags as INCONCLUSIVE
regardless of p-value, and prints a power warning. Results may still be reported
as preliminary or exploratory — not as confirmatory.

---

## 5. Bootstrap CI Configuration

| Parameter | Value | Rationale |
|-----------|-------|----------|
| n_bootstrap | 10,000 | Standard for publication |
| CI level | 95% | Standard |
| Method | Percentile bootstrap | Appropriate for entropy/accuracy metrics |
| Seed | 42 (from `nope_analysis/seeds.py`) | Reproducibility |

Bootstrap CI is reported for every metric in stats.json.
For probe accuracy, bootstrap CI is computed on the test-fold mean, not on the
full distribution (accounts for the fact that k-fold folds are not independent).

---

## 6. Reporting Standards

Every table or figure in the paper that reports a comparison must include:

1. n per group
2. Mean ± SD (or 95% CI for small samples)
3. Welch t-statistic and degrees of freedom
4. Holm-corrected p-value (not raw p)
5. Cohen's d with label (negligible/small/medium/large)
6. Decision tag (VALIDATED / FAILED / INCONCLUSIVE)

**Format template** (copy into paper LaTeX):
```
Global: M=X.XX (SD=X.XX, n=N) vs SWA: M=X.XX (SD=X.XX, n=N);
Welch t(df)=X.XX, p_Holm=0.XXX, d=0.XX (medium)
```

---

## 7. Pre-registration Policy

A hypothesis is **pre-registered** if PLAN.md is committed to git **before**
any experiment results are generated. The git timestamp of PLAN.md vs. the
first results.jsonl entry establishes the order.

A hypothesis is **retrofitted** if PLAN.md was written after results existed.
All retrofitted criteria must be marked `retrofitted: true` in the PLAN.md
`Decision Criteria` YAML block and flagged in ANALYSIS.md "Deviations" section.

Retrofitted hypotheses may be included in the paper only as:
- Exploratory findings
- Pilot results motivating future confirmatory work
- Not as primary conclusions

---

*Version: 1.0 — 2026-04-26*
