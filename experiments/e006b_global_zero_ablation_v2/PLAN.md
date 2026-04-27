# PLAN — e006b Global Zero-Output Ablation v2

**Status**: PENDING
**Script**: `nope_analysis/experiments/exp3_global_zero_ablation_v2.py`
**Output**: `outputs/e006b_global_zero_ablation_v2/`
**RQ**: RQ3 — Does NoPE Global contribute causally to long-range dependency?
**Registration**: PRE-REGISTERED 2026-04-27
**Redesign of**: e006 (FAILED — incorrect test selection: paired data tested as independent groups)

---

## HARKing Risk Declaration (required per R4)

Effect direction is pre-known from e006 post-hoc analysis (ΔNLL = +0.1407 nats, post-hoc
one-sample t=30.15, p=1.65e-17). This experiment is therefore **partially exploratory**:
the test design is pre-registered and correct, but the direction of the alternative hypothesis
is informed by prior results. This must be disclosed in the paper as a pre-registered
replication under known direction, not as a blind confirmatory test.

`retrofitted: false` — the test design (paired/one-sample) was decided before this PLAN.md
was written; only the direction of the result is pre-known.

---

## Hypothesis

**H3c_null**: The mean per-sequence ΔNLL (ablated − baseline) is not significantly
different from zero (μ = 0).

**H3c_alt**: The mean ΔNLL is significantly greater than zero, indicating that zeroing
Global attention output consistently degrades per-token prediction quality.

---

## Key fix from e006

e006 passed `(nll_ablated, nll_baseline)` as two independent groups to Welch t-test.
This inflates the t-statistic denominator with text-to-text variance (~0.28 nats) that
swamps the within-sequence treatment effect (~0.14 nats).

**Correct design**: compute `delta_nll[i] = nll_ablated[i] − nll_baseline[i]` per sequence,
then apply a one-sample t-test against μ=0 (equivalent to paired t-test).

---

## Method

1. Load EXAONE-4.5-33B with `nope_analysis.loader.load_model_and_tokenizer()`
2. Register forward hooks on each Global layer's self-attention output
3. Hook replaces attention output tensor with zeros (before residual add)
4. Run baseline forward pass (no hook) and ablated forward pass on each sequence
5. Compute `delta_nll[i] = mean_nll_ablated[i] − mean_nll_baseline[i]` per sequence
6. Primary statistical test: one-sample t-test on delta_nll against μ=0
7. Report: t-statistic, p-value, Cohen's d_z, bootstrap 95% CI on mean delta_nll
8. Call `auto_validate.py` at end

## Sample size

- **n = 50 sequences** (up from e006's n=20; power analysis: with d_z=6.74 from post-hoc,
  n=5 would be sufficient, but n=50 provides robust CI and tests for variance)
- Corpus: WikiText-103 (25 sequences, seq_len=2048) + KLUE-MRC (25 sequences, seq_len=2048)
- Seed: 42

## Beyond-window measurement (secondary, if memory allows)

Attempt sequences at seq_len=[4096, 5120, 6144] to measure ablation effect specifically
at token positions > 4096. With 2× A100 (170GB), seq_len=5120 may be feasible.
If OOM, document as Note (does not affect primary within-window result).

## Decision Criteria

```yaml
primary_metric: delta_nll
test: one_sample_t  # one-sample t-test on per-sequence delta_nll, μ=0
threshold_p: 0.05
threshold_effect: 0.3  # Cohen's d_z (medium)
min_n: 50
verdict_logic:
  VALIDATED: p < 0.05 AND d_z > 0.3 AND n >= 50
  FAILED: p >= 0.05 OR d_z <= 0.3
  INCONCLUSIVE: n < 50 (OOM or corpus failure)
retrofitted: false
```

## VRAM budget

- Model (bfloat16): ~65GB across 2× A100
- Hook overhead per sequence (seq_len=2048): ~200MB
- Two forward passes per sequence: ~400MB overhead
- Total: ~66GB — fits in 170GB. seq_len=5120 estimated ~1.2GB overhead → ~67GB total.
