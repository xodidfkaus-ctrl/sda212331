# PLAN — e006 Global Zero-Output Ablation

**Status**: PENDING
**Script**: `nope_analysis/experiments/exp3_swa_ablation.py`
**Output**: `outputs/exp3_swa_ablation/` (not yet created)
**RQ**: RQ3 — Does NoPE Global contribute causally to long-range dependency?
**Registration**: PRE-REGISTERED 2026-04-26

---

## Hypothesis

**H3c_null**: Zeroing the self-attention output of all Global layers does not
significantly degrade perplexity.

**H3c_alt**: Removing Global attention output causes a statistically significant
and practically meaningful perplexity increase, indicating that Global attention
is a causally necessary component (not redundant).

---

## Method

1. Load EXAONE-4.5-33B
2. Register forward hooks on Global layers' self-attention output
3. Hook replaces the attention output tensor with zeros (before residual add)
4. Run forward pass on WikiText-103 / KLUE-MRC sequences
5. Compare per-token NLL: ablated vs baseline

**Note**: This is methodologically more aggressive than e005 (zeros the entire
attention output, not just the mask). Comparison of e005 and e006 results will
reveal whether the effect is due to the attention pattern or the attention computation
itself. This comparison is scientifically valuable regardless of direction.

### Corpus

Same as e005: ≥ 20 sequences per language (≥ 40 total), 4,096–8,192 tokens.
Same seed: `EXPERIMENT_SEEDS["e006"]` from `nope_analysis/seeds.py`.

---

## Decision Criteria

```criteria
metric: perplexity_delta
direction: ablated > baseline
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 20
n_simultaneous_tests: 1
bootstrap_n: 10000
retrofitted: false
```

---

## Stopping rules

Same as e005. Run e005 first to confirm full A100 80GB is available before running e006.

---

## Required before running

- [ ] e005 completed (results in hand for comparison)
- [ ] Full A100 80GB confirmed
- [ ] PLAN.md committed
- [ ] `auto_validate.py` called at exit
