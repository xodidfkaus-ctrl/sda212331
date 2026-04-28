# PLAN — e009 Needle-in-Haystack Retrieval

**Experiment ID**: e009
**Slug**: needle_in_haystack
**Script**: `nope_analysis/experiments/exp9_needle_in_haystack.py`
**Output**: `outputs/e009_needle_in_haystack/`
**RQ**: RQ3 behavioral validation — does Global attention causally enable retrieval beyond the SWA window?
**Registration**: PRE-REGISTERED 2026-04-28
**retrofitted**: false

---

## Motivation

e005b and e006b validated RQ3 via perplexity proxy (mechanistic).
No behavioral test exists in this study. A hostile reviewer will ask:
"Does the perplexity change actually translate to real task failure?"

This experiment provides the behavioral counterpart:
if Global ablation causes retrieval failure at positions > 4096 tokens,
that is direct causal evidence that Global enables real long-range retrieval.

HANDOVER Section 10 already identifies this as the recommended external validity test.

---

## Hypothesis

**H_NIH_alt** (Global enables long-range retrieval):
> Retrieval accuracy at needle position > 4096 tokens drops significantly
> when Global attention output is zeroed, relative to baseline.
> Metric: hit_rate(baseline) - hit_rate(ablated) > 0, p ≤ 0.05 (binomial/McNemar).

**H_NIH_null**:
> No significant retrieval difference between baseline and Global-ablated model.

---

## Method

### Needle construction
- Needle: unique key-value string inserted into haystack text
  - Format: "The verification code for this section is EXAONE[6-digit-random]."
  - Chosen to be unambiguous, not in training data
- Retrieval query: "What is the verification code mentioned in the text?"
- Hit criterion: generated output contains the exact 6-digit code

### Haystack construction
- Source: EDGAR corpus (`en_edgar`, avg 10,600 tokens per section)
- Select documents with tokenized length ≥ 6000 tokens
- Insert needle at token position 4200–5000 (beyond SWA window boundary of 4096)
- Total input length: 5500–7000 tokens

### Ablation
- Same hook approach as e006b: zero out `self_attn` output at all 16 Global layers
- Registered via `register_forward_hook` during `model.generate()`

### Generation
- `model.generate(input_ids, max_new_tokens=30, do_sample=False)`
- Temperature=0 (greedy) for reproducibility

### Statistical test
- **McNemar's test** (paired: same haystack × needle pair, baseline vs ablated)
- n = 50 unique (haystack, needle) pairs
- Report: hit_rate_baseline, hit_rate_ablated, McNemar chi², p-value, odds ratio

---

## Decision Criteria

```yaml
experiment_id: e009
hypothesis: H_NIH_alt
test: McNemar
direction: baseline_hit_rate > ablated_hit_rate
thresholds:
  p_value: 0.05
  min_pairs: 50
  min_baseline_hit_rate: 0.3   # if baseline itself can't retrieve, test is uninformative
retrofitted: false
```

- VALIDATED: p ≤ 0.05, baseline_hit_rate ≥ 0.3, direction correct
- FAILED: p > 0.05
- INCONCLUSIVE: baseline_hit_rate < 0.3 (model can't retrieve even without ablation → needle setup invalid)

---

## Runtime estimate
- 50 pairs × 2 conditions (baseline + ablated) × ~5s per generate = ~8 min
- Model already loaded in memory from e008

## Required before running
- [ ] e008 completed (model unloaded, VRAM freed)
- [ ] PLAN.md committed
- [ ] `auto_validate.py` updated for McNemar test support (or manual stats)
