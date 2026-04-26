# PLAN — e001 Attention Entropy (Short Input)

**Status**: DONE
**Script**: `nope_analysis/experiments/exp1_attention_entropy.py`
**Output**: `outputs/exp1_attention_entropy/`
**RQ**: RQ2 — Do SWA and Global exhibit different attention patterns?
**Registration**: RETROFITTED (results existed before this PLAN.md was written)

---

## Hypothesis

**Exploratory — no formal pre-registration.**

Working assumption at run time: Global (NoPE) attention layers, lacking positional
encoding, might attend more uniformly than SWA layers even on short inputs, producing
lower entropy or different attention distance.

This was the first experiment in the project. No decision criteria were pre-registered.

---

## Method

- Input: 3 short prompts (14–26 tokens) in Korean
- Model: EXAONE-4.5-33B loaded with `attn_implementation='eager'`
- Measure per head, per layer: attention entropy H = -sum(p log p),
  mean attention distance = sum(pos * attn_weight)
- Group by layer type: Global (indices 3,7,11,...,63) vs SWA (all others)
- Single forward pass per prompt; aggregate across heads and prompts

---

## Decision Criteria

```criteria
metric: entropy
direction: "any difference between global_nope and swa"
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 10
n_simultaneous_tests: 1
bootstrap_n: 10000
retrofitted: true
note: "Exploratory. No pre-registered hypothesis. Decision criteria written post-hoc."
```

---

## Stopping rules

N/A — experiment already completed.

---

## Analysis code

`nope_analysis/analysis/statistical_tests.py` — Welch t-test, Cohen's d, bootstrap CI.
Note: `auto_validate.py` was not called in the original run (created later).
Statistical tests must be applied retroactively in ANALYSIS.md.
