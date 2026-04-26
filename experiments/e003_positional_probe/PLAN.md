# PLAN — e003 Positional Probe (Short Input, Same-split)

**Status**: DONE
**Script**: `nope_analysis/experiments/exp2_positional_probe.py`
**Output**: `outputs/exp2_positional_probe/`
**RQ**: RQ1 — Do NoPE Global layers encode positional information?
**Registration**: RETROFITTED (results existed before this PLAN.md)

---

## Hypothesis

Working hypothesis at run time: NoPE Global layers, lacking RoPE, will have lower
linear probe accuracy for predicting token position from hidden states, compared
to SWA layers (which use RoPE and therefore embed explicit positional information).

**Predicted direction**: probe_accuracy(Global) < probe_accuracy(SWA)

This prediction was falsified by the result (both = 1.000).

---

## Method

- Input: short prompts (estimated 20–50 tokens; exact prompts not recorded)
- For each layer, extract hidden states h_i for each token
- Train `sklearn.LogisticRegression` to classify token position from h_i
- Accuracy reported as fraction of correct position classifications
- CRITICAL FLAW: train and test on the same data — no held-out test set

---

## Decision Criteria

```criteria
metric: probe_accuracy
direction: global_nope < swa  # predicted before run; falsified
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 10
n_simultaneous_tests: 1
bootstrap_n: 10000
retrofitted: true
note: |
  No train/test split was applied. accuracy=1.000 for both groups is a data
  leakage artifact, not a finding. This experiment is superseded by e004.
```

---

## Stopping rules

N/A — experiment completed. Superseded by e004.
