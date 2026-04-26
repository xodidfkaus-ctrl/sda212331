# PLAN — e004 Positional Probe v2 (Train/Test Split)

**Status**: DONE
**Script**: `nope_analysis/experiments/exp2b_positional_probe_v2.py`
**Output**: `outputs/exp2b_positional_probe_v2/`
**RQ**: RQ1 — Do NoPE Global layers encode positional information?
**Registration**: RETROFITTED (results existed before this PLAN.md)

---

## Hypothesis

Motivated by e003 data leakage. Working hypothesis: with a proper train/test split,
Global layers will show lower probe accuracy than SWA layers, since NoPE Global
layers lack RoPE and may not encode position as strongly.

**Predicted direction**: probe_accuracy(Global) < probe_accuracy(SWA)

Result: Global ≈ SWA (within 0.004 at all lengths). Hypothesis not confirmed.

---

## Method

- 30 prompts, 3 lengths (64, 128, 256 tokens), from [corpus not recorded]
- 80/20 train/test split (not k-fold)
- `sklearn.LogisticRegression`, max_iter=300
- 10-class classification (position bin)
- Metric: test accuracy (fraction of test tokens correctly classified by position bin)
- Run on CPU (sklearn limitation; A100 idle)

---

## Decision Criteria

```criteria
metric: test_probe_accuracy
direction: global_nope < swa  # predicted direction
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 30
n_simultaneous_tests: 3  # three length conditions: 64, 128, 256
bootstrap_n: 10000
retrofitted: true
note: |
  n=30 prompts with 80/20 split: test set has ~6 samples per class (60 total for 10-class).
  Severely underpowered. Overfit gap ~0.45 (train=1.000, test=0.52-0.55).
  Result is suggestive but not paper-grade. Superseded by e004_upgraded (planned)
  and e008 (delta probe).
```

---

## Stopping rules

N/A — experiment completed.
Note: this experiment should be upgraded (≥300 prompts, PyTorch, k-fold) before
citing in the paper. See HANDOVER.md Section 6 "Exp 2b Upgraded."
