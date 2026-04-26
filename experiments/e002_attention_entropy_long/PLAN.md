# PLAN — e002 Attention Entropy (Long Input)

**Status**: DONE
**Script**: `nope_analysis/experiments/exp1b_long_input.py`
**Output**: `outputs/exp1b_long_input/`
**RQ**: RQ2 — Do SWA and Global exhibit different attention patterns?
**Registration**: RETROFITTED (results existed before this PLAN.md)

---

## Hypothesis

Motivated by e001 null result. Working hypothesis at run time:
At sequence lengths approaching or exceeding the SWA window (4,096 tokens), the
structural difference between Global (no window) and SWA (4,096-token window) should
produce measurably different entropy and attention distance.

**Predicted direction**: Global entropy ≤ SWA at lengths well below window (uniform
attention → lower entropy than SWA's focused window); Global distance > SWA at lengths
approaching window (Global attends farther, SWA is bounded).

Note: The entropy direction prediction was inverted by the actual result at 2,048
tokens (Global > SWA). This was not anticipated before running.

---

## Method

- Input: Single text per length condition (5 conditions: 128, 256, 512, 1024, 2048 tokens)
- Text: [not recorded in summary — likely repeated pattern or natural text]
- Model: EXAONE-4.5-33B, `attn_implementation='eager'`
- Measure: attention entropy and mean attention distance per head per layer
- Group: Global (16 layers) vs SWA (48 layers)
- Single forward pass per length

---

## Decision Criteria

```criteria
metric: entropy
direction: "global_nope vs swa (direction not pre-specified)"
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 10
n_simultaneous_tests: 5
bootstrap_n: 10000
retrofitted: true
note: |
  Single text input per length condition. n per group = n_layers × n_heads for that
  one text (not multiple independent texts). Between-length comparison confounded
  by document identity. Must rerun with >=10 distinct documents per length for
  confirmatory analysis.
```

---

## Stopping rules

N/A — experiment already completed.
