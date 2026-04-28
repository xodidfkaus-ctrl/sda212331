# PLAN — e010 Attention Locality Analysis

**Experiment ID**: e010
**Slug**: attention_locality
**Script**: `nope_analysis/experiments/exp10_attention_locality.py`
**Output**: `outputs/e010_attention_locality/`
**RQ**: RQ3 mechanistic supplement — why does Global attention distance < SWA at long sequences (e007b direction reversal)?
**Registration**: PRE-REGISTERED 2026-04-28
**retrofitted**: false

---

## Motivation

e007b found that Global attention distance < SWA distance at all lengths ≥ 2048,
with the gap growing with sequence length (d = -1.481 at 8192 tokens).
This was pre-registered as direction Global > SWA → FAILED.

The attn_collapse_diag (n=3, diagnostic, not pre-registered) showed:
- sink_frac: Global ≈ SWA (0.039 vs 0.038) → attention sink hypothesis FALSIFIED
- local_frac: Global ≈ SWA → local token hypothesis also not supported
- mean_distance: Global < SWA, same direction as e007b

The true cause of direction reversal is not yet understood.
This experiment tests the next candidate explanation:

**SWA window-forcing hypothesis**:
> SWA is geometrically forced to attend within the 4096-token window.
> For a query at position p and uniform attention within [p-4096, p],
> mean distance ≈ 2048 regardless of p.
> Global, unconstrained, can attend locally when that is more informative,
> pulling its mean distance below SWA's geometrically-enforced baseline.

If this is correct:
- Beyond-window attention fraction (pos < current - 4096) in Global should be LOW
- SWA beyond-window fraction should be near 0 (by definition — window mask)
- The DIFFERENCE in distance is explained by SWA's forced window, not Global's local bias

---

## Hypothesis

**H_LOC_alt** (window-forcing explains direction reversal):
> The fraction of Global attention directed to positions beyond the SWA window
> (token distance > 4096) is < 5% of total attention weight.
> This low beyond-window usage is consistent with the finding that Global's
> causal contribution beyond the window is ~10% (e005b).

**H_LOC_null**:
> Global uses beyond-window positions substantially (≥ 15% of attention weight).
> The direction reversal in e007b would then require a different explanation.

---

## Method

### Measurement
For each layer, for each query token at position p:
```
beyond_window_frac = sum(attn_weights[p, :p-4096]) / sum(attn_weights[p, :])
within_window_frac = sum(attn_weights[p, p-4096:p]) / sum(attn_weights[p, :])
```

Use sparse Q·K dot-product hook (same approach as e007b, stride=128 sampling).

### Conditions
- seq_len: [4096+512, 4096+2048, 8192] = [4608, 6144, 8192]
  (only lengths where beyond-window is possible)
- Layer types: Global (16 layers) vs SWA (48 layers, beyond-window = 0 by mask)
- n = 30 sequences per length condition (WikiText-103 + EDGAR)

### Statistical test
- One-sample t-test: H0: mean beyond_window_frac = 0.15 (null threshold)
- Test direction: observed < 0.15 → supports H_LOC_alt
- Report: mean beyond_window_frac per Global layer, per length
- Also report: correlation between layer depth and beyond_window_frac

---

## Decision Criteria

```yaml
experiment_id: e010
hypothesis: H_LOC_alt
test: one_sample_t
direction: beyond_window_frac < 0.15
thresholds:
  p_value: 0.05
  min_n: 30
  null_mean: 0.15
retrofitted: false
```

- VALIDATED: mean beyond_window_frac < 0.15, p ≤ 0.05
- FAILED: mean beyond_window_frac ≥ 0.15
- INCONCLUSIVE: n insufficient or OOM at target seq_len

## Contribution to paper

If VALIDATED: e007b direction reversal is explained mechanistically.
The paper can state: "Global layers have the capacity for beyond-window attention,
but rarely exercise it (< X%). The ~10% causal contribution (e005b) is consistent
with this low utilization rate."

If FAILED: beyond-window attention is substantial but doesn't produce distance advantage.
Alternative explanation needed. Document in NEGATIVE_RESULTS.md.

---

## Runtime estimate
- 30 × 3 lengths × sparse hook (stride=128) × ~2–3s per pass = ~5 min
- Model load required (~15 min)

## Required before running
- [ ] e008 completed
- [ ] PLAN.md committed
- [ ] Verify sparse hook implementation from e007b is reusable
