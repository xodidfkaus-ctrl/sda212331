# PLAN — e008 Delta-Attribution Probe

**Status**: PENDING
**Script**: `nope_analysis/experiments/exp2c_delta_probe.py` (not yet written)
**Output**: `outputs/exp2c_delta_probe/` (not yet created)
**RQ**: RQ1 — Does Global actively encode position (H1) or propagate SWA signal (H2)?
**Registration**: PRE-REGISTERED 2026-04-26

---

## Hypothesis

**H1_null** (Global propagates only):
> The probe accuracy delta at Global layers (acc(h_out) − acc(h_in)) is not
> significantly different from zero, and not significantly larger than the delta
> observed at SWA layers.

**H1_alt** (Global independently encodes):
> At one or more Global layer indices, the probe accuracy delta is significantly
> greater than zero (p_Holm ≤ 0.01, Cohen's d ≥ 0.5), AND the Global layer delta
> is significantly larger than the mean SWA layer delta.

---

## Method

1. Load EXAONE-4.5-33B
2. Register forward hooks to capture `h_in[i]` and `h_out[i]` at each layer i
   - `h_in[i]` = residual stream entering layer i (before attention sublayer)
   - `h_out[i]` = residual stream exiting layer i (after attention + residual add)
3. Run forward pass on ≥ 300 prompts (WikiText-103 / KLUE-MRC, 3 lengths: 64/128/256)
4. For each layer i and each length:
   a. Train PyTorch linear probe on h_in[i] (80% of prompts)
   b. Train PyTorch linear probe on h_out[i] (80% of prompts)
   c. Evaluate both on held-out 20%
   d. delta_acc[i] = mean test_acc(h_out[i]) − mean test_acc(h_in[i]) over k-fold folds
5. Compare delta_acc at Global layers vs delta_acc at SWA layers

### Probe specification (PyTorch, GPU)

```python
probe = nn.Linear(hidden_size, n_bins)  # n_bins = 10 position classes
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
# Train for 100 epochs with cross-entropy loss
# k-fold: k=5, stratified by position class
```

### Corpus

- ≥ 300 prompts total from WikiText-103 (English) and KLUE-MRC (Korean)
- Balanced: 150 per language
- Seed: `EXPERIMENT_SEEDS["e008"]`

---

## Decision Criteria

```criteria
metric: probe_accuracy_delta
direction: h_out > h_in  # for Global layers
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
min_n_per_group: 300
n_simultaneous_tests: 16  # one per Global layer index
bootstrap_n: 10000
retrofitted: false
note: |
  Primary comparison: delta_acc at Global layers vs zero.
  Secondary comparison: delta_acc at Global layers vs delta_acc at SWA layers.
  Both must be significant to accept H1_alt.
```

---

## Stopping rules

1. If PyTorch probe does not converge → increase max epochs to 300, reduce lr to 1e-4
2. If VRAM insufficient for 300 prompts × 256 tokens → batch extraction in chunks of 50
3. If n < 100 prompts due to corpus availability → tag result as EXPLORATORY

---

## Required before running

- [ ] e004 completed (e008 is the confirmatory follow-up)
- [ ] Full A100 80GB confirmed
- [ ] PLAN.md committed to git (this file)
- [ ] PyTorch probe implemented (not sklearn — see HANDOVER.md Section 7c)
- [ ] `auto_validate.py` called at exit

---

## Analysis code template

```python
from nope_analysis.analysis.auto_validate import validate_experiment
validate_experiment(
    experiment_id="e008",
    comparisons={
        f"delta_acc_layer_{i}": (delta_global[i], delta_zero_baseline)
        for i in global_layer_indices
    },
    output_dir=Path("outputs/exp2c_delta_probe"),
)
```
