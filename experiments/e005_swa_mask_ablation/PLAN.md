# PLAN — e005 SWA Mask Injection Ablation

**Status**: PENDING
**Script**: `nope_analysis/experiments/exp3b_swa_mask_ablation.py`
**Output**: `outputs/exp3b_swa_mask_ablation/` (not yet created)
**RQ**: RQ3 — Does NoPE Global contribute causally to long-range dependency?
**Registration**: PRE-REGISTERED 2026-04-26 (git commit ef34bf7 or later)

---

## Hypothesis

**H3b_null**: Forcing Global layers to attend like SWA (same window mask) does not
significantly change the model's perplexity on tokens at positions > 4,096.

**H3b_alt**: Constraining Global to a SWA-like window causes a statistically
significant and practically meaningful perplexity increase at positions > 4,096,
indicating that NoPE Global's unrestricted attention provides causal benefit for
long-range dependency.

---

## Method

1. Load EXAONE-4.5-33B with `attn_implementation='eager'`
2. Register `register_forward_pre_hook` on each Global layer's attention module
3. Hook injects a sliding window causal mask (window = 4,096) into the attention
   score matrix before softmax, replacing the full causal mask
4. Run forward pass on each test sequence (WikiText-103 / KLUE-MRC corpus)
5. Compute per-token negative log likelihood
6. Compare NLL at positions > 4,096 vs positions ≤ 4,096 between:
   - Baseline: unmodified model
   - Ablated: SWA mask injected into Global layers
7. Comparison: ablated_NLL_at_long_range vs baseline_NLL_at_long_range

### Corpus

- WikiText-103 (English): ≥ 20 sequences of 4,096–8,192 tokens
- KLUE-MRC (Korean): ≥ 20 sequences of 4,096–8,192 tokens
- Total: ≥ 40 sequences (≥ 20 per language)
- Seed: `EXPERIMENT_SEEDS["e005"]` from `nope_analysis/seeds.py`

---

## Decision Criteria

```criteria
metric: perplexity_delta_long_range
direction: ablated > baseline
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
conditions:
  token_positions: "> 4096"
min_n_per_group: 20
n_simultaneous_tests: 1
bootstrap_n: 10000
retrofitted: false
```

**Secondary metric** (reported but not used for verdict): perplexity at positions
≤ 4,096 (should NOT change significantly — serves as sanity check).

---

## Stopping rules

1. OOM during forward pass at 8,192 tokens → reduce to 5,120 tokens (>4,096 still
   tests the SWA window boundary). Document the reduction here.
2. If VRAM < 80GB → experiment cannot run. Request full A100 instance.
3. If hook causes NaN outputs → debug attention mask dtype (likely float/bool mismatch;
   see exp3b_swa_mask_ablation.py for known fix).

---

## Required before running

- [ ] Corpus downloaded: `python3 -c "from nope_analysis.corpus.downloader import download_all; download_all()"`
- [ ] Full A100 80GB confirmed: `nvidia-smi --query-gpu=memory.total --format=csv,noheader`
- [ ] PLAN.md committed to git (this file) ← required by CLAUDE.md design rule
- [ ] `auto_validate.py` called at experiment exit ← required by CLAUDE.md design rule

---

## Analysis code

Call at experiment exit:
```python
from nope_analysis.analysis.auto_validate import validate_experiment
validate_experiment(
    experiment_id="e005",
    comparisons={
        "perplexity_delta_long_range": (ablated_nll_list, baseline_nll_list),
    },
    output_dir=Path("outputs/exp3b_swa_mask_ablation"),
)
```
