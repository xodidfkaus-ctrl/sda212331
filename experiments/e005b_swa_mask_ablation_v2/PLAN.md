# PLAN — e005b SWA Mask Injection Ablation v2

**Status**: PENDING
**Script**: `nope_analysis/experiments/exp3b_swa_mask_ablation_v2.py`
**Output**: `outputs/e005b_swa_mask_ablation_v2/`
**RQ**: RQ3 — Does NoPE Global contribute causally to long-range dependency?
**Registration**: PRE-REGISTERED 2026-04-27
**Redesign of**: e005 (INCONCLUSIVE — OOM at seq_len ≥ 5120; within-window only; wrong measurement granularity)

---

## HARKing Risk Declaration

e005 produced zero beyond-window data points (OOM). No effect was observed.
e005b is not informed by any result signal — it is a straightforward re-run enabled by
the hardware upgrade (2× A100, 170GB). Within-window ΔNLL=0 was expected and is
not a result to adjust around.

Three deviations from e005's PLAN.md are fixed here:
1. Per-token NLL at positions > 4096 (not whole-sequence comparison)
2. Real corpus (WikiText-103 / KLUE-MRC, now cached) instead of synthetic fallback
3. auto_validate.py call regardless of whether beyond-window group is populated

`retrofitted: false`

---

## Hypothesis

**H3b_null**: Forcing Global layers to attend like SWA (window=4096 mask) does not
significantly increase per-token NLL at positions > 4,096.

**H3b_alt**: Constraining Global to a SWA-like window causes a statistically significant
and practically meaningful NLL increase at positions > 4,096, indicating that NoPE
Global's unrestricted attention provides causal benefit for long-range dependency.

---

## Key fixes from e005

1. **Per-token NLL filtering**: collect per-token log-likelihood; compute mean NLL only
   for token positions > 4096 (not whole-sequence mean). This correctly isolates the
   beyond-window regime where the ablation has a theoretical effect.
2. **Real corpus**: WikiText-103 (English) + KLUE-MRC (Korean). EDGAR sections (avg
   10,600 tokens) preferred for beyond-window measurement as they are naturally long.
3. **Paired design**: each sequence run twice (baseline, masked); analyze delta_nll per
   sequence to remove text-level variance (lesson from e006 FAILED).
4. **auto_validate.py always called**: even if beyond-window group is smaller than planned.

---

## Method

1. Load EXAONE-4.5-33B with `nope_analysis.loader.load_model_and_tokenizer()`
2. Register `register_forward_pre_hook` on each Global layer's attention module
3. Hook injects sliding window causal mask (window=4096) before softmax
4. For each sequence:
   a. Run baseline forward pass → collect per-token log-likelihood tensor
   b. Run masked forward pass → collect per-token log-likelihood tensor
   c. Compute delta_nll per token: masked_nll[t] − baseline_nll[t]
   d. Split by token position: within_window (t ≤ 4096) vs beyond_window (t > 4096)
5. Primary metric: mean delta_nll at positions > 4096
6. Statistical test: one-sample t-test on per-sequence mean delta_nll (beyond-window)
7. Call `auto_validate.py` at end (with available data, even if n < target)

## Sample size and sequence lengths

- Target: n = 20 sequences at seq_len = [5120, 6144, 7168, 8192]
- Corpus: EDGAR (en_edgar, avg 10,600 tokens → truncate to target length)
  - EDGAR sections naturally exceed 4096 tokens, avoiding stitching artifacts
- Seed: 42

## Decision Criteria

```yaml
primary_metric: delta_nll_beyond_window  # mean delta NLL at token positions > 4096
test: one_sample_t  # on per-sequence mean delta_nll at beyond-window positions, μ=0
threshold_p: 0.05
threshold_effect: 0.5  # Cohen's d_z (medium, per RQ3 operational definition Section 2b)
min_n: 10  # beyond-window sequences
verdict_logic:
  VALIDATED: p < 0.05 AND d_z > 0.5 AND n >= 10
  FAILED: p >= 0.05 OR d_z <= 0.5 (with n >= 10)
  INCONCLUSIVE: n < 10 (OOM prevents beyond-window measurement)
retrofitted: false
```

## VRAM budget

- Model (bfloat16): ~65GB across 2× A100
- Hook overhead (seq_len=8192, eager attn not used — hook only intercepts mask): ~50MB
- Two forward passes per sequence: ~130MB overhead per sequence at 8192t
- Total: ~66GB — fits within 170GB
- Key: this hook intercepts the mask argument, NOT the full attention matrix.
  No n² memory required. OOM risk is low.
- torch.cuda.empty_cache() between sequences
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
