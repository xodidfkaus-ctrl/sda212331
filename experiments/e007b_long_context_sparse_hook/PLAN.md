# PLAN — e007b Long Context Sparse Hook Analysis

**Status**: PENDING
**Script**: `nope_analysis/experiments/exp4_long_context_sparse_hook.py`
**Output**: `outputs/e007b_long_context_sparse_hook/`
**RQ**: RQ3 — Does Global attention distance diverge from SWA at lengths > SWA window?
**Registration**: PRE-REGISTERED 2026-04-27
**Redesign of**: e007 (INCONCLUSIVE — OOM from output_attentions=True memory fragmentation)

---

## HARKing Risk Declaration

Effect direction is pre-known from e007's single completed sample (Global dist=549.5,
SWA dist=441.6 at 2048t; gap=107.9). This is consistent with e002/e001b direction.
e007b tests the same hypothesis with a feasible memory approach. Result direction is
partially anticipated but the statistical significance at beyond-window lengths (≥4096)
is genuinely unknown (e007 had zero beyond-window samples).

`retrofitted: false` — sparse hook design was specified in e007's ANALYSIS.md before
this PLAN.md was written.

---

## Hypothesis

**H3d_null**: Global and SWA attention distance do not diverge significantly at lengths
≥ 4,096 tokens. Global distance growth rate is not higher than SWA's.

**H3d_alt**: At sequence lengths ≥ 4,096 tokens, Global attention distance continues
to grow while SWA distance plateaus near the 4,096-token window limit, with divergence
statistically significant at ≥ 2 length conditions.

---

## Key fix from e007

`output_attentions=True` stores the full n×n attention matrix per layer.
At 2048t: 2048² × 64 layers × 2 bytes = 536MB — causes CUDA memory fragmentation after
the first forward pass, blocking re-allocation for the second sample.

**Fix — Q·K sparse hook (Option A from e007 ANALYSIS.md)**:
Register hooks on Q and K projection outputs. Sample a subset of query positions
(stride=128 → ~64 queries per 8192-token sequence). Compute Q_subset · K^T
(64 × seq_len per layer instead of seq_len × seq_len). Apply causal mask and softmax.
Compute attention distance from sampled positions only.

Memory at 8192t: 64 × 8192 × 64 layers × 2 bytes ≈ 67MB (vs. 8192² × 64 × 2 = 8.6GB).

---

## Method

1. Load EXAONE-4.5-33B with `nope_analysis.loader.load_model_and_tokenizer()`
2. Register hooks on Q and K projection outputs of each attention layer
   - Hook captures Q[::stride, :] (sampled query rows) and full K
   - Computes sparse attention scores: Q_sub @ K.T / sqrt(head_dim)
   - Applies causal mask (upper-triangular fill with -inf at sampled positions)
   - Applies softmax → sparse_attn (n_sampled × seq_len)
   - Computes mean attention distance = sum(pos × attn_weight, dim=-1).mean()
3. stride = 128 (every 128th query position sampled)
4. Test at 5 length conditions: [2048, 4096, 5120, 6144, 8192] tokens
5. n ≥ 10 distinct texts per length condition (from WikiText-103 / KLUE-MRC)
6. Separate analysis for Global layers vs SWA layers
7. Statistical test per length condition: Welch t-test + Cohen's d (Global vs SWA distance)
8. Saturation test: compare SWA distance at 4096 vs 6144, 8192 (should not grow)
9. Call `auto_validate.py` at end

## Sample size

- n = 10 per length condition × 5 conditions = 50 total forward passes
- Corpus: WikiText-103 (long passages) stitched to reach target lengths
- Seed: 42

## Decision Criteria

```yaml
primary_metric: attn_distance_divergence
test: welch_t  # Global vs SWA distance at each length condition
threshold_p: 0.05
threshold_effect: 0.5  # Cohen's d (medium)
min_n_per_condition: 5
min_conditions_significant: 2  # per RQ3 operational definition (Section 2b)
verdict_logic:
  VALIDATED: significant divergence (p<0.05, d>0.5) at >= 2 length conditions >= 4096t
  FAILED: no significant divergence at any beyond-window condition
  INCONCLUSIVE: n < 5 at any tested beyond-window condition, or all OOM
retrofitted: false
```

## VRAM budget

- Model (bfloat16): ~65GB across 2× A100
- Sparse hook overhead per forward pass (8192t): ~67MB (negligible)
- Total: ~65GB — comfortably within 170GB
- torch.cuda.empty_cache() between each sample to prevent fragmentation
- Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for defragmentation
