# ANALYSIS — e007 Long Context Hook Analysis

**Experiment ID**: e007
**Script**: `nope_analysis/experiments/exp4_long_context.py`
**Output**: `outputs/exp4_long_context/`
**RQ**: RQ3 — Does Global attention distance diverge from SWA at lengths > SWA window?
**Date run**: 2026-04-27
**auto_validate verdict**: **INCONCLUSIVE** (force_verdict; n=1 sample, all beyond-window OOM)

---

## Primary Result

```
[attn_distance_divergence] global_nope=549.5±122.0 vs swa=441.6±112.2
diff=+107.9  t=3.123  p=0.0046  d=0.941(large)
Overall: INCONCLUSIVE — force_verdict in PLAN.md (underpowered: n_samples=1)
```

`outputs/exp4_long_context/stats.json` — `overall_verdict: "INCONCLUSIVE"`

---

## OOM Failure — Worse Than Predicted

**Prediction**: Model (~67GB) + per-layer attention hook overhead ≈ ~70GB < 85.2GB available.

**Actual**: OOM at the **second sample of 2048 tokens**. Only 1 sample completed.

| Length | Completed samples | Status |
|--------|-------------------|--------|
| 2048 | **1/5** | sample 0 ok, OOM on sample 1 |
| 3072–8192 | 0/5 | OOM on sample 0 |

**Root cause**: CUDA memory fragmentation. After the first forward pass, freed blocks are
held fragmented by the CUDA caching allocator. `torch.cuda.empty_cache()` between samples
(exp4_long_context.py:189) did not defragment sufficiently. The second allocation for
contiguous attention matrix blocks failed even with theoretically sufficient total free memory.

`output_attentions=True` forces eager attention (full n² attention matrix per layer),
which is incompatible with multi-sample runs on this hardware budget after model load.

---

## Single Data Point (2048 tokens, 1 sample)

| Metric | Global (NoPE) | SWA | Gap |
|--------|--------------|-----|-----|
| Attention distance (tokens) | **549.5 ± 122.0** | **441.6 ± 112.2** | **+107.9** |
| n_layers | 16 | 48 | — |
| t-stat | 3.123 | p=0.0046 | d=0.941 (large) |

**Warning — pseudo-replication**: Variance above is cross-layer, not cross-sample. 16 Global
layers and 48 SWA layers from the same sequence are not independent observations. The
statistical test cannot be interpreted as sample-level evidence.

**Direction replication**: e002 showed Global dist=643.7, SWA dist=489.1 at 2048 tokens
(different corpus). Same direction, similar magnitude. Consistent but not confirmatory.

---

## Contribution to RQ3 Verdict

- **Condition 1 (e005)**: INCONCLUSIVE (OOM at ≥5120)
- **Condition 2 (e006)**: FAILED (test selection error; beyond-window n=0)
- **Condition 3 (e007)**: INCONCLUSIVE (OOM at ≥3072; n=1 within-window only)

**RQ3 verdict to date**: 0 of 3 conditions confirmed. Beyond-window measurement has
failed in all three conditions due to OOM. Current conclusion:
**no confirmatory evidence for Global causal long-range contribution at tested lengths.**

---

## OOM Mitigation for e007b

`output_attentions=True` is incompatible with the VRAM budget for multi-sample runs.

**Recommended — Option A (Q·K sparse hook)**:
Register hooks on Q and K projection outputs. Sample a subset of query positions
(stride=128 → 64 queries per 8192-token sequence). Compute Q_subset · K^T (64 × seq
instead of seq × seq). Memory at 8192 tokens: 42MB per layer vs 5.4GB full.
Attention distance from sampled positions is unbiased if stride is uniform.

Pre-register before implementing. This is a design change from e007 — new experiment ID required.

---

## Exploratory Observations (not confirmatory)

1. At 2048 tokens (within SWA window), Global layers attend farther than SWA (gap=107.9 tokens).
   Suggests Global architecture intrinsically favors longer-range attention even inside the window.
2. Cross-layer variance is large (σ≈112–122 tokens), motivating per-layer analysis in e007b.
3. Both e002 and e007 show the same direction at 2048 tokens despite different corpora.

---

*ANALYSIS written: 2026-04-27. Reviewer: Claude Code (gatekeeper mode).*
