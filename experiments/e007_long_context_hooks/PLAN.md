# PLAN — e007 Long Context Hook Analysis

**Status**: PENDING
**Script**: `nope_analysis/experiments/exp4_long_context.py`
**Output**: `outputs/exp4_long_context/` (not yet created)
**RQ**: RQ3 — Does NoPE Global continue to diverge from SWA at lengths > SWA window?
**Registration**: PRE-REGISTERED 2026-04-26

---

## Hypothesis

**H3d_null**: Global and SWA attention distance do not diverge significantly at
lengths ≥ 4,096 tokens; Global distance growth rate is not higher than SWA's.

**H3d_alt**: At sequence lengths ≥ 4,096 tokens, Global attention distance
continues to grow with sequence length while SWA distance plateaus near the
4,096-token window limit, with the divergence statistically significant at
≥ 2 length conditions.

---

## Method

1. Register forward hooks to extract per-head attention weights at each layer
2. Compute mean attention distance = sum(pos × attn_weight) for each head
3. Test at 4 length conditions: [2048, 4096, 6144, 8192] tokens
4. ≥ 5 distinct texts per length condition (memory-constrained)
5. Compare Global vs SWA attention distance at each length
6. Test for saturation: SWA distance at 6144 and 8192 should not significantly
   exceed SWA distance at 4096 (window constraint); Global should continue growing

### Corpus

WikiText-103 sequences truncated/padded to target lengths.
Seed: `EXPERIMENT_SEEDS["e007"]`.

---

## Decision Criteria

```criteria
metric: attn_distance_divergence
direction: global_nope > swa
accept:
  min_abs_cohen_d: 0.5
  max_p_holm_corrected: 0.01
  min_length_conditions_significant: 2
min_n_per_group: 5
n_simultaneous_tests: 4
bootstrap_n: 10000
retrofitted: false
note: |
  n=5 is below standard minimum (30). Results tagged EXPLORATORY regardless of p-value.
  This is a memory-constrained experiment (each forward pass requires ~70GB at 8192 tokens).
  The 5-sequence minimum is the practical limit on a single A100 80GB.
```

---

## Stopping rules

1. OOM at 8,192 tokens → remove that condition; report on [2048, 4096, 6144] only
2. OOM at 6,144 tokens → report on [2048, 4096] only; note severe limitation
3. If n < 5 for any condition → tag that condition as INSUFFICIENT_DATA, exclude
   from Holm correction, include in exploratory section of paper

---

## Required before running

- [ ] e005 and e006 completed (to contextualize RQ3 verdict)
- [ ] Full A100 80GB confirmed
- [ ] PLAN.md committed
- [ ] `auto_validate.py` called at exit
