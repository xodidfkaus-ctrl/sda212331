# ANALYSIS — e005b SWA Mask Injection Ablation v2

**Experiment ID**: e005b
**Script**: `nope_analysis/experiments/exp3b_swa_mask_ablation_v2.py`
**Output**: `outputs/e005b_swa_mask_ablation_v2/`
**RQ**: RQ3 — Does NoPE Global contribute causally to long-range dependency?
**Date run**: 2026-04-27
**GPU**: 2× NVIDIA A100 80GB PCIe
**auto_validate verdict**: **VALIDATED**

---

## Primary Result

```
[delta_nll_beyond_window] mean=+0.0172±0.0165 vs null=0.0
t=4.676  p=0.0002  d_z=1.046(large)  ✓ significant
Overall: VALIDATED — criteria met (p<0.05, d_z>0.5, n=20)
```

`outputs/e005b_swa_mask_ablation_v2/stats.json` — `overall_verdict: "VALIDATED"`

---

## Results by Length

| seq_len | Δ_within (sanity) | Δ_beyond (primary) | n_beyond_tokens |
|---------|-------------------|--------------------|-----------------|
| 5120 | 0.0000 | +0.0166, −0.0024, +0.0080, −0.0016, +0.0042 | 1023 |
| 6144 | 0.0000 | +0.0348, +0.0071, +0.0108, −0.0012, +0.0113 | 2047 |
| 7168 | 0.0000 | +0.0403, +0.0174, +0.0093, +0.0010, +0.0300 | 3071 |
| 8192 | 0.0000 | +0.0559, +0.0307, +0.0179, +0.0123, +0.0418 | 4095 |

**Δ_within = 0.0000 exactly**: expected and correct. For token positions ≤ 4096, the
injected SWA mask is identical to the standard causal mask, so outputs are bit-identical.
This confirms the hook is correctly implemented.

**Δ_beyond trend**: The mean beyond-window effect increases with sequence length:
5120t (~0.005) < 6144t (~0.012) < 7168t (~0.020) < 8192t (~0.032). More tokens beyond
the window means a larger cumulative NLL increase when those positions lose access to
pre-window context.

**Overall** (n=20, pooled): mean Δ_beyond = +0.0172 nats, d_z = 1.046.

---

## Interpretation

H3b_null is rejected: forcing Global layers to attend only within the SWA window causes
a statistically significant and practically large increase in per-token NLL at positions
beyond 4,096 tokens (t=4.68, p=0.0002, d_z=1.05).

**Critical nuance — magnitude**:
The absolute effect (+0.017 nats) is ~10× smaller than e006b's full-ablation effect
(+0.169 nats). This ratio reveals the internal structure of Global attention's contribution:

| Ablation | What it removes | Δ_NLL |
|----------|----------------|-------|
| e006b: zero all Global output | All Global attention (within + beyond window) | +0.169 nats |
| e005b: SWA mask on Global | Only Global's beyond-window attention component | +0.017 nats |
| Difference (residual) | Within-window Global contribution | ~+0.152 nats |

**Approximately 90% of Global's causal contribution comes from within-window attention**
(attending to tokens within the same 4,096-token context window), and ~10% comes from
specifically beyond-window attention (tokens > 4,096 positions back).

This is consistent with e007b's finding that Global attention distance does not dramatically
exceed SWA at long sequences: Global is not primarily a "look far back" mechanism. It
contributes within-window, with a smaller but genuine incremental contribution beyond the
SWA window.

---

## Steel-man of H3b_null (why the null might still hold)

**Attention sink hypothesis**: Global layers might be forwarding attention through the
beyond-window mask injection, but most of those beyond-window attention weights are on
"sink tokens" (BOS, repeated tokens at position 0) that carry little semantic content.
If e005b is measuring attention to semantically vacuous sink tokens rather than true
long-range context, the Δ_beyond effect could be partly an artifact.

**Counter-argument**: The effect size is large (d_z=1.05), the direction is consistent
across 17/20 samples, and it grows with sequence length. A pure sink-token artifact would
not predict a length-dependent magnitude increase. The evidence supports a genuine
(if modest) beyond-window contribution.

---

## RQ3 Verdict — H3_alt NOW SUPPORTED

| # | Condition | Experiment | Verdict |
|---|-----------|-----------|---------|
| 1 | SWA mask → PPL↑ at pos > 4,096 | **e005b** | **✅ VALIDATED** |
| 2 | Zero Global output → PPL↑ | e006b | ✅ VALIDATED |
| 3 | Global distance > SWA at lengths > 4,096 | e007b | INCONCLUSIVE |

**RQ3 tally: 2 VALIDATED / 1 INCONCLUSIVE.**
H3_alt requires ≥ 2 of 3 conditions VALIDATED. **Threshold met.**

**Conclusion**: There is moderate pre-registered evidence that NoPE Global Attention
makes a causal contribution to long-range dependency beyond what SWA alone provides.
The contribution exists at both global (e006b) and specifically beyond-window (e005b) levels,
with the beyond-window component being statistically significant but modest in absolute magnitude.

---

## Scope and Limitations

1. **Corpus**: EDGAR only (English financial text). Korean generalization unverified.
2. **Length regime**: 5,120–8,192 tokens. Not tested at 262K (the model's claimed context
   length). The 3.1% coverage limitation applies.
3. **Magnitude interpretation**: +0.017 nats is the average across all beyond-window tokens
   at lengths tested. At longer sequences (e.g., 128K tokens), the cumulative effect could
   be larger, but this is extrapolation.
4. **SWA window size**: SWA_WINDOW=4,096 is fixed. The effect at narrower windows (e.g.,
   1,024 tokens) would likely be larger (more tokens "excluded" by the mask). The 4,096
   window was chosen to match the architecture spec.

---

## Deviations from Plan

### Deviation 1 — Corpus: EDGAR only (no WikiText-103, no KLUE-MRC)

Same as e006b Deviation 1. EDGAR was the only cached corpus covering seq_len > 4,096.
Impact: monolingual English results. Direction finding robust; magnitude may not generalize.

### Deviation 2 — Initial run failure: device mismatch on 2-GPU split

**Cause**: `window_mask` created on cuda:0; attention layers on cuda:1 (2-GPU device_map='auto').
**Fix**: Added `.to(old_mask.device)` in pre_hook (committed 7c54bfe).
**Impact on results**: None — the fixed run produced valid data. Added to HANDOVER.md as
infra note for future sessions.

---

*ANALYSIS written: 2026-04-27. Reviewer: Claude Code (gatekeeper mode).*
