# ANALYSIS — e002 Attention Entropy (Long Input)

**Verdict**: INCONCLUSIVE
**Run date**: before 2026-04-20 (exact date not recorded)
**GPU**: NVIDIA A100 MIG 3g.40gb (40GB)
**Duration**: ~15–30 minutes (5 length conditions, each a fresh forward pass)

---

## Results

From `outputs/exp1b_long_input/summary.json`:

| Length | Global Entropy | SWA Entropy | Δ Entropy | Global Dist | SWA Dist | Δ Dist |
|--------|---------------|-------------|-----------|-------------|----------|--------|
| 128 | 1.7706 (±0.5000) | 1.8119 (±0.5199) | −0.041 | 33.91 | 31.71 | +2.20 |
| 256 | 2.0517 (±0.5875) | 2.1638 (±0.5809) | −0.112 | 70.23 | 64.66 | +5.57 |
| 512 | 2.4877 (±0.6524) | 2.6145 (±0.5762) | −0.127 | 144.11 | 127.11 | +17.00 |
| 1024 | 2.9815 (±0.7240) | 3.0456 (±0.5755) | −0.064 | 305.54 | 254.85 | +50.69 |
| 2048 | **3.5414** (±0.7763) | **3.4497** (±0.5912) | **+0.092** | **643.43** | **489.09** | **+154.34** |

---

## Statistical Tests

*Applied retroactively. Original run did not call auto_validate.py.*

**Critical problem**: Only a single text input was used per length condition. The
n_a and n_b for the Welch t-test are n_Global_layers × n_heads and n_SWA_layers × n_heads
for that one text — not independent samples from a population of texts.

This means the test has adequate nominal n (16 × 40 = 640 for Global, 48 × 40 = 1,920
for SWA), but the samples are not independent (all from the same document). The test
statistic is meaningful only for characterizing that one text, not for generalizing.

**Estimated effect size at 2,048 tokens** (from summary):
- Entropy: Δ = +0.092, pooled SD ≈ 0.69 → d ≈ 0.13 (negligible)
- Attn distance: Δ = +154.3, pooled SD ≈ large → d not computable without per-sample data

**Conclusion from available data**: The point estimates show a consistent trend
(Global distance > SWA across all lengths; entropy reversal at 2,048), but effect
size for entropy is negligible (d < 0.2) and no independent-sample variance is available.

---

## Decision

**INCONCLUSIVE** — Trend observed in point estimates; cannot confirm statistical
significance or effect size with single-text-per-condition design. The "direction
reversal at 2,048 tokens" is an observed point, not a confirmed finding.

**What must change before citing**: Rerun with ≥ 10 distinct texts per length condition
and apply `auto_validate.py` at exit.

---

## Deviations from Plan

⚠️ **HARKING RISK** — The following were determined post-hoc:

1. **"First meaningful reversal" language**: The phrase "first meaningful reversal"
   in HANDOVER.md was coined after seeing that 2,048 tokens was where Global > SWA.
   No statistical test supported "meaningful." This language has been corrected in
   HANDOVER.md (2026-04-26 audit).

2. **Entropy direction was unpredicted**: The original working hypothesis expected
   Global ≤ SWA (more uniform → lower entropy). The reversal at 2,048 was not
   anticipated and the interpretation was formed after seeing it.

3. **Single document design**: No design document specified how many texts to use.
   The choice of 1 text per length was opportunistic, not power-calculated.

4. **Hypothesis about 4,096+ saturation**: "Beyond 4,096 tokens, SWA distance
   saturates while Global continues to grow" was generated entirely from the 2,048
   data point. This is speculation, not a confirmed trend.

---

## Post-hoc Rationale

The trend in the data (distance gap grows with length, entropy reverses near window)
is consistent with the architectural hypothesis and warrants follow-up. However,
the single-document design makes this exploratory. The findings should be framed
as "preliminary observations motivating e007" rather than "evidence for H2_alt."

---

## Follow-up

→ e007 (long context hooks): tests the saturation hypothesis at 4,096–8,192 tokens
with multiple documents
