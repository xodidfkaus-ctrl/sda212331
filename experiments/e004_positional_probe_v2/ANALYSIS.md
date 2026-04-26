# ANALYSIS — e004 Positional Probe v2 (Train/Test Split)

**Verdict**: INCONCLUSIVE (underpowered; overfit gap too large)
**Run date**: before 2026-04-25
**GPU**: NVIDIA A100 MIG 3g.40gb (40GB); sklearn ran on CPU (~107 min)
**Duration**: ~107 minutes (CPU-bound sklearn; GPU utilization 0%)

---

## Results

From `outputs/exp2b_positional_probe_v2/summary.json`:

| Length | Global train | Global test | SWA train | SWA test | Global overfit gap | SWA overfit gap |
|--------|-------------|-------------|-----------|----------|-------------------|-----------------|
| 64 | 1.000 | 0.524 | 1.000 | 0.520 | 0.476 | 0.480 |
| 128 | 1.000 | 0.554 | 1.000 | 0.552 | 0.446 | 0.448 |
| 256 | 1.000 | 0.545 | 0.998 | 0.542 | 0.455 | 0.456 |

Random baseline: 0.100 (10-class uniform)

---

## Statistical Tests

*Applied retroactively.*

Comparison: Global test accuracy vs SWA test accuracy across all lengths.

| Length | Global test | SWA test | Δ | Estimated d |
|--------|------------|----------|---|-------------|
| 64 | 0.524 | 0.520 | +0.004 | ~0.01 (negligible) |
| 128 | 0.554 | 0.552 | +0.002 | ~0.01 (negligible) |
| 256 | 0.545 | 0.542 | +0.003 | ~0.01 (negligible) |

**Effect size**: Global vs SWA difference is negligible (d ≈ 0.01). The two groups
are effectively identical in test accuracy — consistent with both H1 (Global encodes
independently) and H2 (Global propagates SWA's signal unchanged).

**Sample size problem**: n=30 prompts → ~24 test samples with 80/20 split → 2-3 test
samples per position class in a 10-class problem. This is not enough to estimate
test accuracy reliably. Bootstrap CI would be very wide (estimated: ± 0.05–0.10).

**Overfit gap analysis**: Gap = 0.45–0.48 is severe. With n=30, the probe memorizes
training prompts rather than learning a generalizable positional signal. "Effect is
real but effect size limited" (previous HANDOVER.md language) is not defensible.
Corrected in HANDOVER.md (2026-04-26 audit).

---

## Decision

**INCONCLUSIVE** — primarily due to underpowering (n=30 << min_n=300 for
publication-grade probe), not because the finding is absent. The above-chance
test accuracy (0.52–0.55 vs 0.10 baseline) is suggestive, but the overfit gap
makes the estimate unreliable.

**Cannot claim**: "Global layers encode position at the same level as SWA"
**Can state**: "Preliminary evidence that both Global and SWA hidden states encode
position above chance; requires replication with n≥300 and k-fold CV."

---

## Deviations from Plan

⚠️ **HARKING RISK**:

1. **"Effect is real" claim**: The original HANDOVER.md framed the overfit gap as
   "limited effect size." This is a post-hoc rationalization of a methodological
   flaw. Corrected to "unreliable estimate" in the 2026-04-26 audit.

2. **Direction not pre-registered**: The working hypothesis was Global < SWA. The
   result (Global ≈ SWA) was then reframed as "interesting finding" (identical
   accuracy implies position info present in Global). This reframing was post-hoc.

3. **No k-fold**: A single 80/20 split was used. Which prompts land in test vs train
   affects the result and was not pre-specified.

4. **sklearn CPU probe**: Not a HARKing issue but a design flaw that should have been
   caught before running. See CHECKLIST.md Item 1.

---

## Post-hoc Rationale

The suggestive result (both above chance, both similar) is scientifically interesting
and correctly motivates e008 (delta probe). It should be reported as preliminary,
explicitly flagged as n=30 / retrofitted / pending replication. The overfit gap should
be reported transparently — it is an honest methodological limitation.

---

## Follow-up

→ e004_upgraded: rerun with ≥300 prompts, PyTorch GPU probe, k-fold CV (must
   be registered as a new experiment with PLAN.md committed before running)
→ e008 (delta-attribution probe): resolves H1 vs H2 that this experiment cannot
