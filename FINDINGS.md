# Findings — EXAONE 4.5 NoPE Research

This file is the authoritative record of paper-grade findings.

**Rules**:
1. Every bullet must cite at least one experiment ID in brackets, e.g. `[e002]`.
2. No finding may appear here without a corresponding ANALYSIS.md with a verdict.
3. INCONCLUSIVE results are included only as "preliminary observations" — labeled explicitly.
4. HANDOVER.md contains no conclusions. Research findings live here only.
5. Negative results belong in NEGATIVE_RESULTS.md, not here.

---

## RQ1 — Positional Information in NoPE Global Layers

### F1.1 — Both Global and SWA residual streams contain above-chance positional information [e004] ⚠️ PRELIMINARY

Both Global (NoPE) and SWA (RoPE) hidden states support linear position classification
at test accuracy 0.52–0.55 across lengths 64–256 tokens, well above the 10-class
random baseline (0.10).

**Verdict basis**: e004 — INCONCLUSIVE (underpowered: n=30 prompts, sklearn CPU probe,
severe overfit gap ≈ 0.45)

**Implication**: Positional information is present in both groups. This is consistent
with two competing mechanisms (Global generates vs. propagates from SWA); e008 is
required to distinguish them.

**Claims this does NOT support**:
- Whether Global layers actively encode position or merely carry SWA's signal (→ e008)
- The magnitude of positional encoding at lengths > 256 tokens
- Paper-grade statistical claim (n insufficient; must be upgraded to n ≥ 300 with GPU probe)

**Upgrade path**: e008 (delta-attribution probe, n=300, PyTorch GPU probe, PRE-REGISTERED)

---

### F1.2 — H1 vs H2 resolution: PENDING [e008]

**Status**: e008 not yet run. Finding will be populated after auto_validate.py
writes `outputs/exp2c_delta_probe/stats.json`.

**Pre-registered decision rule** (from PREREGISTRATION.md):
- H1_alt accepted if: delta_acc(h_out − h_in) > 0 at ≥ 1 Global layer, Cohen's d ≥ 0.5,
  p_Holm ≤ 0.01, n ≥ 300
- H1_null retained if: delta ≈ 0 across all Global layer indices

---

## RQ2 — Attention Pattern Divergence (SWA vs Global)

### F2.1 — No divergence at inputs well within SWA window [e001] (informative null)

At 14–26 tokens (<<SWA window of 4,096), Global and SWA attention entropy and mean
attention distance are nearly identical (d ≈ negligible).

**Interpretation**: Mechanistically expected — when all tokens fit within the SWA
window, Global and SWA attend to the same context. The architectural difference is
structurally irrelevant at this length regime.

**Verdict**: INCONCLUSIVE (no formal statistical test; effect size negligible from
point estimates)

---

### F2.2 — Attention distance gap grows with sequence length; direction reversal in entropy at 2,048 tokens [e002] ⚠️ PRELIMINARY OBSERVATION ONLY

Point estimates from a single-document run:

| Length | Global dist | SWA dist | Δ dist | Δ entropy |
|--------|-------------|----------|--------|-----------|
| 128 | 33.9 | 31.7 | +2.2 | −0.041 |
| 256 | 70.2 | 64.7 | +5.6 | −0.112 |
| 512 | 144.1 | 127.1 | +17.0 | −0.127 |
| 1024 | 305.5 | 254.9 | +50.7 | −0.064 |
| 2048 | **643.4** | **489.1** | **+154.3** | **+0.092** |

**Critical caveat**: Based on a **single text per length condition**. No variance
estimate, no independent replication, no statistical test. The entropy reversal at
2,048 tokens is a point observation, not a confirmed finding.

**Verdict**: INCONCLUSIVE (single-document design; d not estimable for between-condition
comparison)

**Do not cite** this table as evidence for H2_alt. Cite it as "preliminary observation
motivating e007."

**Upgrade path**: e007 (long context hooks at 2,048–8,192 tokens with ≥ 5 distinct texts,
PRE-REGISTERED)

---

### F2.3 — SWA distance saturation above 4,096 tokens: PENDING [e007]

**Status**: e007 not yet run.

---

## RQ3 — Causal Contribution of Global to Long-Range Dependency

### F3.1 — Causal ablation verdict: INCONCLUSIVE (1 of 3 complete) [e005, e006, e007]

**Decision rule** (2-of-3 majority, from HANDOVER.md Section 2b and PREREGISTRATION.md):
1. [e005] SWA mask injection → perplexity increase at pos > 4,096, d ≥ 0.5 → **INCONCLUSIVE** (OOM; zero beyond-window data)
2. [e006] Global zero-output → perplexity degradation, d ≥ 0.5 → PENDING
3. [e007] Global distance growth diverges from SWA saturation at ≥ 2 length conditions → PENDING

H3_alt (Global contributes causally) requires ≥ 2 of 3 conditions with d ≥ 0.5.
Current tally: 0 VALIDATED / 1 INCONCLUSIVE / 2 PENDING.

**e005 note**: The experiment failed to collect any beyond-window data. OOM at seq_len ≥ 5120
on A100 80GB PCIe with `attn_implementation='eager'`. The design requires a narrower SWA window
or a different memory strategy before it can produce evidence for or against H3b_alt.
See ANALYSIS.md Deviation 1 and "Recommended Next Steps."

**Critical implication for e006 and e007**: Even if both return VALIDATED, the 2-of-3 rule
would be satisfied. However, if either also returns INCONCLUSIVE/FAILED, H3_alt cannot be
supported. With e005 eliminated, the RQ3 verdict depends entirely on e006 + e007.

---

## Architecture Characterization (non-hypothesis findings)

### FA.1 — EXAONE 4.5 uses LLLG pattern: 48 SWA + 16 Global layers [HANDOVER.md, Phase 1]

The model alternates 3 SWA layers per 1 Global (NoPE) layer across 64 main layers.
Global layer indices: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63.
This is an architectural fact from the EXAONE 4.5 Technical Report (arXiv 2604.08644),
not an empirical finding of this study.

### FA.2 — Weight statistics distinguish Global from SWA layers [Phase 1 / analyze.py]

Phase 1 weight statistics analysis found differences in weight std distributions
between Global (NoPE) and SWA layers by layer depth. See `outputs/stats.jsonl` and
`outputs/report.md`. This was exploratory and is not a primary paper claim.

---

## Scope Limitations (findings this study cannot make)

The following conclusions are **out of scope** regardless of e005–e008 results.
See HANDOVER.md Section 10 for full discussion.

- "NoPE enables 262K-token recall" — no experiment tests at > 8,192 tokens
- "Mechanistic findings at 2K generalize to 262K" — extrapolation, not measured
- "EXAONE 4.5 outperforms other models on long-context tasks" — no comparative benchmarks
- "RoPE vs NoPE is the cause of observed differences" — no control model tested

---

*Last updated: 2026-04-26 | Pending findings: F1.2, F2.3, F3.1 (e005 INCONCLUSIVE; e006–e008 not yet run)*
