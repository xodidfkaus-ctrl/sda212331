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

### F2.3 — SWA distance saturation above 4,096 tokens: FAILED [e007b] ⚠️ PRELIMINARY OBSERVATION ONLY

**e007 status**: INCONCLUSIVE (2026-04-27). OOM at seq_len ≥ 3,072 after the first sample
at 2,048 tokens. `output_attentions=True` caused CUDA memory fragmentation incompatible
with multi-sample runs. Single data point (1 sample, 2,048 tokens, cross-layer variance
treated as proxy for cross-sample): Global dist=549.5, SWA dist=441.6, gap=+107.9 tokens.
Direction consistent with e002 but not confirmatory — pseudo-replication, no between-sample
variance.

**e007b status**: FAILED (2026-04-27). Sparse Q·K hook (stride=128), n=10 per length
condition. Direction reversed at all lengths: SWA > Global, gap grows to −26.2 tokens at
8,192 tokens (d=−1.481, p_holm=0.023). Verdict FAILED after auto_validate direction fix
(bug: previous version used abs(d) ignoring sign).

**Do not cite e007/e007b as evidence for H2_alt.** F2.3 remains a preliminary observation.

---

## RQ3 — Causal Contribution of Global to Long-Range Dependency

### F3.1 — Causal ablation verdict: H3_alt SUPPORTED (2 of 3 conditions VALIDATED) [e005b, e006b, e007b]

**Decision rule** (2-of-3 majority, from HANDOVER.md Section 2b and PREREGISTRATION.md):

| # | Condition | Experiment | Verdict | Key result |
|---|-----------|-----------|---------|------------|
| 1 | SWA mask → PPL↑ at pos > 4,096 | **e005b** | **✅ VALIDATED** | Δ_beyond=+0.017 nats, t=4.68, p=0.0002, d_z=1.05 |
| 2 | Zero Global output → PPL↑ | **e006b** | **✅ VALIDATED** | ΔNLL=+0.169 nats, t=21.95, p≈2e-27, d_z=3.10 |
| 3 | Global distance > SWA at lengths > 4,096 | e007b | FAILED | Direction reversed: SWA > Global (d=−1.481 at 8192t) |

**H3_alt SUPPORTED: 2 of 3 conditions met.** Moderate pre-registered confirmatory evidence.

---

**Finding F3.1a — Global attention is causally necessary [e006b]**

Zeroing all 16 Global layer attention outputs (before residual add) causes a large, consistent
NLL increase across all 50 sequences tested (ΔNLL=+0.169 nats, d_z=3.10, p≈2e-27).
Effect is larger at sequences beyond the SWA window (+0.218 nats) than within (+0.136 nats).

**Correct interpretation**: Global attention output is causally necessary for NLL quality.
Removing it degrades model performance across all sequence lengths tested.

**NOT supported by e006b alone**: that this effect is specifically from beyond-window
attention (token positions > 4,096). e005b provides that evidence.

---

**Finding F3.1b — Global's beyond-window attention provides incremental causal benefit [e005b]**

Forcing Global layers to attend only within the SWA window (injecting window=4,096 mask)
causes a statistically significant NLL increase at token positions > 4,096 tokens
(Δ_beyond=+0.017 nats, d_z=1.05, p=0.0002, n=20). Within-window tokens are unaffected
(Δ_within=0.000), confirming the hook correctly isolates the beyond-window regime.

**Magnitude context**:
| Ablation | Removes | ΔNLL |
|----------|---------|------|
| e006b: zero all Global output | All Global attention | +0.169 nats |
| e005b: SWA-mask Global | Beyond-window attention only | +0.017 nats |
| Implied within-window contribution | — | ~+0.152 nats |

~90% of Global's causal value is within-window; ~10% is specifically beyond-window.
The beyond-window contribution is real but modest in absolute terms.

---

**Finding F3.1c — Global attention distance does not exceed SWA at long contexts [e007b, exploratory]**

Contrary to H3d_alt, Global attention mean distance is consistently < SWA distance at all
tested lengths (2,048–8,192 tokens). The gap grows with length: SWA exceeds Global by −26.2
tokens at 8,192 tokens (d=−1.481, large effect). Pre-registered test FAILED (direction
opposite to hypothesis). The result is interpreted as a SWA window-forcing artifact: SWA is
mechanically constrained to attend within its 4,096-token window, giving a near-constant mean
distance ≈ 2,048 tokens per query position at long sequences, while Global (NoPE) can
distribute attention more flexibly (including nearby tokens). This is an exploratory
characterization finding — not pre-registered in this direction, requires replication with
diverse corpus.

---

**Prior experiments (superseded by b-variants)**:
- e005 INCONCLUSIVE (OOM), e006 FAILED (wrong test), e007 INCONCLUSIVE (OOM).

---

**Paper framing guidance**:
The two validated conditions (e005b + e006b) together support: "NoPE Global attention
contributes causally to model quality, including a statistically significant incremental
benefit from attending beyond the SWA window." The effect at beyond-window positions is
real but small in absolute terms (+0.017 nats). The paper should not overstate this as
"dramatically improves long-range performance" — it provides modest, consistent benefit.

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

*Last updated: 2026-04-27 (session 5) | **H3_alt SUPPORTED** — e005b + e006b both VALIDATED (2/3); e007b FAILED (direction reversed, auto_validate direction fix applied); F1.2 awaits e008; F2.3 exploratory only*
