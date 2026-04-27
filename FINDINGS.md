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

### F2.3 — SWA distance saturation above 4,096 tokens: PENDING [e007b] ⚠️ PRELIMINARY OBSERVATION ONLY

**e007 status**: INCONCLUSIVE (2026-04-27). OOM at seq_len ≥ 3,072 after the first sample
at 2,048 tokens. `output_attentions=True` caused CUDA memory fragmentation incompatible
with multi-sample runs. Single data point (1 sample, 2,048 tokens, cross-layer variance
treated as proxy for cross-sample): Global dist=549.5, SWA dist=441.6, gap=+107.9 tokens.
Direction consistent with e002 but not confirmatory — pseudo-replication, no between-sample
variance.

**e007b status**: PENDING (PRE-REGISTERED 2026-04-27). Redesigned with sparse Q·K hook
(stride=128) instead of `output_attentions=True`. Memory at 8,192 tokens: ~42 MB/layer
vs 5.4 GB full. Script: `nope_analysis/experiments/exp4_long_context_sparse_hook.py`.

**Do not cite e007 results as evidence for H2_alt.** Awaiting e007b.

---

## RQ3 — Causal Contribution of Global to Long-Range Dependency

### F3.1 — Causal ablation verdict: IN PROGRESS (1 of 3 confirmed) [e005b, e006b, e007b]

**Decision rule** (2-of-3 majority, from HANDOVER.md Section 2b and PREREGISTRATION.md):

| # | Condition | Experiment | Verdict | Notes |
|---|-----------|-----------|---------|-------|
| 1 | SWA mask → PPL↑ at pos > 4,096 | e005b (redesign of e005) | **PENDING** | script written, not yet run |
| 2 | Zero Global output → PPL↑ | **e006b** (redesign of e006) | **✅ VALIDATED** | ΔNLL=+0.169, t=21.95, p≈2e-27, d_z=3.10 |
| 3 | Global distance diverges from SWA at ≥2 lengths >4,096 | e007b (redesign of e007) | **PENDING** | script written, running |

H3_alt (Global contributes causally) requires ≥ 2 of 3 conditions VALIDATED.
**Current tally: 1 VALIDATED / 0 INCONCLUSIVE / 2 PENDING.**

**e006b finding**: Zeroing all 16 Global layer attention outputs causes a statistically
significant and practically large increase in per-token NLL (ΔNLL=+0.169 nats, d_z=3.10, n=50).
Effect is larger beyond the SWA window (+0.218 nats at seq>4,096) than within (+0.136 nats),
consistent with H3_alt. However, e006b does NOT distinguish within-window from beyond-window
causal contribution at the token level — the beyond-window comparison is per-sequence, not
per-token-position. Per-token filtering is e005b's task.

**Scope warning**: e006b proves Global attention is causally necessary (somewhere). It does
NOT by itself prove Global contributes to long-range dependency specifically. The long-range
causal claim requires ≥ 1 of {e005b, e007b} to also VALIDATE.

**Prior experiments (superseded)**:
- e005 INCONCLUSIVE: OOM at all beyond-window lengths; zero primary-metric data.
- e006 FAILED: correct effect present (post-hoc d_z=6.74) but wrong test selected (independent
  groups instead of paired); FAILED verdict upheld per Rule R5.
- e007 INCONCLUSIVE: OOM from `output_attentions=True` after 1 sample at 2,048 tokens.

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

*Last updated: 2026-04-27 (session 4) | e006b VALIDATED (F3.1 condition 2); e007b running; e005b pending; F1.2/F2.3 await e008/e007b*
