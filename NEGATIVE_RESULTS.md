# Negative Results — EXAONE 4.5 NoPE Research

This document records all hypotheses that were rejected, experiments that were aborted,
and results that were generated but not used in the main paper conclusions.

Negative results are a first-class scientific output. Recording them prevents
duplication, enables honest meta-analysis, and is required by ACL/EMNLP reproducibility
checklists.

---

## 1. Rejected Hypotheses

### [e003] Global layers encode position more weakly than SWA layers

**Original prediction**: Probe accuracy(Global) < probe accuracy(SWA)

**Result**: Both groups = 1.000 (train accuracy only; data leakage — e003 was superseded)

**Why rejected**: The working assumption was that NoPE Global layers would encode
positional information weakly because they lack RoPE. The result was the opposite:
both achieved perfect accuracy. However, this was train-set accuracy with no
held-out test, making the result invalid rather than informative.

**Status**: Superseded by e004 (train/test split). The qualitative finding (both
groups encode position) was replicated in e004 at above-chance test accuracy.

**Implication**: The hypothesis that Global layers *lack* positional information
(the originally expected finding) was definitively ruled out. This makes the
mechanistic question (H1 vs H2 in e008) more interesting, not less.

---

### [e004] Train/test split: Global probe accuracy < SWA probe accuracy

**Original prediction**: probe_accuracy(Global) < probe_accuracy(SWA)

**Result**: Global ≈ SWA at all tested lengths (within 0.004):

| Length | Global test | SWA test |
|--------|-------------|----------|
| 64 | 0.524 | 0.520 |
| 128 | 0.554 | 0.552 |
| 256 | 0.545 | 0.542 |

**Why rejected**: The predicted Global < SWA direction was not observed. Both groups
encode positional information nearly identically when measured at the residual stream.

**Implication**: The near-identical probe accuracy is consistent with two competing
mechanistic explanations (H1: Global encodes independently; H2: Global propagates
SWA's encoding unchanged). This result does not distinguish them — it motivates e008.

**Caution**: n=30 prompts, 80/20 split, sklearn CPU probe, severe overfit (train ≈ 1.0).
The INCONCLUSIVE verdict reflects underpowering, not a clean null.

---

### [e001] Short inputs: any meaningful entropy/distance difference between Global and SWA

**Original prediction**: Some difference expected (direction unspecified)

**Result**: Entropy 0.9301 vs 0.9422 (d ≈ negligible); distance 7.18 vs 7.10 (d ≈ negligible)

**Why rejected**: At 14–26 tokens, all positions fit within the SWA window (4,096 tokens).
The architectural distinction between Global (no window) and SWA (window = 4,096) is
irrelevant when the input is shorter than the window. This null result is mechanistically
expected, not a failure of the hypothesis.

**Implication**: Motivated e002 (test at longer inputs where window constraint matters).

---

### [e006] Global zero-output ablation — FAILED by independent Welch test (test selection error)

**Original prediction**: Zeroing Global layer self-attention output causes statistically significant perplexity increase (H3c_alt); pre-registered threshold: Cohen's d > 0.5, p_holm < 0.01, n ≥ 20.

**Result**: Welch independent-groups t-test: t=1.587, p=0.121, d=0.502 — **not significant**.
Post-hoc paired analysis (exploratory): t=30.15, p=1.65e-17, d_z=6.74 — strongly significant, but conducted after FAILED verdict.

**Why rejected (per Rule R5)**: The pre-registered comparison was called as independent groups, which is wrong for a paired experiment. Text-to-text PPL variance (~0.28 nats) swamped the treatment effect (~0.14 nats) in the denominator. The FAILED verdict was upheld per Rule R5 — changing the test after a FAILED result is p-hacking.

**Critical secondary limitation**: beyond_swa_window n=0 (OOM at ≥5120 tokens). All 20 measurements were within the SWA window — the experiment did not test the scientific question of interest (long-range dependency beyond 4,096 tokens).

**Implication**: The within-window direction is clear and consistent (ΔNLL > 0 for all 20 samples, d_z=6.74 by paired analysis). Global attention contributes to prediction within the SWA window. The confirmatory claim for RQ3 (beyond-window effect) requires e006b with (1) paired t-test pre-registered, (2) OOM mitigation for seq_len > 4,096.

**Source**: `outputs/exp3_swa_ablation/stats.json` — `overall_verdict: "FAILED"`. Full analysis: `experiments/e006_global_zero_ablation/ANALYSIS.md`.

---

### [e007] Long context hook analysis — INCONCLUSIVE (OOM after 1 sample)

**Original prediction**: Global attention distance diverges from SWA at ≥ 2 length
conditions above 4,096 tokens (H3d_alt); d ≥ 0.5, p_Holm ≤ 0.01 at ≥ 2 lengths.

**Result**: OOM at the second sample of 2,048 tokens. Only 1 sample completed at 2,048 tokens;
all lengths ≥ 3,072 failed on sample 0. `output_attentions=True` caused CUDA memory
fragmentation — after the first forward pass, freed blocks remained fragmented and subsequent
contiguous allocation for full attention matrices failed.

Single data point (n=1, 2,048 tokens): Global dist=549.5 ± 122.0 vs SWA dist=441.6 ± 112.2,
gap=+107.9. Note: variance is cross-layer (pseudo-replication), not cross-sample.
auto_validate verdict: INCONCLUSIVE (force_verdict flag; n=1 < min_n=5).

**Why not rejected as FAILED**: Zero beyond-window data collected. The experiment could not
test the primary question (divergence at seq ≥ 4,096). This is a data collection failure, not
a negative result.

**Implication**: Direction is consistent with e002 observation (same direction, similar magnitude,
different corpus). Motivates e007b (sparse Q·K hook, stride=128, pre-registered 2026-04-27).

**Source**: `outputs/exp4_long_context/stats.json` — `overall_verdict: "INCONCLUSIVE"`.
Full analysis: `experiments/e007_long_context_hooks/ANALYSIS.md`.

---

### [e007b] Long context sparse hook — INCONCLUSIVE (direction reversed)

**Original prediction**: Global attention distance > SWA distance at ≥ 2 length conditions
above 4,096 tokens (H3d_alt); d ≥ 0.5, p_Holm ≤ 0.05.

**Result**: Direction reversed at all tested lengths. Global distance < SWA distance,
with the gap growing with sequence length (−0.7 tokens at 2,048t → −26.2 tokens at 8,192t).
At 8,192 tokens: t=−3.64, p_holm=0.002 in the wrong direction. Overall verdict: INCONCLUSIVE.

**Why not FAILED**: A significant difference exists but in the opposite direction from H3d_alt.
INCONCLUSIVE is correct — the null (no difference) is also not confirmed.

**Known methodological limitation**: auto_validate received per-layer distances (n=160 for
Global, n=480 for SWA at each length), not per-sample means (n=10). t-statistics are inflated
~4–7×. The directional finding (SWA > Global) is robust (9/10 samples agree at 8,192t), but
exact p-values are unreliable. Fix committed in 7c54bfe (sample-level aggregation); stats.json
was NOT regenerated — requires a rerun to produce corrected statistics.

**Mechanistic interpretation**: SWA window-forcing effect. At long sequences, SWA is
constrained to attend within its 4,096-token window, enforcing a near-constant mean distance
≈ 2,048 tokens per query position. Global (NoPE), without this constraint, distributes
attention more flexibly and may concentrate on nearby tokens or attention sinks, giving lower
mean distance. This is exploratory and not pre-registered in this direction.

**Implication for RQ3**: e007b condition (H3d) contributes 0 to the H3_alt tally.
H3_alt is supported via e005b + e006b (2/3 majority), independent of e007b.

**Source**: `outputs/e007b_long_context_sparse_hook/stats.json` — `overall_verdict: "INCONCLUSIVE"`.

---

## 2. Aborted Experiments

### [pre-e005] Initial Exp 3b run — OOM at seq_len = 6144 on A100 MIG 3g.40gb (40GB)

**Date**: before 2026-04-26

**Cause**: eager attention on EXAONE-4.5-33B (65GB model) with seq_len = 6,144 exceeded
available VRAM on the A100 MIG 3g.40gb slice (40GB usable). Forward pass never completed.

**Resolution**: Experiment redesigned (e005) to require full A100 80GB. The MIG 40GB
slice is documented as insufficient for experiments with seq_len > 2,048.

**VRAM budget note**: Model bfloat16 ≈ 65GB. Eager attention at 6,144 tokens adds
~3GB activation memory (n_layers × seq² × heads). Full A100 80GB has 12GB headroom;
MIG 40GB slice does not.

---

### [pre-e004] Exp 2b first run — 107-minute CPU hang

**Date**: before 2026-04-25

**Cause**: sklearn.LogisticRegression does not use GPU. The 30-prompt × 64-layer
fitting loop ran entirely on CPU (32 vCPUs) for ~107 minutes while the A100 sat idle.

**Resolution**: Documented in HANDOVER.md Section 7c. Future probe experiments (e008)
must use the PyTorch GPU probe instead of sklearn.

**Design rule added**: CLAUDE.md and CHECKLIST.md updated: "Probe fitting must use
PyTorch (GPU), not sklearn (CPU)."

---

## 3. Results Not Used in Main Conclusions

### [e003] Perfect probe accuracy (1.000)

Not used because of data leakage (train = test). The number itself is meaningless.
Retained in outputs/ for audit trail. Superseded by e004.

### [e002] "Reversal at 2,048 tokens" point estimate

The entropy reversal (Global 3.54 > SWA 3.45 at 2,048 tokens) is not used as a
confirmed finding because:
- Only one text per length condition (no variance estimate)
- No statistical test was run at experiment time
- Effect size d ≈ 0.13 (negligible) from pooled SD estimate
- Cannot rule out document-specific artifact

This is retained in ANALYSIS.md as a motivating observation for e007, not as evidence.

### [e001] Entropy/distance point estimates at short inputs

The absolute values (entropy ≈ 0.93, distance ≈ 7.1 tokens) describe behavior at
14–26 tokens. Not used in main conclusions because:
- Input length is far below the SWA window (not the regime of interest)
- No statistical test was run
- 3 prompts is too few for any inference

---

## 4. Template for Future Entries

When a hypothesis is rejected, add a subsection here within one week of the
experiment completing. Format:

```
### [e{NNN}] {Hypothesis name}

**Original prediction**: {exact hypothesis from PLAN.md}
**Result**: {quantitative outcome}
**Why rejected**: {mechanistic or statistical reason}
**Implication**: {what this tells us; does it motivate follow-up?}
**Source**: outputs/{slug}/stats.json — overall_verdict: FAILED
```

When an experiment is aborted, add:

```
### [pre-e{NNN}] {Short description}

**Date**: {date}
**Cause**: {OOM / bug / corpus unavailable / etc.}
**Resolution**: {redesign / deferred / cancelled}
**Evidence**: {logs, error messages, VRAM readings}
```

---

*Last updated: 2026-04-26*
