# ANALYSIS — e008 Delta-Attribution Probe

**Experiment ID**: e008
**Script**: `nope_analysis/experiments/exp2c_delta_probe.py`
**Output**: `outputs/exp2c_delta_probe/`
**RQ**: RQ1 — Does Global actively encode position (H1_alt) or propagate SWA's signal (H1_null)?
**Date run**: 2026-04-28
**GPU**: 2× NVIDIA A100 80GB PCIe
**auto_validate verdict**: **INCONCLUSIVE**

---

## Primary Result

```
Overall verdict: INCONCLUSIVE
n_comparisons: 16 (one per Global layer)
All 16 comparisons: INCONCLUSIVE

Global layers: mean Δ = −0.0061 ± 0.0040  (n=48, 45/48 negative)
SWA layers:   mean Δ = +0.0076 ± 0.0191  (n=144)
```

`outputs/exp2c_delta_probe/stats.json` — `overall_verdict: "INCONCLUSIVE"`

---

## Results by Sequence Length

| seq_len | Global mean Δ | SWA mean Δ | Global n | SWA n |
|---------|--------------|------------|----------|-------|
| 64 | −0.0056 ± 0.0041 | +0.0070 ± 0.0153 | 16 | 48 |
| 128 | −0.0052 ± 0.0038 | +0.0076 ± 0.0188 | 16 | 48 |
| 256 | −0.0077 ± 0.0038 | +0.0083 ± 0.0225 | 16 | 48 |

The sign pattern is consistent across all three length conditions: Global Δ is negative
at every seq_len, SWA Δ is positive at every seq_len.

---

## Per-Layer Delta (Global layers only)

| Layer | Mean Δ (3 seq_lens) | Direction |
|-------|---------------------|-----------|
| 3 | +0.000538 | ↑ (only positive Global layer) |
| 7 | −0.008659 | ↓ |
| 11 | −0.009358 | ↓ |
| 15 | −0.004857 | ↓ |
| 19 | −0.004523 | ↓ |
| 23 | −0.007374 | ↓ |
| 27 | −0.006966 | ↓ |
| 31 | −0.002865 | ↓ |
| 35 | −0.011558 | ↓ |
| 39 | −0.012561 | ↓ |
| 43 | −0.006931 | ↓ |
| 47 | −0.004284 | ↓ |
| 51 | −0.003477 | ↓ |
| 55 | −0.004592 | ↓ |
| 59 | −0.002969 | ↓ |
| 63 | −0.007960 | ↓ |

15 of 16 Global layers show negative delta. Layer 3 (first Global layer) shows
a negligible positive delta (+0.00054), not significantly different from zero.

---

## Why INCONCLUSIVE

**Root cause: design mismatch between pre-registration and implementation.**

The pre-registered criterion specified `min_n_per_group: 300` (referring to prompts).
However, `validate_experiment` was called with delta_acc values aggregated per seq_len
per Global layer — yielding **n=3 per comparison** (one value per seq_len), not n=300.

With n=3 and 16 simultaneous tests (Holm correction), the effective p threshold per
comparison is p ≤ 0.01/16 = 0.000625 for the first test, making rejection impossible
with n=3 regardless of effect size.

Individual raw p-values before Holm correction:
- Layer 19: p = 0.0083 (large effect d = −8.88)
- Layer 39: p = 0.0073 (d = −9.50)
- Layer 47: p = 0.0125 (d = −7.24)
- Layer 43: p = 0.0164 (d = −6.30)
- After Holm correction: all p_holm ≥ 0.117 → none pass threshold

The underpowering is a design flaw in the script's interface to auto_validate, not
a failure of the probe itself. The probe trained successfully; the validation input
was incorrectly structured.

---

## Directional Signal (Exploratory — NOT pre-registered in this direction)

**⚠️ HARKing risk: the following is post-hoc characterization. Do not cite as confirmatory.**

Despite the INCONCLUSIVE verdict, the directional pattern is:

1. **Direction consistently opposes H1_alt**: H1_alt required delta > 0 at ≥ 1 Global
   layer. Observed mean Δ = −0.0061 (negative). The direction supports H1_null.

2. **15/16 Global layers show negative delta**: The probability of observing 15/16 negative
   by chance under H1_alt (which predicts positive) is (0.5)^15 × C(16,15) = 0.00024.
   This is a strong directional signal, but it is unplanned (not a pre-registered test).

3. **SWA layers show the opposite sign (+0.0076)**: SWA attention adds positional
   decodability (as expected — SWA uses RoPE). Global layers do not; they slightly
   reduce it. This is consistent with a picture where SWA injects positional information
   and Global diffuses or passes it through without augmenting it.

4. **Magnitude**: Global Δ ≈ −0.006 on a base accuracy of ~0.47–0.50. The absolute
   reduction is small (~1.2% of base accuracy), suggesting Global mildly dilutes but
   does not destroy positional information.

**Tentative interpretation (exploratory)**: The evidence directionally favors H1_null —
Global does not actively encode positional information. Global attention operations appear
to slightly reduce position decodability rather than increasing it. However, this
interpretation cannot be formally claimed without a properly powered confirmatory test.

---

## Steel-man of H1_alt (why it might still hold)

1. **Probe granularity**: 10-bin classification over seq_len ≤ 256 may be too coarse to
   detect subtle positional encoding. Global might encode fine-grained positional features
   that don't survive 10-class binning.

2. **Within-window regime only**: seq_len [64, 128, 256] is well within the SWA window
   (4,096 tokens). Global's positional role might only emerge beyond the SWA window where
   SWA cannot contribute. The negative delta here may reflect that, within the window,
   SWA fully handles positioning and Global genuinely need not add any.

3. **Layer 3 exception**: The first Global layer (layer 3) shows a slightly positive delta
   (+0.00054), consistent with some positional injection at early layers before SWA has
   fully saturated.

---

## Methodological Post-Mortem

The correct implementation for validate_experiment should have passed per-prompt or
per-fold delta_acc values to achieve n ≥ 15 (5 folds × 3 seq_lens) or n = 15 per layer.
Still below min_n=300, but would allow Holm correction to function.

A properly powered upgrade would:
1. Pass per-fold delta values (n=5 per seq_len → n=15 total) to get past n=3
2. Or compare Global Δ vs SWA Δ as a two-sample test (n=48 vs n=144) — achievable
   with current data, not pre-registered
3. Or run with more seq_len conditions (e.g., 8 lengths instead of 3)

**Status: this experiment is not re-run-able under R5 (no fishing for significance).**
The INCONCLUSIVE result stands. The directional signal is documented as exploratory.

---

## Relation to Other Findings

- **Consistent with F3.1b (e005b)**: If 90% of Global's causal contribution is
  within-window, and Global doesn't add positional information beyond SWA, this suggests
  Global's within-window contribution is not positional encoding — it may be semantic
  integration or context aggregation.

- **Consistent with e007b direction reversal**: Global's shorter attention distance than
  SWA is consistent with Global not needing to reach beyond the SWA window to do its job.

- **RQ1 conclusion**: H1_null direction supported by directional evidence (exploratory).
  Formal confirmation requires a redesigned experiment with proper n.

---

*Generated: 2026-04-28 | Verdict: INCONCLUSIVE | Direction: H1_null (exploratory)*
