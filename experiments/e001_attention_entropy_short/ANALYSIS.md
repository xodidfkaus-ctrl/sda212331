# ANALYSIS — e001 Attention Entropy (Short Input)

**Verdict**: INCONCLUSIVE
**Run date**: before 2026-04-20 (exact date not recorded)
**GPU**: NVIDIA A100 MIG 3g.40gb (40GB) — confirmed in HANDOVER.md
**Duration**: <5 minutes (short input, no model loading from scratch each time)

---

## Results

From `outputs/exp1_attention_entropy/summary.json`:

| Metric | Global (NoPE) | SWA | Difference |
|--------|--------------|-----|------------|
| Entropy mean | 0.9301 | 0.9422 | −0.0121 |
| Entropy std | 0.3550 | 0.3608 | — |
| Attn distance mean | 7.178 tokens | 7.100 tokens | +0.078 |

---

## Statistical Tests

*Applied retroactively (not run at experiment time).*

The original run aggregated entropy over all layers and all heads. The summary.json
reports means over n = 16 Global layers × n_heads × n_prompts and n = 48 SWA layers
× n_heads × n_prompts. Exact per-sample counts are in results.jsonl.

Directional conclusion from point estimates only:
- Entropy: SWA slightly higher (0.9422 vs 0.9301), d ≈ small/negligible
- Attn distance: Global slightly higher (7.18 vs 7.10), d ≈ negligible

**Statistical significance**: not tested at run time. Retroactive test pending on
raw results.jsonl. Given d ≈ 0.03 (estimated), result is expected to be FAILED
(negligible effect) or INCONCLUSIVE (insufficient n per condition).

---

## Decision

**INCONCLUSIVE** — point estimates show no meaningful difference at short inputs.
Expected given that 14–26 tokens fit well within the SWA window (4,096 tokens),
so Global and SWA see identical context. The null result is informative (motivates
e002 with longer inputs).

---

## Deviations from Plan

⚠️ **HARKING RISK** — All of the following were determined post-hoc:

1. **No pre-registration**: The "no difference at short inputs" conclusion was formed
   after seeing the results. The hypothesis that short inputs would show no difference
   was generated from the data, not stated before running.

2. **No statistical tests at run time**: The `summary.json` contains only means.
   No Welch t-test, Cohen's d, or bootstrap CI was computed during the original run.

3. **Sample size not justified**: 3 prompts × (16 Global + 48 SWA layers) × 40 heads.
   No power analysis was conducted.

4. **Decision to run e002**: The decision to extend to longer inputs was made after
   observing the null result — classic exploratory-iterative design, not confirmatory.

---

## Post-hoc Rationale

The null result at short inputs is scientifically valid and not contingent on the
post-hoc nature of the framing. Short inputs are physically constrained to be entirely
within the SWA window, so the null result is mechanistically expected. The exploratory
nature of this experiment does not invalidate the finding — it limits it to
"hypothesis-generating" status rather than "confirmatory."

---

## Follow-up

→ e002 (long input entropy): tests whether divergence emerges at lengths > SWA window
