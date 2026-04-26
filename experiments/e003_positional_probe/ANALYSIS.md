# ANALYSIS — e003 Positional Probe (Short Input, Same-split)

**Verdict**: INCONCLUSIVE (superseded — data leakage)
**Run date**: before 2026-04-20
**GPU**: NVIDIA A100 MIG 3g.40gb (40GB)
**Duration**: ~107 minutes (sklearn CPU-only; A100 idle at 0%)

---

## Results

From `outputs/exp2_positional_probe/summary.json`:

| Group | Mean probe accuracy | Std |
|-------|-------------------|-----|
| Global (NoPE) | **1.000** | 0.000 |
| SWA | **1.000** | 0.000 |

---

## Statistical Tests

**Not applicable** — accuracy = 1.000 for both groups with std = 0.000.
Cohen's d = 0 / 0 = undefined (NaN).

The perfect accuracy is a data leakage artifact, not a true signal:
the probe was trained and evaluated on the same set of tokens. A sufficiently
powerful linear classifier will perfectly memorize a finite training set.

---

## Decision

**INCONCLUSIVE** — the result is uninterpretable due to data leakage.
This experiment is superseded by e004 (train/test split).

The only valid conclusion: hidden states at all layers contain enough information
to linearly separate token positions in-sample. This is trivially true for any
deep network with sufficient capacity.

---

## Deviations from Plan

⚠️ **DESIGN FLAW — not a deviation from plan, but the plan itself was flawed:**

1. **No train/test split**: The critical flaw was not identified before running.
   The "data leakage = bug" diagnosis was made post-hoc after seeing accuracy = 1.000.

2. **sklearn CPU probe**: 107 minutes of CPU compute on a GPU machine. The inefficiency
   was not anticipated. See HANDOVER.md Section 7c and CHECKLIST.md Item 1.

3. **Original interpretation**: The HANDOVER.md originally stated "preceding SWA layers
   inject positional info → Global layers propagate it forward" as a confident
   interpretation. This was a post-hoc narrative formed from a leaky result. The
   interpretation has been corrected in HANDOVER.md (2026-04-26 audit).

4. **Falsified prediction**: The working hypothesis was that Global < SWA. The result
   (both = 1.000) could not confirm or deny this — it only confirmed leakage.

---

## Post-hoc Rationale

The experiment motivated the correct diagnosis (need for train/test split) and
led directly to e004, which is a methodologically improved version. The result
should be reported in the paper as a negative methodological finding — "naive
in-sample probing is uninformative; train/test split is required."

---

## Follow-up

→ e004 (probe v2, train/test split): methodologically corrected version
→ e008 (delta-attribution probe): resolves H1/H2 that e003/e004 cannot distinguish
