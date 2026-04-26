# ANALYSIS — e005 SWA Mask Injection Ablation

**Status**: DONE
**Verdict**: INCONCLUSIVE
**Run date**: 2026-04-26
**GPU**: NVIDIA A100 80GB PCIe (81920 MiB)
**Duration**: ~45 min (model load ~7 min + 20 within-window sequences; OOM stopped all beyond-window attempts)
**Seed**: 42

---

## Results

### Primary metric: perplexity delta at positions > 4,096

| Group | n | Mean ΔNLL | Status |
|-------|---|-----------|--------|
| beyond_swa_window (pos > 4096) | **0** | — | **OOM — no data** |
| within_swa_window (pos ≤ 4096) | 20 | 0.0000 | Expected (sanity check) |

Beyond-window data collection failed completely. Both tested lengths (5120, 6144) caused
out-of-memory immediately on the first sample. The experiment produced **zero data points
for the primary metric**.

### Per-length within-window summary (sanity check only)

| seq_len | PPL baseline | PPL SWA-masked | ΔNLL | corpus |
|---------|-------------|----------------|------|--------|
| 1024 | 1.3283 | 1.3283 | 0.0000 | fallback (synthetic) |
| 2048 | 1.1474 | 1.1474 | 0.0000 | fallback (synthetic) |
| 3072 | 1.0999 | 1.0999 | 0.0000 | fallback (synthetic) |
| 4096 | 1.0776 | 1.0776 | 0.0000 | fallback (synthetic) |

Within-window ΔNLL = 0.0000 exactly across all 20 samples. This is **expected and correct**:
when seq_len ≤ SWA_WINDOW (4096), the injected SWA causal mask is identical to the standard
causal mask, so outputs are bit-identical. This confirms the hook is not corrupting within-window
computations.

### Hook verification (before main run)

Verified with `tiny_window=1` injection on a 160-token sentence:
- NLL before: 2.4063
- NLL after (window=1): 2.4403
- ΔNLL: +0.0340 (hook confirmed operational)

---

## Statistical Tests

`auto_validate.py` was **not called** — skipped due to empty beyond-window group.

```
[stats] within_window n=20, beyond_window n=0
[stats] Skipping t-test: beyond_window group empty (need pos>4096 data).
[auto_validate] Skipped — insufficient data (beyond or within group empty)
```

No stats.json was written to `outputs/exp3b_swa_mask_ablation/`.

---

## Decision

**Verdict: INCONCLUSIVE**

The experiment cannot return VALIDATED or FAILED because the primary measurement
(NLL at positions > 4,096 under SWA mask ablation) was never obtained. The hardware
constraint is fundamental: the A100 80GB runs out of memory immediately when attempting
to run an eager-attention forward pass at seq_len ≥ 5120 with the 65GB EXAONE-4.5-33B
model. There is no within-session workaround for this without changing the experimental
design.

The PLAN.md stopping rule for OOM was: "OOM at 8,192 → reduce to 5,120." That stopping
rule assumed seq_len=5120 was feasible. It was not. The plan had no stopping rule for
the case where all beyond-window lengths fail. This gap in the stopping rules means the
experiment terminates with null data, not a negative result.

This is **not a FAILED result** for H3b_alt. It is a **data collection failure**.
The null hypothesis cannot be retained on zero observations.

---

## Deviations from Plan

### Deviation 1 — OOM at ALL beyond-window lengths (Critical)

**Planned**: Sequences of 4,096–8,192 tokens; stopping rule reduces to 5,120 if 8,192 OOMs.
**Actual**: OOM at seq_len=5120 (first attempt) and seq_len=6144 (second attempt). No
beyond-window data collected.

**Root cause**: PLAN.md stopping rule assumed seq_len=5120 is feasible on A100 80GB.
Empirically, `attn_implementation='eager'` requires O(seq_len²) attention matrix memory.
At seq_len=5120, this exceeds available VRAM with the 65GB model resident.

**Impact**: Primary metric entirely missing. Experiment verdict forced to INCONCLUSIVE.

**Required action before rerunning**: Either (a) chunked/sliding forward pass that avoids
materializing full attention at once, or (b) reduce model parallelism to free ~15GB VRAM,
or (c) accept that this experiment requires seq_len ≤ 4096 and redesign the measurement
around a narrower SWA window (window < 4096 on within-window sequences).

### Deviation 2 — Corpus fallback used instead of WikiText-103 / KLUE-MRC (Moderate)

**Planned**: WikiText-103 (English) + KLUE-MRC (Korean), ≥ 40 sequences total.
**Actual**: Both corpora unavailable (not cached in this session). Synthetic repeated-text
fallback used for all 20 sequences.

**Impact on within-window sanity check**: ΔNLL = 0.0000 regardless of corpus, so this
deviation does not affect the sanity check. However, the very low baseline perplexity
(1.07–1.34) confirms the synthetic text is trivially predictable. Had beyond-window data
been obtainable, the near-zero perplexity would severely reduce the signal-to-noise ratio
for detecting an ablation effect.

**Impact on primary metric**: Moot — no beyond-window data was collected regardless of corpus.

### Deviation 3 — Whole-sequence NLL vs. per-token NLL at positions > 4,096 (Moderate)

**Planned**: Per-token NLL *specifically at positions > 4,096* (PLAN.md Method step 5–6).
**Actual**: Whole-sequence NLL comparison between baseline and masked model. The script
labels sequences by their total length (> or ≤ 4096), not by filtering to tokens at positions
> 4096 within longer sequences.

**Impact**: For within-window sequences (all that ran), this distinction is irrelevant.
For beyond-window sequences (none ran), this would have been a measurement error — the plan
called for token-level filtering, but the implementation did sequence-level comparison.
If e005 is redesigned and rerun, the per-token filtering must be implemented to match PLAN.md.

### Deviation 4 — auto_validate.py not called at exit (Minor)

**Planned**: `auto_validate.py` called at end of every experiment (CLAUDE.md R2).
**Actual**: Skipped because the beyond-window group was empty. The call was guarded:
```python
if beyond and within:
    validate_experiment(...)
else:
    print("[auto_validate] Skipped — insufficient data")
```

The guard is correct behavior (calling validate_experiment with an empty group would fail),
but it means no stats.json was written and the pre-registration order check was not executed.
The order check would have passed (PLAN.md committed 2026-04-26, before this run).

---

## Post-hoc Rationale

All deviations arose from infrastructure constraints (VRAM, corpus availability), not
from post-hoc design changes after seeing results. The within-window ΔNLL=0 result was
fully expected and predicted by the experimental design. No hypothesis was adjusted
after observing data.

The stopping rule gap (no rule for OOM-at-5120) will be added to the revised PLAN.md
if e005 is redesigned.

---

## Steel-man of Null Hypothesis

The following arguments support H3b_null (Global mask injection does *not* degrade
long-range perplexity), which we cannot rule out on current evidence:

1. **Residual stream dominance**: Even if Global attention is masked to SWA range, the
   residual connections carry forward all token representations from prior layers. SWA layers
   at earlier depths may have already aggregated sufficient long-range context into the residual.
   Ablating Global's attention window may not remove the information — it was already deposited.

2. **Global attention sparsity**: Prior work on hybrid SWA/Global architectures shows Global
   layers often concentrate attention weight on a small set of "sink" tokens (beginning/end of
   sequence). If real attention mass at positions > 4096 is already negligible in unmodified
   Global layers, forcing a 4096-window mask changes nothing operationally.

3. **Synthetic corpus nullity**: The fallback corpus (repeated text, PPL ≈ 1.07) has near-zero
   entropy. Even if long-range Global attention provides real benefit, there is no uncertainty
   left to reduce at positions > 4096 when the text is maximally predictable. The experiment
   might have found ΔNLL ≈ 0 on the synthetic corpus even if beyond-window runs had been possible.

4. **NLL sensitivity floor**: For tasks where the baseline NLL is 1.07–1.34 nats, a d ≥ 0.5
   difference between ablated and baseline NLL requires an absolute difference of roughly
   0.5 × σ. With σ → 0 on nearly-deterministic text, the t-statistic is undefined and no
   effect would be detectable by design.

**Implication**: If e005 is redesigned, it must use high-entropy text (natural language
prose, NOT repeated/synthetic sequences) to have any chance of detecting an effect.
The corpus deviation alone may have been disqualifying regardless of the OOM issue.

---

## Contribution to RQ3 Verdict

This experiment is condition 1 of 3 in the RQ3 operational definition:

- e005: **INCONCLUSIVE** (no beyond-window data; OOM-constrained)
- e006: PENDING
- e007: PENDING

Per HANDOVER.md Section 2b, H3_alt requires ≥ 2 of 3 conditions VALIDATED.
e005 contributes **0** to the tally. e006 and e007 must both return VALIDATED for
H3_alt to be supported; either failing yields H3_alt INCONCLUSIVE or FAILED.

---

## Recommended Next Steps

1. **Do not rerun e005 without redesign.** The current design cannot produce beyond-window
   data on this hardware. Attempting a rerun is p-hacking if redesign choices are informed
   by seeing ΔNLL=0 on within-window sequences.

2. **Redesign options (each requires a new PLAN.md pre-registered before any code changes)**:
   - **e005r (recommended)**: Use narrower SWA window (e.g., 512 tokens) so that the mask
     injection produces a detectable effect within seq_len ≤ 4096. This tests "Global
     contributes to cross-segment dependency within accessible memory." Changes the hypothesis
     from "> 4096 tokens" to "tokens outside a 512-token local window."
   - **e005s (alternative)**: Use Flash Attention with custom mask kernels that avoid
     materializing the full attention matrix. Requires re-implementing the hook to work
     with non-eager attention. High engineering cost; changes the attention computation path.

3. **Proceed to e006**: Global zero-output ablation does not require seq_len > 4096; it
   operates on whole-sequence perplexity with Global outputs zeroed. May be feasible within
   current VRAM limits.
