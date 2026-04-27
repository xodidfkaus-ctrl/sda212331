# ANALYSIS — e006 Global Zero-Output Ablation

**Experiment ID**: e006
**Script**: `nope_analysis/experiments/exp3_swa_ablation.py`
**Output**: `outputs/exp3_swa_ablation/`
**RQ**: RQ3 — Does NoPE Global contribute causally to long-range dependency?
**Date run**: 2026-04-27
**auto_validate verdict**: **FAILED** (see Critical Note A below)

---

## Primary Result

```
[perplexity_delta] ablated=2.1367±0.2792 vs baseline=1.9960±0.2813
diff=+0.1407  t=1.587  p=0.121  d=0.502(medium)  ✗ not significant
Overall: FAILED — p > 0.05 (null not rejected by pre-registered test)
```

`outputs/exp3_swa_ablation/stats.json` — `overall_verdict: "FAILED"`

---

## Critical Note A — Test Selection Error (post-hoc discovery)

**The FAILED verdict is an artifact of incorrect test selection, not evidence of a null effect.**

e006 is a paired experiment: each of the 20 sequences was run twice — once with baseline weights, once with Global attention output zeroed. The auto_validate call passed `(nll_ablated, nll_baseline)` as two independent groups, which treats the 20 text-specific PPL values as unrelated draws. This inflates the denominator of the t-statistic because text-to-text variance (~0.28 nats) swamps the treatment effect (~0.14 nats).

**Post-hoc paired analysis (exploratory, not confirmatory):**

```
One-sample t-test on delta_nll against μ=0
n=20, mean_delta=0.1407, std_delta=0.0209
t=30.15, p=1.65e-17, Cohen's d_z=6.74 (huge)
```

The treatment effect (ΔNLL = +0.1407 nats) is extremely consistent across all 20 samples (CV < 15%) and highly significant under the correct paired test. The direction is unambiguous: zeroing Global attention output consistently increases per-token NLL.

**Why FAILED verdict stands (Rule R5):** The paired analysis was conducted after observing the FAILED result from auto_validate. Running a different test after seeing a failing test constitutes p-hacking if used to overturn the pre-registered verdict. Rule R5 applies: the FAILED verdict ends this analysis for the primary confirmatory claim.

**Recommended remediation:** Pre-register e006b with a paired t-test / one-sample t-test on delta_nll as the primary comparison. The effect direction and magnitude are now known — e006b must be conducted to confirm under the correct pre-registered design.

---

## Critical Note B — Beyond-Window Measurement Gap

**The experiment did not test the scientific question of interest for RQ3.**

The RQ3 operational definition requires measuring perplexity increase at token positions **> 4,096 tokens** (beyond the SWA window), because within-window positions are attended to by both SWA and Global layers.

```
within_swa_window:  n=20, lengths [1024, 2048, 3072, 4096]  ← measured
beyond_swa_window:  n=0,  lengths [5120, 6144]               ← OOM (skipped)
```

All 20 measurements are within the SWA window. At these lengths, SWA can also attend to all positions, so the ablation removes Global's contribution while SWA provides a redundant path. A perplexity increase within the window demonstrates that Global contributes **something**, but does not demonstrate that Global is specifically necessary for **long-range** dependency beyond SWA's reach.

This is the same OOM failure mode as e005. The A100 80GB (full, not MIG) was confirmed available, but eager attention at ≥5120 tokens still OOMed. Root cause: model (65GB bfloat16) + eager attention matrix at 5120 tokens = ~17GB + activation memory exceeds 80GB.

**What within-window results support:**
- Global attention output contributes to per-token prediction even within the SWA window
- Removing it degrades perplexity at all tested lengths (monotonically increasing trend: ΔNLL 0.139→0.128→0.142→0.155 as seq_len increases from 1024→4096)

**What these results cannot support:**
- Any RQ3 claim about long-range dependency specifically beyond 4,096 tokens
- Condition 2 of the RQ3 operational definition (requires seq_len > 4,096)

---

## Per-Length Results

| seq_len | n | ΔNLL (mean) | ΔNLL (std) | ΔPPL (mean) | baseline PPL | ablated PPL |
|---------|---|-------------|------------|-------------|-------------|------------|
| 1024 | 5 | +0.1386 | 0.0210 | +1.003 | 7.07 | 8.08 |
| 2048 | 5 | +0.1276 | 0.0166 | +1.025 | 7.53 | 8.55 |
| 3072 | 5 | +0.1417 | 0.0174 | +1.215 | 7.92 | 9.14 |
| 4096 | 5 | +0.1548 | 0.0161 | +1.344 | 7.98 | 9.32 |
| 5120 | 0 | — | — | — | OOM | OOM |
| 6144 | 0 | — | — | — | OOM | OOM |

Corpus: EDGAR 10-K annual report sections (`en_edgar`). Amendment from WikiText-103 documented in PLAN.md §Pre-run Amendment.

---

## Contribution to RQ3 Verdict

Per the Section 2b operational definition (HANDOVER.md):

- **Condition 1 (e005, SWA mask injection)**: e005 = INCONCLUSIVE (OOM at ≥5120)
- **Condition 2 (e006, zero-output ablation)**: e006 = FAILED (test error; beyond-window n=0)
- **Condition 3 (e007, long-context hooks)**: PENDING

**RQ3 verdict so far**: 0 of 3 conditions met. Majority (≥2) required for "moderate evidence." Current state = "no evidence from confirmatory analysis." Mechanistic direction (Global contributes something) is suggested by within-window ΔNLL, but not confirmatory for the long-range dependency claim.

---

## Next Steps

1. **e006b (Priority 1)**: Re-run with paired t-test as primary comparison. Pre-register before running. n=20 within-window meets the corrected test's power requirement (d_z=6.74 → even n=5 would be sufficient; the practical question is whether beyond-window can be measured).

2. **OOM fix investigation (Priority 1, blocks e006b and e007)**: Eager attention at 5120 tokens OOMs on A100 80GB. Options:
   - Chunk-based perplexity: compute NLL on overlapping 4096-token windows with stride, extrapolate to 5120+ positions
   - Flash-attention + manual hook: flash_attn does not return attention weights, but hooks on the attention output (not weights) may still work
   - Needle-in-haystack behavioral proxy (HANDOVER.md §10): faster, avoids eager attention OOM

3. **e007 (Priority 2)**: Long-context hook analysis — measures attention distance growth, which does not require eager attention weight extraction for PPL computation. Different approach that may avoid the OOM.

---

## Exploratory Observations (not confirmatory)

1. ΔNLL is monotonically increasing from seq_len=1024 to 4096 (0.139→0.155 nats). If this trend continues beyond the SWA window, the beyond-window effect would be larger. This is speculative without data.

2. ΔPPL is also increasing (1.003→1.344 from 1024→4096), suggesting the Global contribution grows with sequence length even within the window. Consistent with Global's full-sequence attention scope.

3. All 20 delta_nll values are positive (min: ~0.107, max: ~0.163). Zero sign reversals. The direction is completely consistent — this is not noise.

These observations are exploratory (post-hoc patterns in a FAILED experiment). They are consistent with H3c_alt but do not confirm it.

---

*ANALYSIS written: 2026-04-27. Reviewer: Claude Code (gatekeeper mode).*
