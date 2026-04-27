# ANALYSIS — e006b Global Zero-Output Ablation v2

**Experiment ID**: e006b
**Script**: `nope_analysis/experiments/exp3_global_zero_ablation_v2.py`
**Output**: `outputs/e006b_global_zero_ablation_v2/`
**RQ**: RQ3 — Does NoPE Global contribute causally to long-range dependency?
**Date run**: 2026-04-27
**GPU**: 2× NVIDIA A100 80GB PCIe
**auto_validate verdict**: **VALIDATED**

---

## Primary Result

```
[delta_nll] mean=0.1688±0.0544 vs null=0.0
diff=+0.1688  t=21.947  p=2.2e-27  d_z=3.104(large)  ✓ significant
Overall: VALIDATED — criteria met (p<0.05, d_z>0.3, n=50)
```

`outputs/e006b_global_zero_ablation_v2/stats.json` — `overall_verdict: "VALIDATED"`

---

## Results by Window Position

| Group | n | Mean ΔNLL | Std |
|-------|---|-----------|-----|
| Within SWA window (seq ≤ 4096) | 30 | **+0.1363** | ~0.04 |
| Beyond SWA window (seq > 4096) | 20 | **+0.2175** | ~0.04 |
| All sequences | 50 | **+0.1688** | 0.0544 |

Key observation: beyond-window ΔNLL (+0.2175) is **59% larger** than within-window ΔNLL (+0.1363). This suggests Global attention contributes more when sequences exceed the SWA window, consistent with H3c_alt.

Corpus: `en_edgar` (EDGAR 10-K annual reports, avg ~10,600 tokens). No synthetic fallback used.
Lengths tested: 2048, 3072, 4096 (within) + 5120, 6144 (beyond).

---

## Interpretation

The null hypothesis (Global attention output is causally irrelevant) is rejected at extreme significance (t=21.95, p≈2e-27, d_z=3.10). Global attention output removal consistently degrades NLL across all 50 sequences with essentially zero variance in direction (all ΔNLL > 0).

**Correct interpretation**: "Zeroing Global attention output causes a statistically significant and practically large increase in per-token NLL (ΔNLL=+0.169 nats, d_z=3.10)."

**NOT supported**: "NoPE Global provides long-range benefit" — this experiment does not distinguish within-window from beyond-window causal contribution at the token level. The beyond-window comparison above is observational (per-sequence, not per-token position filtering). Per-token filtering is implemented in e005b.

---

## Design note vs. e006 (FAILED)

e006 used Welch t-test on (nll_ablated, nll_baseline) as independent groups.
Text-to-text PPL variance (~0.28 nats) dominated the within-sequence treatment effect (~0.14 nats), giving t=1.587, p=0.121 → FAILED.

e006b uses one-sample t-test on `delta_nll = nll_ablated − nll_baseline` per sequence.
This removes text-level variance from the denominator. Result: t=21.95, p≈2e-27.
Both experiments measured the same underlying effect; e006 simply used the wrong test.

---

## RQ3 Operational Definition Status (Section 2b)

| Condition | Experiment | Status |
|-----------|-----------|--------|
| Exp 3b (SWA mask → PPL increase at >4096) | e005b | PENDING |
| Exp 3 (zero Global → PPL degradation) | **e006b** | **VALIDATED ✓** |
| Exp 4 (Global distance diverges at >4096) | e007b | PENDING |

1 of 3 conditions confirmed. Need ≥ 2 for "moderate evidence" conclusion.

---

## Deviations from Plan

### Deviation 1 — Corpus: EDGAR only (no WikiText-103, no KLUE-MRC)

**Planned**: WikiText-103 (25 sequences, seq_len=2048) + KLUE-MRC (25 sequences, seq_len=2048).
**Actual**: `en_edgar` (EDGAR 10-K annual reports) only across all 50 sequences at lengths
2048–6144. KLUE-MRC was not cached; WikiText-103 sequences at required lengths were not
pre-loaded. EDGAR was used as primary fallback as it covers the required length range.

**Impact on primary result**: The one-sample t-test is within-sequence (baseline vs ablated),
so corpus identity affects text-level baseline PPL but not the within-sequence ΔNLL estimate.
The paired design removes corpus-level variance. The verdict (VALIDATED) is robust to this
deviation.

**Limitation**: All evidence for e006b comes from English financial text (EDGAR). Generalizability
to other languages (Korean) and domains (news, fiction, technical) is unverified. The ΔNLL
magnitude (0.169 nats) should not be interpreted as a universal constant.

**Required in paper**: Disclosure that corpus is monolingual English (financial). Ideally
replicate with Korean text before final submission.

---

## HARKing Declaration (as noted in PLAN.md)

Effect direction was pre-known from e006's post-hoc analysis (d_z=6.74 post-hoc).
e006b is a pre-registered replication under known direction. The statistical conclusion
is valid (correct test, pre-registered criteria), but the paper must disclose that
this is a directional replication, not a blind confirmatory test.

---

*ANALYSIS written: 2026-04-27. Reviewer: Claude Code (gatekeeper mode).*
