# Experiment Log — EXAONE 4.5 NoPE Research

Chronological record of all experiments. Every row must link to a committed PLAN.md and ANALYSIS.md.
Do not edit result tags retroactively — append a superseding row instead.

**Result tags**:
- `VALIDATED` — primary metric met d ≥ 0.5 AND Holm-corrected p ≤ 0.01 with n ≥ min_n
- `FAILED` — p > 0.05 OR |d| < 0.2 with adequate n
- `INCONCLUSIVE` — underpowered, marginal effect, or design flaw preventing verdict
- `SUPERSEDED` — replaced by a later experiment; finding not used directly

---

## Table

| ID | Slug | RQ | Hypothesis tested | Registration | Verdict | Date | GPU | PLAN | ANALYSIS | stats.json |
|----|------|----|-------------------|--------------|---------|------|-----|------|----------|------------|
| e001 | attention_entropy_short | RQ2 | Short-input entropy/distance: Global vs SWA (no difference predicted) | RETROFITTED | INCONCLUSIVE | before 2026-04-20 | A100 MIG 3g.40gb (40GB) | [PLAN](experiments/e001_attention_entropy_short/PLAN.md) | [ANALYSIS](experiments/e001_attention_entropy_short/ANALYSIS.md) | not run |
| e002 | attention_entropy_long | RQ2 | Long-input (128–2048 tok): Global entropy/distance > SWA above SWA window half-point | RETROFITTED | INCONCLUSIVE | before 2026-04-20 | A100 MIG 3g.40gb (40GB) | [PLAN](experiments/e002_attention_entropy_long/PLAN.md) | [ANALYSIS](experiments/e002_attention_entropy_long/ANALYSIS.md) | not run |
| e003 | positional_probe | RQ1 | Probe accuracy: Global < SWA (NoPE → weaker position encoding) | RETROFITTED | INCONCLUSIVE | before 2026-04-20 | A100 MIG 3g.40gb (40GB) | [PLAN](experiments/e003_positional_probe/PLAN.md) | [ANALYSIS](experiments/e003_positional_probe/ANALYSIS.md) | not run |
| e004 | positional_probe_v2 | RQ1 | Train/test split probe: Global test acc < SWA test acc | RETROFITTED | INCONCLUSIVE | before 2026-04-25 | A100 MIG 3g.40gb (40GB) | [PLAN](experiments/e004_positional_probe_v2/PLAN.md) | [ANALYSIS](experiments/e004_positional_probe_v2/ANALYSIS.md) | not run |
| e005 | swa_mask_ablation | RQ3 | SWA mask injected into Global: perplexity at pos > 4096 increases (d ≥ 0.5) | PRE-REGISTERED 2026-04-26 | INCONCLUSIVE | 2026-04-26 | A100 80GB PCIe | [PLAN](experiments/e005_swa_mask_ablation/PLAN.md) | [ANALYSIS](experiments/e005_swa_mask_ablation/ANALYSIS.md) | not written (OOM — zero beyond-window samples) |
| e006 | global_zero_ablation | RQ3 | Zero Global attn output: perplexity degrades (d ≥ 0.5) | PRE-REGISTERED 2026-04-26 | FAILED | 2026-04-27 | 2× A100 80GB PCIe | [PLAN](experiments/e006_global_zero_ablation/PLAN.md) | [ANALYSIS](experiments/e006_global_zero_ablation/ANALYSIS.md) | independent-groups Welch t wrong test |
| e006b | global_zero_ablation_v2 | RQ3 | Zero Global attn output: mean ΔNLL > 0 (one-sample t, d_z ≥ 0.3) | PRE-REGISTERED 2026-04-27 | VALIDATED | 2026-04-27 | 2× A100 80GB PCIe | [PLAN](experiments/e006b_global_zero_ablation_v2/PLAN.md) | [ANALYSIS](experiments/e006b_global_zero_ablation_v2/ANALYSIS.md) | [stats.json](outputs/e006b_global_zero_ablation_v2/stats.json) |
| e007 | long_context_hooks | RQ3 | Global attn distance diverges from SWA at ≥ 2 length conditions above 4096 | PRE-REGISTERED 2026-04-26 | INCONCLUSIVE | 2026-04-27 | 2× A100 80GB PCIe | [PLAN](experiments/e007_long_context_hooks/PLAN.md) | [ANALYSIS](experiments/e007_long_context_hooks/ANALYSIS.md) | output_attentions OOM after 1 sample |
| e005b | swa_mask_ablation_v2 | RQ3 | SWA mask → ΔNLL > 0 at pos > 4096 (one-sample t, d_z ≥ 0.3) | PRE-REGISTERED 2026-04-27 | PENDING | — | — | [PLAN](experiments/e005b_swa_mask_ablation_v2/PLAN.md) | — | — |
| e007b | long_context_sparse_hook | RQ3 | Global attn distance diverges from SWA at lengths ≥ 4096 (sparse QK hook) | PRE-REGISTERED 2026-04-27 | INCONCLUSIVE | 2026-04-27 | 2× A100 80GB PCIe | [PLAN](experiments/e007b_long_context_sparse_hook/PLAN.md) | [ANALYSIS](experiments/e007b_long_context_sparse_hook/ANALYSIS.md) | [stats.json](outputs/e007b_long_context_sparse_hook/stats.json) |
| e008 | delta_probe | RQ1 | H1_alt: delta probe acc at Global out > Global in (d ≥ 0.5, p_Holm ≤ 0.01) | PRE-REGISTERED 2026-04-26 | PENDING | — | — | [PLAN](experiments/e008_delta_probe/PLAN.md) | [ANALYSIS](experiments/e008_delta_probe/ANALYSIS.md) | — |

---

## Notes on retrofitted experiments (e001–e004)

All four completed experiments were designed and run without pre-registration.
PLAN.md and ANALYSIS.md were written retroactively on 2026-04-26 to establish a
baseline record. The retrofitted status is marked in both PLAN.md `Decision Criteria`
blocks (`retrofitted: true`) and ANALYSIS.md "Deviations" sections with explicit
HARKing risk flags.

These results are:
- Usable as **exploratory/pilot findings** (hypothesis-generating)
- Usable as **motivation** for pre-registered follow-ups (e005–e008)
- **Not usable** as primary confirmatory evidence in the paper's main claims

---

## How to add a new row

1. Create `experiments/e{NNN}_{slug}/PLAN.md` and commit it **before** running the experiment.
2. Run experiment → call `auto_validate.py` → `outputs/{slug}/stats.json` is written.
3. Write `experiments/e{NNN}_{slug}/ANALYSIS.md` and update `STATUS` to `DONE`.
4. Add a row here with the verdict from `stats.json`.
5. If the finding is paper-grade, add to `FINDINGS.md` with this experiment's ID.

*Last updated: 2026-04-27 (session 4) — e006b VALIDATED, e007b/e005b PENDING*
