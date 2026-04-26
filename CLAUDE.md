# EXAONE 4.5 Weight Analysis — Claude Code Context

## Research Purpose
EXAONE 4.5 (33B, LG AI Research) internal structure analysis.
Goals: algorithm understanding, baseline for future EXAONE model comparison, ecosystem expansion.

## Architecture Facts (hard numbers)
- Hidden: 5120 | Intermediate: 27392 | Layers: 64 + 1 MTP
- Hybrid attention: 16 × (3 SWA + 1 Global/NoPE)
- Global layers at indices: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
- GQA: 40 Q / 8 KV heads | Head dim: 128
- Reordered Norm (QK-Reorder-LN): RMSNorm on Q/K inputs before attention + RMSNorm after attention output before residual (not standard Pre/Post)
- Vision encoder: 1.2B, 2D RoPE (separate from LM)
- Vocab: 153,600 | Context: 262,144

## Pre-registration Rules (HARD — do not break under any circumstances)

These rules exist to prevent HARKing (Hypothesizing After Results are Known), which
is the primary scientific integrity risk in iterative ML research.

**R1. No new experiment without a committed PLAN.md.**
Before running any experiment script, `experiments/e{NNN}_{slug}/PLAN.md` must exist
and be committed to git. The git timestamp of PLAN.md must precede the first line
written to `outputs/*/results.jsonl`. Violation = retroactive retrofitting.

**R2. No experiment marked DONE without committed ANALYSIS.md and stats.json.**
`auto_validate.py` must be called at the end of every experiment script. It writes
`outputs/{slug}/stats.json`. This file and the corresponding ANALYSIS.md must both
be committed before `experiments/e{NNN}_{slug}/STATUS` is updated to `DONE`.

**R3. Findings live in FINDINGS.md, not HANDOVER.md.**
HANDOVER.md contains experiment metadata, setup instructions, and architecture facts.
Conclusions and research findings belong in FINDINGS.md, each citing an experiment ID.

**R4. Retrofitted hypotheses must be marked as such.**
If PLAN.md is written after any results exist, set `retrofitted: true` in the
`Decision Criteria` block. These results may be used only as exploratory/preliminary
findings in the paper — never as primary confirmatory evidence.

**R5. Do not re-run experiments to fish for significance.**
A FAILED or INCONCLUSIVE verdict from auto_validate.py ends the experiment for that
metric. Changing parameters (n_prompts, threshold, metric definition) after seeing
a FAILED result to obtain VALIDATED constitutes p-hacking. Document the FAILED
result in NEGATIVE_RESULTS.md and move on.

---

## Design Rules (do not break)
1. **Streaming load**: never load full model into memory. One tensor at a time via safetensors.
2. **Vision encoder separate**: always report vision stats independently from LM stats.
3. **SWA vs Global split**: always distinguish the two attention types in plots and tables.
4. **SVD is optional**: `--no-spectral` skips it. Don't make SVD mandatory in new code.
5. **Append-only stats.jsonl**: never overwrite, always append for reproducibility.
6. **run_config.json**: always write run parameters before analysis starts.

## Priority Order
- P0: weight stats (std, abs_mean, sparsity, l2) per tensor — DONE
- P0: SWA vs Global attention comparison — DONE
- P0: Reordered Norm analysis — DONE
- P1: effective rank / stable rank per layer — DONE
- P1: MTP layer vs main layer comparison
- P2: cross-model comparison scaffold (for EXAONE 5.x when released)
- P2: vision encoder deep dive (2D RoPE weight geometry)

## Out of Scope
- Inference / generation benchmarks
- Fine-tuning or weight modification
- Any training runs

## Extension TODO (for Claude Code to pick up)
- [ ] Add `--compare` flag: load two stats.jsonl files and diff by category
- [ ] Add KV head vs Q head weight distribution comparison
- [ ] Add histogram export per category (for publication figures)
- [ ] Add MTP layer dedicated report section
- [ ] Korean tokenizer analysis (vocab 153,600 breakdown)
