# Research Operating Checklist — EXAONE 4.5 NoPE Project

Apply these 14 checks during the project. Treat them as standing rules, not one-time advice.

**When to read:**
- Items 1–6: re-read before every experiment run
- Items 7–9: re-read at the start of every session
- Items 10–12: re-read before every paper draft revision
- Items 13–14: pin to the top of the experiment notebook permanently

---

## Pre-experiment checks (before every run)

### 1. Verify the GPU is actually being used

If a 100-minute experiment seems suspicious on an A100, it is probably running on CPU somewhere. The Exp 2b sklearn case is a real example. Run `nvidia-smi` every 30 seconds during the experiment. 0% GPU utilization = code problem. ~100% = normal.

```bash
watch -n 30 nvidia-smi
```

### 2. Confirm `attn_implementation='eager'`

With flash_attn, attention weights are not returned and results silently come out as zeros or None — no error, no warning. If results look off, suspect this first before debugging anything else. Verify in `nope_analysis/loader.py`: the `load_model_and_tokenizer()` call must pass `attn_implementation='eager'`.

### 3. Confirm all 64 + 1 layers loaded

Truncated loading can silently drop some Global layers. Always print and check before trusting any layer-indexed result:

```python
print(len(model.model.layers))  # must be 65 (64 main + 1 MTP)
```

---

## Skepticism when results arrive

### 4. Distrust "too clean" results

Accuracy = 1.000 (Exp 2) is a 99% data leakage signal, not a success. Suspiciously clean results mean suspect a bug first. Always:
- Compare against a random baseline
- Verify the train/test split was applied correctly
- Check that the probe was not trained and tested on the same indices

### 5. Always report effect size, p-value, AND sample size together

For any "Global > SWA" or similar claim, all three are required:

| What | How | Threshold |
|------|-----|-----------|
| Statistical significance | Welch t-test | p < 0.05 |
| Practical significance | Cohen's d | d >= 0.5 |
| Sample size | n per group | n >= 30 minimum; n >= 100–300 for venue submission |

Missing any one will be flagged by reviewers. Use `compare_groups()` from `nope_analysis/analysis/statistical_tests.py`.

### 6. Distinguish "passes through" from "encodes"

This is the RQ1 issue. "Information exists in this layer's hidden state" and "this layer creates the information" are different claims. Before writing "Layer X does Y," always ask: "Could this be propagated from an earlier layer?" The Exp 2c delta-attribution probe exists specifically to answer this question — do not skip it.

---

## Easy to forget every session

### 7. Push results to git immediately after every experiment

Elice Cloud wipes `model_cache/` on session end. `outputs/` can also be lost if the instance is reassigned. Push after every experiment completes — do not batch pushes.

```bash
git add outputs/ && git commit -m "exp_name: brief result" && git push
```

### 8. Verify which GPU you actually got each session

Elice does not always allocate the same instance. Yesterday: A100 80GB full. Today: MIG 40GB slice possible. If VRAM < 80GB, experiments from Exp 3b onward will OOM without warning.

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

If VRAM < 80GB: stop and request a different instance before running any heavy experiment.

### 9. Dependency install order is strict

Wrong order overwrites transformers and removes the `exaone4_5` module silently. Always:

```bash
pip install -r requirements.txt
pip install git+https://github.com/nuxlear/transformers.git@add-exaone4_5
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify with: `python3 -c "from nope_analysis.loader import load_config; load_config(); print('OK')"`

---

## Critical for paper writing

### 10. Acknowledge the "262K marketed vs 8K experimental" gap early

When a reviewer asks "Why are long-context experiments capped at 8K?", a prepared answer must already exist in the paper. The answer is in `HANDOVER.md` Section 10. Do not add this section last-minute — a reviewer who finds the gap without a prepared response will reject on external validity grounds alone.

### 11. Track NoPE-adjacent prior work continuously

This area moves fast. Papers like "RoPE to NoPE and Back Again" appear regularly. Monitor weekly:
- Google Scholar Alerts: "NoPE attention", "no positional embedding", "hybrid sliding window attention"
- arxiv cs.CL new submissions
- Search terms: `"NoPE"`, `"no positional encoding"`, `"SWA global attention"`

If a similar mechanistic analysis paper drops near submission time, contribution claims weaken substantially and the related work section needs immediate update.

### 12. Clarify the EXAONE license before submitting

EXAONE is NC-licensed (non-commercial). Academic paper publication is allowed. Redistribution of model derivatives is not. If analysis artifacts include model outputs or derived weights, license inheritance is ambiguous. Email LG AI Research before submission to confirm "academic publication OK" in writing — this is a one-time action that prevents post-publication problems.

---

## Meta — the most important

### 13. Write the claimable conclusion BEFORE running the experiment

Do not run the experiment first and then ask "what does this mean?" Before every experiment, write one sentence:

> "If [observed condition], we will claim [specific conclusion]. If not, we will conclude [null result]."

Example for Exp 4:
> "If SWA attention distance saturates beyond 4,096 tokens while Global distance continues to grow, with p < 0.05 and Cohen's d >= 0.5 at >= 2 length conditions, we will claim Global Attention provides long-range coverage that SWA cannot. Otherwise, we will conclude no measurable coverage difference at tested lengths."

This single line prevents HARKing (Hypothesizing After Results are Known), which is a desk-rejection reason at most venues.

### 14. Steel-man the null hypothesis

The current hypothesis is "NoPE Global contributes to long-range dependency." Seriously consider it could be wrong:
- Maybe Global contributes little and SWA's residual dominates
- Maybe positional information in Global layers is an artifact of the residual stream, not Global computation
- Maybe the entropy difference at 2,048 tokens is noise

A null result — "NoPE Global does not contribute measurably at the tested length range" — is still a publishable finding. It would be the first empirical test of the assumption, and a negative result with adequate sample size and statistical rigor is accepted at ACL/EMNLP workshops. Do not design experiments to confirm the hypothesis; design them to test it.

---

*Version: 2026-04-26*
