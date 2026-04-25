# EXAONE 4.5 Research Topics

> This document lists all current and planned research topics.
> New Claude Code sessions must read this file to understand the full research scope.
>
> **Model**: LGAI-EXAONE/EXAONE-4.5-33B
> **Paper**: arXiv:2604.08644 (EXAONE 4.5 Technical Report)
> **Final goal**: arXiv paper → ACL / EMNLP workshop submission

---

## What the researcher needs to prepare

**"Auto"** = code handles it automatically. **"Researcher"** = must be provided manually.

| Topic | Text input | Images | Extra prep |
|-------|-----------|--------|------------|
| RQ1–3 (all current experiments) | auto | not needed | none |
| A. Lost in the Middle | auto | not needed | none |
| B. MTP layer analysis | auto | not needed | none |
| **C. Reasoning mode** | auto | not needed | **math/logic problem samples recommended** |
| D. Korean specialization | auto | not needed | none |
| **E. Vision encoder × NoPE** | auto | **required** | see Topic E section below |
| F. KV Head specialization | auto | not needed | none |
| G. Reordered Norm | auto | not needed | none |
| H. Layer importance | auto | not needed | none |
| **I. Memory retrieval vs genuine reasoning** | auto | not needed | **AIME problem samples + counterfactual variants recommended** |

---

### Topic C — Reasoning mode: recommended prep

Current experiments only do a forward pass with fixed input.
Reasoning mode analysis requires the model to **actually generate tokens** — a different setup.

**Useful to provide**:
- 10–20 math problems (AIME level or below)
- 10–20 logical reasoning problems
- Korean-language problem samples (for Korean reasoning analysis)

Experiments can run without these (code has built-in defaults), but researcher-provided problems improve result diversity and credibility.

---

### Topic I — Memory retrieval vs reasoning: recommended prep

The core of the counterfactual test is problems **not in the training data**.
Auto-generated variants work, but researcher-prepared ones are higher quality.

**Useful to provide**:
```
Type 1: Math (AIME style)
  - 10 original problems (from AIME 2024 or 2026)
  - Variants with only the numbers changed (same structure, different values)
  e.g. "Find integers x, y satisfying x² + y² = 100..."
    → "Find integers x, y satisfying x² + y² = 169..."

Type 2: Logical reasoning
  - 5 premise-conclusion logic problems
  - Variants with slightly changed premises

Type 3: Korean reasoning (optional)
  - 5 Korean math or logic problems
```

**Works without prep**: code has 3–5 built-in problems.

---

### Topic E — Vision encoder × NoPE: images required

Image files cannot be auto-generated — **real image files are mandatory**.

**Image types and storage location**:
```
sda212331/test_images/          ← create this folder, place images here (git-excluded)
├── document_korean.jpg         ← document/newspaper image with Korean text
├── document_english.jpg        ← document image with English text
├── chart_or_graph.jpg          ← chart/graph image (with numerical data)
├── natural_scene.jpg           ← general photo (people, scenery)
└── mixed_text_image.jpg        ← mixed text+image (slide, poster)
```

**Why these categories**:
- Document images: high text content → language layers expected to reference image tokens heavily
- Natural photos: no text → different image token attention pattern expected
- Comparing these reveals Global layers' multimodal processing strategy

**Image requirements**: JPG or PNG, any size (Vision Encoder auto-resizes)

---

## Research background: why this model

EXAONE 4.5 uses a SWA (Sliding Window Attention) + Global (NoPE) hybrid architecture.
What is novel is the specific design choice of applying NoPE **only to Global Attention layers**
(not to SWA layers) — to our knowledge, no prior mechanistic analysis of this exact configuration exists in the literature.

A notable weakness appears in official benchmarks:
```
AA-LCR (long-context reasoning):  EXAONE 4.5 = 50.6  vs  GPT-5 mini = 68.0  (gap: 17.4)
```
The central research motivation is to determine whether the NoPE architecture contributes to this gap.

---

## PART 1: Current research (RQ1 – RQ3)

### RQ1 — Do NoPE layers encode positional information?

If token position can be linearly decoded from Global layer hidden states (despite no positional embedding),
it means preceding SWA layers implicitly pass positional information forward.

| Experiment | Method | Status |
|-----------|--------|--------|
| Exp 2 | Layer-wise linear probe → position classification (5 prompts, no train/test split) | ✅ Done |
| Exp 2b | Train/test split, 30 prompts, 64/128/256 tokens, domain breakdown | ✅ Done |

**Current result**:
- Exp 2: accuracy=1.0 for both → confirmed positional encoding, but unreliable (no train/test split)
- Exp 2b: test accuracy 0.52–0.55 for both Global and SWA (random baseline=0.10) → **RQ1 confirmed**
- No meaningful difference between Global (NoPE) and SWA at any length
- **Conclusion**: NoPE Global layers encode positional information at the same level as SWA layers via propagation from preceding RoPE layers.

---

### RQ2 — Do SWA and Global have functionally different attention patterns?

Measured via attention entropy (spread) and attention distance (how far the model looks).
Key question: does the difference grow with input length?

| Experiment | Method | Status |
|-----------|--------|--------|
| Exp 1 | Short input (14–26 tokens) entropy measurement | ✅ Done |
| Exp 1b | Long input (128–2048 tokens) entropy + distance measurement | ✅ Done |

**Current results**:
- Exp 1: Global=0.930, SWA=0.942 → no difference at short lengths (both fit inside the SWA window)
- Exp 1b: at 2,048 tokens — Global entropy (3.54) > SWA (3.45) reversal; attention distance gap = 154 tokens
- **New hypothesis**: gap will accelerate beyond 4,096 tokens → motivates Exp 4

---

### RQ3 — Does NoPE Global actually contribute to long-range dependency?

Measured by how much perplexity increases when Global layers are disabled.
Key: does the effect grow beyond 4,096 tokens (the SWA window boundary)?

| Experiment | Method | Status |
|-----------|--------|--------|
| Exp 3b ⭐ | Inject SWA mask into Global layers → measure perplexity change (1,024–6,144 tokens) | 🔄 Pending |
| Exp 3 | Zero out Global self_attn output → measure perplexity change (1,024–6,144 tokens) | 🔄 Pending |
| Exp 4 | Hook-based attention stats at 2,048–8,192 tokens | 🔄 Pending |

---

## PART 2: Future research topics (priority order)

---

### Topic A — "Lost in the Middle" × NoPE  ★ highest priority

**Motivation**: directly explains AA-LCR score of 50.6. Highest potential paper impact.

**Research question**: how differently do Global and SWA reference the beginning / middle / end of long input?

**Method**:
- Divide input into 3 segments: first 1/3 (early), middle 1/3, last 1/3 (recent)
- Sum attention weights directed to each segment
- Compare per-segment reference ratio: Global vs SWA
- Observe trend across lengths (1K, 4K, 8K, 32K tokens)

**Expected finding**: SWA concentrates on the end; Global attends more evenly or reaches the beginning.
If "Lost in the Middle" is worse in SWA layers → Global provides compensation.

**Code**: `nope_analysis/experiments/expA_lost_in_middle.py` (not yet written)

---

### Topic B — MTP layer analysis  ★ highest novelty

**Motivation**: to our knowledge, no prior mechanistic analysis of EXAONE 4.5's MTP (Multi-Token Prediction) layer exists.
(Note: EXAONE 4.5 adopts MTP from two sources cited in the paper: DeepSeek-V3 [DeepSeek-AI] and
Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction", ICML 2024 [Meta Research].
EXAONE 4.5's specific MTP implementation and its interaction with the NoPE architecture has not been analyzed.)

**Research questions**:
1. How does the MTP layer's attention pattern differ from the 64 main layers?
2. How much does perplexity change if the MTP layer is removed?

**Method**:
- Extract MTP layer hidden states and compare with layer 64
- Measure MTP attention entropy and distance
- MTP ablation: zero out MTP layer output → measure loss change

**Note**: MTP layer access path is speculative — confirm at runtime before use:
```python
# Verify the actual attribute path before running experiments
assert hasattr(model, 'language_model'), "unexpected model structure"
# Expected: model.language_model.model.mtp_layers[0]
# — confirm via: print([n for n, _ in model.named_modules() if 'mtp' in n.lower()])
```

**Code**: `nope_analysis/experiments/expB_mtp_analysis.py` (not yet written)

---

### Topic C — Attention pattern changes in reasoning mode

**Motivation**: EXAONE 4.5 defaults to `enable_thinking=True`. Do Global layers behave differently during thinking token generation?

**Research question**: does NoPE Global leverage long-range dependency more during reasoning (think tokens)?

**Method**:
- Run model in reasoning mode on math/logic problems → identify think token spans
- Compare attention entropy/distance: think span vs answer span
- Check whether Global layers attend more broadly during think spans

**Note**: requires inference mode (`model.generate()`) — current experiments only use forward pass.

**Code**: `nope_analysis/experiments/expC_reasoning_attention.py` (not yet written)

---

### Topic D — Korean vs English layer specialization

**Motivation**: EXAONE shows competitive Korean performance (KMMMU: EXAONE 42.7 vs Qwen3-VL 32B 37.8;
note Qwen3.5-27B scores 51.7, so EXAONE ranks 2nd among compared VLMs).
Korean is SOV (verb at end) → longer-range dependencies are structurally required.

**Research question**: are certain Global layers specialized for Korean syntactic processing?

**Method**:
- Prepare parallel Korean/English prompts (translation pairs)
- Language identification probe on layer-wise hidden states
- Check whether Global attention distance is longer for Korean than English
- Compare perplexity increase from Global ablation: Korean vs English

**Code**: `nope_analysis/experiments/expD_korean_english.py` (not yet written)

---

### Topic E — Vision encoder × NoPE interaction

**Motivation**: EXAONE 4.5 is a VLM. When image tokens enter language layers, do Global (NoPE) layers reference image tokens more than SWA layers?

**Research question**: are NoPE layers the core layers for multimodal integration?

**Method**:
- Construct image + text input (image tokens + question tokens)
- Sum attention weights from text tokens to image tokens, per layer
- Compare cross-modal attention: Global vs SWA layers
- Identify which layers reference images most during answer token generation

**Note**: must understand how vision encoder processes inputs; image token position range needs runtime confirmation.

**Code**: `nope_analysis/experiments/expE_vision_nope.py` (not yet written)

---

### Topic F — KV head role specialization

**Motivation**: in GQA, 40 Q-heads share 8 KV-heads. If the 8 KV-heads have different roles, specialization patterns should differ between Global and SWA.

**Research question**: do Global layer KV-heads specialize more distinctly than SWA KV-heads?

**Method**:
- Measure attention distance distribution of 8 KV-heads per layer
- Measure inter-head cosine similarity as a specialization proxy
- Compare head specialization patterns: Global vs SWA

**Code**: `nope_analysis/experiments/expF_kv_head_specialization.py` (not yet written)

---

### Topic G — Reordered Norm effect quantification

**Motivation**: EXAONE 4.0/4.5 uses QK-Reorder-LN (non-standard compared to Pre/Post Norm):
(1) RMSNorm on Q/K inputs **before** attention, and (2) RMSNorm on attention output **before** the residual add.
This double-norm design stabilizes deep layers by controlling variance growth (source: EXAONE 4.0 paper, Figure 2).
What quantitative effect does this have on the residual stream representations?

**Method**:
- Track L2 norm of residual stream per layer
- Measure hidden state change magnitude before/after norm
- Analyze norm magnitude trend as layers deepen

**Code**: `nope_analysis/experiments/expG_reordered_norm.py` (not yet written)

---

### Topic H — Layer-wise importance (fine-grained ablation)

**Motivation**: Exp 3 disables all Global layers simultaneously. Layer-wise importance likely varies.

**Research question**: which matters more — shallow Global (layers 3, 7) or deep Global (layers 59, 63)?

**Method**:
- Ablate each of the 16 Global layers individually → measure perplexity change per layer
- Generate importance heatmap (layer index × input length)
- Identify which Global layers handle long-range dependency

**Code**: `nope_analysis/experiments/expH_layer_importance.py` (not yet written)

---

### Topic I — Memory retrieval vs genuine reasoning detection  ★ highest societal impact

**Motivation**: whether LLMs retrieve patterns from training data or reason in real-time has not been
mechanistically proven. EXAONE 4.5 is a particularly good model for this investigation.

```
AIME 2024 (Jan 2024):  pre-cutoff   ← problems in training data (knowledge cutoff: Nov 2024)
AIME 2025 (Jan 2025):  post-cutoff  ← problems after knowledge cutoff
AIME 2026 (Jan 2026):  post-cutoff  ← problems well after knowledge cutoff
```

Note: EXAONE 4.5 knowledge cutoff is Nov 2024 (from EXAONE 4.0 Table 1). AIME 2025 (held Jan–Feb 2025)
is already after the cutoff, not before. A pre/post comparison therefore uses AIME 2024 as the
"potentially memorized" baseline and AIME 2025/2026 as the "post-cutoff" group.

If scores are similar across pre- and post-cutoff sets, this is suggestive (not conclusive) evidence
of genuine reasoning capability — problem difficulty distributions may differ, and counterfactual
verification is required before drawing strong conclusions.

⚠️ AIME 2025 score of 92.9 is not confirmed in the EXAONE 4.5 technical report (arXiv:2604.08644).
Use only scores directly cited in the paper; verify before citing in any publication.

**Research question**: is EXAONE 4.5's answer generation memory retrieval or step-by-step reasoning?
What role do NoPE Global layers play in this process?

**Experiment design**:

1. **Counterfactual variant test**
   - Change only the numbers in math problems → measure accuracy change
   - If memory retrieval: accuracy drops sharply on variants
   - If genuine reasoning: variants solved correctly too

2. **Logit Lens analysis**
   - Check prediction distribution after each layer
   - If answer emerges in shallow layers: memory retrieval suspected
   - If prediction converges gradually in deep layers: reasoning pattern

3. **Think token span × NoPE analysis** (connects to Topic C)
   - Generate with `enable_thinking=True` → compare think span vs answer span
   - Hypothesis: Global (NoPE) layers reference problem context more during reasoning spans
   - "Long-range attention activation = mechanistic signal of genuine reasoning"

4. **Pre/post-cutoff accuracy pattern comparison**
   - Compare AIME 2024 (pre-cutoff, Jan 2024) vs AIME 2025/2026 (post-cutoff, after Nov 2024) by problem type
   - If only certain types show a score gap → those types are memory retrieval candidates

**Connection to our research**:
```
RQ2/RQ3: NoPE Global integrates long-range information
    ↓
Topic C:  Attention patterns change in reasoning mode
    ↓
Topic I:  Those attention changes are mechanistic evidence of genuine reasoning
```
→ One coherent story: "EXAONE 4.5's NoPE Global integrates long-range context during reasoning,
   providing the structural basis for genuine reasoning rather than memory retrieval."

**Code**: `nope_analysis/experiments/expI_reasoning_vs_memory.py` (not yet written)

---

## Full research map

```
EXAONE 4.5 NoPE Research
│
├── [Active] NoPE fundamentals
│   ├── RQ1: positional encoding (Exp 2, 2b)
│   ├── RQ2: attention pattern differences (Exp 1, 1b)
│   └── RQ3: long-range contribution (Exp 3b, Exp 3, Exp 4)
│
├── [Future] Performance gap analysis
│   ├── Topic A: Lost in the Middle × NoPE  ← explains AA-LCR 50.6
│   └── Topic D: Korean specialization       ← EXAONE identity
│
├── [Future] Architecture deep dives
│   ├── Topic B: MTP layer                  ← highest novelty
│   ├── Topic F: KV head specialization
│   ├── Topic G: Reordered Norm
│   └── Topic H: layer-wise importance
│
├── [Future] Multimodal / reasoning
│   ├── Topic C: Reasoning mode attention
│   └── Topic E: Vision × NoPE interaction
│
└── [Future] Reasoning authenticity
    └── Topic I: Memory retrieval vs genuine reasoning  ← highest societal impact
```

---

*Last updated: 2026-04-25*
