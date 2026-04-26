# MODEL_FACTS.md — EXAONE 4.5 33B Verifiable Reference

**Purpose**: Quick-lookup for verified architectural facts. Every entry cites its source.
**Rule R6**: Check this file BEFORE making any architectural claim. Do not use memory.
**Last updated**: 2026-04-26
**Sources**:
- `[MC]` = `papers/model_card_EXAONE-4.5-33B.md` (retrieved 2026-04-26 from HuggingFace)
- `[TR §N]` = `papers/2604.08644/fulltext.txt` Section N (arXiv:2604.08644)

---

## Architecture

| Fact | Value | Source |
|------|-------|--------|
| Total parameters | ~33B (marketed) | [MC §Model Configuration] |
| LM parameters | 31.7B | [MC §Model Configuration] |
| Vision encoder parameters | 1.29B | [MC §Model Configuration] |
| Hidden dimension | 5,120 | [MC §Model Configuration] |
| Intermediate (MLP) size | 27,392 | [MC §Model Configuration] |
| Main decoder layers | 64 | [MC §Model Configuration] |
| MTP (Multi-Token Prediction) layers | 1 | [MC §Model Configuration] |
| Total layer count (including MTP) | 65 | [MC §Model Configuration] |
| Hybrid attention pattern | 16 × (3 SWA + 1 Global) = 48 SWA + 16 Global | [MC §Model Configuration] |
| SWA Q-heads | 40 | [MC §Model Configuration] |
| SWA KV-heads | 8 | [MC §Model Configuration] |
| SWA head dimension | 128 (both Q and KV) | [MC §Model Configuration] |
| SWA window size | 4,096 tokens | [MC §Model Configuration] |
| SWA positional encoding | RoPE (implied; only Global is NoPE) | [MC §Model Configuration] |
| Global attention Q-heads | 40 | [MC §Model Configuration] |
| Global attention KV-heads | 8 | [MC §Model Configuration] |
| Global attention head dimension | 128 (both Q and KV) | [MC §Model Configuration] |
| Global attention positional encoding | **NoPE — No RoPE applied** | [MC §Model Configuration] |
| Global layer indices (0-based) | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 | Derived from 16×(3SWA+1G) pattern starting at index 0; consistent with [MC] |
| Normalization type | Reordered Norm: applied after Attention/MLP, before residual connection | [MC §Model Configuration] |
| Vocab size | 153,600 | [MC §Model Configuration] |
| Model type | Causal LM + Vision Encoder (VLM) | [MC §Model Configuration] |
| LM backbone | EXAONE 4.0 32B architecture | [TR §Introduction, line 33] |
| MTP module | From DeepSeek-V3 / Moon et al. [14, 19] | [TR §Architecture] |

---

## Context

| Fact | Value | Source |
|------|-------|--------|
| Maximum context length | 262,144 tokens (= 256K in binary) | [MC §Model Configuration] |
| Context length in technical report | "256K tokens" | [TR §Training, line 212] |
| Context extension method | Integrated into SFT stage (not a separate post-training stage) | [TR §Training, line 212–213] |
| Inference context for benchmarks | 256K reported; H200 single-GPU or 4×A100-40GB for TP | [MC §Quickstart] |

---

## Vision Encoder

| Fact | Value | Source |
|------|-------|--------|
| Vision encoder parameters | 1.29B (also reported as ~1.2B) | [MC]; [TR line 33: "1.2B parameter vision encoder"] |
| Vision encoder type | Custom, trained from scratch | [TR §Architecture, line 94] |
| Vision positional encoding | 2D RoPE | [TR §Architecture, line 98]; [MC §Model Configuration] |
| Vision attention type | GQA | [MC §Model Configuration] |
| Vision-LM merger | Separate merger module (type not specified in fulltext) | [TR §Architecture] |

---

## Training

| Fact | Value | Source |
|------|-------|--------|
| Training scale (tokens) | Not explicitly stated in available fulltext | — |
| Knowledge cutoff | December 2024 | [MC §Model Configuration] |
| Languages | Multilingual; Korean emphasized; tokenizer from K-EXAONE | [TR §Architecture, line 110–111] |
| Modalities | Text + Vision (multimodal) | [TR] |
| Training pipeline | Two-stage pre-training → SFT → RLVR/preference learning | [TR §Training] |
| Context extension in pipeline | Integrated into SFT (not a standalone post-training stage) | [TR §Training, line 212–213] |
| MTP at inference | Disabled at inference time (benchmark evals only) | [TR §Evaluation, line 329] |

---

## License

| Fact | Value | Source |
|------|-------|--------|
| License name | EXAONE AI Model License Agreement 1.2 - NC | [MC §License]; [TR Appendix B] |
| Licensor | LG Management Development Institute Co., Ltd. | [TR Appendix B §1.4] |
| **PERMITTED** | Research and educational use | [TR Appendix B §2.1a] |
| **PERMITTED** | Publishing academic papers / presenting results | [TR Appendix B §2.1b] |
| **PERMITTED** | Creating derivatives for research/education | [TR Appendix B §2.1c] |
| **PERMITTED** | Non-commercial competition participation | [TR Appendix B §2.1a] |
| **PERMITTED** | Distributing model/derivatives with license copy | [TR Appendix B §2.1d] |
| **FORBIDDEN** | Commercial use (products/services generating revenue) | [TR Appendix B §3.1] |
| **FORBIDDEN** | Developing competing models using this model | [TR Appendix B §3.1] |
| **FORBIDDEN** | Redistributing without license copy | [TR Appendix B §2.1d] |
| Output ownership | Licensor claims no rights in outputs; Licensee solely responsible | [TR Appendix B §4.2] |
| Attribution requirement | Cite model name + version in any publication | [TR Appendix B §4.3] |
| **Key implication for this project** | Academic publication is explicitly permitted (§2.1b). Confirm in writing with LG AI Research before submission if derivatives are distributed (see CHECKLIST.md item 12). | [TR Appendix B §2.1b, §3.1] |

---

## Comparison Points (cited in technical report or model card)

| EXAONE 4.5 aspect | Compared to | Relationship | Source |
|---|---|---|---|
| Language backbone | EXAONE 4.0 32B | EXAONE 4.5 LM inherits EXAONE 4.0 architecture | [TR §Introduction, line 33] |
| MTP module | DeepSeek-V3 / Moon et al. | Borrowed MTP design | [TR §Architecture, refs 14, 19] |
| Reasoning/thinking mode | Qwen3 | Uses same `reasoning_parser qwen3` at inference | [MC §Quickstart, vLLM example] |
| Vision encoder | "Existing vision encoders did not meet requirements" | Custom trained from scratch instead | [TR §Architecture, line 94] |
| Korean | K-EXAONE tokenizer reused | Vocab and tokenizer design inherited | [TR §Architecture, line 110–111] |

**Note**: Claims like "first open SWA+NoPE model" are NOT in the technical report or model card.
Do not make novelty comparison claims without a specific citation.

---

## Facts NOT in available reference materials

The following facts are frequently assumed but **not present** in the model card or
the available sections of the technical report. Do not state these without verification:

- Exact total training token count
- Pre-training data composition / sources
- Whether SWA layers also have NoPE as an option
- RoPE base frequency (theta) for SWA layers
- Exact MTP module architecture details (number of draft heads, etc.)
- Whether context extension (256K) applies equally to vision+text and text-only
- Inference batch size / throughput numbers under specific hardware configs

If any of these facts are needed, extract from `papers/2604.08644/fulltext.txt`
or contact `contact_us@lgresearch.ai`.
