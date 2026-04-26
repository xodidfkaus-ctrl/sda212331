> **Retrieved**: 2026-04-26
> **Source**: https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B
> **Note**: Content preserved exactly as of retrieval date. Verify against upstream for updates.

---

# EXAONE-4.5-33B Model Card

## EXAONE 4.5

We introduce EXAONE 4.5, the first open-weight vision language model developed by LG AI Research. Integrating a dedicated visual encoder into the existing EXAONE 4.0 framework, we expand the model's capability toward multimodality. EXAONE 4.5 features 33 billion parameters in total, including 1.2 billion parameters from the vision encoder. EXAONE 4.5 achieves competitive performance in general benchmark while outperforming SOTA models of similar size in document understanding and Korean contextual reasoning, inheriting powerful language capabilities from our previous language models.

For more details, please refer to the [technical report](http://arxiv.org/abs/2604.08644), [blog](https://www.lgresearch.ai/blog/view?seq=641) and [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.5).

### Model Configuration

- Model Type: Causal Language Model + Vision Encoder
- Number of Parameters (Language Model): 31.7B
- Number of Parameters (Vision Encoder): 1.29B
- Hidden Dimension: 5,120
- Intermediate size: 27,392
- Number of Layers: 64 Main layers + 1 MTP layers
  - Hybrid Attention Pattern: 16 x (3 Sliding window attention + 1 Global attention)
  - Reordered Norm: Apply normalization after Attention/MLP, and before residual connection
- Sliding Window Attention
  - Number of Attention Heads: 40 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - Sliding Window Size: 4096
- Global Attention
  - Number of Attention Heads: 40 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - No Rotary Positional Embedding Used (NoPE)
- Vision Encoder
  - Grouped Query Attention (GQA)
  - 2D RoPE for vision embeddings
- Vocab Size: 153,600
- Context Length: 262,144 tokens
- Knowledge Cutoff: Dec 2024 (2024/12)

## Evaluation Results

### Vision-Language Tasks

| Task Category | EXAONE 4.5 33B | GPT-5 mini | Qwen3-VL 32B | Qwen3-VL 235B | Qwen3.5 27B |
|---|---|---|---|---|---|
| **STEM / Puzzle** |
| MMMU | 78.7 | 79.0 | 78.1 | 80.6 | 82.3 |
| MMMU-Pro | 68.6 | 67.3 | 68.1 | 69.3 | 75.0 |
| MedXpertQA-MM | 42.1 | 34.4 | 41.6 | 47.6 | 62.4 |
| MathVision | 75.2 | 71.9 | 70.2 | 74.6 | 86.0 |
| MathVista (mini) | 85.0 | 79.1 | 85.9 | 85.8 | 87.8 |
| WeMath | 79.1 | 70.3 | 71.6 | 74.8 | 84.0 |
| LogicVista | 73.8 | 70.3 | 70.9 | 72.2 | 77.0 |
| BabyVision | 18.8 | 20.9 | 17.4 | 22.2 | 44.6 |
| **Document Understanding** |
| AI2D | 89.0 | 88.2 | 88.9 | 89.2 | 92.9 |
| ChartQAPro | 62.2 | 60.9 | 61.4 | 61.2 | 66.8 |
| CharXiv (RQ) | 71.7 | 68.6 | 65.2 | 66.1 | 79.5 |
| OCRBench v2 | 63.2 | 55.8 | 68.4 | 66.8 | 67.3 |
| OmniDocBench v1.5 | 81.2 | 77.0 | 83.1 | 84.5 | 88.9 |
| **General** |
| MMStar | 74.9 | 74.1 | 79.4 | 78.7 | 81.0 |
| BLINK | 68.8 | 67.7 | 68.5 | 67.1 | 71.6 |
| HallusionBench | 63.7 | 63.2 | 67.4 | 66.7 | 70.0 |
| **Korean** |
| KMMMU | 42.7 | 42.6 | 37.8 | 42.1 | 51.7 |
| K-Viscuit | 80.1 | 78.5 | 78.5 | 83.9 | 84.0 |
| KRETA | 91.9 | 94.8 | 90.3 | 92.8 | 96.5 |

### Language-only Tasks

| Task Category | EXAONE 4.5 33B | GPT-5 mini | K-EXAONE 236B | Qwen3-VL 235B | Qwen3.5 27B |
|---|---|---|---|---|---|
| **Reasoning** |
| AIME 2025 | 92.9 | 91.1 | 92.8 | 89.7 | 93.5 |
| AIME 2026 | 92.6 | 92.4 | 92.2 | 89.4 | 90.8 |
| GPQA-Diamond | 80.5 | 82.3 | 79.1 | 77.1 | 85.5 |
| LiveCodeBench v6 | 81.4 | 78.1 | 80.7 | 70.1 | 80.7 |
| MMLU-Pro | 83.3 | 83.3 | 83.8 | 83.8 | 86.1 |
| **Agentic Tool Use** |
| τ2-Bench (Retail) | 77.9 | 78.3 | 78.6 | 67.0 | 84.7 |
| τ2-Bench (Airline) | 56.5 | 60.0 | 60.4 | 62.0 | 67.5 |
| τ2-Bench (Telecom) | 73.0 | 74.1 | 73.5 | 44.7 | 99.3 |
| **Instruction Following** |
| IFBench | 62.6 | 74.0 | 67.3 | 59.2 | 76.5 |
| IFEval | 89.6 | 92.8 | 89.7 | 88.2 | 95.0 |
| **Long Context Understanding** |
| AA-LCR | 50.6 | 68.0 | 53.5 | 58.7 | 67.3 |
| **Korean** |
| KMMLU-Pro | 67.6 | 72.5 | 67.3 | 71.1 | 73.0 |
| KoBALT | 52.1 | 63.6 | 61.8 | 51.1 | 54.9 |

## Quickstart

### Serving EXAONE 4.5

For better inference speed and memory usage, it is preferred to serve the model using optimized inference engines. The EXAONE 4.5 model is supported by various frameworks, including TensorRT-LLM, vLLM, SGLang, and llama.cpp. Support will be expanded in the future.

Practically, you can serve the EXAONE 4.5 model with 256K context length on **single H200 GPU**, or **4x A100-40GB GPUs** by using a tensor-parallelism.

### TensorRT-LLM

TensorRT-LLM provides zero day support for EXAONE 4.5. Transformers library of our fork is required to utilize EXAONE 4.5 model. You can install Transformers by running the following commands:

```bash
pip install git+https://github.com/nuxlear/transformers.git@add-exaone4_5
```

Please refer to the official [installation guide](https://github.com/NVIDIA/TensorRT-LLM?tab=readme-ov-file#getting-started), and [EXAONE documentations](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/exaone), and [EXAONE 4.5 PR](https://github.com/NVIDIA/TensorRT-LLM/pull/12873) for the detail.

After you install the TensorRT-LLM, you can launch the server with the following code snippet. You can remove unnecessary arguments from the snippet.

```bash
trtllm-serve LGAI-EXAONE/EXAONE-4.5-33B \
    —tp_size 2 \
    —port 8000 \
    —reasoning_parser qwen3
```

An OpenAI-compatible API server will be available at [http://localhost:8000/v1](http://localhost:8000/v1).

### vLLM

Both Transformers and vLLM of our forks are required to utilize EXAONE 4.5 model. You can install the requirements by running the following commands:

```bash
uv pip install git+https://github.com/lkm2835/vllm.git@add-exaone4_5
uv pip install git+https://github.com/nuxlear/transformers.git@add-exaone4_5
```

After you install the vLLM, you can launch the server with the following code snippet. You can remove unnecessary arguments from the snippet.

```bash
vllm serve LGAI-EXAONE/EXAONE-4.5-33B \
    --served-model-name EXAONE-4.5-33B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --limit-mm-per-prompt '{"image": 64}' \
    --speculative_config '{
        "method": "mtp",
        "num_speculative_tokens": 3
    }'
```

An OpenAI-compatible API server will be available at [http://localhost:8000/v1](http://localhost:8000/v1).

### SGLang

Both Transformers and SGLang of our forks are required to utilize EXAONE 4.5 model. You can install the requirements by running the following commands:

```bash
uv pip install 'git+https://github.com/lkm2835/sglang.git@add-exaone4_5#subdirectory=python&egg=sglang[all]'
uv pip install git+https://github.com/nuxlear/transformers.git@add-exaone4_5
```

After you install the SGLang, you can launch the server with the following code snippet. You can remove unnecessary arguments from the snippet.

```bash
python -m sglang.launch_server \
    --model-path LGAI-EXAONE/EXAONE-4.5-33B \
    --served-model-name EXAONE-4.5-33B \
    --port 8000 \
    --tp-size 2 \
    --mem-frac 0.81 \
    --reasoning-parser qwen3 \
    --tool-call-parser hermes \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

An OpenAI-compatible API server will be available at [http://localhost:8000/v1](http://localhost:8000/v1).

### Using EXAONE 4.5

After launching the OpenAI-compatible server with EXAONE 4.5, you can seamlessly use the model via API with a single code integration, even though the serving framework has changed. To use OpenAI Python SDK and following examples, you should install the `openai` library on your environment.

> To achieve the expected performance, we recommend using the following configurations:
>
> - We recommend to use `temperature=1.0`, `top_p=0.95`, `presence_penalty=1.5` for general purpose.
> - We recommend to use `temperature=0.6`, `top_p=0.95`, `presence_penalty=1.5`, `top_k=20` for OCR/document-related tasks, and Korean inputs.
> - We recommend to use `temperature=1.0`, `top_p=0.95` for text-only inputs.
> - Different from EXAONE-4.0, EXAONE 4.5 uses `enable_thinking=True` as default. Thus, you need to set `enable_thinking=False` when you want to use non-reasoning mode.
> - EXAONE 4.5 prefers using `\boxed{}` format to answer the question. We recommend using this format with the corresponding format instruction for better parsing accuracy.

## Limitation

EXAONE 4.5 models, like all existing multimodal models, have certain limitations and may occasionally generate inappropriate responses. The multimodal model generates responses based on the output probability of tokens, and it is determined during learning from training data. While we make every effort to exclude personal, harmful, and biased information from the training data, some problematic content may still be included, potentially leading to undesirable responses. Please note that the text generated by EXAONE 4.5 models does not reflect the views of LG AI Research.

- Inappropriate answers may be generated, which contain personal, harmful or other inappropriate information.
- Biased responses may be generated, which are associated with age, gender, race, and so on.
- The generated responses rely heavily on statistics from the training data, which can result in the generation of semantically or syntactically incorrect sentences.
- Since the models do not reflect the latest information, the responses may be false or contradictory.

LG AI Research strives to reduce potential risks that may arise from EXAONE 4.5 models. Users are not allowed to engage in any malicious activities (e.g., keying in illegal information) that may induce the creation of inappropriate outputs violating LG AI's ethical principles when using EXAONE 4.5 models.

## License

The model is licensed under [EXAONE AI Model License Agreement 1.2 - NC](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B/blob/main/LICENSE)

## Citation

```bibtex
@article{exaone-4.5,
  title={EXAONE 4.5 Technical Report},
  author={{LG AI Research}},
  journal={arXiv preprint arXiv:2604.08644},
  year={2026}
}
```

## Contact

LG AI Research Technical Support: [contact_us@lgresearch.ai](mailto:contact_us@lgresearch.ai)
