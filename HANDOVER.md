# EXAONE 4.5 NoPE Research — Session Handover

> **To Claude Code reading this**: Read this document first at the start of every new session.
> Complete the **Section 0 checklist** before doing anything else.
> Skipping the checklist causes experiments to run on CPU (hours instead of minutes) or fail due to missing model.

> **Full research topic list** → see `RESEARCH_TOPICS.md`

---

## 0-1. What the researcher needs to prepare

Most experiments generate text automatically — no manual input needed.
**Two exceptions:**

| When | What to prepare | Where |
|------|----------------|-------|
| Before Topic E (Vision × NoPE) | 5 image files (document/chart/photo, JPG/PNG) | Create `sda212331/test_images/` and place files there |
| Before Topic C (Reasoning mode) | 10–20 math/logic problem samples (optional, code has defaults) | Paste as text to Claude Code |

Image requirements → see "Topic E" section in `RESEARCH_TOPICS.md`.

---

## 0. Pre-run Checklist (run every session before anything else)

Run the steps **in order**. All must pass before starting experiments.

### Step 1 — GPU check (most important)

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); [print(f'  GPU{i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

**Expected output:**
```
CUDA: True
GPUs: 2
  GPU0: NVIDIA A100 80GB PCIe
  GPU1: NVIDIA A100 80GB PCIe
```

**❌ If `CUDA: False` — stop immediately. Do not run any experiment.**

Diagnosis and fix:
```bash
# Check cause
python3 -c "import torch; torch.cuda.is_available()" 2>&1 | grep -i "warn\|error"

# Most common cause: PyTorch / CUDA driver version mismatch
nvidia-smi | grep "CUDA Version"

# Fix for CUDA 12.x driver
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python3 -c "import torch; print(torch.cuda.is_available())"
```

> **Why this matters**: A forward pass on the 33B model takes 3–4 hours on CPU, 5–10 minutes on GPU.

---

### Step 2 — Model cache check

```bash
ls /home/elicer/sda212331/model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/snapshots/
```

**Expected**: a long hash folder name (e.g. `58d6616...`)

**❌ Missing**: re-download the model (deleted on session restart).
```bash
cd /home/elicer/sda212331
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('LGAI-EXAONE/EXAONE-4.5-33B', cache_dir='./model_cache', ignore_patterns=['*.bin','*.pt'])
"
# Takes ~20 minutes
```

---

### Step 3 — Dependency check

```bash
python3 -c "from nope_analysis.loader import load_config; cfg = load_config(); print('OK:', cfg.model_type)"
```

**Expected**: `OK: exaone4_5` or similar

**❌ ImportError**: reinstall packages
```bash
pip install transformers safetensors scikit-learn matplotlib -q
```

---

### Step 4 — Git auth setup (required every session)

**This environment is ephemeral.** Git credentials are lost on every session restart.
Ask the user for their GitHub PAT (Personal Access Token) and configure it:

```bash
git config credential.helper store
echo "https://xodidfkaus-ctrl:PAT_HERE@github.com" > ~/.git-credentials
git remote set-url origin https://github.com/xodidfkaus-ctrl/sda212331.git
```

If the user hasn't provided a PAT, ask:
> "I need a GitHub PAT to push results. Please generate one at github.com/settings/tokens with `public_repo` scope."

**Never embed the PAT directly in the remote URL** — it will be exposed in `git remote -v` output.

---

### Step 5 — Git sync check

```bash
cd /home/elicer/sda212331
git log --oneline -3
git status
```

If there are unpushed results from the previous session, push them first.

---

### Checklist summary

| Item | Command | Pass condition |
|------|---------|---------------|
| GPU active | `python3 -c "import torch; print(torch.cuda.is_available())"` | `True` |
| GPU count | `python3 -c "import torch; print(torch.cuda.device_count())"` | `2` |
| Model cache | `ls model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/snapshots/` | folder exists |
| Dependencies | `python3 -c "from nope_analysis.loader import load_config; load_config()"` | no error |
| Git auth | `git push --dry-run 2>&1` | no error |
| Git sync | `git status` | no unpushed results |

---

## 1. Project basics

| Item | Value |
|------|-------|
| GitHub | https://github.com/xodidfkaus-ctrl/sda212331 |
| Environment | Elice Cloud — **ephemeral, resets on restart** |
| GPU | **A100 80GB × 1 sufficient** (model=65GB < 80GB; `device_map='auto'` adapts automatically). ×2 is overkill for this research — no speed benefit at batch_size=1. |
| Model path | `/home/elicer/sda212331/model_cache/` (64GB, **deleted on session end**) |
| Working directory | `/home/elicer/sda212331/` |

---

## 2. Research goal and core questions

### One-line summary
> Empirically analyze how **NoPE (No Positional Embedding) Global Attention layers** in EXAONE 4.5 behave differently from SWA layers.

### Background
- EXAONE 4.5 uses a SWA (Sliding Window Attention) + Global (NoPE) hybrid architecture
- The specific design of applying NoPE **only to Global Attention** (not SWA) has no prior mechanistic analysis in the literature
- **Final goal**: arXiv paper → ACL/EMNLP workshop submission

### Three core research questions

| RQ | Question | Experiments |
|----|----------|-------------|
| **RQ1** | Do NoPE Global layers encode positional information? | Exp 2, Exp 2b |
| **RQ2** | Do SWA (RoPE) and Global (NoPE) have functionally different attention patterns? | Exp 1, Exp 1b |
| **RQ3** | Does NoPE Global actually contribute to long-range dependency (262K context)? | Exp 3b, Exp 3, Exp 4 |

### Methodology note — this is exploratory research
Each experiment's result informs the design of the next one. **The experiment plan is not fixed.**

```
Run Exp 1
  → find pattern in results (e.g. layers 32–48 differ)
  → generate hypothesis
  → design Exp 2 (re-measure that range by input length)
  → find Korean/English difference
  → design Exp 3 ...
```

---

## 3. EXAONE 4.5 Architecture (memorize these)

| Item | Value |
|------|-------|
| Total parameters | 33B (LM backbone: 32B from EXAONE 4.0 + Vision encoder: 1.2B) |
| Layers | 64 main + 1 MTP |
| Attention pattern | `LLLG` × 16 = 48 SWA + 16 Global |
| SWA window | 4,096 tokens (RoPE applied, sees only the most recent 4,096) |
| Global Attention | No window limit, **no RoPE (NoPE)** |
| Global layer indices | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 |
| GQA | 40 Q-heads / 8 KV-heads / head dim 128 |
| Reordered Norm (QK-Reorder-LN) | RMSNorm on Q/K inputs before attention + RMSNorm after attention output before residual (non-standard — source: EXAONE 4.0 paper) |
| Vocab | 153,600 | Context | 262,144 tokens |

**What is NoPE?** Not applying RoPE (Rotary Position Embedding). Global layers are designed to capture long-range dependencies without positional encoding because they attend to the full sequence. However, **preceding SWA layers inject positional information into representations via RoPE** — empirically confirmed in Exp 2.

---

## 4. Project file structure

```
sda212331/
│
├── CLAUDE.md                          ← Claude Code design rules (do not break)
├── HANDOVER.md                        ← this file
├── README.md                          ← minimal project description
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← excludes model_cache/, __pycache__/
│
├── analyze.py                         ← [Phase 1] weight statistics analysis entry point
│                                         streams model tensors, computes per-tensor stats
│                                         python analyze.py --model-path ./model_cache
│
├── run.sh                             ← shell script to run analyze.py
├── run_nope.sh                        ← runs Exp 1 + Exp 2 sequentially + auto git push
│
├── src/                               ← [Phase 1] weight stats modules (used by analyze.py)
│   ├── loader.py                      ← safetensors streaming loader, tensor classifier
│   │                                     classify_tensor(): categorizes tensors
│   │                                     iter_tensors(): streaming yield to prevent OOM
│   ├── stats.py                       ← tensor statistics (std, abs_mean, sparsity, effective_rank, stable_rank)
│   │                                     StatsAggregator: append-only write to stats.jsonl
│   ├── viz.py                         ← visualization (4 chart types)
│   └── report.py                      ← generates markdown analysis report
│
├── nope_analysis/                     ← [Phase 2] NoPE behavior analysis experiments
│   ├── loader.py                      ← full model load (for attention extraction)
│   │                                     ⚠️ includes CONFIG_MAPPING patch — always use this loader
│   │                                     load_model_and_tokenizer(): forces attn_implementation='eager'
│   │                                     get_global_layer_indices(), get_swa_layer_indices()
│   ├── analysis/
│   │   └── statistical_tests.py      ← statistical testing utilities
│   │                                    compare_groups(): Welch t-test + Cohen's d + bootstrap CI
│   │                                    analyze_jsonl(): analyze experiment results.jsonl directly
│   ├── corpus/
│   │   └── downloader.py             ← real text corpus downloader
│   │                                    WikiText-103 (English) / KLUE-MRC (Korean) auto-download+cache
│   │                                    build_input_from_corpus(): creates token tensors for experiments
│   └── experiments/
│       ├── exp1_attention_entropy.py  ← Exp 1: short input (14–26 tokens) entropy [DONE]
│       ├── exp1b_long_input.py        ← Exp 1b: long input (128–2048 tokens) entropy + distance [DONE]
│       ├── exp2_positional_probe.py   ← Exp 2: layer-wise positional linear probe [DONE]
│       ├── exp2b_positional_probe_v2.py ← Exp 2b: improved probe (train/test split, 30 prompts) [RUNNING]
│       ├── exp3_swa_ablation.py       ← Exp 3: zero-out Global attn output ablation [PENDING]
│       ├── exp3b_swa_mask_ablation.py ← Exp 3b: ⭐ SWA mask injection ablation (methodologically superior)
│       │                                   injects SWA window mask into Global layers via forward hook
│       │                                   → converts Global to "behave like SWA"
│       │                                   → isolates pure NoPE long-range contribution
│       └── exp4_long_context.py       ← Exp 4: Global vs SWA attention divergence at 4096+ tokens [PENDING]
│
├── outputs/                           ← all experiment results (pushed to GitHub)
│   ├── exp1_attention_entropy/        ← Exp 1 results [DONE]
│   ├── exp1b_long_input/              ← Exp 1b results [DONE]
│   ├── exp2_positional_probe/         ← Exp 2 results [DONE]
│   └── exp2b_positional_probe_v2/     ← Exp 2b results [RUNNING / pending]
│
└── model_cache/                       ← 64GB model files (git-excluded, deleted on session end)
```

---

## 5. Completed experiment results

### [Phase 1] Weight statistics analysis (`analyze.py`)
- **Purpose**: baseline statistics of internal weights (std, effective rank, sparsity)
- **Key finding**: weight std distributions of Global (NoPE) vs SWA layers differ by layer depth
- **Output**: `outputs/stats.jsonl`, `outputs/summary.json`, 4 charts

### Exp 1 — Attention Entropy (short input, 14–26 tokens) → **RQ2**
| Metric | Global (NoPE) | SWA |
|--------|--------------|-----|
| Entropy | 0.9301 | 0.9422 |
| Attn Distance | 7.18 tokens | 7.10 tokens |
- **Conclusion**: no difference — short inputs fit entirely within the SWA window (4,096), so structural differences don't emerge
- **Next**: Exp 1b (long input)

### Exp 1b — Attention Entropy (long input, 128–2048 tokens) → **RQ2** [DONE]

| Length | Global Entropy | SWA Entropy | Global Dist | SWA Dist |
|--------|---------------|-------------|-------------|----------|
| 128  | 1.7735 | 1.8121 | 33.9  | 31.7  |
| 256  | 2.0534 | 2.1636 | 70.2  | 64.7  |
| 512  | 2.4912 | 2.6152 | 143.9 | 127.1 |
| 1024 | 2.9823 | 3.0447 | 305.7 | 255.1 |
| 2048 | **3.5419** | **3.4501** | **643.7** | **489.1** |

- **Key finding 1**: at 2,048 tokens, Global entropy (3.54) > SWA entropy (3.45) — first meaningful reversal
- **Key finding 2**: attention distance gap grows sharply with length (2.2 at 128 tokens → **154.6** at 2,048)
  - Global: attends uniformly across full sequence (distance approaches seq_len/2)
  - SWA: distance growth slows as length increases (window constraint)
- **New hypothesis**: beyond 4,096 tokens, SWA distance saturates while Global continues to grow → motivates Exp 4

### Exp 2 — Positional Probing (short input) → **RQ1**
| Metric | Global (NoPE) | SWA |
|--------|--------------|-----|
| Probe Accuracy | **1.000** | **1.000** |
- **Key finding**: even NoPE layers perfectly encode positional information in hidden states
- **Interpretation**: preceding SWA layers inject positional info via RoPE → Global layers propagate it forward
- **Limitation**: trained and evaluated on same data (possible overfitting), short sentences only → **Exp 2b needed**

---

## 6. Remaining experiments (priority order)

### Exp 2b — Improved Positional Probe (Priority 1) → **RQ1** [CURRENTLY RUNNING]
- **Why needed**: Exp 2's accuracy=1.0 is unreliable — no train/test split, same data for train and eval
- **Improvements**: train/test split (80/20), 30 prompts, lengths 64/128/256, domain breakdown (Korean/English/math/code)
- **Goal**: reliable answer to "do NoPE layers encode less positional info than SWA layers?"

### Exp 3b — SWA Mask Injection Ablation ⭐ (Priority 2, methodologically superior) → **RQ3**
- Injects SWA window mask into Global layers via `register_forward_pre_hook`
- Converts Global → "behaves like SWA" → isolates pure NoPE long-range attention effect
- Uses real text (WikiText-103 / KLUE-MRC) to avoid repetition bias
- Includes automatic statistical tests (t-test, Cohen's d)

### Exp 3 — Global Zero-Output Ablation (Priority 3) → **RQ3**
- Replaces Global layer self_attn output with zeros → removes all attention contribution
- **Note**: methodologically aggressive (removes attention entirely, not SWA-like conversion)
- **Value**: comparing Exp 3b vs Exp 3 results reveals the difference between the two ablation strategies

### Exp 4 — Long Context Hook Analysis (Priority 4) → **RQ3**
- Collects attention stats via hooks for 2,048–8,192 tokens (avoids OOM)
- Confirms whether SWA distance saturates above 4,096 while Global keeps growing
- Reproduces "Lost in the Middle" effect on EXAONE 4.5

---

## 7. How to run

```bash
cd /home/elicer/sda212331

# Check model cache (after session restart)
ls model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/

# Phase 1 weight analysis (already done)
python analyze.py --model-path ./model_cache

# NoPE experiments (run individually)
python3 nope_analysis/experiments/exp1_attention_entropy.py
python3 nope_analysis/experiments/exp1b_long_input.py
python3 nope_analysis/experiments/exp2_positional_probe.py
python3 nope_analysis/experiments/exp2b_positional_probe_v2.py
python3 nope_analysis/experiments/exp3b_swa_mask_ablation.py   # ⭐ run before exp3
python3 nope_analysis/experiments/exp3_swa_ablation.py
python3 nope_analysis/experiments/exp4_long_context.py

# Pre-download corpus (run once before Exp 3b / Exp 4)
python3 -c "from nope_analysis.corpus.downloader import download_all; download_all()"

# Statistical tests (after experiment completes)
python3 -c "
from nope_analysis.analysis.statistical_tests import analyze_jsonl, save_stats_report
from pathlib import Path
results = analyze_jsonl(Path('outputs/exp4_long_context/results.jsonl'))
save_stats_report(results, Path('outputs/exp4_long_context/stats.json'))
"

# Push results to GitHub
git add outputs/ nope_analysis/ HANDOVER.md
git commit -m "exp_name: result summary"
git push
```

### Model loading — do not break this rule
```python
# ❌ forbidden
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)  # fails: config_type mismatch

# ✅ correct
from nope_analysis.loader import load_model_and_tokenizer  # includes CONFIG patch
model, tokenizer = load_model_and_tokenizer()
# internally uses Exaone4_5_ForConditionalGeneration + attn_implementation='eager'
```

---

## 7b. LG EXAONE official papers (`papers/`)

6 papers by LG AI Research are stored in `papers/`.
Each folder: `{arxiv_id}/paper.pdf` + `fulltext.txt` + `images/` + `metadata.json`

| arXiv ID | Title | Year | Relevance |
|----------|-------|------|-----------|
| 2604.08644 | **EXAONE 4.5 Technical Report** ⭐ | 2026 | the model we analyze — NoPE, SWA, 262K context details |
| 2507.11407 | EXAONE 4.0 Technical Report | 2025 | base LM of EXAONE 4.5, Reasoning mode design (Topic C background) |
| 2601.01739 | K-EXAONE Technical Report | 2026 | MoE architecture, Korean specialization (Topic D background) |
| 2503.12524 | EXAONE Deep: Reasoning Enhanced | 2025 | reasoning-enhanced model, directly relevant to Topics C and I |
| 2412.04862 | EXAONE 3.5 Technical Report | 2024 | long-context design baseline (Exp 3/4 background) |
| 2408.03541 | EXAONE 3.0 7.8B | 2024 | early architecture design philosophy |

```python
# Use paper text as experiment corpus
text = Path('papers/2604.08644/fulltext.txt').read_text(encoding='utf-8')
```

---

## 8. Design rules (do not break)

1. Always load model from `model_cache/` (specify HuggingFace cache path directly)
2. Save experiment results to `outputs/{exp_name}/` then push to git immediately
3. `attn_implementation='eager'` is required — flash_attn does not return attention weights
4. Vision encoder analysis must be reported separately from LM analysis
5. Every experiment saves: `summary.json` + `results.jsonl` + charts
6. `stats.jsonl` is append-only — never overwrite (reproducibility)
7. SVD can be skipped with `--no-spectral` — not required

---

## 9. Comprehension check for Claude Code

**Answer these before starting work in a new session.**

1. What is NoPE, and why is it applied only to Global Attention layers in EXAONE 4.5?

2. What does the `LLLG` pattern mean? List all Global layer indices.

3. Create a table mapping RQ1/RQ2/RQ3 to their corresponding experiments.

4. What does accuracy=1.000 in Exp 2 mean, and why is it unreliable?

5. What is the difference between Exp 1 and Exp 1b? Why does short input fail to reveal differences between SWA and Global?

6. Why is `AutoModelForCausalLM` forbidden here? What is the correct loading method?

7. What is the difference between `src/loader.py` and `nope_analysis/loader.py`?

8. Explain the exploratory nature of this research — why is the experiment plan not fixed?

---

*Last updated: 2026-04-25*
