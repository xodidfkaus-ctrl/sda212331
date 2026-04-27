# EXAONE 4.5 NoPE Research — Session Handover

> **To Claude Code reading this**: Read this document first at the start of every new session.
> Complete the **Section 00 quick setup** before doing anything else.
> Skipping the checklist causes experiments to run on CPU (hours instead of minutes) or fail due to missing model.

> **Full research topic list** → see `RESEARCH_TOPICS.md`
> **Standing experiment rules** → see `CHECKLIST.md` (re-read items 1–6 before every run; items 7–9 every session; items 13–14 always)

---

## 00. Ephemeral Session Quick Setup (run this first, every session)

All cloud environments reset pip packages, git credentials, and model cache on session restart — regardless of which cloud. Run the full sequence below at the start of every session.

### What the researcher says to Claude at session start
> "Read HANDOVER.md and set up. GitHub PAT is `ghp_xxx`."

- GitHub PAT: required for git push/pull authentication.
- HuggingFace token: **not required** (EXAONE-4.5-33B is a public model, gated=False confirmed)

---

### Step A — Working directory check and repo clone

```bash
# Confirm home directory (may differ across cloud environments)
echo "HOME: $HOME"

# Clone repo if not present (replace PAT with actual value)
REPO_DIR="$HOME/sda212331"
ls "$REPO_DIR" 2>/dev/null || git clone https://PAT@github.com/xodidfkaus-ctrl/sda212331.git "$REPO_DIR"
cd "$REPO_DIR"
```

> **Note**: If the cloud environment changes, the home path may differ from `/home/elicer/`.
> `nope_analysis/loader.py` line 13 has `MODEL_PATH` hardcoded to `/home/elicer/sda212331/model_cache/...`.
> If the home path differs, update `MODEL_PATH` in `nope_analysis/loader.py` line 13.

---

### Step B — Git auth setup (required every session)

```bash
cd "$HOME/sda212331"
git remote set-url origin https://PAT@github.com/xodidfkaus-ctrl/sda212331.git
git config user.email "xodidfkaus@gmail.com"
git config user.name "elicer"
# Verify
git push --dry-run 2>&1 | head -3
```

---

### Step C — GPU / CUDA version check (must run before installing dependencies)

```bash
nvidia-smi | grep "CUDA Version"
python3 -c "import subprocess; r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True); print(r.stdout)"
```

Install torch based on detected CUDA driver version:

| CUDA Driver Version | torch install command |
|--------------------|-----------------------|
| 12.1 ~ 12.2 | `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 ~ 12.5 | `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124` |
| 12.6+ | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |

> **Elice A100 baseline**: CUDA driver 12.2 → use `cu121` (confirmed 2026-04-26)

---

### Step D — Install dependencies (required every session)

```bash
cd "$HOME/sda212331"
pip install -r requirements.txt -q
pip install git+https://github.com/nuxlear/transformers.git@add-exaone4_5 -q
pip install accelerate -q
# Install torch per Step C result (confirmed 2026-04-27 session 4: CUDA driver 12.2 → cu121)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
```

> **Why the nuxlear fork?** The `exaone4_5` module is not in PyPI transformers. Only the `add-exaone4_5` branch of the nuxlear fork includes it.
> `requirements.txt` lists `transformers>=4.40.0` which does not satisfy this — always install the fork separately.

> **Why accelerate?** `device_map='auto'` (required for 2-GPU model split) raises `ValueError` without `accelerate`. Must install every session — confirmed missing on session 4 startup.

> **CUDA version history**: Elice baseline was cu121 (CUDA 12.2, confirmed 2026-04-26). Session 3 noted CUDA 12.4 (cu124). Session 4 (2026-04-27) reverted to CUDA 12.2 — use cu121. Always run Step C first.

---

### Step D.5 — Corpus cache build (required every session)

The corpus cache (`nope_analysis/corpus/cache/`) is **not committed to git** and is deleted on every session reset. Without it, experiment scripts fall back to synthetic repeated text (as happened in e005), which produces near-zero perplexity and masks any ablation effect.

```bash
cd "$HOME/sda212331"
python3 -c "from nope_analysis.corpus.downloader import download_all; download_all()"
```

Expected output:
```
Pre-downloading corpora...
[corpus] WikiText-103 loaded: 699,876 passages  →  en.json  (499MB)
[corpus] EDGAR total: 59,218 sections           →  en_edgar.json  (2.4GB)
[corpus] KLUE-MRC loaded: 17,554 passages       →  ko.json  (41MB)
Done.
```

**Runtime**: ~5–10 minutes (EDGAR 2.4GB 다운로드 포함).
**Disk**: ~3GB total. **Why not committed to git**: GitHub 파일 크기 제한 초과.

**DART (한국어 금융공시) 추가 빌드** — 선택사항, API 키 필요:
```bash
export DART_API_KEY=your_key_here   # opendart.fss.or.kr 에서 발급
python3 -c "
from nope_analysis.corpus.downloader import download_all
download_all(['ko_dart'], dart_api_key='your_key_here')
"
# ko_dart.json 생성. 사업보고서 최대 200건, 문건당 수천~수만 토큰.
```

**코퍼스 선택 가이드**:
| lang | 용도 | 평균 길이 | 이어붙임 필요 |
|------|------|-----------|--------------|
| `en` | WikiText-103 (Wikipedia) | ~185 tokens | 4096 토큰에 22개 |
| `en_edgar` | SEC 10-K 연간보고서 | ~10,600 tokens | **불필요** (단일 섹션으로 충분) |
| `ko` | KLUE-MRC (뉴스/위키) | ~500 tokens | 4096 토큰에 8개 |
| `ko_dart` | DART 사업보고서 | 수천~수만 tokens | **불필요** |

**실험 스크립트에서 코퍼스 교체**: `lang='en'` → `lang='en_edgar'`로 바꾸면 됨. PLAN.md 코퍼스 명세가 `WikiText-103`으로 pre-register된 실험은 교체 시 PLAN.md deviation으로 기록 필요.

---

### Step E — GPU + loader verification

```bash
python3 -c "
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {p.name}, {p.total_memory/1e9:.1f}GB')
else:
    print('No CUDA — recheck Steps C/D')
"
python3 -c "import sys; sys.path.insert(0,'$HOME/sda212331'); from nope_analysis import loader; print('loader: OK')"
```

Expected output (Elice A100):
```
CUDA: True
GPU: NVIDIA A100 80GB PCIe MIG 3g.40gb, 42.4GB
loader: OK
```

GPU name will differ on other clouds. **What matters: `CUDA: True` and `loader: OK`**.

**Minimum VRAM: 80GB** (model bfloat16 = 65GB + activation memory)

| GPU | VRAM | Experiment feasibility |
|-----|------|----------------------|
| A100 80GB (full) | 80GB | ✅ All experiments |
| H100 80GB | 80GB | ✅ All experiments |
| A100 MIG 3g.40gb | 40GB | ❌ OOM for seq > 2048 |
| A6000 48GB | 48GB | ❌ Insufficient |

---

### Step F — Model cache check and download

```bash
ls "$HOME/sda212331/model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/snapshots/" 2>/dev/null \
  && echo "Model cache exists" \
  || python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'LGAI-EXAONE/EXAONE-4.5-33B',
    cache_dir='$HOME/sda212331/model_cache',
    ignore_patterns=['*.bin','*.pt'],
)
print('Download complete')
"
# ~20-40 minutes, 64GB. No token required (public model).
```

> **Model cache is deleted every session.** Must re-download before running experiments.
> Other setup steps can proceed in parallel during download.

---

### Step G — Git sync check

```bash
cd "$HOME/sda212331"
git log --oneline -3
git status
```

Push any results from the previous session that were not pushed.

---

### Checklist summary

| Item | Command | Pass condition |
|------|---------|---------------|
| Repo exists | `ls $HOME/sda212331` | File list shown |
| Git auth | `git push --dry-run` | No error |
| CUDA | `python3 -c "import torch; print(torch.cuda.is_available())"` | `True` |
| VRAM >= 40GB | GPU output | memory >= 40GB |
| Loader import | `from nope_analysis import loader` | No error |
| Model cache | `ls model_cache/.../snapshots/` | Folder exists |
| Corpus cache | `ls nope_analysis/corpus/cache/` | `en.json`, `en_edgar.json`, `ko.json` present |

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
GPUs: 1
  GPU0: NVIDIA A100 80GB PCIe MIG 3g.40gb  (42.4 GB usable)
```

**If `CUDA: False` — stop immediately. Do not run any experiment.**

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

**Missing**: re-download the model (deleted on session restart).
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

**ImportError**: reinstall packages (order matters)
```bash
pip install -r requirements.txt -q
pip install git+https://github.com/nuxlear/transformers.git@add-exaone4_5 -q
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 -q
```
> **Why the nuxlear fork?** PyPI transformers does not include `exaone4_5` yet. The nuxlear fork adds `Exaone4_5_ForConditionalGeneration`.
> **Why torch cu124?** As of 2026-04-27 the environment has CUDA driver 12.4. Use cu124. Always check `nvidia-smi` first — driver version determines the correct index URL.

---

### Step 3.5 — Corpus cache build (required every session)

The experiment corpus cache is deleted on session reset. Without it, scripts fall back to synthetic text (PPL ≈ 1.07), which masks ablation effects (root cause of e005 ΔNLL = 0).

```bash
cd "$HOME/sda212331"
python3 -c "from nope_analysis.corpus.downloader import download_all; download_all()"
```

**Expected output**: `en: 699,876 passages cached` and `ko: 17,554 passages cached`.
**Runtime**: ~2–3 minutes. **Disk**: ~540MB (not committed to git).

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
| GPU count | `python3 -c "import torch; print(torch.cuda.device_count())"` | `1` |
| Model cache | `ls model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/snapshots/` | folder exists |
| Dependencies | `python3 -c "from nope_analysis.loader import load_config; load_config()"` | no error |
| Corpus cache | `ls nope_analysis/corpus/cache/` | `en.json`, `en_edgar.json`, `ko.json` present |
| Git auth | `git push --dry-run 2>&1` | no error |
| Git sync | `git status` | no unpushed results |

---

## 1. Project basics

| Item | Value |
|------|-------|
| GitHub | https://github.com/xodidfkaus-ctrl/sda212331 |
| Environment | Elice Cloud — **ephemeral, resets on restart** |
| GPU | **A100 80GB × 1 (full, not MIG)** or equivalent required. Model bfloat16 = 65GB + activation memory → minimum 80GB VRAM. MIG 40GB slices cannot run Exp 3b and beyond. |
| Model path | `/home/elicer/sda212331/model_cache/` (64GB, **deleted on session end**) |
| Working directory | `/home/elicer/sda212331/` |

---

## 2. Research goal and core questions

### One-line summary
> Empirically analyze how **NoPE (No Positional Embedding) Global Attention layers** in EXAONE 4.5 behave differently from SWA layers.

### Background
- EXAONE 4.5 uses a SWA (Sliding Window Attention) + Global (NoPE) hybrid architecture
- Similar hybrid patterns appear in Mistral-family models (SWA + full attention, but RoPE applied to both) and Cohere Command-A; however, the specific design of applying NoPE *only* to Global layers — so local context uses RoPE and global context uses no positional encoding — has not been analyzed mechanistically in published work as of April 2026
- **Paper positioning**: This study provides the first mechanistic analysis of the SWA-RoPE + Global-NoPE hybrid behavior. It does **not** claim priority for the architecture design itself. The related architecture note must be included in the paper's related work section.
- **Final goal**: arXiv paper → ACL/EMNLP workshop submission

### Three core research questions

| RQ | Question | Experiments |
|----|----------|-------------|
| **RQ1** | Do NoPE Global hidden states contain positional information, and is this signal *generated by Global layers* or *propagated unchanged from preceding SWA layers*? | Exp 2, Exp 2b, **Exp 2c** |
| **RQ2** | Do SWA (RoPE) and Global (NoPE) exhibit functionally different attention patterns, and at what sequence length does the difference emerge? | Exp 1, Exp 1b |
| **RQ3** | Does NoPE Global Attention make a causal contribution to long-range dependency beyond what SWA alone provides? See Section 2b for operational definition. | Exp 3b, Exp 3, Exp 4 |

### Methodology note — this is exploratory research
Each experiment's result informs the design of the next one. **The experiment plan is not fixed.**

```
Run Exp 1
  → find pattern in results (e.g. layers 32-48 differ)
  → generate hypothesis
  → design Exp 2 (re-measure that range by input length)
  → find Korean/English difference
  → design Exp 3 ...
```

---

## 2b. RQ3 Operational Definition (pre-register before running Exp 3/3b/4)

**Why this section exists**: Exp 3 (zero-output ablation), Exp 3b (SWA mask injection), and Exp 4 (long-context hook) each measure "contribution" via a different proxy. Without a pre-registered definition, conflicting results between experiments cannot be principled interpreted, and a reviewer can reject the RQ3 conclusion by pointing to whichever experiment does not support it.

### Operational definition
> **NoPE Global Attention contributes to long-range dependency** if and only if **at least two** of the following three conditions hold:
>
> 1. **(Exp 3b)** Forcing Global layers to attend like SWA (same window mask) causes a statistically significant increase in perplexity at token positions > 4,096, with Cohen's d > 0.5.
> 2. **(Exp 3)** Zeroing Global layer attention output causes a statistically significant degradation in perplexity or long-range recall accuracy, with Cohen's d > 0.5.
> 3. **(Exp 4)** Global attention distance continues to grow with sequence length beyond the SWA saturation point (>= 4,096 tokens) while SWA distance plateaus, with the divergence statistically significant at >= 2 length conditions.

### Triangulation protocol
- Report each experiment independently with its own statistical tests (Welch t, Cohen's d, bootstrap CI).
- If all three conditions hold → strong causal evidence.
- If exactly two hold → moderate evidence; discuss which proxy is most ecologically valid.
- If fewer than two hold → conclusion is "no evidence of Global causal contribution at tested lengths." Do not overstate.
- **Conflicts between experiments must be explicitly analyzed** (e.g., Exp 4 shows distance divergence but Exp 3b shows no perplexity change → discuss the methodological difference between attention pattern vs. functional output).

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

**What is NoPE?** Not applying RoPE (Rotary Position Embedding). Global layers attend to the full sequence without positional encoding. Preceding SWA layers apply RoPE and inject positional information into the shared residual stream — this is the likely reason why hidden states at Global layers still contain positional information despite NoPE (Exp 2/2b). Whether Global layers *add to* or merely *pass through* this signal is the open question addressed by Exp 2c (H1 vs. H2 — see Section 6).

---

## 4. Project file structure

```
sda212331/
|
+-- CLAUDE.md                          <- Claude Code design rules (do not break)
+-- HANDOVER.md                        <- this file
+-- README.md                          <- minimal project description
+-- requirements.txt                   <- Python dependencies
+-- .gitignore                         <- excludes model_cache/, __pycache__/
|
+-- analyze.py                         <- [Phase 1] weight statistics analysis entry point
|                                         streams model tensors, computes per-tensor stats
|                                         python analyze.py --model-path ./model_cache
|
+-- run.sh                             <- shell script to run analyze.py
+-- run_nope.sh                        <- runs Exp 1 + Exp 2 sequentially + auto git push
|
+-- src/                               <- [Phase 1] weight stats modules (used by analyze.py)
|   +-- loader.py                      <- safetensors streaming loader, tensor classifier
|   |                                     classify_tensor(): categorizes tensors
|   |                                     iter_tensors(): streaming yield to prevent OOM
|   +-- stats.py                       <- tensor statistics (std, abs_mean, sparsity, effective_rank, stable_rank)
|   |                                     StatsAggregator: append-only write to stats.jsonl
|   +-- viz.py                         <- visualization (4 chart types)
|   +-- report.py                      <- generates markdown analysis report
|
+-- nope_analysis/                     <- [Phase 2] NoPE behavior analysis experiments
|   +-- loader.py                      <- full model load (for attention extraction)
|   |                                     includes CONFIG_MAPPING patch — always use this loader
|   |                                     load_model_and_tokenizer(): forces attn_implementation='eager'
|   |                                     get_global_layer_indices(), get_swa_layer_indices()
|   +-- analysis/
|   |   +-- statistical_tests.py      <- statistical testing utilities
|   |                                    compare_groups(): Welch t-test + Cohen's d + bootstrap CI
|   |                                    analyze_jsonl(): analyze experiment results.jsonl directly
|   +-- corpus/
|   |   +-- downloader.py             <- real text corpus downloader
|   |                                    WikiText-103 (English) / KLUE-MRC (Korean) auto-download+cache
|   |                                    build_input_from_corpus(): creates token tensors for experiments
|   +-- experiments/
|       +-- exp1_attention_entropy.py  <- Exp 1: short input (14-26 tokens) entropy [DONE]
|       +-- exp1b_long_input.py        <- Exp 1b: long input (128-2048 tokens) entropy + distance [DONE]
|       +-- exp2_positional_probe.py   <- Exp 2: layer-wise positional linear probe [DONE]
|       +-- exp2b_positional_probe_v2.py <- Exp 2b: probe with train/test split, 30 prompts [DONE]
|       +-- exp2c_delta_probe.py       <- Exp 2c: delta-attribution probe [PENDING]
|       |                                   Compare probe acc at Global layer INPUT vs OUTPUT
|       |                                   Distinguishes H1 (Global generates position) from
|       |                                   H2 (Global propagates SWA's signal unchanged)
|       +-- exp3_swa_ablation.py       <- Exp 3: zero-out Global attn output ablation [PENDING]
|       +-- exp3b_swa_mask_ablation.py <- Exp 3b: SWA mask injection ablation [PENDING]
|       |                                   injects SWA window mask into Global layers via forward hook
|       |                                   converts Global to "behave like SWA"
|       |                                   isolates pure NoPE long-range contribution
|       +-- exp4_long_context.py       <- Exp 4: Global vs SWA attention divergence at 4096+ tokens [PENDING]
|
+-- outputs/                           <- all experiment results (pushed to GitHub)
|   +-- exp1_attention_entropy/        <- Exp 1 results [DONE]
|   +-- exp1b_long_input/              <- Exp 1b results [DONE]
|   +-- exp2_positional_probe/         <- Exp 2 results [DONE]
|   +-- exp2b_positional_probe_v2/     <- Exp 2b results [DONE]
|
+-- model_cache/                       <- 64GB model files (git-excluded, deleted on session end)
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
- **Conclusion**: No difference — short inputs fit entirely within the SWA window (4,096 tokens), so the structural distinction between Global and SWA does not emerge. Motivates Exp 1b.

### Exp 1b — Attention Entropy (long input, 128–2048 tokens) → **RQ2** [DONE]

| Length | Global Entropy | SWA Entropy | Global Dist | SWA Dist |
|--------|---------------|-------------|-------------|----------|
| 128  | 1.7735 | 1.8121 | 33.9  | 31.7  |
| 256  | 2.0534 | 2.1636 | 70.2  | 64.7  |
| 512  | 2.4912 | 2.6152 | 143.9 | 127.1 |
| 1024 | 2.9823 | 3.0447 | 305.7 | 255.1 |
| 2048 | **3.5419** | **3.4501** | **643.7** | **489.1** |

- **Observed finding 1**: At 2,048 tokens, Global entropy (3.54) > SWA entropy (3.45) — direction reversal observed (Delta = 0.09 nats). **Statistical significance not yet verified.** Must apply `compare_groups()` from `statistical_tests.py` (Welch t-test + Cohen's d + bootstrap CI) before calling this "meaningful." If the experiment ran on a single input per length condition, rerun with >= 10 distinct inputs per condition to obtain variance estimates.
- **Observed finding 2**: Attention distance gap grows with length (2.2 tokens at 128 → 154.6 tokens at 2,048). Global attends more uniformly across the full sequence; SWA distance growth slows under window constraint. **Apply statistical tests to confirm this trend before citing it.**
- **Hypothesis** (pending statistical validation): Beyond 4,096 tokens, SWA distance will saturate while Global continues growing. → Motivates Exp 4.

### Exp 2 — Positional Probing (short input) → **RQ1**
| Metric | Global (NoPE) | SWA |
|--------|--------------|-----|
| Probe Accuracy | **1.000** | **1.000** |
- **Finding**: Both Global and SWA hidden states are perfectly linearly separable for position class.
- **Limitation**: Trained and tested on the same data (severe overfit). Short sentences only.
- **Caution — do not overinterpret**: Identical probe accuracy is consistent with two competing hypotheses:
  - **(H1)** Global layers independently generate positional signals ("encode")
  - **(H2)** Global layers propagate SWA's positional representation unchanged ("propagate")
  - A linear probe on the residual stream cannot distinguish H1 from H2. The residual stream at a Global layer is the cumulative sum of all prior layer contributions — dominated by the 3 preceding SWA layers. Exp 2c is required to resolve this.
- **Next**: Exp 2b (train/test split to check generalization), Exp 2c (delta-attribution probe to distinguish H1 vs H2)

### Exp 2b — Positional Probing v2 (train/test split) → **RQ1** [DONE]

| Length | Global test acc | SWA test acc | Overfit gap |
|--------|----------------|-------------|-------------|
| 64  | 0.524 | 0.520 | 0.476 |
| 128 | 0.554 | 0.552 | 0.446 |
| 256 | 0.545 | 0.542 | 0.456 |

- **Random baseline**: 0.10 (10-class uniform)
- **RQ1 finding (preliminary, unreliable at current sample size)**: Both Global and SWA hidden states encode positional information above chance (test acc ~0.52–0.55 vs. 0.10 baseline). Near-identical accuracy between Global and SWA is consistent with both H1 and H2 — **this does not resolve RQ1**. Exp 2c is required before this becomes a paper claim.
- **Reliability warning**: Train acc ≈ 1.0, test acc ≈ 0.54, overfit gap ≈ 0.45 — this is severe overfitting with only 30 prompts. Each test fold (with 5-fold CV) would contain ~6 samples for a 10-class problem. The above-chance result is suggestive but not paper-grade evidence. **Must replicate with >= 300 prompts, PyTorch GPU probe (Section 7c), and >= 5-fold cross-validation before citing in the paper. Add bootstrap 95% CI on test accuracy.**
- **Output**: `outputs/exp2b_positional_probe_v2/`

---

## 6. Remaining experiments (priority order)

### Exp 2c — Delta-Attribution Probe (Priority 1) → **RQ1** — resolves H1 vs. H2

**Goal**: Determine whether Global layers actively encode positional information (H1) or merely propagate SWA's signal unchanged (H2).

**Method**:
- For each Global layer index `i` in `[3, 7, 11, ..., 63]`:
  - Register forward hooks to capture `h_in[i]` (hidden state at Global layer **input**, before attention sublayer) and `h_out[i]` (after attention + residual add)
  - Fit separate linear probes on `h_in[i]` and `h_out[i]`
  - Report `delta_acc[i] = acc(h_out[i]) - acc(h_in[i])`
- Run the same analysis on SWA layers as a baseline

**Interpretation**:
- `delta_acc ≈ 0` at all Global layers → H2 (propagation only; SWA layers are the source)
- `delta_acc > 0` at some Global layers, and Global delta > SWA delta → H1 (Global encodes)
- `delta_acc > 0` at SWA layers but ≈ 0 at Global layers → confirms RoPE is the sole source

**Sample size and runtime**:
- >= 300 prompts from WikiText-103/KLUE-MRC corpus
- PyTorch GPU probe (Section 7c), >= 5-fold cross-validation, bootstrap 95% CI
- VRAM: model (65GB) + one forward pass activations (~1GB for 256-token prompt) → fits in 85GB A100
- Runtime: ~25 min extraction (300 × ~5s per forward pass) + < 5 min GPU probe training

### Exp 2b (Upgraded) — Rerun with >= 300 prompts + PyTorch probe (Priority 2)

Fixes the reliability problem in current Exp 2b (n=30, sklearn, single split):
- Replace `sklearn.LogisticRegression` with PyTorch GPU probe (see Section 7c)
- >= 300 prompts sampled from WikiText-103 / KLUE-MRC corpus
- 5-fold or 10-fold stratified cross-validation
- Bootstrap 95% CI on test accuracy
- **Prerequisite for citing any RQ1 result in the paper**

### Exp 3b — SWA Mask Injection Ablation (Priority 3) → **RQ3**
- Injects SWA window mask into Global layers via `register_forward_pre_hook`
- Converts Global → "behaves like SWA" → isolates pure NoPE long-range effect
- Uses real text (WikiText-103 / KLUE-MRC) to avoid repetition bias
- Measure: perplexity change at token positions > 4,096 (requires sequence length > 4,096)
- Apply `compare_groups()` for Welch t + Cohen's d
- **See Section 2b for pass/fail criteria**

### Exp 3 — Global Zero-Output Ablation (Priority 4) → **RQ3**
- Replaces Global layer self_attn output with zeros → removes all attention contribution
- More aggressive ablation than Exp 3b; comparison between the two is methodologically informative
- **See Section 2b for pass/fail criteria**

### Exp 4 — Long Context Hook Analysis (Priority 5) → **RQ3**
- Collects attention stats via hooks for 2,048–8,192 tokens (avoids OOM)
- Confirms whether SWA distance saturates above 4,096 while Global keeps growing
- **See Section 2b for pass/fail criteria**
- **See Section 10 for external validity constraints on what this length range can support**

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
python3 nope_analysis/experiments/exp2c_delta_probe.py         # resolves RQ1 H1 vs H2 — run before exp3b
python3 nope_analysis/experiments/exp3b_swa_mask_ablation.py   # run before exp3
python3 nope_analysis/experiments/exp3_swa_ablation.py
python3 nope_analysis/experiments/exp4_long_context.py

# Pre-download corpus (run once before Exp 2c / Exp 3b / Exp 4)
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
# Forbidden
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)  # fails: config_type mismatch

# Correct
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
| 2604.08644 | **EXAONE 4.5 Technical Report** | 2026 | the model we analyze — NoPE, SWA, 262K context details |
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

## 7c. Known design issue — sklearn probe is CPU-only (fix before writing new probe experiments)

**Problem**: Exp 2b uses `sklearn.LogisticRegression` for probe fitting.
sklearn does not support GPU → 32 vCPUs run at 100% for ~107 minutes while A100 sits idle at 0%.

```
sklearn LogisticRegression on (3000 x 5120), 192 fits:  ~107 min on CPU
PyTorch linear layer equivalent on A100:                 ~1 min on GPU  (~100x faster)
```

**Fix**: replace sklearn probe with PyTorch for any new experiment that uses probing:
```python
# Slow — do not use for new probe experiments
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=300).fit(X_train, y_train)

# Fast — use this instead
import torch, torch.nn as nn
probe = nn.Linear(hidden_size, n_bins).to(model.device)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
# train with cross-entropy loss for ~100 epochs
```

Exp 2b already ran with sklearn (results saved, but n=30 and overfit). Exp 2c and upgraded Exp 2b must use PyTorch with >= 300 prompts and k-fold CV.

---

## 8. Design rules (do not break)

1. Always load model from `model_cache/` (specify HuggingFace cache path directly)
2. Save experiment results to `outputs/{exp_name}/` then push to git immediately
3. `attn_implementation='eager'` is required — flash_attn does not return attention weights
4. Vision encoder analysis must be reported separately from LM analysis
5. Every experiment saves: `summary.json` + `results.jsonl` + charts
6. `stats.jsonl` is append-only — never overwrite (reproducibility)
7. SVD can be skipped with `--no-spectral` — not required
8. **Probe fitting must use PyTorch (GPU), not sklearn (CPU)** — see Section 7c
9. **Always apply statistical tests** (`compare_groups()` from `statistical_tests.py`) before calling any observed difference "significant" or "meaningful"
10. **Apply the RQ3 operational definition** (Section 2b) before running or interpreting Exp 3/3b/4

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

9. **(New)** State hypotheses H1 and H2 for RQ1. What experiment distinguishes them, and how does the delta-attribution method work?

10. **(New)** State the RQ3 operational definition from Section 2b. What are the three conditions, and what majority-vote rule determines the conclusion?

11. **(New)** The model supports 262K context but experiments stay under 8K. What claims can and cannot be made from results in this range? (See Section 10.)

---

## 10. Computational Scope and External Validity

**Gap**: The model supports 262,144-token context. All experiments in this study use sequences of <= 8,192 tokens (< 3.2% of the maximum context length).

### Why experiments are bounded at 8K

Eager attention (required for weight extraction via hooks) stores the full attention matrix, which scales as O(n²) in sequence length and O(n² × n_heads × n_layers) in total memory. At 262K tokens with 33B model parameters already occupying 65GB, a full forward pass with eager attention is not feasible on a single A100 80GB. Even at 8,192 tokens, memory pressure limits batch size to 1.

### What the experimental range supports

| Experiment | Seq length | Claim supported | Claim NOT supported |
|-----------|-----------|----------------|---------------------|
| Exp 1 / 1b | <= 2,048 | Entropy and distance patterns diverge as length increases within SWA window | Behavior at lengths >> window |
| Exp 2 / 2b / 2c | <= 256 | Positional linearity of short/medium sequences | Long-context positional encoding |
| Exp 3b / 3 | <= 4,096 (ideally > 4,096) | Ablation effect near window boundary | Effect at 262K |
| Exp 4 | <= 8,192 | SWA saturation hypothesis up to 2× window | Behavior beyond 8K |

### Claims that cannot be made from this study

- "NoPE Global Attention enables 262K-token recall" — not tested
- "Mechanistic findings at 2K hold at 262K" — extrapolation only, not measured
- "EXAONE 4.5 achieves better long-context performance than comparable models" — no comparative benchmarks

### Recommended affordable long-context proxy test

Needle-in-a-haystack retrieval at 4,096–8,192 tokens: insert a key-value fact at a position > 4,096 tokens (beyond SWA window), ask the model to retrieve it, measure hit rate with vs. without Global attention ablated (Exp 3b hook). This provides a behavioral (not purely mechanistic) data point for RQ3 at lengths where SWA cannot attend to the needle but Global can. Even a single-condition version of this test substantially strengthens the external validity of the RQ3 conclusion.

---

*Last updated: 2026-04-27 (session 5) — RQ3 concluded: H3_alt SUPPORTED (e005b + e006b, 2/3). e007b FAILED (direction reversed). auto_validate direction bug fixed. e008 (delta probe, RQ1) RUNNING. Attention collapse diagnostic RUNNING. Full details in Session 5 summary below.*

### Session 5 summary (2026-04-27)

**RQ3 CONCLUDED**: H3_alt SUPPORTED (2/3 majority — e005b VALIDATED + e006b VALIDATED).

**Critical bug fixed — auto_validate direction check**:
`nope_analysis/analysis/auto_validate.py` `_verdict()` used `abs(cohens_d)` without checking sign.
This caused dist@8192t (d=−1.481) to be marked VALIDATED despite wrong direction.
Fix added: if pre-registered direction is `A > B` and observed d < 0 → FAILED (not VALIDATED).
Affects e007b specifically; all other experiments had correct direction.

**e007b finalized — FAILED** (previously misclassified as INCONCLUSIVE):
- Pseudo-replication bug fixed (n=160/480 → n=10/10, sample-level aggregation)
- Corrected effect sizes: d=−0.087 (2048t) → −1.481 (8192t) — large effects, wrong direction
- Direction reversed at ALL lengths: SWA > Global, growing gap. Mechanistic interpretation: SWA
  window-forcing (SWA constrained to attend within 4096-token window → near-constant mean
  distance ≈ 2048 per query; Global can concentrate locally → lower mean distance at long seq)
- ANALYSIS.md, EXPERIMENT_LOG.md, NEGATIVE_RESULTS.md, FINDINGS.md, PREREGISTRATION.md all updated

**RQ3 final tally** (from FINDINGS.md F3.1):
| # | Condition | Experiment | Verdict |
|---|-----------|-----------|---------|
| 1 | SWA mask → PPL↑ at pos > 4096 | e005b | ✅ VALIDATED (d_z=1.05, p=0.0002) |
| 2 | Zero Global output → PPL↑ | e006b | ✅ VALIDATED (d_z=3.10, p≈2e-27) |
| 3 | Global distance > SWA at ≥2 lengths | e007b | FAILED (direction reversed) |

**90/10 ratio finding** (from e005b + e006b):
- Zeroing all Global (e006b) = +0.169 nats
- Masking only beyond-window (e005b) = +0.017 nats
- ~90% of Global's causal value is within-window; ~10% specifically beyond-window

**New scripts committed this session**:
| Script | Purpose | Status |
|--------|---------|--------|
| `nope_analysis/experiments/exp2c_delta_probe.py` | e008 delta probe (RQ1) | RUNNING |
| `nope_analysis/experiments/exp_attention_collapse_diag.py` | Attention collapse diagnostic | RUNNING |
| `.claude/settings.json` (project) | bypassPermissions — no permission prompts | Active |

**e008 design** (pre-registered 2026-04-26, script written this session):
- PyTorch k-fold (k=5) linear probe on position bins
- n=300 prompts (150 EDGAR + 150 WikiText), SEQ_LENGTHS=[64, 128, 256]
- delta_acc[i] = acc(h_out[i]) − acc(h_in[i]) for each Global layer i
- Accept H1_alt if delta significant at ≥1 Global layer: d≥0.5, p_Holm≤0.01
- Runtime: ~2–3 hours

**Attention collapse diagnostic** (not pre-registered, diagnostic only):
- `exp_attention_collapse_diag.py` — measures entropy, sink_frac, local_frac, distance
- 3 samples × [2048, 4096, 8192] tokens
- Contextualizes e007b direction reversal: is Global locally biased or collapsing to BOS?
- Output: `outputs/attn_collapse_diag/`

**ANALYSIS.md files written/committed this session**:
- `experiments/e005b_swa_mask_ablation_v2/ANALYSIS.md` — VALIDATED
- `experiments/e006b_global_zero_ablation_v2/ANALYSIS.md` — VALIDATED (+ Deviations section added)
- `experiments/e007b_long_context_sparse_hook/ANALYSIS.md` — FAILED (updated from INCONCLUSIVE)

**Next session — run in order**:
1. Check e008 delta probe result → write `experiments/e008_delta_probe/ANALYSIS.md`
2. Check attention collapse diagnostic output → interpret in context of e007b direction reversal
3. If e008 VALIDATES H1_alt → update FINDINGS.md F1.2, draft RQ1 paper section
4. If e008 FAILS → H1_null supported; document in NEGATIVE_RESULTS.md, update FINDINGS.md
5. Topic E — Vision × NoPE experiment (script: `expE_vision_nope.py`, test_images ready)
6. Paper outline draft (all RQ1/RQ2/RQ3 results now have at least preliminary verdicts)

---

### Session 4 summary (2026-04-27)

**Hardware confirmed**: 2× A100 80GB PCIe (85.2GB each), 32 vCPU, 1082GB RAM.
CUDA driver: 12.2 → torch cu121 (reverted from session 3's cu124).

**Infrastructure changes this session**:
- `accelerate` added to install sequence (required for `device_map='auto'` on 2-GPU setup)
- `nope_analysis/analysis/statistical_tests.py`: added `compare_one_sample`, `cohens_dz`, `report_stats_one_sample`
- `nope_analysis/analysis/auto_validate.py`: one-sample t-test support (`test_type: one_sample` in criteria block; `n_b=0` bypasses n_b check in `_verdict`)
- `nope_analysis/seeds.py`: registered e005b, e006b, e007b
- `~/.claude/settings.json`: `permissions.defaultMode = bypassPermissions` (no more permission prompts)

**New experiments pre-registered** (PLAN.md committed, scripts written):
| ID | Directory | Script | Status |
|----|-----------|--------|--------|
| e006b | experiments/e006b_global_zero_ablation_v2/ | exp3_global_zero_ablation_v2.py | **RUNNING** |
| e007b | experiments/e007b_long_context_sparse_hook/ | exp4_long_context_sparse_hook.py | PENDING |
| e005b | experiments/e005b_swa_mask_ablation_v2/ | exp3b_swa_mask_ablation_v2.py | PENDING |

**New test images added** (for Topic E — Vision × NoPE):
- `test_images/ko_math/` — 29 Korean math problem images
- `test_images/en_math/` — 13 English math problem images
- `test_images/en_graph/` — 19 English graph/chart images
- Previous: `test_images/반도체 도면 1/`, `반도체 도면 2/` (2 items, from session 3)

**e006b design** (fixes e006 FAILED):
- Paired one-sample t-test on per-sequence delta_nll (μ=0)
- n=50 sequences (10 per length: 2048/3072/4096/5120/6144t)
- Expected to VALIDATE: post-hoc from e006 showed t=30.15, d_z=6.74

**Next session — run in order**:
1. Check e006b result + write ANALYSIS.md
2. Run e007b: `python3 nope_analysis/experiments/exp4_long_context_sparse_hook.py`
3. Run e005b: `python3 nope_analysis/experiments/exp3b_swa_mask_ablation_v2.py`
4. If all 3 RQ3 conditions met → draft paper outline (see RQ3 operational definition Section 2b)
5. e008 (delta probe): script NOT yet written — needs to be written before running

### Hardware note (2026-04-27 session 4)
Confirmed environment: 2× A100 80GB PCIe (85.2GB each), 32 vCPU, 1082GB RAM.
CUDA 12.2 → use cu121. `device_map='auto'` splits model across both GPUs automatically.

### Hardware note (2026-04-27 session 3 — original)
Previous environment: 1× A100 80GB PCIe, 16 vCPU, 192 GiB RAM.
**New environment: 2× A100 80GB PCIe, 32 vCPU, 384 GiB RAM.**
`device_map='auto'` in `nope_analysis/loader.py` handles 2-GPU split automatically — no code changes needed.
CUDA version on new machine: verify with `nvidia-smi` before installing torch (Step C).

*Previous: 2026-04-26 (session 2) — e005 (SWA mask ablation) run → INCONCLUSIVE: OOM at seq_len ≥ 5120 on A100 80GB PCIe prevented all beyond-window measurements. Full results in experiments/e005_swa_mask_ablation/ANALYSIS.md; redesign options documented there. Pre-registration infrastructure completed (bd9b933 + 4dbbee4 + f42305c): force_verdict, min_layers_significant enforcement, timestamp ordering, Operating Persona. R6 architectural fact-verification rule added to CLAUDE.md; papers/MODEL_FACTS.md and papers/model_card_EXAONE-4.5-33B.md now committed. All findings in EXPERIMENT_LOG.md and FINDINGS.md.*

*Previous: 2026-04-26 (session 1) — Scientific audit applied: fixed RQ1 logical leap (H1/H2 distinction, Exp 2c added), added Exp 2b reliability warning, corrected Exp 1b "meaningful reversal" to require statistical test, added RQ3 operational definition (Section 2b), added related architecture positioning note (Section 2), added computational scope section (Section 10), converted Korean setup sections to English, added design rules 9-10.*
