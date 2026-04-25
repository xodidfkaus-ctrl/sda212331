# EXAONE 4.5 NoPE 연구 인수인계 자료

> **이 문서를 읽는 Claude Code에게**: 새 PC, 새 세션에서 이 프로젝트를 시작할 때 가장 먼저 읽어야 할 문서입니다.
> 반드시 **섹션 0의 실행 전 체크리스트를 통과한 뒤** 섹션 9의 이해 확인 질문에 답하고 작업을 시작하세요.
>
> **전체 연구 주제 목록** → `RESEARCH_TOPICS.md` 참고 (현재 진행 + 향후 8개 주제 전부 기록됨)
> 체크리스트를 건너뛰면 실험이 CPU로 돌아 수 시간이 걸리거나 모델이 없어서 실패합니다.

---

## 0-1. 연구자가 준비해야 하는 것 (한눈에)

대부분의 실험은 코드가 텍스트를 자동 생성하므로 연구자 준비가 필요 없다.
**단 두 가지 예외**가 있다:

| 언제 필요한가 | 무엇을 준비하는가 | 어디에 넣는가 |
|-------------|-----------------|--------------|
| 주제 E (Vision × NoPE) 실험 시작 전 | 이미지 파일 5종 (문서/차트/사진 등 JPG/PNG) | `sda212331/test_images/` 폴더 생성 후 저장 |
| 주제 C (Reasoning 모드) 실험 시작 전 | 수학·논리 문제 샘플 10~20개 (선택 사항, 없어도 기본값으로 실행됨) | Claude Code에게 텍스트로 전달 |

이미지 상세 요구사항 → `RESEARCH_TOPICS.md` 의 "주제 E" 섹션 참고.

---

## 0. 실행 전 체크리스트 (매 세션 시작 시 반드시 실행)

아래 명령을 **순서대로** 실행하고 모두 통과해야 실험을 시작할 수 있습니다.

### Step 1 — GPU 확인 (가장 중요)

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU수:', torch.cuda.device_count()); [print(f'  GPU{i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

**정상 출력:**
```
CUDA: True
GPU수: 2
  GPU0: NVIDIA A100 80GB PCIe
  GPU1: NVIDIA A100 80GB PCIe
```

**❌ `CUDA: False` 로 나오면 즉시 중단 — 실험 절대 실행하지 말 것.**

GPU가 False인 원인과 해결법:
```bash
# 원인 확인
python3 -c "import torch; torch.cuda.is_available()" 2>&1 | grep -i "warn\|error"

# 가장 흔한 원인: PyTorch와 CUDA 드라이버 버전 불일치
# 드라이버 버전 확인
nvidia-smi | grep "CUDA Version"

# 해결: CUDA 12.x 드라이버라면 아래 명령으로 PyTorch 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 재설치 후 다시 확인
python3 -c "import torch; print(torch.cuda.is_available())"
```

> **왜 중요한가**: GPU 없이 33B 모델 forward pass를 돌리면 실험 1개에 3~4시간 걸립니다.
> GPU면 5~10분입니다. GPU 없이 실행하면 안 됩니다.

---

### Step 2 — 모델 캐시 확인

```bash
ls /home/elicer/sda212331/model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/snapshots/
```

**정상**: 긴 해시 폴더 이름이 보임 (e.g. `58d6616...`)

**❌ 없으면**: 모델을 재다운로드해야 합니다 (세션 재시작 시 삭제됨).
```bash
cd /home/elicer/sda212331
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('LGAI-EXAONE/EXAONE-4.5-33B', cache_dir='./model_cache', ignore_patterns=['*.bin','*.pt'])
"
# 약 20분 소요
```

---

### Step 3 — 의존성 확인

```bash
python3 -c "from nope_analysis.loader import load_config; cfg = load_config(); print('OK:', cfg.model_type)"
```

**정상**: `OK: exaone4_5` 또는 유사한 모델 타입 출력

**❌ ImportError 나오면**: 패키지 재설치
```bash
pip install transformers safetensors scikit-learn matplotlib -q
```

---

### Step 4 — Git 상태 확인

```bash
cd /home/elicer/sda212331
git log --oneline -3    # 최근 커밋 확인
git status              # 미커밋 파일 확인
```

이전 세션에서 push 안 된 결과가 있으면 먼저 push하고 시작하세요.

---

### 체크리스트 요약표

| 항목 | 확인 명령 | 통과 기준 |
|------|-----------|-----------|
| GPU 활성화 | `python3 -c "import torch; print(torch.cuda.is_available())"` | `True` |
| GPU 개수 | `python3 -c "import torch; print(torch.cuda.device_count())"` | `2` |
| 모델 캐시 | `ls model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/snapshots/` | 폴더 존재 |
| 의존성 | `python3 -c "from nope_analysis.loader import load_config; load_config()"` | 오류 없음 |
| Git 동기화 | `git status` | 미push 결과 없음 |

---

## 1. 기본 정보

| 항목 | 값 |
|------|-----|
| GitHub | https://github.com/xodidfkaus-ctrl/sda212331 |
| 환경 | 엘리스 클라우드 (NVIDIA A100 80GB × 2, RAM 384GB) |
| 모델 경로 | `/home/elicer/sda212331/model_cache/` (64GB, **세션 종료 시 삭제됨**) |
| 작업 디렉토리 | `/home/elicer/sda212331/` |

---

## 2. 연구 목적 및 핵심 질문

### 한 줄 요약
> EXAONE 4.5의 **NoPE(No Positional Embedding) Global Attention 레이어**가 SWA 레이어와 어떻게 다르게 동작하는지 실증적으로 분석한다.

### 연구 배경
- EXAONE 4.5는 SWA(Sliding Window Attention) + Global(NoPE) 하이브리드 구조를 사용하는 최초 공개 모델 중 하나
- NoPE를 Global Attention에만 적용한 사례는 학술적으로 분석된 바 없음
- **최종 목표**: arXiv 논문 작성 → ACL/EMNLP 워크샵 투고

### 3개의 핵심 연구 질문 (Research Questions)

| RQ | 질문 | 대응 실험 |
|----|------|-----------|
| **RQ1** | NoPE Global 레이어는 위치 정보를 인코딩하는가? | Exp 2, Exp 2b |
| **RQ2** | SWA(RoPE)와 Global(NoPE)의 attention 패턴이 기능적으로 다른가? | Exp 1, Exp 1b |
| **RQ3** | 장거리 의존성(262K 컨텍스트)에서 NoPE가 실제로 기여하는가? | Exp 3, Exp 4 (미작성) |

### 연구 방법론 (중요: 이 연구는 탐색적 접근)
코드 실행 시간보다 **"뭘 봐야 할지 모르는 상태에서 점점 좁혀가는 과정"** 이 핵심이다.

```
실험 1 실행
  → 결과에서 패턴 발견 (예: 레이어 32~48에서 다름)
  → 왜? → 새 가설 생성
  → 실험 2 설계 (그 구간만 입력 길이별로 다시 측정)
  → 결과에서 한국어/영어 차이 발견
  → 실험 3 설계...
```
즉, 각 실험 결과가 다음 실험을 설계하는 입력이 된다. **실험 계획은 고정이 아니다.**

---

## 3. EXAONE 4.5 아키텍처 (필수 암기)

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 33B (LM 31.7B + Vision 1.29B) |
| 레이어 수 | 64 main + 1 MTP |
| 어텐션 패턴 | `LLLG` × 16 = 48 SWA + 16 Global |
| SWA 윈도우 | 4,096 토큰 (RoPE 사용, 최근 4096만 볼 수 있음) |
| Global Attention | 윈도우 제한 없음, **RoPE 없음(NoPE)** |
| Global 레이어 인덱스 | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 |
| GQA | Q헤드 40개 / KV헤드 8개 / 헤드 dim 128 |
| Reordered Norm | Attn/MLP 이후, residual 이전 (비표준 — Pre/Post Norm과 다름) |
| Vocab | 153,600 | Context | 262,144 토큰 |

**NoPE란?** RoPE(Rotary Position Embedding)를 적용하지 않는 것. Global 레이어는 전체 시퀀스를 보기 때문에 위치 임베딩 없이도 장거리 의존성을 포착하도록 설계됨. 그러나 **앞 SWA 레이어들이 위치 정보를 representation에 이미 담아서 올려준다** — Exp 2에서 실증됨.

---

## 4. 프로젝트 파일 구조 및 각 파일 역할

```
sda212331/
│
├── CLAUDE.md                          ← Claude Code 설계 원칙 (변경 금지 규칙 포함)
├── HANDOVER.md                        ← 지금 이 파일 (인수인계)
├── README.md                          ← 최소한의 프로젝트 설명
├── requirements.txt                   ← Python 의존성
├── .gitignore                         ← model_cache/, __pycache__/ 제외
│
├── analyze.py                         ← [1차 분석] 가중치 통계 분석 진입점
│                                         모델을 스트리밍 로드해 tensor별 통계 계산
│                                         python analyze.py --model-path ./model_cache
│
├── run.sh                             ← analyze.py 실행 쉘스크립트
├── run_nope.sh                        ← Exp 1, Exp 2 순차 실행 + git push 자동화
│
├── src/                               ← [1차 분석] 가중치 통계 분석 모듈 (analyze.py에서 사용)
│   ├── loader.py                      ← safetensors 스트리밍 로더, 텐서 분류기
│   │                                     classify_tensor(): 텐서를 카테고리로 분류
│   │                                     iter_tensors(): OOM 방지 스트리밍 yield
│   ├── stats.py                       ← 텐서 통계 계산 (std, abs_mean, sparsity, effective_rank, stable_rank)
│   │                                     StatsAggregator: 결과를 stats.jsonl에 append-only 저장
│   ├── viz.py                         ← 시각화 (4종 차트 생성)
│   │                                     plot_std_by_layer, plot_effective_rank_by_layer,
│   │                                     plot_category_summary, plot_norm_weights
│   └── report.py                      ← 분석 결과 마크다운 리포트 생성
│
├── nope_analysis/                     ← [2차 분석] NoPE 동작 분석 실험 코드
│   ├── loader.py                      ← 모델 풀로드 (attention 추출용)
│   │                                     ⚠️ CONFIG_MAPPING 패치 포함 — 반드시 이걸로 로드
│   │                                     load_model_and_tokenizer(): attn_implementation='eager' 강제
│   │                                     get_global_layer_indices(), get_swa_layer_indices()
│   ├── analysis/
│   │   └── statistical_tests.py      ← 통계 검정 유틸리티
│   │                                    compare_groups(): Welch t-test + Cohen's d + bootstrap CI
│   │                                    analyze_jsonl(): 실험 results.jsonl 직접 분석
│   ├── corpus/
│   │   └── downloader.py             ← 실제 텍스트 코퍼스 다운로더
│   │                                    WikiText-103(영어) / KLUE-MRC(한국어) 자동 다운로드+캐시
│   │                                    build_input_from_corpus(): 실험용 토큰 텐서 생성
│   └── experiments/
│       ├── exp1_attention_entropy.py  ← Exp 1: 짧은 입력(14~26 토큰) entropy 측정 [완료]
│       ├── exp1b_long_input.py        ← Exp 1b: 긴 입력(128~2048 토큰) entropy 측정 [완료]
│       ├── exp2_positional_probe.py   ← Exp 2: 레이어별 위치 정보 linear probe [완료]
│       ├── exp2b_positional_probe_v2.py ← Exp 2b: probe 개선판 (train/test split, 30개 프롬프트) [실행 예정]
│       ├── exp3_swa_ablation.py       ← Exp 3: Global attention output 제거 ablation [실행 예정]
│       ├── exp3b_swa_mask_ablation.py ← Exp 3b: ⭐ SWA 마스크 주입 ablation (방법론 개선판)
│       │                                   Global 레이어에 SWA 윈도우 마스크 강제 주입
│       │                                   → "NoPE 제거"가 아닌 "SWA처럼 동작"으로 변환
│       │                                   → 순수한 NoPE 장거리 기여도 측정
│       └── exp4_long_context.py       ← Exp 4: 4096+ 토큰에서 Global vs SWA attention 분기 측정 [실행 예정]
│
├── outputs/                           ← 모든 실험 결과 (GitHub에 push됨)
│   ├── run_config.json                ← 1차 분석 실행 파라미터
│   ├── stats.jsonl                    ← 모든 텐서 통계 (append-only, 627KB)
│   ├── summary.json                   ← 카테고리별 집계 통계
│   ├── report.md                      ← 자동 생성 분석 리포트
│   ├── attn_std_by_layer.png          ← SWA vs Global 가중치 std 차트
│   ├── category_summary.png           ← 카테고리별 파라미터 수 & std
│   ├── effective_rank_by_layer.png    ← 레이어별 유효 랭크
│   ├── norm_weights.png               ← Reordered Norm 가중치 크기
│   ├── exp1_attention_entropy/        ← Exp 1 결과 [완료]
│   │   ├── results.jsonl
│   │   ├── summary.json
│   │   └── entropy_by_layer.png
│   ├── exp1b_long_input/              ← Exp 1b 결과 [실행중]
│   │   ├── results.jsonl
│   │   ├── summary.json
│   │   ├── entropy_vs_length.png
│   │   └── entropy_by_layer_per_length.png
│   └── exp2_positional_probe/         ← Exp 2 결과 [완료]
│       ├── results.jsonl
│       ├── summary.json
│       └── probe_accuracy_by_layer.png
│
└── model_cache/                       ← 64GB 모델 파일 (Git 제외, 세션 종료 시 삭제됨)
```

---

## 5. 완료된 실험 결과

### [1차] 가중치 통계 분석 (`analyze.py`)
- **목적**: 모델 내부 가중치의 기본 통계 파악 (std, effective rank, sparsity)
- **주요 발견**: Global(NoPE) 레이어와 SWA 레이어의 가중치 std 분포가 레이어 깊이에 따라 다름
- **출력**: `outputs/stats.jsonl`, `outputs/summary.json`, 4종 차트

### Exp 1 — Attention Entropy (짧은 입력, 14~26 토큰) → **RQ2**
| 지표 | Global(NoPE) | SWA |
|------|--------------|-----|
| Entropy | 0.9301 | 0.9422 |
| Attn Distance | 7.18 tokens | 7.10 tokens |
- **결론**: 차이 없음 → 짧은 입력은 SWA 윈도우(4096) 안에 전부 들어가므로 구조적 차이가 드러나지 않음
- **다음 단계**: Exp 1b (긴 입력)

### Exp 1b — Attention Entropy (긴 입력, 128~2048 토큰) → **RQ2** [완료]
- **목적**: 입력 길이 증가에 따라 SWA와 Global entropy/distance가 어떻게 달라지는지

| 길이 | Global Entropy | SWA Entropy | Global Dist | SWA Dist |
|------|---------------|-------------|-------------|----------|
| 128  | 1.7735 | 1.8121 | 33.9  | 31.7  |
| 256  | 2.0534 | 2.1636 | 70.2  | 64.7  |
| 512  | 2.4912 | 2.6152 | 143.9 | 127.1 |
| 1024 | 2.9823 | 3.0447 | 305.7 | 255.1 |
| 2048 | **3.5419** | **3.4501** | **643.7** | **489.1** |

- **핵심 발견 1**: 2048 토큰에서 Global entropy(3.54) > SWA entropy(3.45)로 역전 — 처음으로 유의미한 차이
- **핵심 발견 2**: Attention Distance 격차가 길이에 따라 급격히 증가 (128에서 2.2 차이 → 2048에서 **154.6** 차이)
  - Global: 전체 시퀀스를 고르게 봄 (distance가 seq_len/2에 근접)
  - SWA: 길이가 늘어나도 distance 증가가 더딤 (윈도우 제약의 흔적)
- **새 가설**: 4096 토큰 이상에서는 SWA distance가 포화되고 Global만 계속 증가할 것 → Exp 4 동기
- **다음 단계**: Exp 2b (positional probe 개선), Exp 3 (ablation)

### Exp 2 — Positional Probing (짧은 입력) → **RQ1**
| 지표 | Global(NoPE) | SWA |
|------|--------------|-----|
| Probe Accuracy | **1.000** | **1.000** |
- **핵심 발견**: NoPE 레이어임에도 위치 정보가 hidden state에 완벽하게 인코딩됨
- **해석**: 앞 SWA 레이어들이 RoPE를 통해 이미 위치 정보를 representation에 주입 → Global 레이어가 이를 그대로 전달
- **한계**: 동일 데이터로 학습+평가(과적합 가능성), 짧은 문장만 사용 → **Exp 2b 필요**

---

## 6. 남은 실험 (우선순위 순)

### Exp 2b — Positional Probe 개선 (우선순위 1) → **RQ1**
- **왜 필요한가**: Exp 2의 accuracy=1.0은 신뢰하기 어려움. train/test split 없이 같은 데이터로 학습+평가했기 때문
- **작업 내용**:
  - `exp2_positional_probe.py`를 개선해 train/test split 추가
  - 다양한 길이(128~512 토큰)와 도메인(한국어/영어/수학) 텍스트 사용
  - 목표: "NoPE 레이어가 SWA 레이어보다 위치 정보를 덜 인코딩하는가"에 대한 신뢰할 수 있는 답

### Exp 3 — Global Zero-Output Ablation → **RQ3** [실행 예정]
- Global 레이어의 self_attn output을 0으로 교체 → attention 기여 전체 제거
- **주의**: 방법론이 다소 과격함 (attention을 "제거"하는 것이지 "SWA화"가 아님)
- **비교 실험**: Exp 3b와 결과를 나란히 놓으면 두 ablation 방법의 차이도 분석 가능

### Exp 3b — SWA Mask Injection Ablation ⭐ (방법론 개선판) → **RQ3** [실행 예정]
- **Exp 3의 설계 문제를 수정한 개선판**
- Global 레이어에 SWA 윈도우 마스크를 강제 주입 (register_forward_pre_hook + with_kwargs=True)
- "Global → SWA처럼 동작"으로 변환 → NoPE의 장거리 attention 효과만 순수 분리
- WikiText-103 / KLUE-MRC 실제 텍스트 사용 (반복 텍스트 편향 제거)
- 통계 검정(t-test, Cohen's d) 자동 포함

### Exp 4 — Long Context Hook 분석 → **RQ3** [실행 예정]
- 2048~8192 토큰 범위에서 hook 기반으로 attention 통계 수집 (OOM 방지)
- 4096 이상에서 SWA distance가 포화되고 Global만 계속 증가하는지 확인
- "Lost in the Middle" 현상 EXAONE 4.5 재현 여부

---

## 7. 코드 실행 방법

```bash
cd /home/elicer/sda212331

# 모델 캐시 확인 (세션 재시작 시)
ls model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/

# 가중치 통계 분석 (1차 분석, 이미 완료)
python analyze.py --model-path ./model_cache

# NoPE 실험 개별 실행
python3 nope_analysis/experiments/exp1_attention_entropy.py
python3 nope_analysis/experiments/exp1b_long_input.py
python3 nope_analysis/experiments/exp2_positional_probe.py
python3 nope_analysis/experiments/exp2b_positional_probe_v2.py
python3 nope_analysis/experiments/exp3_swa_ablation.py
python3 nope_analysis/experiments/exp3b_swa_mask_ablation.py   # ⭐ 방법론 개선판 (Exp 3보다 이것 우선)
python3 nope_analysis/experiments/exp4_long_context.py

# 코퍼스 사전 다운로드 (Exp 3b, 4 실행 전 한 번만)
python3 -c "from nope_analysis.corpus.downloader import download_all; download_all()"

# 통계 검정 (실험 완료 후 results.jsonl에 적용)
python3 -c "
from nope_analysis.analysis.statistical_tests import analyze_jsonl, save_stats_report
from pathlib import Path
results = analyze_jsonl(Path('outputs/exp4_long_context/results.jsonl'))
save_stats_report(results, Path('outputs/exp4_long_context/stats.json'))
"

# 결과 GitHub push
git add outputs/ nope_analysis/ HANDOVER.md
git commit -m "실험명: 결과 요약"
git push
```

### 모델 로딩 주의사항 (절대 어기지 말 것)
```python
# ❌ 금지
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)  # config_type 불일치로 오류

# ✅ 올바른 방법
from nope_analysis.loader import load_model_and_tokenizer  # CONFIG 패치 자동 포함
model, tokenizer = load_model_and_tokenizer()
# 내부적으로 Exaone4_5_ForConditionalGeneration 사용 + attn_implementation='eager'
```

---

## 8. 설계 원칙 (변경 금지)

1. 모델은 항상 `model_cache/`에서 로드 (HuggingFace 캐시 경로 직접 지정)
2. 실험 결과는 `outputs/{exp_name}/` 에 저장 후 즉시 git push
3. `attn_implementation='eager'` 필수 — flash_attn은 attention weights 반환 안 함
4. Vision encoder 분석은 LM과 분리하여 별도 보고
5. 모든 실험은 `summary.json` + `results.jsonl` + 차트 세트로 저장
6. `stats.jsonl`은 append-only — 덮어쓰지 말 것 (재현성)
7. SVD는 `--no-spectral`로 생략 가능 — 필수 아님

---

## 9. Claude Code 이해 확인 질문

**새 세션의 Claude Code는 이 질문에 답한 뒤 작업을 시작해야 합니다.**

1. NoPE가 무엇인지, 왜 EXAONE 4.5에서 Global Attention에만 적용됐는지 설명하시오.

2. `LLLG` 패턴의 의미와 Global 레이어의 인덱스를 모두 나열하시오.

3. RQ1/RQ2/RQ3 각각에 대응하는 실험이 무엇인지 표로 정리하시오.

4. Exp 2의 accuracy=1.000이 의미하는 바와, 왜 이것이 신뢰하기 어려운지 설명하시오.

5. Exp 1과 Exp 1b의 차이, 그리고 짧은 입력으로 SWA와 Global의 차이가 안 나오는 이유를 설명하시오.

6. 이 연구에서 `AutoModelForCausalLM`을 쓰면 안 되는 이유와 올바른 로딩 방법을 설명하시오.

7. `src/loader.py`와 `nope_analysis/loader.py`의 차이를 설명하시오.

8. 이 연구의 탐색적 특성을 설명하시오 — 왜 실험 계획이 고정이 아닌가?

---

*마지막 업데이트: 2026-04-25*
