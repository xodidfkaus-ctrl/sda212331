# EXAONE 4.5 연구 주제 목록

> 이 문서는 현재 진행 중인 연구와 향후 진행할 연구 주제를 전부 기록합니다.
> 새 세션의 Claude Code는 이 파일을 읽고 전체 연구 범위를 파악해야 합니다.
>
> **모델**: LGAI-EXAONE/EXAONE-4.5-33B
> **논문**: arXiv:2604.08644 (EXAONE 4.5 Technical Report)
> **최종 목표**: arXiv 논문 작성 → ACL / EMNLP 워크샵 투고

---

## 연구 배경: 왜 이 모델인가

EXAONE 4.5는 SWA(Sliding Window Attention) + Global(NoPE) 하이브리드 구조를 채택한
거의 유일한 공개 모델이다. NoPE(No Positional Embedding)를 Global Attention에만
적용한 사례는 학술적으로 분석된 전례가 없다.

공식 벤치마크에서 주목할 약점이 발견됐다:
```
AA-LCR (장문 추론):  EXAONE 4.5 = 50.6  vs  GPT-5 mini = 68.0  (△17.4)
```
이 격차가 NoPE 구조의 한계에서 비롯되는지 검증하는 것이 연구의 핵심 동기다.

---

## PART 1: 현재 진행 중인 연구 (RQ1 ~ RQ3)

### RQ1 — NoPE 레이어는 위치 정보를 인코딩하는가?

위치 임베딩이 없는 Global 레이어의 hidden state에서 토큰 위치를
예측할 수 있다면 → 앞 SWA 레이어가 위치 정보를 암묵적으로 전달한다는 뜻.

| 실험 | 방법 | 상태 |
|------|------|------|
| Exp 2 | 레이어별 hidden state에 linear probe → position 분류 (5개 프롬프트, 과적합 버전) | ✅ 완료 |
| Exp 2b | train/test 분리, 30개 프롬프트, 64/128/256 토큰, 도메인별 breakdown | 🔄 진행 중 |

**현재 결과**: Exp 2에서 Global/SWA 모두 accuracy=1.0 → NoPE 레이어도 위치 정보 완벽 인코딩 확인.
단, 동일 데이터 학습+평가라 과적합 의심 → Exp 2b에서 검증 중.

---

### RQ2 — SWA와 Global의 attention 패턴이 기능적으로 다른가?

attention entropy(분산도)와 attention distance(얼마나 멀리 보는가)로 비교.
입력이 길어질수록 차이가 커지는지 관찰.

| 실험 | 방법 | 상태 |
|------|------|------|
| Exp 1 | 짧은 입력 (14~26 토큰) entropy 측정 | ✅ 완료 |
| Exp 1b | 긴 입력 (128~2048 토큰) entropy + distance 측정 | ✅ 완료 |

**현재 결과**:
- Exp 1: Global=0.930, SWA=0.942 → 짧은 입력에서 차이 없음 (SWA 윈도우 안에 전부 들어가기 때문)
- Exp 1b: 2048 토큰에서 Global entropy(3.54) > SWA(3.45) 역전, attention distance 격차 154 토큰
- **새 가설**: 4096 토큰 초과 시 격차가 급격히 커질 것 → Exp 4에서 검증

---

### RQ3 — 장거리 의존성에서 NoPE Global이 실제로 기여하는가?

Global 레이어를 끄면 perplexity가 얼마나 올라가는지 측정.
4096 토큰(SWA 윈도우 경계) 초과 시 효과가 더 커지는지가 핵심.

| 실험 | 방법 | 상태 |
|------|------|------|
| Exp 3 | Global self_attn output을 0으로 치환 → perplexity 비교 (1024~6144 토큰) | 🔄 대기 중 |
| Exp 4 | hook 방식으로 4096 초과 구간 attention distance/entropy 측정 (2048~8192 토큰) | 🔄 대기 중 |

---

## PART 2: 향후 연구 주제 (우선순위 순)

---

### 주제 A — "Lost in the Middle" × NoPE  ★ 최우선

**동기**: AA-LCR 50.6의 직접적 원인 규명 가능. 논문 임팩트 최대.

**연구 질문**: 장문 입력의 앞/중간/끝 세 구간을 Global과 SWA가 얼마나 다르게 참조하는가?

**방법**:
- 입력을 3구간으로 나눔: 앞 1/3 (초반), 중간 1/3, 끝 1/3 (최근)
- 각 구간으로 향하는 attention weight 합산 비교
- Global vs SWA별로 구간 참조 비율 분석
- 길이 변화 (1K, 4K, 8K, 32K 토큰)에 따른 추세 관찰

**예상 발견**: SWA는 끝 구간에 집중, Global은 고르게 분포하거나 앞 구간까지 참조.
"Lost in the Middle"이 SWA 레이어에서 더 심하다면 → Global이 보완하는 역할.

**코드 위치**: `nope_analysis/experiments/expA_lost_in_middle.py` (미작성)

---

### 주제 B — MTP 레이어 분석  ★ 논문 신규성 최고

**동기**: MTP(Multi-Token Prediction) 레이어를 실험적으로 분석한 논문이 존재하지 않음.

**연구 질문**:
1. MTP 레이어의 attention 패턴이 메인 64 레이어와 어떻게 다른가?
2. MTP 레이어를 제거하면 perplexity가 얼마나 변하는가?

**방법**:
- MTP 레이어의 hidden state를 추출하고 메인 레이어 64번과 비교
- MTP의 attention entropy, distance 측정
- MTP ablation: MTP 레이어 output을 0으로 치환 → loss 변화

**주의**: MTP 레이어는 `model.language_model.model.mtp_layers[0]` 형태로 접근 예상.
실제 구조는 런타임 확인 필요.

**코드 위치**: `nope_analysis/experiments/expB_mtp_analysis.py` (미작성)

---

### 주제 C — Reasoning 모드에서 attention 패턴 변화

**동기**: EXAONE 4.5는 `enable_thinking=True`가 기본. 사고 토큰 생성 중
Global 레이어가 일반 모드와 다르게 동작하는가?

**연구 질문**: 추론 과정(think 토큰) 중에 NoPE Global이 장거리 의존성을 더 활용하는가?

**방법**:
- reasoning 모드로 문제 풀기 (수학, 논리) → 생성된 think 토큰 구간 식별
- think 구간 vs 일반 답변 구간에서 attention entropy/distance 비교
- Global 레이어가 think 구간에서 더 넓게 attend하는지 확인

**주의**: 모델을 inference 모드로 실행해야 함 (현재 실험들은 forward pass만 사용).
vLLM 또는 transformers generate() 필요.

**코드 위치**: `nope_analysis/experiments/expC_reasoning_attention.py` (미작성)

---

### 주제 D — 한국어 vs 영어 레이어 전문화

**동기**: EXAONE이 한국어에서 특히 강함 (KMMMU 42.7 vs Qwen3-VL 32B 37.8).
한국어는 동사가 문장 끝에 오는 SOV 구조 → 장거리 의존성이 더 중요.

**연구 질문**: 특정 Global 레이어가 한국어 구조를 처리하는 데 전문화되어 있는가?

**방법**:
- 한국어/영어 프롬프트를 동일 내용으로 준비 (번역쌍)
- 레이어별 hidden state에서 언어 식별 probe
- 한국어 입력에서 Global 레이어의 attention distance가 영어보다 더 긴지 측정
- 한국어에서 Global 레이어 ablation 시 perplexity 증가폭이 영어보다 큰지 비교

**코드 위치**: `nope_analysis/experiments/expD_korean_english.py` (미작성)

---

### 주제 E — 비전 인코더 × NoPE 상호작용

**동기**: EXAONE 4.5는 VLM. 이미지 토큰이 언어 레이어로 들어올 때
Global(NoPE) 레이어가 SWA보다 이미지 토큰을 더 많이 참조하는가?

**연구 질문**: NoPE 레이어가 멀티모달 통합의 핵심 레이어인가?

**방법**:
- 이미지+텍스트 입력 구성 (image tokens + question tokens)
- 레이어별로 텍스트 토큰이 이미지 토큰에 부여하는 attention weight 합산
- Global 레이어 vs SWA 레이어의 cross-modal attention 비교
- 이미지 관련 질문의 정답 토큰 생성 시 어느 레이어가 이미지를 가장 많이 참조하는지

**주의**: Vision encoder 입력 처리 방식 파악 필요.
이미지 토큰의 위치 범위를 런타임에서 확인해야 함.

**코드 위치**: `nope_analysis/experiments/expE_vision_nope.py` (미작성)

---

### 주제 F — KV Head 8개의 역할 분업

**동기**: GQA에서 40 Q-head가 8 KV-head를 공유. 8개 KV-head가 서로 다른
역할을 맡는다면 Global/SWA에서 분업 패턴이 다를 것.

**연구 질문**: Global 레이어의 KV head가 SWA보다 더 뚜렷하게 역할 분업하는가?

**방법**:
- 레이어별 8개 KV-head의 attention distance 분포 측정
- head 간 cosine similarity로 전문화 정도 측정
- Global vs SWA에서 head 전문화 패턴 비교

**코드 위치**: `nope_analysis/experiments/expF_kv_head_specialization.py` (미작성)

---

### 주제 G — Reordered Norm 효과 정량화

**동기**: 표준 Pre/Post Norm과 달리 EXAONE은 Attention/MLP 직후, residual 이전에
norm 적용. 이 비표준 설계가 representation에 어떤 영향을 주는가?

**방법**:
- 레이어별 residual stream의 L2 norm 추적
- norm 전/후 hidden state의 변화량 측정
- 레이어가 깊어질수록 norm 크기 변화 추세 분석

**코드 위치**: `nope_analysis/experiments/expG_reordered_norm.py` (미작성)

---

### 주제 H — 레이어별 중요도 (Fine-grained Ablation)

**동기**: Exp 3은 모든 Global 레이어를 동시에 끄는데, 레이어별로 중요도가 다를 것.

**연구 질문**: 얕은 Global (레이어 3, 7)과 깊은 Global (레이어 59, 63) 중 어느 쪽이 더 중요한가?

**방법**:
- Global 레이어 16개를 하나씩 ablation → 각각 perplexity 변화 측정
- 중요도 히트맵 생성 (레이어 인덱스 × 입력 길이)
- "어느 깊이의 Global이 장거리 의존성을 담당하는가" 규명

**코드 위치**: `nope_analysis/experiments/expH_layer_importance.py` (미작성)

---

## 연구 주제 전체 지도

```
EXAONE 4.5 NoPE 연구
│
├── [진행 중] NoPE 기본 분석
│   ├── RQ1: 위치 인코딩 여부 (Exp 2, 2b)
│   ├── RQ2: attention 패턴 차이 (Exp 1, 1b)
│   └── RQ3: 장거리 기여도 (Exp 3, 4)
│
├── [향후] 성능 격차 원인 규명
│   ├── 주제 A: Lost in the Middle × NoPE  ← AA-LCR 50.6 설명
│   └── 주제 D: 한국어 전문화              ← EXAONE 정체성
│
├── [향후] 아키텍처 심층 분석
│   ├── 주제 B: MTP 레이어                ← 논문 신규성 최고
│   ├── 주제 F: KV Head 분업
│   ├── 주제 G: Reordered Norm
│   └── 주제 H: 레이어별 중요도
│
└── [향후] 멀티모달/추론 분석
    ├── 주제 C: Reasoning 모드 attention
    └── 주제 E: Vision × NoPE 상호작용
```

---

*마지막 업데이트: 2026-04-25*
