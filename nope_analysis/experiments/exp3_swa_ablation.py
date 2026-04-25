"""
Experiment 3: SWA Ablation — NoPE Global 레이어가 perplexity에 기여하는가?

연구 질문 (RQ3): 장거리 의존성에서 NoPE Global이 실제로 기여하는가?

방법:
  조건 A (베이스라인): 정상 모델로 perplexity 측정
  조건 B (ablation):  Global 레이어의 attention output을 0으로 치환 후 perplexity 측정
  → A vs B 차이 = NoPE Global의 기여

  입력 길이 1024, 2048, 3072, 4096, 5120, 6144 토큰으로 테스트
  → 4096 이하: SWA가 전체 컨텍스트 커버 가능 → ablation 효과 작을 것
  → 4096 초과: SWA 윈도우 초과 → ablation 효과 클 것 (가설)

결과: outputs/exp3_swa_ablation/
"""
import sys
sys.path.insert(0, '/home/elicer/sda212331')

import json
import torch
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nope_analysis.loader import load_model_and_tokenizer, load_config, get_global_layer_indices

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp3_swa_ablation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LENGTHS = [1024, 2048, 3072, 4096, 5120, 6144]
SWA_WINDOW = 4096

BASE_TEXT = (
    "인공지능 기술의 발전은 현대 사회에 많은 변화를 가져오고 있다. "
    "특히 자연어 처리 분야에서의 혁신은 인간과 기계 사이의 소통 방식을 근본적으로 바꾸고 있으며 "
    "대규모 언어 모델의 등장으로 텍스트 생성 번역 요약 등 다양한 작업에서 인간 수준의 성능을 달성하고 있다. "
    "The development of large language models has fundamentally changed how we approach natural language understanding. "
    "These models learn from vast amounts of text data and can generate coherent contextually appropriate responses. "
    "The hybrid attention mechanism combining sliding window attention and global attention is a key architectural innovation. "
    "Sliding window attention limits each token to attending only within a local window reducing computational complexity. "
    "Global attention layers by contrast allow every token to attend to every other token in the sequence. "
    "NoPE No Positional Embedding is a technique used in global attention layers to remove explicit position encodings. "
    "수학적으로 증명하면 임의의 연속 함수에 대해 적분이 존재함을 보일 수 있다. "
    "언어 모델의 어텐션 메커니즘은 입력 시퀀스의 각 토큰 간 관계를 학습하는 핵심 구성 요소이다. "
)


def build_input(tokenizer, target_len: int):
    repeated = BASE_TEXT * (target_len // 60 + 5)
    tokens = tokenizer(repeated, return_tensors='pt', add_special_tokens=True)
    ids = tokens['input_ids']
    if ids.shape[1] < target_len:
        raise ValueError(f"텍스트 부족: {ids.shape[1]} < {target_len}")
    return ids[:, :target_len]


def compute_perplexity(model, input_ids):
    """cross-entropy loss → perplexity"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    nll = outputs.loss.item()
    return float(np.exp(nll)), nll


def register_ablation_hooks(model, global_layers):
    """
    Global 레이어의 self_attn output을 0으로 교체하는 hook 등록.
    attn_output을 0으로 만들면 residual connection은 유지되고
    attention 기여분만 제거됨.
    """
    handles = []
    global_set = set(global_layers)

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output: (attn_output, attn_weights, past_key_value, ...)
            # attn_output을 0으로 → attention 기여 제거
            zeroed = torch.zeros_like(output[0])
            return (zeroed,) + output[1:]
        return hook

    # 모델 구조 탐색: language_model.model.layers[i].self_attn
    try:
        layers = model.language_model.model.layers
    except AttributeError:
        try:
            layers = model.model.layers
        except AttributeError:
            print("⚠️ 레이어 접근 실패 — 모델 구조 확인 필요")
            return handles

    for i in global_set:
        if i < len(layers):
            handle = layers[i].self_attn.register_forward_hook(make_hook(i))
            handles.append(handle)

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def run():
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    print(f"Global (NoPE) layers: {global_layers}")
    print(f"SWA window: {SWA_WINDOW} tokens")
    print(f"Target lengths: {TARGET_LENGTHS}\n")

    model, tokenizer = load_model_and_tokenizer()

    results = []

    for target_len in TARGET_LENGTHS:
        print(f"\n{'='*50}")
        print(f"Length = {target_len} tokens")

        input_ids = build_input(tokenizer, target_len).to(model.device)

        # 조건 A: 베이스라인
        ppl_base, nll_base = compute_perplexity(model, input_ids)
        print(f"  Baseline  PPL: {ppl_base:.4f}  (NLL: {nll_base:.4f})")

        # 조건 B: Global ablation
        handles = register_ablation_hooks(model, global_layers)
        ppl_ablated, nll_ablated = compute_perplexity(model, input_ids)
        remove_hooks(handles)
        print(f"  Ablated   PPL: {ppl_ablated:.4f}  (NLL: {nll_ablated:.4f})")

        delta_nll = nll_ablated - nll_base
        delta_ppl = ppl_ablated - ppl_base
        print(f"  ΔNLL: {delta_nll:+.4f}  ΔPPL: {delta_ppl:+.4f}  (양수 = Global이 도움)")

        results.append({
            'seq_len': target_len,
            'ppl_baseline': ppl_base,
            'ppl_ablated': ppl_ablated,
            'nll_baseline': nll_base,
            'nll_ablated': nll_ablated,
            'delta_nll': delta_nll,
            'delta_ppl': delta_ppl,
            'exceeds_swa_window': target_len > SWA_WINDOW,
        })

        del input_ids
        torch.cuda.empty_cache()

    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    plot_results(results)
    save_summary(results)


def plot_results(results):
    lengths = [r['seq_len'] for r in results]
    delta_ppls = [r['delta_ppl'] for r in results]
    delta_nlls = [r['delta_nll'] for r in results]
    colors = ['tomato' if r['exceeds_swa_window'] else 'steelblue' for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(lengths)), delta_ppls, color=colors, alpha=0.8)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.axvline(lengths.index(SWA_WINDOW) - 0.5 + (1 if SWA_WINDOW in lengths else 0),
                color='gray', linestyle='--', alpha=0.7, label=f'SWA window ({SWA_WINDOW})')
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels([str(l) for l in lengths])
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('ΔPPL (Ablated - Baseline)')
    ax1.set_title('PPL Increase When Global Attention Removed\n(blue=within SWA window, red=exceeds window)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.plot(lengths, [r['ppl_baseline'] for r in results], 'o-', color='steelblue', label='Baseline', linewidth=2)
    ax2.plot(lengths, [r['ppl_ablated'] for r in results], 's--', color='tomato', label='Global Ablated', linewidth=2)
    ax2.axvline(SWA_WINDOW, color='gray', linestyle='--', alpha=0.7, label=f'SWA window')
    ax2.set_xlabel('Sequence Length (tokens)')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Baseline vs Ablated Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'ablation_perplexity.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/ablation_perplexity.png")


def save_summary(results):
    within = [r for r in results if not r['exceeds_swa_window']]
    beyond = [r for r in results if r['exceeds_swa_window']]

    summary = {
        'within_swa_window': {
            'lengths': [r['seq_len'] for r in within],
            'mean_delta_nll': float(np.mean([r['delta_nll'] for r in within])) if within else None,
            'mean_delta_ppl': float(np.mean([r['delta_ppl'] for r in within])) if within else None,
        },
        'beyond_swa_window': {
            'lengths': [r['seq_len'] for r in beyond],
            'mean_delta_nll': float(np.mean([r['delta_nll'] for r in beyond])) if beyond else None,
            'mean_delta_ppl': float(np.mean([r['delta_ppl'] for r in beyond])) if beyond else None,
        },
        'per_length': {str(r['seq_len']): {
            'ppl_baseline': r['ppl_baseline'],
            'ppl_ablated': r['ppl_ablated'],
            'delta_nll': r['delta_nll'],
            'delta_ppl': r['delta_ppl'],
        } for r in results},
    }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Experiment 3 Summary ===")
    print(f"  SWA 윈도우 내 ({SWA_WINDOW}토큰 이하)  평균 ΔNLL: {summary['within_swa_window']['mean_delta_nll']:.4f}")
    print(f"  SWA 윈도우 초과 ({SWA_WINDOW}토큰 초과) 평균 ΔNLL: {summary['beyond_swa_window']['mean_delta_nll']:.4f}")
    print(f"  → 초과 구간에서 차이가 더 크면: NoPE Global이 장거리 의존성에 기여 확인")
    print(f"\nSummary saved: {OUT_DIR}/summary.json")


if __name__ == '__main__':
    run()
