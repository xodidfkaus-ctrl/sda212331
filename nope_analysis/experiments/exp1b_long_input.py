"""
Experiment 1b: Attention Entropy — Long Input (128~2048 tokens)

Exp 1의 한계: 짧은 입력(14~26 토큰)은 SWA 윈도우(4096) 안에 전부 들어가므로
SWA와 Global의 차이가 나타나지 않음.

연구 질문: 입력 길이가 길어질수록 SWA와 Global(NoPE)의 entropy/attention_distance 차이가
유의미하게 달라지는가?

방법:
- 동일 텍스트를 tokenizer로 잘라 128, 256, 512, 1024, 2048 토큰 길이로 고정
- 각 길이에서 레이어별 entropy 및 attention distance 측정
- SWA vs Global 추세 비교

결과: outputs/exp1b_long_input/
"""
import sys
sys.path.insert(0, '/home/elicer/sda212331')

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nope_analysis.loader import load_model_and_tokenizer, load_config, get_global_layer_indices, get_swa_layer_indices

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp1b_long_input')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 긴 텍스트를 만들기 위한 베이스 텍스트 (반복해서 목표 길이 달성)
BASE_TEXT = (
    "인공지능 기술의 발전은 현대 사회에 많은 변화를 가져오고 있다. "
    "특히 자연어 처리 분야에서의 혁신은 인간과 기계 사이의 소통 방식을 근본적으로 바꾸고 있으며, "
    "대규모 언어 모델의 등장으로 텍스트 생성, 번역, 요약 등 다양한 작업에서 인간 수준의 성능을 달성하고 있다. "
    "EXAONE 4.5는 LG AI Research에서 개발한 멀티모달 언어 모델로, 한국어와 영어 모두에서 뛰어난 성능을 보인다. "
    "The development of large language models has fundamentally changed how we approach natural language understanding. "
    "These models learn from vast amounts of text data and can generate coherent, contextually appropriate responses. "
    "The hybrid attention mechanism combining sliding window attention and global attention is a key architectural innovation. "
    "Sliding window attention limits each token to attending only within a local window, reducing computational complexity. "
    "Global attention layers, by contrast, allow every token to attend to every other token in the sequence. "
    "NoPE, or No Positional Embedding, is a technique used in global attention layers to remove explicit position encodings. "
    "수학적으로 증명하면, 임의의 연속 함수에 대해 적분이 존재함을 보일 수 있다. "
    "언어 모델의 어텐션 메커니즘은 입력 시퀀스의 각 토큰 간 관계를 학습하는 핵심 구성 요소이다. "
    "어텐션 엔트로피는 각 레이어가 입력 토큰들에 얼마나 고르게 주목하는지를 나타내는 지표이다. "
    "높은 엔트로피는 많은 토큰에 분산된 어텐션을, 낮은 엔트로피는 특정 토큰에 집중된 어텐션을 의미한다. "
)

TARGET_LENGTHS = [128, 256, 512, 1024, 2048]


def build_input_of_length(tokenizer, target_len: int) -> torch.Tensor:
    """베이스 텍스트를 반복해서 정확히 target_len 토큰인 input_ids 생성"""
    # 충분히 긴 텍스트 생성
    repeated = BASE_TEXT * (target_len // 50 + 5)
    tokens = tokenizer(repeated, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids']

    if input_ids.shape[1] < target_len:
        raise ValueError(f"텍스트가 너무 짧음: {input_ids.shape[1]} < {target_len}")

    # 정확히 target_len으로 자름
    return input_ids[:, :target_len]


def compute_entropy(attn_weights: torch.Tensor) -> float:
    """attn_weights: (heads, seq, seq) → scalar mean entropy"""
    p = attn_weights.float().clamp(min=1e-10)
    entropy = -(p * p.log()).sum(dim=-1)  # (heads, seq)
    return entropy.mean().item()


def compute_attention_distance(attn_weights: torch.Tensor) -> float:
    """평균 attention 거리 (query와 key 위치 차이의 기대값)"""
    seq_len = attn_weights.shape[-1]
    positions = torch.arange(seq_len, device=attn_weights.device).float()
    q_pos = positions.unsqueeze(-1)
    k_pos = positions.unsqueeze(0)
    dist_matrix = (q_pos - k_pos).abs()
    weighted = (attn_weights.float() * dist_matrix.unsqueeze(0)).sum(dim=-1)
    return weighted.mean().item()


def run():
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    swa_layers = get_swa_layer_indices(cfg)
    global_set = set(global_layers)

    print(f"Global (NoPE) layers: {global_layers}")
    print(f"SWA layers count: {len(swa_layers)}")
    print(f"Target lengths: {TARGET_LENGTHS}\n")

    model, tokenizer = load_model_and_tokenizer()

    all_results = []

    for target_len in TARGET_LENGTHS:
        print(f"\n{'='*50}")
        print(f"Running length = {target_len} tokens...")

        input_ids = build_input_of_length(tokenizer, target_len).to(model.device)
        actual_len = input_ids.shape[1]
        print(f"  Actual sequence length: {actual_len} tokens")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        attentions = outputs.attentions  # tuple of (1, heads, seq, seq)

        for layer_idx, attn in enumerate(attentions):
            attn = attn.squeeze(0)  # (heads, seq, seq)
            layer_type = 'global_nope' if layer_idx in global_set else 'swa'
            entropy = compute_entropy(attn)
            attn_dist = compute_attention_distance(attn)

            all_results.append({
                'seq_len': actual_len,
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'entropy': entropy,
                'attn_distance': attn_dist,
            })

        # 진행 상황 출력
        global_e = [r['entropy'] for r in all_results if r['seq_len'] == actual_len and r['layer_type'] == 'global_nope']
        swa_e = [r['entropy'] for r in all_results if r['seq_len'] == actual_len and r['layer_type'] == 'swa']
        print(f"  Global entropy: {np.mean(global_e):.4f} | SWA entropy: {np.mean(swa_e):.4f}")

        del outputs
        torch.cuda.empty_cache()

    # 결과 저장
    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved: {OUT_DIR}/results.jsonl")

    plot_entropy_vs_length(all_results, global_set)
    build_and_save_summary(all_results)


def plot_entropy_vs_length(results, global_set):
    # 길이별 × 레이어타입별 집계
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        data[r['seq_len']][r['layer_type']].append(r['entropy'])

    dist_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        dist_data[r['seq_len']][r['layer_type']].append(r['attn_distance'])

    lengths = sorted(data.keys())
    global_entropies = [np.mean(data[l]['global_nope']) for l in lengths]
    swa_entropies = [np.mean(data[l]['swa']) for l in lengths]
    global_dists = [np.mean(dist_data[l]['global_nope']) for l in lengths]
    swa_dists = [np.mean(dist_data[l]['swa']) for l in lengths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(lengths, global_entropies, 'o-', color='coral', label='Global (NoPE)', linewidth=2, markersize=8)
    ax1.plot(lengths, swa_entropies, 's-', color='steelblue', label='SWA (RoPE)', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('Mean Attention Entropy')
    ax1.set_title('Attention Entropy vs Sequence Length\nSWA vs Global (NoPE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(lengths)
    ax1.set_xticklabels([str(l) for l in lengths])

    ax2 = axes[1]
    ax2.plot(lengths, global_dists, 'o-', color='coral', label='Global (NoPE)', linewidth=2, markersize=8)
    ax2.plot(lengths, swa_dists, 's-', color='steelblue', label='SWA (RoPE)', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length (tokens)')
    ax2.set_ylabel('Mean Attention Distance (tokens)')
    ax2.set_title('Attention Distance vs Sequence Length\nSWA vs Global (NoPE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(lengths)
    ax2.set_xticklabels([str(l) for l in lengths])

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'entropy_vs_length.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/entropy_vs_length.png")

    # 레이어별 비교 (각 길이에서)
    plot_per_layer(results, global_set, lengths)


def plot_per_layer(results, global_set, lengths):
    n_layers = max(r['layer_idx'] for r in results) + 1
    fig, axes = plt.subplots(len(lengths), 1, figsize=(16, 4 * len(lengths)))
    if len(lengths) == 1:
        axes = [axes]

    for ax, target_len in zip(axes, lengths):
        layer_entropy = defaultdict(list)
        for r in results:
            if r['seq_len'] == target_len:
                layer_entropy[r['layer_idx']].append(r['entropy'])

        layers = list(range(n_layers))
        entropies = [np.mean(layer_entropy.get(i, [0])) for i in layers]
        colors = ['coral' if i in global_set else 'steelblue' for i in layers]

        for i, (e, c) in enumerate(zip(entropies, colors)):
            ax.bar(i, e, color=c, alpha=0.8, width=0.8)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Mean Entropy')
        ax.set_title(f'Seq Len = {target_len} tokens  (coral=Global/NoPE, blue=SWA/RoPE)')
        ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'entropy_by_layer_per_length.png', dpi=120)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/entropy_by_layer_per_length.png")


def build_and_save_summary(results):
    from collections import defaultdict
    by_len_type = defaultdict(lambda: {'entropy': [], 'attn_distance': []})
    for r in results:
        key = (r['seq_len'], r['layer_type'])
        by_len_type[key]['entropy'].append(r['entropy'])
        by_len_type[key]['attn_distance'].append(r['attn_distance'])

    summary = {}
    for (seq_len, layer_type), vals in sorted(by_len_type.items()):
        if seq_len not in summary:
            summary[seq_len] = {}
        summary[seq_len][layer_type] = {
            'entropy_mean': float(np.mean(vals['entropy'])),
            'entropy_std': float(np.std(vals['entropy'])),
            'attn_distance_mean': float(np.mean(vals['attn_distance'])),
        }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment 1b Summary ===")
    print(f"{'Length':>8}  {'Global Entropy':>16}  {'SWA Entropy':>14}  {'Global Dist':>12}  {'SWA Dist':>10}")
    print("-" * 68)
    for seq_len in sorted(summary.keys()):
        g = summary[seq_len].get('global_nope', {})
        s = summary[seq_len].get('swa', {})
        print(
            f"{seq_len:>8}  "
            f"{g.get('entropy_mean', 0):>14.4f}    "
            f"{s.get('entropy_mean', 0):>12.4f}    "
            f"{g.get('attn_distance_mean', 0):>10.1f}    "
            f"{s.get('attn_distance_mean', 0):>8.1f}"
        )
    print(f"\nSummary saved: {OUT_DIR}/summary.json")


if __name__ == '__main__':
    run()
