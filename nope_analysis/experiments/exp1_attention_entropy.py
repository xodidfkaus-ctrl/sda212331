"""
Experiment 1: Attention Entropy — SWA vs Global (NoPE)

연구 질문: NoPE Global 레이어는 SWA 레이어보다 더 넓게(균일하게) attention하는가?
- 높은 entropy = 많은 토큰에 고르게 attention
- 낮은 entropy = 특정 토큰에 집중

결과: outputs/exp1_attention_entropy/
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

from nope_analysis.loader import load_model_and_tokenizer, load_config, get_global_layer_indices, get_swa_layer_indices

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp1_attention_entropy')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 실험용 입력 텍스트 (다양한 길이/언어)
PROMPTS = [
    "인공지능 기술의 발전은 현대 사회에 많은 변화를 가져오고 있다. 특히 자연어 처리 분야에서의 혁신은",
    "The development of large language models has fundamentally changed how we approach natural language understanding.",
    "EXAONE 4.5는 LG AI Research에서 개발한 멀티모달 언어 모델로, 한국어 처리에 특화되어 있으며",
    "수학적으로 증명하면, 임의의 연속 함수 f: [a,b] → R에 대해 리만 적분이 존재함을 보일 수 있다.",
]


def compute_entropy(attn_weights: torch.Tensor) -> float:
    """attn_weights: (heads, seq, seq) → scalar mean entropy"""
    # 각 쿼리 위치에서의 분포 entropy
    p = attn_weights.float().clamp(min=1e-10)
    entropy = -(p * p.log()).sum(dim=-1)  # (heads, seq)
    return entropy.mean().item()


def compute_attention_distance(attn_weights: torch.Tensor) -> float:
    """평균 attention 거리 (query와 key 위치 차이의 기대값)"""
    seq_len = attn_weights.shape[-1]
    positions = torch.arange(seq_len, device=attn_weights.device).float()
    # (heads, seq_q, seq_k)
    q_pos = positions.unsqueeze(-1)  # (seq, 1)
    k_pos = positions.unsqueeze(0)   # (1, seq)
    dist_matrix = (q_pos - k_pos).abs()  # (seq, seq)
    # weighted average distance
    weighted = (attn_weights.float() * dist_matrix.unsqueeze(0)).sum(dim=-1)  # (heads, seq)
    return weighted.mean().item()


def run():
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    swa_layers = get_swa_layer_indices(cfg)
    print(f"Global (NoPE) layers: {global_layers}")
    print(f"SWA layers (first 10): {swa_layers[:10]}...")

    model, tokenizer = load_model_and_tokenizer()

    results = []

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\nPrompt {prompt_idx+1}/{len(PROMPTS)}: {prompt[:50]}...")
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        seq_len = inputs['input_ids'].shape[1]
        print(f"  Sequence length: {seq_len} tokens")

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
        attentions = outputs.attentions  # len = num_layers

        for layer_idx, attn in enumerate(attentions):
            attn = attn.squeeze(0)  # (heads, seq, seq)
            layer_type = 'global_nope' if layer_idx in global_layers else 'swa'
            entropy = compute_entropy(attn)
            attn_dist = compute_attention_distance(attn)

            results.append({
                'prompt_idx': prompt_idx,
                'seq_len': seq_len,
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'entropy': entropy,
                'attn_distance': attn_dist,
            })

        del outputs
        torch.cuda.empty_cache()

    # 저장
    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    plot_results(results, global_layers, swa_layers)
    print_summary(results)


def plot_results(results, global_layers, swa_layers):
    # 레이어별 평균 entropy (전체 프롬프트 평균)
    from collections import defaultdict
    layer_entropy = defaultdict(list)
    layer_dist = defaultdict(list)
    for r in results:
        layer_entropy[r['layer_idx']].append(r['entropy'])
        layer_dist[r['layer_idx']].append(r['attn_distance'])

    n_layers = max(layer_entropy.keys()) + 1
    layers = list(range(n_layers))
    entropies = [np.mean(layer_entropy[i]) for i in layers]
    dists = [np.mean(layer_dist[i]) for i in layers]

    global_set = set(global_layers)
    colors = ['coral' if i in global_set else 'steelblue' for i in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    for i, (e, c) in enumerate(zip(entropies, colors)):
        ax1.bar(i, e, color=c, alpha=0.8, width=0.8)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Mean Attention Entropy')
    ax1.set_title('EXAONE 4.5 — Attention Entropy by Layer\n(coral=Global/NoPE, blue=SWA/RoPE)')
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (d, c) in enumerate(zip(dists, colors)):
        ax2.bar(i, d, color=c, alpha=0.8, width=0.8)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Attention Distance (tokens)')
    ax2.set_title('EXAONE 4.5 — Mean Attention Distance by Layer')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'entropy_by_layer.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/entropy_by_layer.png")


def print_summary(results):
    global_e = [r['entropy'] for r in results if r['layer_type'] == 'global_nope']
    swa_e = [r['entropy'] for r in results if r['layer_type'] == 'swa']
    global_d = [r['attn_distance'] for r in results if r['layer_type'] == 'global_nope']
    swa_d = [r['attn_distance'] for r in results if r['layer_type'] == 'swa']

    summary = {
        'global_nope': {
            'entropy_mean': float(np.mean(global_e)),
            'entropy_std': float(np.std(global_e)),
            'attn_distance_mean': float(np.mean(global_d)),
        },
        'swa': {
            'entropy_mean': float(np.mean(swa_e)),
            'entropy_std': float(np.std(swa_e)),
            'attn_distance_mean': float(np.mean(swa_d)),
        }
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment 1 Summary ===")
    print(f"Global (NoPE) entropy: {summary['global_nope']['entropy_mean']:.4f} ± {summary['global_nope']['entropy_std']:.4f}")
    print(f"SWA         entropy: {summary['swa']['entropy_mean']:.4f} ± {summary['swa']['entropy_std']:.4f}")
    print(f"Global attn distance: {summary['global_nope']['attn_distance_mean']:.1f} tokens")
    print(f"SWA    attn distance: {summary['swa']['attn_distance_mean']:.1f} tokens")


if __name__ == '__main__':
    run()
