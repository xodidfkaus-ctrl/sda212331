"""
Experiment 4: Long Context Attention Analysis (4096 토큰 초과 구간)

연구 질문 (RQ3): SWA 윈도우(4096)를 넘는 입력에서 Global과 SWA의 attention 패턴이
                 어떻게 달라지는가?

핵심 관찰 포인트:
  - 4096 토큰 초과 시 SWA는 초반 토큰들을 더 이상 볼 수 없음
  - Global은 여전히 전체 시퀀스 참조 가능
  - → Attention distance 곡선의 분기 지점이 4096 근방에서 나타나는지 확인

메모리 전략:
  전체 attention 행렬 저장 대신 hook으로 레이어별 통계만 추출 후 즉시 해제.
  - (1, 40, 8192, 8192) × bfloat16 = 5.4GB × 64레이어 = 346GB → 불가
  - hook으로 스칼라 통계만 저장 → MB 수준

입력 길이: 2048, 3072, 4096, 5120, 6144, 8192 토큰
결과: outputs/exp4_long_context/
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

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp4_long_context')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LENGTHS = [2048, 3072, 4096, 5120, 6144, 8192]
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
    "어텐션 엔트로피는 각 레이어가 입력 토큰들에 얼마나 고르게 주목하는지를 나타내는 지표이다. "
)


def build_input(tokenizer, target_len: int):
    repeated = BASE_TEXT * (target_len // 80 + 5)
    tokens = tokenizer(repeated, return_tensors='pt', add_special_tokens=True)
    ids = tokens['input_ids']
    if ids.shape[1] < target_len:
        raise ValueError(f"텍스트 부족: {ids.shape[1]} < {target_len}")
    return ids[:, :target_len]


def compute_entropy(attn: torch.Tensor) -> float:
    p = attn.float().clamp(min=1e-10)
    return (-(p * p.log()).sum(dim=-1)).mean().item()


def compute_attention_distance(attn: torch.Tensor) -> float:
    seq_len = attn.shape[-1]
    positions = torch.arange(seq_len, device=attn.device).float()
    dist_matrix = (positions.unsqueeze(-1) - positions.unsqueeze(0)).abs()
    return (attn.float() * dist_matrix.unsqueeze(0)).sum(dim=-1).mean().item()


def run_with_hooks(model, input_ids, global_set):
    """
    Hook으로 레이어별 attention 통계를 수집한다.
    Attention 행렬 자체는 저장하지 않아 메모리를 절약한다.
    """
    stats = {}  # layer_idx -> {'entropy': float, 'distance': float}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output[1] = attention weights (batch, heads, seq, seq) or None
            attn_w = output[1]
            if attn_w is not None:
                attn = attn_w.squeeze(0).detach()  # (heads, seq, seq)
                stats[layer_idx] = {
                    'entropy': compute_entropy(attn),
                    'distance': compute_attention_distance(attn),
                }
                del attn
            # None을 반환해서 attention 행렬이 output tuple에 축적되지 않게 함
            return (output[0], None) + output[2:]
        return hook

    # hook 등록
    handles = []
    try:
        layers = model.language_model.model.layers
    except AttributeError:
        layers = model.model.layers

    for i, layer in enumerate(layers):
        h = layer.self_attn.register_forward_hook(make_hook(i))
        handles.append(h)

    with torch.no_grad():
        model(input_ids=input_ids, output_attentions=True)

    for h in handles:
        h.remove()

    return stats


def run():
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    global_set = set(global_layers)
    swa_set = set(get_swa_layer_indices(cfg))
    print(f"Global layers: {global_layers}")
    print(f"SWA window: {SWA_WINDOW} tokens")
    print(f"Target lengths: {TARGET_LENGTHS}\n")

    model, tokenizer = load_model_and_tokenizer()

    all_results = []

    for target_len in TARGET_LENGTHS:
        print(f"\n{'='*50}")
        print(f"Length = {target_len} tokens  {'[SWA 윈도우 초과]' if target_len > SWA_WINDOW else '[SWA 윈도우 내]'}")

        input_ids = build_input(tokenizer, target_len).to(model.device)
        print(f"  Forward pass with hooks...")

        layer_stats = run_with_hooks(model, input_ids, global_set)

        for layer_idx, s in layer_stats.items():
            layer_type = 'global_nope' if layer_idx in global_set else 'swa'
            all_results.append({
                'seq_len': target_len,
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'entropy': s['entropy'],
                'attn_distance': s['distance'],
                'exceeds_swa_window': target_len > SWA_WINDOW,
            })

        g_e = np.mean([s['entropy'] for i, s in layer_stats.items() if i in global_set])
        s_e = np.mean([s['entropy'] for i, s in layer_stats.items() if i in swa_set])
        g_d = np.mean([s['distance'] for i, s in layer_stats.items() if i in global_set])
        s_d = np.mean([s['distance'] for i, s in layer_stats.items() if i in swa_set])
        print(f"  Global  entropy={g_e:.4f}  distance={g_d:.1f}")
        print(f"  SWA     entropy={s_e:.4f}  distance={s_d:.1f}")

        del input_ids
        torch.cuda.empty_cache()

    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')

    plot_results(all_results, global_set)
    save_summary(all_results)


def plot_results(results, global_set):
    lengths = sorted(set(r['seq_len'] for r in results))

    # 1) entropy vs length
    g_ent = [np.mean([r['entropy'] for r in results if r['seq_len'] == l and r['layer_type'] == 'global_nope']) for l in lengths]
    s_ent = [np.mean([r['entropy'] for r in results if r['seq_len'] == l and r['layer_type'] == 'swa']) for l in lengths]
    g_dist = [np.mean([r['attn_distance'] for r in results if r['seq_len'] == l and r['layer_type'] == 'global_nope']) for l in lengths]
    s_dist = [np.mean([r['attn_distance'] for r in results if r['seq_len'] == l and r['layer_type'] == 'swa']) for l in lengths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, g_vals, s_vals, ylabel, title in [
        (ax1, g_ent, s_ent, 'Mean Attention Entropy', 'Entropy vs Sequence Length'),
        (ax2, g_dist, s_dist, 'Mean Attention Distance (tokens)', 'Distance vs Sequence Length'),
    ]:
        ax.plot(lengths, g_vals, 'o-', color='coral', label='Global (NoPE)', linewidth=2, markersize=8)
        ax.plot(lengths, s_vals, 's-', color='steelblue', label='SWA (RoPE)', linewidth=2, markersize=8)
        ax.axvline(SWA_WINDOW, color='gray', linestyle='--', alpha=0.8, label=f'SWA window ({SWA_WINDOW})')
        ax.set_xlabel('Sequence Length (tokens)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Exp 4: Long Context — SWA vs Global beyond 4096 tokens', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'long_context_entropy_distance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/long_context_entropy_distance.png")

    # 2) attention distance gap (Global - SWA) vs length
    dist_gap = [g - s for g, s in zip(g_dist, s_dist)]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['tomato' if l > SWA_WINDOW else 'steelblue' for l in lengths]
    ax.bar(range(len(lengths)), dist_gap, color=colors, alpha=0.8)
    ax.axvline(lengths.index(SWA_WINDOW) - 0.5 if SWA_WINDOW in lengths else 2.5,
               color='gray', linestyle='--', alpha=0.7, label=f'SWA window')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(l) for l in lengths])
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Global Distance − SWA Distance (tokens)')
    ax.set_title('Attention Distance Gap: Global vs SWA\n(red=exceeds SWA window,양수=Global이 더 멀리 봄)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'distance_gap.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/distance_gap.png")


def save_summary(results):
    lengths = sorted(set(r['seq_len'] for r in results))
    summary = {}
    for tlen in lengths:
        summary[tlen] = {}
        for ltype in ['global_nope', 'swa']:
            sub = [r for r in results if r['seq_len'] == tlen and r['layer_type'] == ltype]
            summary[tlen][ltype] = {
                'entropy_mean': float(np.mean([r['entropy'] for r in sub])),
                'attn_distance_mean': float(np.mean([r['attn_distance'] for r in sub])),
            }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Experiment 4 Summary ===")
    print(f"{'Length':>8}  {'Global Dist':>12}  {'SWA Dist':>10}  {'Gap':>8}  {'Note'}")
    print("-" * 60)
    for tlen in sorted(summary.keys()):
        g = summary[tlen]['global_nope']
        s = summary[tlen]['swa']
        gap = g['attn_distance_mean'] - s['attn_distance_mean']
        note = '← SWA 윈도우 초과' if tlen > SWA_WINDOW else ''
        print(f"{tlen:>8}  {g['attn_distance_mean']:>12.1f}  {s['attn_distance_mean']:>10.1f}  {gap:>8.1f}  {note}")
    print(f"\nSummary saved: {OUT_DIR}/summary.json")


if __name__ == '__main__':
    run()
