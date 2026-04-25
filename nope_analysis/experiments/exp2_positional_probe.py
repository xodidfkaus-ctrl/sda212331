"""
Experiment 2: Positional Probing — NoPE 레이어가 위치 정보를 암묵적으로 인코딩하는가?

연구 질문: 위치 임베딩이 없는데도 Global 레이어 hidden state에서 토큰 위치를 예측할 수 있는가?
- 예측 가능 → 위치 정보가 암묵적으로 인코딩됨 (구조적 귀납 편향)
- 예측 불가 → 진정한 위치 무관 표현

방법: 레이어별 hidden state에 linear probe (logistic regression) 학습

결과: outputs/exp2_positional_probe/
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

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp2_positional_probe')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "인공지능과 머신러닝은 현대 기술의 핵심 분야로, 다양한 응용 프로그램에서 활용되고 있다.",
    "Large language models have demonstrated remarkable capabilities across a wide range of natural language processing tasks.",
    "수학, 과학, 철학은 인류 지식의 근간을 이루는 학문 분야로서 서로 깊은 연관성을 가지고 있다.",
    "The quick brown fox jumps over the lazy dog, demonstrating all letters of the English alphabet.",
    "기후 변화는 지구 환경에 심각한 영향을 미치고 있으며, 국제적 협력이 필수적으로 요구된다.",
]


def extract_hidden_states(model, tokenizer, prompts):
    """레이어별 hidden states 수집."""
    all_hidden = defaultdict(list)  # layer_idx -> list of (hidden, position_label)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        seq_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=False)

        # outputs.hidden_states: tuple of (batch, seq, hidden) — layer 0 = embedding
        for layer_idx, hidden in enumerate(outputs.hidden_states[1:], start=0):  # skip embedding
            h = hidden.squeeze(0).cpu().float()  # (seq, hidden)
            positions = torch.arange(seq_len).float()
            # normalize positions to [0, 1]
            pos_normalized = positions / (seq_len - 1) if seq_len > 1 else positions
            all_hidden[layer_idx].append((h, pos_normalized))

        del outputs
        torch.cuda.empty_cache()

    return all_hidden


def linear_probe_accuracy(hidden_states_list, n_bins=10):
    """
    Linear probe: 위치를 n_bins 구간으로 분류.
    Returns: accuracy (0~1)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = [], []
    for h, pos in hidden_states_list:
        labels = (pos * n_bins).long().clamp(0, n_bins - 1)
        X.append(h.numpy())
        y.extend(labels.numpy().tolist())

    X = np.vstack(X)
    y = np.array(y)

    # 클래스 균형 확인
    if len(np.unique(y)) < 2:
        return 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 데이터가 많으면 샘플링
    if len(X_scaled) > 2000:
        idx = np.random.choice(len(X_scaled), 2000, replace=False)
        X_scaled, y = X_scaled[idx], y[idx]

    clf = LogisticRegression(max_iter=200, C=1.0, multi_class='auto')
    clf.fit(X_scaled, y)
    return clf.score(X_scaled, y)


def run():
    cfg = load_config()
    global_layers = set(get_global_layer_indices(cfg))
    swa_layers = set(get_swa_layer_indices(cfg))

    model, tokenizer = load_model_and_tokenizer()

    print("Extracting hidden states from all layers...")
    all_hidden = extract_hidden_states(model, tokenizer, PROMPTS)

    print("Running linear probes per layer...")
    results = []
    for layer_idx in sorted(all_hidden.keys()):
        layer_type = 'global_nope' if layer_idx in global_layers else 'swa'
        acc = linear_probe_accuracy(all_hidden[layer_idx])
        results.append({
            'layer_idx': layer_idx,
            'layer_type': layer_type,
            'probe_accuracy': acc,
        })
        print(f"  Layer {layer_idx:2d} ({layer_type:12s}): accuracy = {acc:.3f}")

    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    plot_results(results, global_layers)
    print_summary(results, global_layers)


def plot_results(results, global_layers):
    layers = [r['layer_idx'] for r in results]
    accs = [r['probe_accuracy'] for r in results]
    colors = ['coral' if r['layer_type'] == 'global_nope' else 'steelblue' for r in results]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(layers, accs, color=colors, alpha=0.8, width=0.8)
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Random baseline (10 bins)')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Linear Probe Accuracy')
    ax.set_title('EXAONE 4.5 — Positional Information in Hidden States\n(coral=Global/NoPE, blue=SWA/RoPE)')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'probe_accuracy_by_layer.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/probe_accuracy_by_layer.png")


def print_summary(results, global_layers):
    global_acc = [r['probe_accuracy'] for r in results if r['layer_type'] == 'global_nope']
    swa_acc = [r['probe_accuracy'] for r in results if r['layer_type'] == 'swa']
    summary = {
        'global_nope': {'mean_accuracy': float(np.mean(global_acc)), 'std': float(np.std(global_acc))},
        'swa': {'mean_accuracy': float(np.mean(swa_acc)), 'std': float(np.std(swa_acc))},
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment 2 Summary ===")
    print(f"Global (NoPE) probe accuracy: {summary['global_nope']['mean_accuracy']:.3f} ± {summary['global_nope']['std']:.3f}")
    print(f"SWA         probe accuracy: {summary['swa']['mean_accuracy']:.3f} ± {summary['swa']['std']:.3f}")
    print("(Random baseline = 0.1 for 10-class position bins)")


if __name__ == '__main__':
    run()
