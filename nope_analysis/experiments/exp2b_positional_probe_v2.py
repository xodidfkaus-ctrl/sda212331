"""
Experiment 2b: Positional Probing v2 — train/test split, 다양한 길이/도메인

Exp 2의 한계:
  - 동일 데이터로 학습+평가 → accuracy=1.0은 과적합일 수 있음
  - 짧은 문장(14~26 토큰)만 사용
  - 도메인 다양성 없음

개선:
  - 30개 프롬프트 (한국어/영어/수학/코드)
  - 64 / 128 / 256 토큰 길이로 고정
  - 24 train / 6 test 분리 (80/20)
  - train acc vs test acc 비교로 과적합 탐지
  - 도메인별 breakdown

결과: outputs/exp2b_positional_probe_v2/
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

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp2b_positional_probe_v2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 30개 프롬프트 (도메인별)
PROMPTS = {
    'korean': [
        "인공지능 기술의 발전은 현대 사회에 많은 변화를 가져오고 있으며 특히 자연어 처리 분야에서의 혁신은 인간과 기계 사이의 소통 방식을 근본적으로 바꾸고 있다.",
        "기후 변화는 지구 환경에 심각한 영향을 미치고 있으며 국제적인 협력과 개인의 노력이 함께 필요한 시대적 과제로 떠오르고 있다.",
        "대한민국의 교육 시스템은 높은 학업 성취도를 자랑하지만 동시에 학생들의 창의성과 자기 주도 학습 능력을 키우는 데 있어서 개선이 필요하다는 지적도 있다.",
        "반도체 산업은 현대 경제의 핵심 축으로 자리잡았으며 글로벌 공급망의 안정성 확보가 각국 정부의 중요한 정책 과제가 되었다.",
        "전통 문화와 현대 문화가 공존하는 한국 사회에서는 세대 간의 가치관 차이가 다양한 사회적 논의를 이끌어 내고 있다.",
        "의료 기술의 발전으로 인해 인간의 평균 수명이 크게 늘어났지만 노령화 사회에 따른 복지 비용 증가와 경제 활력 저하는 새로운 도전 과제를 제시한다.",
        "우주 탐사 기술은 민간 기업의 참여로 새로운 전기를 맞이하고 있으며 화성 식민지 건설이라는 인류의 오랜 꿈이 점점 현실에 가까워지고 있다.",
        "언어는 단순한 소통의 도구를 넘어 문화와 사고방식을 담아내는 그릇으로서 한 민족의 정체성을 형성하는 데 중요한 역할을 한다.",
        "경제적 불평등 문제를 해결하기 위해 다양한 정책적 접근이 시도되고 있으며 기본 소득 제도의 도입 가능성에 대한 논의가 활발히 진행되고 있다.",
        "디지털 전환 시대에 데이터 보안과 개인 정보 보호는 기업과 개인 모두에게 중요한 과제로 부상하고 있으며 관련 법제도의 정비가 시급하다.",
    ],
    'english': [
        "The rapid advancement of machine learning has transformed industries ranging from healthcare to finance, enabling systems to process and analyze data at unprecedented scales.",
        "Climate change represents one of the most pressing challenges of our time, requiring coordinated global action to reduce greenhouse gas emissions and develop sustainable energy alternatives.",
        "The philosophy of mind has long grappled with the hard problem of consciousness, seeking to understand how subjective experience arises from physical processes in the brain.",
        "Modern cryptography relies on mathematical problems that are computationally intractable, ensuring that digital communications remain secure against unauthorized access.",
        "The human genome project revolutionized our understanding of genetics and has paved the way for personalized medicine and new treatments for previously incurable diseases.",
        "Urban planning in the twenty-first century must balance economic development with environmental sustainability and social equity to create livable cities for future generations.",
        "The development of quantum computing promises to solve problems that are intractable for classical computers, with potential applications in drug discovery and materials science.",
        "Literature serves as a mirror reflecting the values, struggles, and aspirations of society, allowing readers to explore diverse perspectives and cultivate empathy across cultural boundaries.",
        "The emergence of social media platforms has fundamentally altered how information spreads, creating both unprecedented opportunities for connection and serious risks of misinformation.",
        "Renewable energy technologies such as solar and wind power have achieved cost parity with fossil fuels in many markets, accelerating the global transition to clean energy.",
    ],
    'math': [
        "The fundamental theorem of calculus establishes a profound connection between differentiation and integration, showing that these two operations are essentially inverse to each other.",
        "Prime numbers have fascinated mathematicians for millennia and their distribution follows patterns described by the Riemann hypothesis which remains one of the greatest unsolved problems.",
        "Linear algebra provides the mathematical foundation for modern machine learning with matrix operations enabling efficient computation of neural network forward and backward passes.",
        "The concept of limits is central to analysis allowing us to rigorously define continuity differentiability and the behavior of functions as they approach particular values.",
        "Graph theory studies the properties of networks and finds applications in computer science biology social sciences and operations research among many other fields.",
    ],
    'code': [
        "A recursive function calls itself with modified parameters until a base case is reached making it an elegant solution for problems with inherently recursive structure like tree traversal.",
        "Object oriented programming organizes code around objects that encapsulate data and behavior promoting modularity reusability and maintainability in large software systems.",
        "Asynchronous programming allows applications to perform multiple operations concurrently without blocking execution enabling more responsive user interfaces and efficient use of system resources.",
        "Database normalization reduces data redundancy and improves integrity by organizing tables and relationships according to a set of formal rules called normal forms.",
        "The time complexity of an algorithm describes how its running time grows with input size using big O notation to characterize worst case behavior.",
    ],
}

# 플랫 리스트로 변환 (도메인 레이블 포함)
ALL_PROMPTS = []
PROMPT_DOMAINS = []
for domain, texts in PROMPTS.items():
    for text in texts:
        ALL_PROMPTS.append(text)
        PROMPT_DOMAINS.append(domain)

TARGET_LENGTHS = [64, 128, 256]
TRAIN_RATIO = 0.8
N_BINS = 10


def build_input_of_length(tokenizer, text: str, target_len: int):
    """텍스트를 반복해서 정확히 target_len 토큰으로 만듦"""
    repeated = (text + " ") * (target_len // 20 + 5)
    tokens = tokenizer(repeated, return_tensors='pt', add_special_tokens=True)
    ids = tokens['input_ids']
    if ids.shape[1] < target_len:
        return None
    return ids[:, :target_len]


def extract_hidden_states_for_prompt(model, tokenizer, text: str, target_len: int):
    """단일 프롬프트에서 레이어별 hidden states 추출"""
    input_ids = build_input_of_length(tokenizer, text, target_len)
    if input_ids is None:
        return None
    input_ids = input_ids.to(model.device)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    result = {}
    for layer_idx, hidden in enumerate(outputs.hidden_states[1:], start=0):
        h = hidden.squeeze(0).cpu().float()  # (seq, hidden)
        pos = torch.arange(seq_len).float() / (seq_len - 1)
        result[layer_idx] = (h, pos)

    del outputs
    torch.cuda.empty_cache()
    return result


def run_probe(train_data, test_data, n_bins=N_BINS):
    """
    train_data: list of (hidden, pos) tuples
    test_data: list of (hidden, pos) tuples
    Returns: (train_acc, test_acc)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def to_xy(data):
        X, y = [], []
        for h, pos in data:
            labels = (pos * n_bins).long().clamp(0, n_bins - 1)
            X.append(h.numpy())
            y.extend(labels.numpy().tolist())
        return np.vstack(X), np.array(y)

    X_train, y_train = to_xy(train_data)
    X_test, y_test = to_xy(test_data)

    if len(np.unique(y_train)) < 2:
        return 0.0, 0.0

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 샘플 수 제한
    if len(X_train_s) > 3000:
        idx = np.random.choice(len(X_train_s), 3000, replace=False)
        X_train_s, y_train = X_train_s[idx], y_train[idx]
    if len(X_test_s) > 1000:
        idx = np.random.choice(len(X_test_s), 1000, replace=False)
        X_test_s, y_test = X_test_s[idx], y_test[idx]

    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(X_train_s, y_train)
    return clf.score(X_train_s, y_train), clf.score(X_test_s, y_test)


def run():
    np.random.seed(42)
    cfg = load_config()
    global_set = set(get_global_layer_indices(cfg))
    n_layers = 64

    # train/test split
    n_total = len(ALL_PROMPTS)
    n_train = int(n_total * TRAIN_RATIO)
    idx = np.random.permutation(n_total)
    train_idx = idx[:n_train].tolist()
    test_idx = idx[n_train:].tolist()
    print(f"Prompts: {n_total} total, {n_train} train, {len(test_idx)} test")
    print(f"Lengths: {TARGET_LENGTHS}")

    model, tokenizer = load_model_and_tokenizer()

    all_results = []

    for target_len in TARGET_LENGTHS:
        print(f"\n{'='*50}")
        print(f"Length = {target_len} tokens")

        # 각 프롬프트별 hidden states 수집
        train_hidden = defaultdict(list)  # layer_idx -> list of (h, pos)
        test_hidden = defaultdict(list)

        print(f"  Extracting train ({n_train} prompts)...")
        for i in train_idx:
            hs = extract_hidden_states_for_prompt(model, tokenizer, ALL_PROMPTS[i], target_len)
            if hs is None:
                continue
            for layer_idx, (h, pos) in hs.items():
                train_hidden[layer_idx].append((h, pos))

        print(f"  Extracting test ({len(test_idx)} prompts)...")
        for i in test_idx:
            hs = extract_hidden_states_for_prompt(model, tokenizer, ALL_PROMPTS[i], target_len)
            if hs is None:
                continue
            for layer_idx, (h, pos) in hs.items():
                test_hidden[layer_idx].append((h, pos))

        print(f"  Running probes per layer...")
        for layer_idx in sorted(train_hidden.keys()):
            layer_type = 'global_nope' if layer_idx in global_set else 'swa'
            train_acc, test_acc = run_probe(train_hidden[layer_idx], test_hidden[layer_idx])
            all_results.append({
                'seq_len': target_len,
                'layer_idx': layer_idx,
                'layer_type': layer_type,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfit_gap': train_acc - test_acc,
            })

        # 진행 요약
        g_test = [r['test_acc'] for r in all_results if r['seq_len'] == target_len and r['layer_type'] == 'global_nope']
        s_test = [r['test_acc'] for r in all_results if r['seq_len'] == target_len and r['layer_type'] == 'swa']
        g_train = [r['train_acc'] for r in all_results if r['seq_len'] == target_len and r['layer_type'] == 'global_nope']
        s_train = [r['train_acc'] for r in all_results if r['seq_len'] == target_len and r['layer_type'] == 'swa']
        print(f"  Global — train: {np.mean(g_train):.3f}, test: {np.mean(g_test):.3f}")
        print(f"  SWA    — train: {np.mean(s_train):.3f}, test: {np.mean(s_test):.3f}")

    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')

    plot_results(all_results, global_set, n_layers)
    save_summary(all_results)


def plot_results(results, global_set, n_layers):
    lengths = sorted(set(r['seq_len'] for r in results))

    # 1) 길이별 train/test accuracy (Global vs SWA)
    fig, axes = plt.subplots(1, len(lengths), figsize=(6 * len(lengths), 5))
    if len(lengths) == 1:
        axes = [axes]
    for ax, tlen in zip(axes, lengths):
        subset = [r for r in results if r['seq_len'] == tlen]
        layers = [r['layer_idx'] for r in subset]
        train_accs = [r['train_acc'] for r in subset]
        test_accs = [r['test_acc'] for r in subset]
        colors = ['coral' if r['layer_type'] == 'global_nope' else 'steelblue' for r in subset]

        ax.scatter(layers, train_accs, c=colors, marker='o', alpha=0.6, s=30, label='train')
        ax.scatter(layers, test_accs, c=colors, marker='x', alpha=0.9, s=50, label='test')
        ax.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='random (0.1)')
        ax.set_title(f'seq_len={tlen}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probe Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Exp 2b: Positional Probe (train vs test)\n(coral=Global/NoPE, blue=SWA, o=train, x=test)', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'probe_train_test_by_layer.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/probe_train_test_by_layer.png")

    # 2) 길이별 test accuracy 추세 (Global vs SWA)
    fig, ax = plt.subplots(figsize=(8, 5))
    g_test = [np.mean([r['test_acc'] for r in results if r['seq_len'] == l and r['layer_type'] == 'global_nope']) for l in lengths]
    s_test = [np.mean([r['test_acc'] for r in results if r['seq_len'] == l and r['layer_type'] == 'swa']) for l in lengths]
    ax.plot(lengths, g_test, 'o-', color='coral', label='Global (NoPE)', linewidth=2, markersize=8)
    ax.plot(lengths, s_test, 's-', color='steelblue', label='SWA (RoPE)', linewidth=2, markersize=8)
    ax.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='random baseline')
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Mean Test Accuracy')
    ax.set_title('Positional Probe Test Accuracy vs Length\n(train/test split, 30 prompts)')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'test_acc_vs_length.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/test_acc_vs_length.png")


def save_summary(results):
    from collections import defaultdict
    summary = {}
    lengths = sorted(set(r['seq_len'] for r in results))
    for tlen in lengths:
        summary[tlen] = {}
        for ltype in ['global_nope', 'swa']:
            sub = [r for r in results if r['seq_len'] == tlen and r['layer_type'] == ltype]
            summary[tlen][ltype] = {
                'train_acc_mean': float(np.mean([r['train_acc'] for r in sub])),
                'test_acc_mean': float(np.mean([r['test_acc'] for r in sub])),
                'overfit_gap_mean': float(np.mean([r['overfit_gap'] for r in sub])),
            }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Experiment 2b Summary ===")
    print(f"{'Length':>8}  {'Type':>12}  {'Train Acc':>10}  {'Test Acc':>10}  {'Overfit Gap':>12}")
    print("-" * 60)
    for tlen in sorted(summary.keys()):
        for ltype in ['global_nope', 'swa']:
            s = summary[tlen][ltype]
            print(f"{tlen:>8}  {ltype:>12}  {s['train_acc_mean']:>10.3f}  {s['test_acc_mean']:>10.3f}  {s['overfit_gap_mean']:>12.3f}")
    print(f"\nSummary saved: {OUT_DIR}/summary.json")


if __name__ == '__main__':
    run()
