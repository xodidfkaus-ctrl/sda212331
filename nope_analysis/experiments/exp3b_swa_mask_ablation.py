"""
Experiment 3b: SWA Mask Ablation — 방법론 개선판

실험 목적:
  Exp 3 (zero-output ablation)의 설계 문제를 수정:
    - Exp 3: Global 레이어의 attn_output = 0 → attention 기여 자체를 제거 (너무 강함)
    - Exp 3b: Global 레이어에 SWA 윈도우 마스크를 강제 주입
             → "Global을 일반 SWA처럼 동작시킴"
             → NoPE(위치 무관 전역 attention)의 효과만 순수 분리

방법:
  register_forward_pre_hook(with_kwargs=True)로 attention_mask를 주입.
  EXAONE의 self_attn은 (query, key, value, attention_mask, ...) 를 받음.
  SWA 마스크: 현재 토큰에서 최대 SWA_WINDOW 이전 토큰만 볼 수 있는 causal band mask.

결과: outputs/exp3b_swa_mask_ablation/

비교 대상:
  - Exp 3 결과 (outputs/exp3_swa_ablation/summary.json) 와 병기하면
    zero-output vs SWA-mask 두 방법의 차이도 확인 가능
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
from nope_analysis.corpus.downloader import build_input_from_corpus
from nope_analysis.analysis.statistical_tests import compare_groups, report_stats, save_stats_report

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp3b_swa_mask_ablation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LENGTHS = [1024, 2048, 3072, 4096, 5120, 6144]
SWA_WINDOW = 4096


def build_swa_causal_mask(seq_len: int, window: int, device) -> torch.Tensor:
    """
    Causal + sliding-window additive mask.
    값: 0 (허용) / -inf (차단)
    Shape: (1, 1, seq_len, seq_len)  — broadcast over batch and heads
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq)
    # causal: j <= i,  window: i - j < window
    allowed = (j <= i) & ((i - j) < window)
    mask = torch.where(allowed, torch.zeros(1, device=device), torch.full((1,), float('-inf'), device=device))
    return mask.view(1, 1, seq_len, seq_len)


def register_swa_mask_hooks(model, global_layers, seq_len: int, device):
    """
    Global 레이어의 forward 직전에 attention_mask를 SWA 마스크로 교체.

    EXAONE self_attn forward 시그니처에서 attention_mask는 kwargs로 전달됨.
    with_kwargs=True 훅으로 kwargs를 직접 수정한다.
    """
    swa_mask = build_swa_causal_mask(seq_len, SWA_WINDOW, device)
    global_set = set(global_layers)
    handles = []

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            # attention_mask가 kwargs에 있으면 교체, args에 있으면 교체
            if 'attention_mask' in kwargs:
                kwargs['attention_mask'] = swa_mask
            elif len(args) >= 4:
                args = args[:3] + (swa_mask,) + args[4:]
            return args, kwargs
        return hook

    try:
        layers = model.language_model.model.layers
    except AttributeError:
        layers = model.model.layers

    for i in global_set:
        if i < len(layers):
            h = layers[i].self_attn.register_forward_pre_hook(make_hook(i), with_kwargs=True)
            handles.append(h)

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def compute_perplexity(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    nll = outputs.loss.item()
    return float(np.exp(nll)), nll


def run():
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    print(f"Global (NoPE) layers: {global_layers}")
    print(f"SWA window: {SWA_WINDOW} tokens")
    print(f"Target lengths: {TARGET_LENGTHS}\n")
    print("Method: SWA mask injection (replaces Global NoPE attention mask with SWA band mask)")
    print("This converts Global layers to behave like SWA, isolating the NoPE long-range effect.\n")

    model, tokenizer = load_model_and_tokenizer()
    device = model.device

    results = []

    for target_len in TARGET_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Length = {target_len} tokens  {'[SWA 윈도우 초과]' if target_len > SWA_WINDOW else '[SWA 윈도우 내]'}")

        # Real corpus input (mixed en/ko)
        try:
            input_ids = build_input_from_corpus('en', target_len, tokenizer).to(device)
            lang_used = 'en (WikiText-103)'
        except Exception as e:
            print(f"  [warn] English corpus failed ({e}), trying Korean...")
            try:
                input_ids = build_input_from_corpus('ko', target_len, tokenizer).to(device)
                lang_used = 'ko (KLUE-MRC)'
            except Exception as e2:
                print(f"  [warn] Korean corpus also failed ({e2}), using fallback BASE_TEXT")
                BASE_TEXT = (
                    "The development of large language models has fundamentally changed natural language processing. "
                    "These models learn from vast amounts of text and can generate coherent contextually appropriate responses. "
                    "The hybrid attention mechanism combining sliding window attention and global attention is a key innovation. "
                    "Sliding window attention limits each token to attending only within a local window of fixed size. "
                    "Global attention layers allow every token to attend to every other token in the full sequence. "
                    "NoPE removes explicit position encodings from global attention layers in transformer models. "
                    "인공지능 기술의 발전은 현대 사회에 많은 변화를 가져오고 있다. "
                    "특히 자연어 처리 분야에서의 혁신은 인간과 기계 사이의 소통 방식을 근본적으로 바꾸고 있다. "
                )
                repeated = BASE_TEXT * (target_len // 60 + 10)
                ids = tokenizer(repeated, return_tensors='pt', add_special_tokens=True)['input_ids']
                input_ids = ids[:, :target_len].to(device)
                lang_used = 'fallback'

        print(f"  Input: {input_ids.shape[1]} tokens, corpus={lang_used}")

        # Condition A: baseline
        ppl_base, nll_base = compute_perplexity(model, input_ids)
        print(f"  Baseline   PPL: {ppl_base:.4f}  NLL: {nll_base:.4f}")

        # Condition B: SWA mask injected into Global layers
        handles = register_swa_mask_hooks(model, global_layers, target_len, device)
        ppl_swa, nll_swa = compute_perplexity(model, input_ids)
        remove_hooks(handles)
        print(f"  SWA-masked PPL: {ppl_swa:.4f}  NLL: {nll_swa:.4f}")

        delta_nll = nll_swa - nll_base
        delta_ppl = ppl_swa - ppl_base
        print(f"  ΔNLL: {delta_nll:+.4f}  ΔPPL: {delta_ppl:+.4f}  (양수 = NoPE Global의 장거리 기여 확인)")

        results.append({
            'seq_len': target_len,
            'corpus': lang_used,
            'ppl_baseline': ppl_base,
            'ppl_swa_masked': ppl_swa,
            'nll_baseline': nll_base,
            'nll_swa_masked': nll_swa,
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
    run_stats(results)


def plot_results(results):
    lengths = [r['seq_len'] for r in results]
    delta_ppls = [r['delta_ppl'] for r in results]
    delta_nlls = [r['delta_nll'] for r in results]
    colors = ['tomato' if r['exceeds_swa_window'] else 'steelblue' for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(len(lengths)), delta_ppls, color=colors, alpha=0.8)
    ax1.axhline(0, color='black', linewidth=0.8)
    try:
        idx = lengths.index(SWA_WINDOW)
        ax1.axvline(idx + 0.5, color='gray', linestyle='--', alpha=0.7, label=f'SWA window ({SWA_WINDOW})')
    except ValueError:
        pass
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels([str(l) for l in lengths])
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('ΔPPL (SWA-Masked − Baseline)')
    ax1.set_title('PPL Increase When Global→SWA Mask Applied\n(blue=within SWA window, red=exceeds window)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.plot(lengths, [r['ppl_baseline'] for r in results], 'o-', color='steelblue', label='Baseline (NoPE Global)', linewidth=2)
    ax2.plot(lengths, [r['ppl_swa_masked'] for r in results], 's--', color='tomato', label='SWA Mask Applied', linewidth=2)
    ax2.axvline(SWA_WINDOW, color='gray', linestyle='--', alpha=0.7, label=f'SWA window')
    ax2.set_xlabel('Sequence Length (tokens)')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Baseline vs SWA-Masked Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Exp 3b: SWA Mask Ablation — NoPE Global Long-range Contribution', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'swa_mask_ablation_perplexity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/swa_mask_ablation_perplexity.png")


def save_summary(results):
    within = [r for r in results if not r['exceeds_swa_window']]
    beyond = [r for r in results if r['exceeds_swa_window']]

    within_nll = float(np.mean([r['delta_nll'] for r in within])) if within else None
    beyond_nll = float(np.mean([r['delta_nll'] for r in beyond])) if beyond else None

    summary = {
        'method': 'swa_mask_injection',
        'description': 'Global layers forced to use SWA window mask; NoPE long-range contribution measured via PPL delta',
        'within_swa_window': {
            'lengths': [r['seq_len'] for r in within],
            'mean_delta_nll': within_nll,
            'mean_delta_ppl': float(np.mean([r['delta_ppl'] for r in within])) if within else None,
        },
        'beyond_swa_window': {
            'lengths': [r['seq_len'] for r in beyond],
            'mean_delta_nll': beyond_nll,
            'mean_delta_ppl': float(np.mean([r['delta_ppl'] for r in beyond])) if beyond else None,
        },
        'per_length': {str(r['seq_len']): {
            'ppl_baseline': r['ppl_baseline'],
            'ppl_swa_masked': r['ppl_swa_masked'],
            'delta_nll': r['delta_nll'],
            'delta_ppl': r['delta_ppl'],
            'corpus': r['corpus'],
        } for r in results},
    }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Experiment 3b Summary (SWA Mask Ablation) ===")
    print(f"{'Length':>8}  {'Baseline PPL':>14}  {'SWA-Masked PPL':>16}  {'ΔNLL':>8}  {'Note'}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['seq_len']):
        note = '← SWA 초과' if r['exceeds_swa_window'] else ''
        print(f"{r['seq_len']:>8}  {r['ppl_baseline']:>14.4f}  {r['ppl_swa_masked']:>16.4f}  {r['delta_nll']:>8.4f}  {note}")

    if within_nll is not None and beyond_nll is not None:
        print(f"\n  SWA 이내  평균 ΔNLL: {within_nll:.4f}")
        print(f"  SWA 초과  평균 ΔNLL: {beyond_nll:.4f}")
        ratio = beyond_nll / within_nll if within_nll != 0 else float('inf')
        print(f"  비율 (초과/이내): {ratio:.2f}x  (>1이면 NoPE가 장거리에서 더 중요함)")
    print(f"\nSummary saved: {OUT_DIR}/summary.json")


def run_stats(results):
    """
    Statistical test: does ΔNLL differ significantly between within vs beyond SWA window?
    Uses compare_groups from statistical_tests.py.
    """
    within = [r['delta_nll'] for r in results if not r['exceeds_swa_window']]
    beyond = [r['delta_nll'] for r in results if r['exceeds_swa_window']]

    if len(within) < 2 or len(beyond) < 2:
        print("\n[stats] Not enough data points for t-test (need ≥2 per group).")
        return

    from nope_analysis.analysis.statistical_tests import compare_groups, report_stats, save_stats_report
    stat_results = [
        compare_groups(beyond, within, label_a='beyond_4096', label_b='within_4096', metric='delta_nll')
    ]
    print(f"\n=== Statistical Tests ===")
    for r in stat_results:
        print(report_stats(r))
    save_stats_report(stat_results, OUT_DIR / 'stats.json')


if __name__ == '__main__':
    run()
