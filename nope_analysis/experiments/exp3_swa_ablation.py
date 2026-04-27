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
from nope_analysis.seeds import set_all_seeds, EXPERIMENT_SEEDS

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp3_swa_ablation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LENGTHS = [1024, 2048, 3072, 4096, 5120, 6144]
SWA_WINDOW = 4096
N_SAMPLES = 5       # samples per length condition; n=20 within-window meets min_n_per_group
BASE_SEED = EXPERIMENT_SEEDS["e006"]  # 42


def build_input_for_sample(tokenizer, target_len: int, sample_idx: int):
    """
    Build a token tensor from real corpus text.
    Priority: en_edgar (EDGAR 10-K, avg ~10K tokens, no concatenation needed)
              → en (WikiText-103, fallback)
              → synthetic (last resort, flagged in results)

    Different sample_idx values yield different corpus sections via seed offset.
    """
    from nope_analysis.corpus.downloader import get_text_sample
    seed = BASE_SEED + sample_idx

    for lang in ['en_edgar', 'en']:
        try:
            text = get_text_sample(lang, min_tokens=target_len + 256,
                                   tokenizer=tokenizer, seed=seed)
            ids = tokenizer(text, return_tensors='pt',
                            add_special_tokens=True)['input_ids']
            if ids.shape[1] >= target_len:
                return ids[:, :target_len], lang
        except Exception as e:
            print(f"[corpus] {lang} seed={seed} failed: {e}")

    # Synthetic fallback — low entropy, results unreliable
    print(f"[corpus] WARNING: synthetic fallback for sample_idx={sample_idx}, "
          "len={target_len}. Mark results as unreliable.")
    _SYNTHETIC = (
        "The development of large language models has fundamentally changed natural "
        "language understanding. Sliding window attention limits context to a local "
        "window. Global NoPE layers attend to the full sequence without positional encoding. "
    )
    repeated = _SYNTHETIC * (target_len // 40 + 10)
    ids = tokenizer(repeated, return_tensors='pt',
                    add_special_tokens=True)['input_ids']
    return ids[:, :target_len], 'synthetic'


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
    set_all_seeds(BASE_SEED)
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    print(f"Global (NoPE) layers: {global_layers}")
    print(f"SWA window: {SWA_WINDOW} tokens")
    print(f"Target lengths: {TARGET_LENGTHS}")
    print(f"Samples per length: {N_SAMPLES}  (seed base: {BASE_SEED})\n")

    model, tokenizer = load_model_and_tokenizer()

    results = []

    for target_len in TARGET_LENGTHS:
        print(f"\n{'='*50}")
        print(f"Length = {target_len} tokens  "
              f"{'[beyond SWA window]' if target_len > SWA_WINDOW else '[within SWA window]'}")

        for sample_idx in range(N_SAMPLES):
            print(f"  Sample {sample_idx+1}/{N_SAMPLES}", end="  ")
            input_ids, corpus_lang = build_input_for_sample(tokenizer, target_len, sample_idx)
            input_ids = input_ids.to(model.device)
            print(f"corpus={corpus_lang}", end="  ")

            # 조건 A: 베이스라인
            ppl_base, nll_base = compute_perplexity(model, input_ids)

            # 조건 B: Global ablation
            handles = register_ablation_hooks(model, global_layers)
            ppl_ablated, nll_ablated = compute_perplexity(model, input_ids)
            remove_hooks(handles)

            delta_nll = nll_ablated - nll_base
            delta_ppl = ppl_ablated - ppl_base
            print(f"ΔNLL={delta_nll:+.4f}  ΔPPL={delta_ppl:+.4f}")

            results.append({
                'seq_len': target_len,
                'sample_idx': sample_idx,
                'corpus': corpus_lang,
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
    lengths = sorted(set(r['seq_len'] for r in results))

    # Aggregate across samples
    mean_delta_ppl = [np.mean([r['delta_ppl'] for r in results if r['seq_len'] == l])
                      for l in lengths]
    std_delta_ppl  = [np.std( [r['delta_ppl'] for r in results if r['seq_len'] == l])
                      for l in lengths]
    mean_ppl_base  = [np.mean([r['ppl_baseline'] for r in results if r['seq_len'] == l])
                      for l in lengths]
    mean_ppl_abl   = [np.mean([r['ppl_ablated']  for r in results if r['seq_len'] == l])
                      for l in lengths]
    colors = ['tomato' if l > SWA_WINDOW else 'steelblue' for l in lengths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(range(len(lengths)), mean_delta_ppl, color=colors,
                   alpha=0.8, yerr=std_delta_ppl, capsize=4)
    ax1.axhline(0, color='black', linewidth=0.8)
    if SWA_WINDOW in lengths:
        ax1.axvline(lengths.index(SWA_WINDOW) + 0.5, color='gray', linestyle='--',
                    alpha=0.7, label=f'SWA window ({SWA_WINDOW})')
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels([str(l) for l in lengths])
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('Mean ΔPPL ± std (Ablated − Baseline)')
    ax1.set_title(f'PPL Increase When Global Attention Removed\n'
                  f'(n={N_SAMPLES}/length; blue=within window, red=beyond)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.plot(lengths, mean_ppl_base, 'o-', color='steelblue', label='Baseline', linewidth=2)
    ax2.plot(lengths, mean_ppl_abl,  's--', color='tomato',   label='Global Ablated', linewidth=2)
    ax2.axvline(SWA_WINDOW, color='gray', linestyle='--', alpha=0.7, label='SWA window')
    ax2.set_xlabel('Sequence Length (tokens)')
    ax2.set_ylabel('Mean Perplexity')
    ax2.set_title('Baseline vs Ablated Perplexity (mean across samples)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'ablation_perplexity.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved: {OUT_DIR}/ablation_perplexity.png")


def save_summary(results):
    within = [r for r in results if not r['exceeds_swa_window']]
    beyond = [r for r in results if r['exceeds_swa_window']]
    lengths = sorted(set(r['seq_len'] for r in results))
    corpus_used = list(set(r['corpus'] for r in results))

    def group_stats(group):
        if not group:
            return {'n': 0, 'mean_delta_nll': None, 'std_delta_nll': None,
                    'mean_delta_ppl': None, 'lengths': []}
        nlls = [r['delta_nll'] for r in group]
        ppls = [r['delta_ppl'] for r in group]
        return {
            'n': len(group),
            'lengths': sorted(set(r['seq_len'] for r in group)),
            'mean_delta_nll': float(np.mean(nlls)),
            'std_delta_nll':  float(np.std(nlls)),
            'mean_delta_ppl': float(np.mean(ppls)),
            'std_delta_ppl':  float(np.std(ppls)),
        }

    per_length = {}
    for l in lengths:
        grp = [r for r in results if r['seq_len'] == l]
        nlls = [r['delta_nll'] for r in grp]
        ppls = [r['delta_ppl'] for r in grp]
        per_length[str(l)] = {
            'n': len(grp),
            'mean_ppl_baseline': float(np.mean([r['ppl_baseline'] for r in grp])),
            'mean_ppl_ablated':  float(np.mean([r['ppl_ablated']  for r in grp])),
            'mean_delta_nll':    float(np.mean(nlls)),
            'std_delta_nll':     float(np.std(nlls)),
            'mean_delta_ppl':    float(np.mean(ppls)),
            'std_delta_ppl':     float(np.std(ppls)),
            'corpus': list(set(r['corpus'] for r in grp)),
        }

    summary = {
        'n_samples_per_length': N_SAMPLES,
        'corpus_used': corpus_used,
        'within_swa_window': group_stats(within),
        'beyond_swa_window': group_stats(beyond),
        'per_length': per_length,
    }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Experiment e006 Summary (n={N_SAMPLES}/length) ===")
    print(f"  Corpus: {corpus_used}")
    w = summary['within_swa_window']
    b = summary['beyond_swa_window']
    if w['mean_delta_nll'] is not None:
        print(f"  Within  SWA window (n={w['n']}):  mean ΔNLL={w['mean_delta_nll']:+.4f} ± {w['std_delta_nll']:.4f}")
    if b['mean_delta_nll'] is not None:
        print(f"  Beyond  SWA window (n={b['n']}):  mean ΔNLL={b['mean_delta_nll']:+.4f} ± {b['std_delta_nll']:.4f}")
    else:
        print(f"  Beyond SWA window: no data (OOM expected at 5120+ tokens)")
    print(f"\nSummary saved: {OUT_DIR}/summary.json")


if __name__ == '__main__':
    run()
