"""
e006b: Global Zero-Output Ablation v2

Fix from e006: paired/one-sample t-test on per-sequence delta_nll instead of
independent-groups Welch t-test. e006's FAILED verdict was a test selection error.

RQ3: Does NoPE Global Attention contribute causally to long-range dependency?
"""
import sys
sys.path.insert(0, '/home/elicer/sda212331')

import json, torch, numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nope_analysis.loader import load_model_and_tokenizer, load_config, get_global_layer_indices
from nope_analysis.seeds import set_all_seeds, EXPERIMENT_SEEDS
from nope_analysis.analysis.auto_validate import validate_experiment

OUT_DIR = Path('/home/elicer/sda212331/outputs/e006b_global_zero_ablation_v2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LENGTHS = [2048, 3072, 4096, 5120, 6144]
SWA_WINDOW = 4096
N_SAMPLES = 10  # per length → 50 total sequences
BASE_SEED = EXPERIMENT_SEEDS["e006b"]

os_env_set = False
try:
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os_env_set = True
except Exception:
    pass


def build_input(tokenizer, target_len, sample_idx):
    from nope_analysis.corpus.downloader import get_text_sample
    seed = BASE_SEED + sample_idx
    for lang in ['en_edgar', 'en', 'ko']:
        try:
            text = get_text_sample(lang, min_tokens=target_len + 256,
                                   tokenizer=tokenizer, seed=seed)
            ids = tokenizer(text, return_tensors='pt', add_special_tokens=True)['input_ids']
            if ids.shape[1] >= target_len:
                return ids[:, :target_len], lang
        except Exception as e:
            print(f"[corpus] {lang} seed={seed}: {e}")
    # synthetic fallback
    chunk = ("The architecture uses sliding window attention for local context "
             "and global NoPE attention for long-range dependencies. ") * 200
    ids = tokenizer(chunk, return_tensors='pt', add_special_tokens=True)['input_ids']
    return ids[:, :target_len], 'synthetic'


def register_zero_hooks(model, global_layers):
    handles = []
    for path in ['model.language_model.layers', 'language_model.layers', 'model.layers']:
        try:
            layers = model
            for attr in path.split('.'):
                layers = getattr(layers, attr)
            break
        except AttributeError:
            layers = None
    if layers is None:
        print("WARNING: could not locate layers")
        return handles

    def make_hook(idx):
        def hook(module, inp, output):
            return (torch.zeros_like(output[0]),) + output[1:]
        return hook

    for i in set(global_layers):
        if i < len(layers):
            handles.append(layers[i].self_attn.register_forward_hook(make_hook(i)))
    return handles


def compute_per_token_nll(model, input_ids):
    """Returns mean NLL over the sequence (scalar)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return float(out.loss.item())


def run():
    set_all_seeds(BASE_SEED)
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    print(f"e006b — Global Zero Ablation v2 (one-sample paired design)")
    print(f"Global layers: {global_layers}")
    print(f"N_SAMPLES per length: {N_SAMPLES}, total target: {N_SAMPLES * len(TARGET_LENGTHS)}\n")

    model, tokenizer = load_model_and_tokenizer()

    results = []

    for tlen in TARGET_LENGTHS:
        label = '[beyond SWA window]' if tlen > SWA_WINDOW else '[within SWA window]'
        print(f"\n{'='*55}\nLength = {tlen}  {label}")

        for sidx in range(N_SAMPLES):
            print(f"  sample {sidx+1}/{N_SAMPLES}", end='  ')
            ids, corpus = build_input(tokenizer, tlen, sidx)
            ids = ids.to(model.device)
            print(f"corpus={corpus}", end='  ')

            handles = []
            try:
                nll_base = compute_per_token_nll(model, ids)
                handles = register_zero_hooks(model, global_layers)
                nll_abl = compute_per_token_nll(model, ids)
                for h in handles: h.remove()
                handles = []
            except RuntimeError as e:
                for h in handles: h.remove()
                torch.cuda.empty_cache()
                if 'out of memory' in str(e).lower():
                    print(f"OOM — skip remaining at len={tlen}")
                    break
                raise

            delta = nll_abl - nll_base
            print(f"baseline={nll_base:.4f}  ablated={nll_abl:.4f}  Δ={delta:+.4f}")

            results.append({
                'seq_len': tlen,
                'sample_idx': sidx,
                'corpus': corpus,
                'nll_baseline': nll_base,
                'nll_ablated': nll_abl,
                'delta_nll': delta,
                'exceeds_swa_window': tlen > SWA_WINDOW,
            })

            del ids
            torch.cuda.empty_cache()

    # Save results
    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    # Separate within/beyond-window
    within  = [r['delta_nll'] for r in results if not r['exceeds_swa_window']]
    beyond  = [r['delta_nll'] for r in results if r['exceeds_swa_window']]
    all_deltas = [r['delta_nll'] for r in results]

    print(f"\n=== e006b Summary ===")
    print(f"  Within window  n={len(within)}  mean_Δ={np.mean(within):+.4f}" if within else "  Within: no data")
    print(f"  Beyond window  n={len(beyond)}  mean_Δ={np.mean(beyond):+.4f}" if beyond else "  Beyond: no data (OOM)")
    print(f"  All sequences  n={len(all_deltas)}  mean_Δ={np.mean(all_deltas):+.4f}")

    # Summary JSON
    summary = {
        'n_total': len(results),
        'n_within': len(within),
        'n_beyond': len(beyond),
        'mean_delta_within': float(np.mean(within)) if within else None,
        'mean_delta_beyond': float(np.mean(beyond)) if beyond else None,
        'mean_delta_all': float(np.mean(all_deltas)) if all_deltas else None,
        'std_delta_all': float(np.std(all_deltas, ddof=1)) if len(all_deltas) > 1 else None,
        'corpus_mix': list(set(r['corpus'] for r in results)),
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    plot_results(results)

    # auto_validate — primary: one-sample t on all delta_nll
    if len(all_deltas) >= 10:
        validate_experiment(
            experiment_id="e006b",
            comparisons={"delta_nll": (all_deltas, [])},
            output_dir=OUT_DIR,
            label_a="delta_nll",
            label_b="null_mu=0",
        )
    else:
        print(f"[auto_validate] Skipped — n={len(all_deltas)} < 10 (insufficient data)")

    print(f"\nOutputs: {OUT_DIR}")


def plot_results(results):
    lengths = sorted(set(r['seq_len'] for r in results))
    per_len = {l: [r['delta_nll'] for r in results if r['seq_len'] == l] for l in lengths}

    means = [np.mean(per_len[l]) for l in lengths]
    stds  = [np.std(per_len[l], ddof=1) if len(per_len[l]) > 1 else 0 for l in lengths]
    colors = ['tomato' if l > SWA_WINDOW else 'steelblue' for l in lengths]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(lengths)), means, color=colors, alpha=0.8, yerr=stds, capsize=4)
    ax.axhline(0, color='black', linewidth=0.8)
    if SWA_WINDOW in lengths:
        ax.axvline(lengths.index(SWA_WINDOW) + 0.5, color='gray',
                   linestyle='--', alpha=0.7, label=f'SWA window ({SWA_WINDOW}t)')
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([str(l) for l in lengths])
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Mean ΔNLL (ablated − baseline) ± std')
    ax.set_title('e006b: Global Zero-Ablation — Paired ΔNLL per Length\n'
                 '(blue=within SWA window, red=beyond)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'delta_nll_per_length.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved.")


if __name__ == '__main__':
    run()
