"""
e007b: Long Context Sparse Hook Analysis

Fix from e007: Q·K sparse hook (stride=128) instead of output_attentions=True.
output_attentions=True caused CUDA memory fragmentation after the first sample.
Sparse hook stores only 42MB per layer vs 5.4GB (full attention matrix at 8192t).

RQ3: Does Global attention distance diverge from SWA at lengths > SWA window?
"""
import sys
sys.path.insert(0, '/home/elicer/sda212331')

import json, torch, numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nope_analysis.loader import load_model_and_tokenizer, load_config, get_global_layer_indices, get_swa_layer_indices
from nope_analysis.seeds import set_all_seeds, EXPERIMENT_SEEDS
from nope_analysis.analysis.auto_validate import validate_experiment

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

OUT_DIR = Path('/home/elicer/sda212331/outputs/e007b_long_context_sparse_hook')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LENGTHS = [2048, 4096, 5120, 6144, 8192]
SWA_WINDOW = 4096
N_SAMPLES = 10
STRIDE = 128   # sample every STRIDE-th query position
BASE_SEED = EXPERIMENT_SEEDS["e007b"]


def build_input(tokenizer, target_len, sample_idx):
    from nope_analysis.corpus.downloader import get_text_sample
    seed = BASE_SEED + sample_idx
    for lang in ['en_edgar', 'en']:
        try:
            text = get_text_sample(lang, min_tokens=target_len + 256,
                                   tokenizer=tokenizer, seed=seed)
            ids = tokenizer(text, return_tensors='pt', add_special_tokens=True)['input_ids']
            if ids.shape[1] >= target_len:
                return ids[:, :target_len], lang
        except Exception as e:
            print(f"[corpus] {lang} seed={seed}: {e}")
    chunk = ("Large language models use sliding window attention for efficiency. "
             "Global NoPE layers attend to the full sequence context. ") * 300
    ids = tokenizer(chunk, return_tensors='pt', add_special_tokens=True)['input_ids']
    return ids[:, :target_len], 'synthetic'


def sparse_attention_distance(q_proj, k_proj, head_dim, n_heads, causal=True):
    """
    Q·K sparse attention distance using sampled query positions.

    q_proj: (seq, n_heads * head_dim) — full Q projections
    k_proj: (seq, n_kv_heads * head_dim) — full K projections
    Returns mean attention distance (scalar float).
    """
    seq_len = q_proj.shape[0]
    sample_positions = torch.arange(0, seq_len, STRIDE, device=q_proj.device)
    n_sampled = len(sample_positions)

    # Reshape: (seq, n_heads, head_dim)
    # Handle GQA: n_kv_heads may != n_heads
    n_kv_heads = k_proj.shape[-1] // head_dim
    q = q_proj.view(seq_len, n_heads, head_dim)[sample_positions]  # (n_sampled, n_heads, d)
    k = k_proj.view(seq_len, n_kv_heads, head_dim)  # (seq, n_kv_heads, d)

    # GQA: expand k to match n_heads
    groups = n_heads // n_kv_heads
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)  # (seq, n_heads, d)

    # Compute scores: (n_sampled, n_heads, seq)
    scale = head_dim ** -0.5
    scores = torch.einsum('shd,thd->sht', q.float(), k.float()) * scale

    # Apply causal mask: position sample_positions[i] can only attend to positions <= sample_positions[i]
    if causal:
        pos_q = sample_positions.unsqueeze(1)   # (n_sampled, 1)
        pos_k = torch.arange(seq_len, device=q.device).unsqueeze(0)  # (1, seq)
        mask = pos_k > pos_q  # (n_sampled, seq) — True where future
        scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

    attn = torch.softmax(scores, dim=-1)  # (n_sampled, n_heads, seq)

    # Attention distance = expected position difference
    pos_k = torch.arange(seq_len, device=attn.device).float()
    pos_q_expanded = sample_positions.float().unsqueeze(1).unsqueeze(1)  # (n_sampled, 1, 1)
    dist = (pos_k - pos_q_expanded).abs()  # (n_sampled, 1, seq) broadcast
    mean_dist = (attn * dist).sum(dim=-1).mean().item()

    return mean_dist


def run_with_sparse_hooks(model, input_ids, global_set, swa_set, cfg):
    """Register hooks on Q/K projections to compute sparse attention distance."""
    head_dim = 128   # EXAONE 4.5: head_dim=128 per MODEL_FACTS.md
    n_q_heads = 40   # EXAONE 4.5: 40 Q heads
    n_kv_heads = 8   # EXAONE 4.5: 8 KV heads

    layer_q = {}
    layer_k = {}
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
        raise RuntimeError("Could not locate model layers")

    def make_q_hook(i):
        def hook(module, inp, output):
            layer_q[i] = output.detach()[:, :, :]  # (batch, seq, dim) or (seq, dim)
        return hook

    def make_k_hook(i):
        def hook(module, inp, output):
            layer_k[i] = output.detach()[:, :, :]
        return hook

    for i in sorted(global_set | swa_set):
        if i >= len(layers):
            continue
        attn = layers[i].self_attn
        # Hook on q_proj and k_proj
        handles.append(attn.q_proj.register_forward_hook(make_q_hook(i)))
        handles.append(attn.k_proj.register_forward_hook(make_k_hook(i)))

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in handles:
        h.remove()

    stats = {}
    seq_len = input_ids.shape[1]

    for i in sorted(layer_q.keys()):
        if i not in layer_k:
            continue
        try:
            # q/k shape: (batch, seq, dim) → squeeze batch
            q = layer_q[i].squeeze(0)  # (seq, n_heads*head_dim)
            k = layer_k[i].squeeze(0)

            if q.shape[0] != seq_len:
                # Some models output (n_heads, seq, head_dim); reshape
                q = q.reshape(seq_len, -1)
                k = k.reshape(seq_len, -1)

            dist = sparse_attention_distance(q, k, head_dim, n_q_heads)
            layer_type = 'global_nope' if i in global_set else 'swa'
            stats[i] = {'distance': dist, 'layer_type': layer_type}
        except Exception as e:
            print(f"  [layer {i}] hook computation failed: {e}")

    del layer_q, layer_k
    return stats


def run():
    set_all_seeds(BASE_SEED)
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    swa_layers = get_swa_layer_indices(cfg)
    global_set = set(global_layers)
    swa_set = set(swa_layers)

    print(f"e007b — Long Context Sparse Hook (stride={STRIDE})")
    print(f"Global layers: {global_layers}")
    print(f"Target lengths: {TARGET_LENGTHS}")
    print(f"N_SAMPLES per length: {N_SAMPLES}\n")

    model, tokenizer = load_model_and_tokenizer()

    all_results = []

    for tlen in TARGET_LENGTHS:
        label = '[beyond SWA]' if tlen > SWA_WINDOW else '[within SWA]'
        print(f"\n{'='*55}\nLength = {tlen}  {label}")

        for sidx in range(N_SAMPLES):
            print(f"  sample {sidx+1}/{N_SAMPLES}", end='  ')
            ids, corpus = build_input(tokenizer, tlen, sidx)
            ids = ids.to(model.device)
            print(f"corpus={corpus}  forward...", end='  ', flush=True)

            try:
                layer_stats = run_with_sparse_hooks(model, ids, global_set, swa_set, cfg)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                if 'out of memory' in str(e).lower():
                    print(f"OOM — skip remaining at len={tlen}")
                    break
                raise

            g_dist = np.mean([s['distance'] for i, s in layer_stats.items() if s['layer_type'] == 'global_nope'])
            s_dist = np.mean([s['distance'] for i, s in layer_stats.items() if s['layer_type'] == 'swa'])
            print(f"Global={g_dist:.1f}  SWA={s_dist:.1f}  gap={g_dist-s_dist:+.1f}")

            for i, s in layer_stats.items():
                all_results.append({
                    'seq_len': tlen,
                    'sample_idx': sidx,
                    'corpus': corpus,
                    'layer_idx': i,
                    'layer_type': s['layer_type'],
                    'attn_distance': s['distance'],
                    'exceeds_swa_window': tlen > SWA_WINDOW,
                })

            del ids
            torch.cuda.empty_cache()

    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')

    plot_results(all_results)
    save_summary(all_results)

    # auto_validate — one comparison per length condition
    comparisons = {}
    for tlen in TARGET_LENGTHS:
        g_vals = [r['attn_distance'] for r in all_results
                  if r['seq_len'] == tlen and r['layer_type'] == 'global_nope']
        s_vals = [r['attn_distance'] for r in all_results
                  if r['seq_len'] == tlen and r['layer_type'] == 'swa']
        if len(g_vals) >= 2 and len(s_vals) >= 2:
            comparisons[f'dist@{tlen}t'] = (g_vals, s_vals)

    if comparisons:
        validate_experiment(
            experiment_id="e007b",
            comparisons=comparisons,
            output_dir=OUT_DIR,
            label_a="global_nope",
            label_b="swa",
        )
    else:
        print("[auto_validate] Skipped — insufficient data")

    print(f"\nOutputs: {OUT_DIR}")


def plot_results(results):
    lengths = sorted(set(r['seq_len'] for r in results))

    g_dist = [np.mean([r['attn_distance'] for r in results
                       if r['seq_len'] == l and r['layer_type'] == 'global_nope']) for l in lengths]
    s_dist = [np.mean([r['attn_distance'] for r in results
                       if r['seq_len'] == l and r['layer_type'] == 'swa']) for l in lengths]
    gap = [g - s for g, s in zip(g_dist, s_dist)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(lengths, g_dist, 'o-', color='coral', label='Global (NoPE)', linewidth=2)
    ax1.plot(lengths, s_dist, 's-', color='steelblue', label='SWA (RoPE)', linewidth=2)
    ax1.axvline(SWA_WINDOW, color='gray', linestyle='--', alpha=0.7, label=f'SWA window')
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('Mean Attention Distance (tokens)')
    ax1.set_title(f'e007b: Sparse Hook — Attention Distance\n(stride={STRIDE})')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    colors = ['tomato' if l > SWA_WINDOW else 'steelblue' for l in lengths]
    ax2.bar(range(len(lengths)), gap, color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(range(len(lengths)))
    ax2.set_xticklabels([str(l) for l in lengths])
    ax2.set_xlabel('Sequence Length (tokens)')
    ax2.set_ylabel('Global − SWA Distance (tokens)')
    ax2.set_title('Distance Gap: Global vs SWA')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'sparse_hook_distance.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved.")


def save_summary(results):
    lengths = sorted(set(r['seq_len'] for r in results))
    summary = {}
    for l in lengths:
        g = [r['attn_distance'] for r in results if r['seq_len'] == l and r['layer_type'] == 'global_nope']
        s = [r['attn_distance'] for r in results if r['seq_len'] == l and r['layer_type'] == 'swa']
        summary[str(l)] = {
            'n_samples': len(set(r['sample_idx'] for r in results if r['seq_len'] == l)),
            'global_mean': float(np.mean(g)) if g else None,
            'swa_mean':    float(np.mean(s)) if s else None,
            'gap':         float(np.mean(g) - np.mean(s)) if g and s else None,
        }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Length':>8}  {'Global':>10}  {'SWA':>10}  {'Gap':>8}")
    print('-' * 45)
    for l in lengths:
        s = summary[str(l)]
        if s['global_mean'] is not None:
            print(f"{l:>8}  {s['global_mean']:>10.1f}  {s['swa_mean']:>10.1f}  {s['gap']:>+8.1f}")


if __name__ == '__main__':
    run()
