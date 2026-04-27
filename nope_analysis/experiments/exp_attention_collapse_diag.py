"""
Attention Collapse / Local Bias Diagnostic

Checks whether Global (NoPE) layers exhibit attention collapse or local bias
at long sequences (2048–8192 tokens) by measuring:
  1. Attention entropy per layer (Global vs SWA)
  2. Attention sink fraction (mass on position 0, BOS token)
  3. Local attention fraction (mass within 64 tokens of query position)
  4. Mean attended position by decile (where in the sequence is attention concentrated?)

Not a formal pre-registered experiment — diagnostic only.
Results inform interpretation of e007b reversed direction finding.
"""

import sys
sys.path.insert(0, '/home/elicer/sda212331')

import json
import numpy as np
import torch
from pathlib import Path
from nope_analysis.loader import load_model_and_tokenizer, get_global_layer_indices, load_config
from nope_analysis.corpus.downloader import get_text_sample

OUT_DIR = Path('/home/elicer/sda212331/outputs/attn_collapse_diag')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_LENGTHS = [2048, 4096, 8192]
N_SAMPLES = 3          # small — diagnostic only
STRIDE = 128           # sample every STRIDE query positions
LOCAL_WINDOW = 64      # "local" = within 64 tokens


def make_sparse_q_hook(layer_idx, q_store):
    def hook(module, input, output):
        # output shape: (batch, n_heads, seq, head_dim)
        q_store[layer_idx] = output.detach().float()
    return hook


def make_sparse_k_hook(layer_idx, k_store):
    def hook(module, input, output):
        k_store[layer_idx] = output.detach().float()
    return hook


def compute_attention_stats(q, k, query_stride=STRIDE, local_window=LOCAL_WINDOW):
    """
    q: (batch, n_q_heads, seq, head_dim)
    k: (batch, n_kv_heads, seq, head_dim)
    Returns dict of per-sample stats.
    """
    seq_len = q.shape[2]
    n_q_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim = q.shape[3]
    scale = head_dim ** -0.5

    # GQA: expand kv heads to match q heads
    if n_kv_heads != n_q_heads:
        repeat = n_q_heads // n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)

    sampled_q_positions = list(range(0, seq_len, query_stride))
    q_sub = q[:, :, sampled_q_positions, :]  # (1, heads, n_sampled, head_dim)

    entropies, sink_fracs, local_fracs, distances = [], [], [], []

    for qi, qpos in enumerate(sampled_q_positions):
        # Causal: only attend up to qpos
        k_causal = k[:, :, :qpos + 1, :]  # (1, heads, qpos+1, head_dim)
        q_vec = q_sub[:, :, qi:qi+1, :]   # (1, heads, 1, head_dim)

        scores = torch.matmul(q_vec, k_causal.transpose(-2, -1)) * scale  # (1, heads, 1, qpos+1)
        attn = torch.softmax(scores, dim=-1).squeeze()  # (heads, qpos+1)

        if attn.dim() == 1:
            attn = attn.unsqueeze(0)

        # Mean over heads
        attn_mean = attn.mean(0)  # (qpos+1,)
        n_keys = attn_mean.shape[0]

        # Entropy
        ent = -(attn_mean * (attn_mean + 1e-12).log()).sum().item()
        entropies.append(ent)

        # Attention sink (position 0)
        sink_frac = attn_mean[0].item()
        sink_fracs.append(sink_frac)

        # Local fraction (within LOCAL_WINDOW of query position)
        lo = max(0, qpos - local_window)
        local_mass = attn_mean[lo:].sum().item()
        local_fracs.append(local_mass)

        # Mean attended position (distance from query)
        positions = torch.arange(n_keys, dtype=torch.float32, device=attn_mean.device)
        dist = (attn_mean * (qpos - positions)).sum().item()
        distances.append(dist)

    return {
        'mean_entropy': float(np.mean(entropies)),
        'mean_sink_frac': float(np.mean(sink_fracs)),
        'mean_local_frac': float(np.mean(local_fracs)),
        'mean_distance': float(np.mean(distances)),
        'n_sampled_positions': len(sampled_q_positions),
    }


def run():
    print("=== Attention Collapse / Local Bias Diagnostic ===\n")
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    global_set = set(global_layers)

    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    # Get layer access path
    lm_layers = model.model.language_model.layers
    n_layers = len(lm_layers)
    swa_set = set(range(n_layers)) - global_set

    print(f"Global layers: {global_layers}")
    print(f"SWA layers (first 8): {sorted(swa_set)[:8]}...")
    print(f"Test lengths: {TEST_LENGTHS}, N_SAMPLES: {N_SAMPLES}\n")

    all_results = []

    for tlen in TEST_LENGTHS:
        print(f"\n{'='*55}")
        print(f"Length = {tlen}")

        for sidx in range(N_SAMPLES):
            try:
                text = get_text_sample('en_edgar', tlen, sidx)
            except Exception:
                text = get_text_sample('en', tlen, sidx)

            ids = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=tlen)['input_ids'].to(model.device)
            actual_len = ids.shape[1]

            q_store, k_store = {}, {}
            handles = []

            try:
                for layer_idx in range(n_layers):
                    layer = lm_layers[layer_idx]
                    handles.append(layer.self_attn.q_proj.register_forward_hook(
                        make_sparse_q_hook(layer_idx, q_store)))
                    handles.append(layer.self_attn.k_proj.register_forward_hook(
                        make_sparse_k_hook(layer_idx, k_store)))

                with torch.no_grad():
                    model(ids)

            finally:
                for h in handles:
                    h.remove()

            # Compute stats per layer
            layer_stats = {}
            for li in range(n_layers):
                if li not in q_store or li not in k_store:
                    continue
                q = q_store[li]
                k = k_store[li]
                if q.shape[2] < 2:
                    continue
                try:
                    stats = compute_attention_stats(q, k)
                    stats['layer_idx'] = li
                    stats['layer_type'] = 'global_nope' if li in global_set else 'swa'
                    stats['seq_len'] = actual_len
                    stats['sample_idx'] = sidx
                    layer_stats[li] = stats
                    all_results.append(stats)
                except Exception as e:
                    print(f"  layer {li} error: {e}")

            # Print per-sample summary
            g_stats = [v for v in layer_stats.values() if v['layer_type'] == 'global_nope']
            s_stats = [v for v in layer_stats.values() if v['layer_type'] == 'swa']

            if g_stats and s_stats:
                print(f"  sample {sidx+1}  len={actual_len}")
                for metric in ['mean_entropy', 'mean_sink_frac', 'mean_local_frac', 'mean_distance']:
                    g_val = np.mean([x[metric] for x in g_stats])
                    s_val = np.mean([x[metric] for x in s_stats])
                    print(f"    {metric:20s}  Global={g_val:.4f}  SWA={s_val:.4f}  gap={g_val-s_val:+.4f}")

            del q_store, k_store
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY — Global vs SWA per metric per length")
    print(f"{'='*70}")
    for tlen in TEST_LENGTHS:
        subset = [r for r in all_results if r['seq_len'] == tlen]
        g = [r for r in subset if r['layer_type'] == 'global_nope']
        s = [r for r in subset if r['layer_type'] == 'swa']
        if not g or not s:
            continue
        print(f"\nLength={tlen}  (n_global_obs={len(g)}, n_swa_obs={len(s)})")
        for metric in ['mean_entropy', 'mean_sink_frac', 'mean_local_frac', 'mean_distance']:
            gm = np.mean([x[metric] for x in g])
            sm = np.mean([x[metric] for x in s])
            flag = "⚠️ COLLAPSE?" if metric == 'mean_entropy' and gm < sm * 0.8 else ""
            flag = flag or ("⚠️ SINK?" if metric == 'mean_sink_frac' and gm > 0.3 else "")
            flag = flag or ("⚠️ LOCAL?" if metric == 'mean_local_frac' and gm > 0.7 else "")
            print(f"  {metric:22s}  Global={gm:.4f}  SWA={sm:.4f}  gap={gm-sm:+.4f}  {flag}")

    # Save
    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved: {OUT_DIR}/results.jsonl")

    # Collapse verdict
    print(f"\n{'='*55}")
    print("COLLAPSE ASSESSMENT:")
    all_g = [r for r in all_results]
    g8192 = [r for r in all_results if r['seq_len'] == 8192 and r['layer_type'] == 'global_nope']
    s8192 = [r for r in all_results if r['seq_len'] == 8192 and r['layer_type'] == 'swa']
    if g8192 and s8192:
        g_ent = np.mean([x['mean_entropy'] for x in g8192])
        s_ent = np.mean([x['mean_entropy'] for x in s8192])
        g_sink = np.mean([x['mean_sink_frac'] for x in g8192])
        g_local = np.mean([x['mean_local_frac'] for x in g8192])
        print(f"At 8192 tokens:")
        print(f"  Global entropy={g_ent:.4f}  SWA entropy={s_ent:.4f}")
        print(f"  Global sink_frac={g_sink:.4f}  (>0.3 = possible collapse)")
        print(f"  Global local_frac={g_local:.4f}  (>0.7 = local bias)")
        if g_sink > 0.3:
            print("  ⚠️  HIGH SINK FRACTION — attention collapse to BOS suspected")
        elif g_local > 0.7:
            print("  ⚠️  HIGH LOCAL FRACTION — local bias present")
        elif g_ent < s_ent * 0.8:
            print("  ⚠️  LOW ENTROPY relative to SWA — possible collapse")
        else:
            print("  ✓  No strong collapse signal — SWA window-forcing more likely explanation")


if __name__ == '__main__':
    run()
