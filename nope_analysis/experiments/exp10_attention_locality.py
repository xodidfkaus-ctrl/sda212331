"""
e010: Attention Locality Analysis

Why does Global attention distance < SWA at long sequences? (e007b direction reversal)

Hypothesis: SWA window-forcing effect.
SWA is geometrically constrained to attend within 4096 tokens → mean distance ≈ 2048.
Global, unconstrained, attends locally when informative → mean distance below SWA's
window-enforced baseline. Key question: what fraction of Global attention goes
BEYOND the SWA window (> 4096 tokens away)?

Method: sparse Q·K hook (reused from e007b) measuring per-position attention weights,
decomposed into beyond-window vs within-window fractions.
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
from nope_analysis.corpus.downloader import get_text_sample
from nope_analysis.analysis.auto_validate import validate_experiment

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

OUT_DIR = Path('/home/elicer/sda212331/outputs/e010_attention_locality')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SWA_WINDOW   = 4096
TARGET_LENGTHS = [4608, 6144, 8192]
N_SAMPLES    = 30
STRIDE       = 128
BASE_SEED    = EXPERIMENT_SEEDS.get("e010", 20100428)
HEAD_DIM     = 128
N_HEADS      = 40
N_KV_HEADS   = 8


def build_input(tokenizer, target_len: int, sample_idx: int):
    seed = BASE_SEED + sample_idx
    for lang in ['en_edgar', 'en']:
        try:
            text = get_text_sample(lang, min_tokens=target_len + 256,
                                   tokenizer=tokenizer, seed=seed)
            ids = tokenizer(text, return_tensors='pt',
                            add_special_tokens=True)['input_ids']
            if ids.shape[1] >= target_len:
                return ids[:, :target_len], lang
        except Exception:
            pass
    chunk = ("Sliding window attention and global NoPE attention "
             "process information at different scales in the model. ") * 600
    ids = tokenizer(chunk, return_tensors='pt',
                    add_special_tokens=True)['input_ids']
    return ids[:, :target_len], 'synthetic'


def sparse_beyond_window_frac(q_proj, k_proj, layer_type: str,
                               seq_len: int) -> dict:
    """
    Compute beyond-window and within-window attention fractions.

    For SWA layers: beyond_frac = 0 by definition (window mask applied by model).
    For Global layers: measure actual attention weight distribution.

    Returns dict with beyond_frac, within_frac, mean_distance.
    """
    sample_positions = torch.arange(0, seq_len, STRIDE, device=q_proj.device)
    # Only query positions where beyond-window tokens exist (pos > SWA_WINDOW)
    sample_positions = sample_positions[sample_positions > SWA_WINDOW]
    if len(sample_positions) == 0:
        return {'beyond_frac': 0.0, 'within_frac': 1.0, 'mean_distance': 0.0}

    n_kv = k_proj.shape[-1] // HEAD_DIM
    groups = N_HEADS // n_kv

    q = q_proj.view(seq_len, N_HEADS, HEAD_DIM)[sample_positions]
    k = k_proj.view(seq_len, n_kv, HEAD_DIM)
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)

    scale = HEAD_DIM ** -0.5
    # Compute attention scores for sampled positions
    scores = torch.einsum('qhd,khd->qhk', q, k) * scale  # (n_q, n_heads, seq)

    # Causal mask
    mask = torch.full((len(sample_positions), seq_len), float('-inf'),
                      device=q.device)
    for i, pos in enumerate(sample_positions):
        if layer_type == 'swa':
            start = max(0, int(pos) - SWA_WINDOW + 1)
        else:
            start = 0
        mask[i, start:int(pos) + 1] = 0.0

    scores = scores + mask.unsqueeze(1)  # broadcast over heads
    attn = torch.softmax(scores, dim=-1)  # (n_q, n_heads, seq)

    # For each query, compute beyond-window fraction
    beyond_fracs = []
    distances = []
    for i, pos in enumerate(sample_positions):
        pos_int = int(pos)
        window_start = max(0, pos_int - SWA_WINDOW + 1)
        # beyond-window = tokens at distance > SWA_WINDOW from query
        beyond_weight = attn[i, :, :window_start].sum(dim=-1).mean().item()
        within_weight = attn[i, :, window_start:pos_int + 1].sum(dim=-1).mean().item()
        beyond_fracs.append(beyond_weight)

        # Mean attention distance
        positions_t = torch.arange(pos_int + 1, device=attn.device).float()
        dist = (positions_t.unsqueeze(0) * attn[i, :, :pos_int + 1]).sum(dim=-1)
        mean_dist = (pos_int - dist).mean().item()
        distances.append(mean_dist)

    return {
        'beyond_frac': float(np.mean(beyond_fracs)),
        'within_frac': float(1 - np.mean(beyond_fracs)),
        'mean_distance': float(np.mean(distances)),
        'n_query_positions': len(sample_positions),
    }


def main():
    print("e010 — Attention Locality Analysis")
    print(f"N_SAMPLES={N_SAMPLES}, TARGET_LENGTHS={TARGET_LENGTHS}, STRIDE={STRIDE}\n")

    set_all_seeds(BASE_SEED)

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    cfg = load_config()
    global_indices = set(get_global_layer_indices())
    n_layers = cfg.num_hidden_layers
    print(f"Model loaded. Layers: {n_layers}\n")

    results = []

    for seq_len in TARGET_LENGTHS:
        print(f"\n{'='*60}")
        print(f"seq_len = {seq_len}")
        for sample_idx in range(N_SAMPLES):
            input_ids, lang = build_input(tokenizer, seq_len, sample_idx)
            input_ids = input_ids.to('cuda:0')

            layer_data = {}

            def make_hook(li):
                def hook(module, args, kwargs_out):
                    # Capture Q and K projections
                    if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                        with torch.no_grad():
                            # args[0] is hidden_states
                            hidden = args[0]
                            q = module.q_proj(hidden)[0].detach().cpu()  # (seq, nH*d)
                            k = module.k_proj(hidden)[0].detach().cpu()
                            ltype = 'global_nope' if li in global_indices else 'swa'
                            layer_data[li] = (q, k, ltype)
                return hook

            hooks = []
            for li in range(n_layers):
                h = model.language_model.model.layers[li].self_attn.register_forward_pre_hook(
                    make_hook(li), with_kwargs=True)
                hooks.append(h)

            try:
                with torch.no_grad():
                    model(input_ids)
            finally:
                for h in hooks:
                    h.remove()

            for li, (q, k, ltype) in layer_data.items():
                q = q.to('cuda:0')
                k = k.to('cuda:0')
                metrics = sparse_beyond_window_frac(q, k, ltype, seq_len)
                q = q.cpu(); k = k.cpu()

                r = {
                    'layer_idx': li,
                    'layer_type': ltype,
                    'seq_len': seq_len,
                    'sample_idx': sample_idx,
                    'lang': lang,
                    **metrics,
                }
                results.append(r)
                with open(OUT_DIR / 'results.jsonl', 'a') as f:
                    f.write(json.dumps(r) + '\n')

            if sample_idx % 5 == 0:
                # Print running average for Global layers
                g_beyond = [r['beyond_frac'] for r in results
                            if r['layer_type'] == 'global_nope'
                            and r['seq_len'] == seq_len]
                if g_beyond:
                    print(f"  sample {sample_idx:02d} | lang={lang} | "
                          f"Global beyond_frac (running mean): {np.mean(g_beyond):.4f}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY — Beyond-window attention fraction by layer type and length")

    summary_rows = []
    for sl in TARGET_LENGTHS:
        for ltype in ['global_nope', 'swa']:
            vals = [r['beyond_frac'] for r in results
                    if r['seq_len'] == sl and r['layer_type'] == ltype]
            if not vals:
                continue
            mean_bf = np.mean(vals)
            std_bf = np.std(vals)
            print(f"  seq_len={sl} {ltype:12s}: beyond_frac={mean_bf:.4f}±{std_bf:.4f} (n={len(vals)})")
            summary_rows.append({
                'seq_len': sl, 'layer_type': ltype,
                'mean_beyond_frac': float(mean_bf),
                'std_beyond_frac': float(std_bf),
                'n': len(vals),
            })

    # Overall Global beyond_frac
    global_beyond = [r['beyond_frac'] for r in results
                     if r['layer_type'] == 'global_nope']
    mean_global_beyond = float(np.mean(global_beyond)) if global_beyond else 0

    # One-sample t-test: H0: mean = 0.15
    from scipy.stats import ttest_1samp
    if global_beyond:
        t_stat, p_val = ttest_1samp(global_beyond, popmean=0.15,
                                     alternative='less')
    else:
        t_stat, p_val = float('nan'), float('nan')

    verdict = (
        'VALIDATED' if (not np.isnan(p_val) and p_val <= 0.05
                        and mean_global_beyond < 0.15)
        else 'FAILED' if (not np.isnan(p_val) and mean_global_beyond >= 0.15)
        else 'INCONCLUSIVE'
    )

    summary = {
        'experiment_id': 'e010',
        'n_results': len(results),
        'mean_global_beyond_frac': mean_global_beyond,
        'ttest_1samp_stat': float(t_stat),
        'ttest_1samp_p': float(p_val),
        'null_threshold': 0.15,
        'verdict': verdict,
        'by_length': summary_rows,
        'interpretation': (
            'Window-forcing hypothesis SUPPORTED: Global rarely uses beyond-window positions.'
            if verdict == 'VALIDATED'
            else 'Global uses beyond-window substantially — alternative explanation needed.'
        ),
    }

    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Overall Global beyond_frac: {mean_global_beyond:.4f}")
    print(f"  One-sample t (H0=0.15, alt=less): t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Verdict: {verdict}")
    print(f"  {summary['interpretation']}")

    validate_experiment(
        experiment_id='e010',
        results_path=OUT_DIR / 'results.jsonl',
        repo_root=Path('/home/elicer/sda212331'),
    )

    print(f"\nOutputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
