"""
e005b: SWA Mask Injection Ablation v2

Fixes from e005:
1. Per-token NLL at positions > 4096 (not whole-sequence comparison)
2. Real corpus required (EDGAR avg 10,600t — no stitching needed for most targets)
3. Paired design: compute delta_nll per sequence → one-sample t-test
4. auto_validate.py always called

RQ3: Does NoPE Global Attention provide causal benefit for tokens beyond the SWA window?
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

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

OUT_DIR = Path('/home/elicer/sda212331/outputs/e005b_swa_mask_ablation_v2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LENGTHS = [5120, 6144, 7168, 8192]
SWA_WINDOW = 4096
N_SAMPLES = 5   # per length → 20 total (increase if memory allows)
BASE_SEED = EXPERIMENT_SEEDS["e005b"]


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
    raise RuntimeError(
        f"Could not obtain real corpus text at len={target_len} sample={sample_idx}. "
        "Synthetic fallback is disabled for e005b — beyond-window measurement requires real text."
    )


def register_swa_mask_hooks(model, global_layers, seq_len):
    """
    Inject a sliding window (SWA_WINDOW=4096) causal mask into Global layers.
    Hooks intercept the attention score matrix BEFORE softmax and mask out
    positions beyond SWA_WINDOW from each query.

    Note: this intercepts the attention module's forward_pre_hook to modify the
    attention_mask argument. EXAONE uses additive masks (0 / -inf convention).
    """
    handles = []
    global_set = set(global_layers)

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

    # Build the window mask once (additive: 0=attend, -inf=mask)
    # Shape: (seq_len, seq_len) — for query pos i, mask all keys j where |i-j| > SWA_WINDOW
    window_mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        lo = max(0, i - SWA_WINDOW + 1)
        hi = i + 1
        window_mask[i, lo:hi] = 0.0
    # Apply causal: also mask future tokens (already -inf in full init, just keep lower triangle +window)
    window_mask = window_mask.to(model.device)

    def make_pre_hook(layer_idx):
        def pre_hook(module, args, kwargs):
            # Inject attention_mask into kwargs
            # Try to replace attention_mask with our window mask
            if 'attention_mask' in kwargs:
                old_mask = kwargs['attention_mask']
                if old_mask is not None:
                    # Add our window constraint on top of existing mask
                    kwargs['attention_mask'] = old_mask + window_mask.unsqueeze(0).unsqueeze(0)
                else:
                    kwargs['attention_mask'] = window_mask.unsqueeze(0).unsqueeze(0)
            return args, kwargs
        return pre_hook

    for i in global_set:
        if i < len(layers):
            h = layers[i].self_attn.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True)
            handles.append(h)

    return handles


def compute_per_token_nll_tensor(model, input_ids):
    """
    Returns per-token NLL as a 1D tensor (length = seq_len - 1).
    Token t's NLL = -log P(token[t] | token[0..t-1]).
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq, vocab)
    logits = logits[0, :-1, :].float()   # (seq-1, vocab)
    targets = input_ids[0, 1:]           # (seq-1,)
    nll_per_token = torch.nn.functional.cross_entropy(
        logits, targets, reduction='none'
    )
    return nll_per_token.cpu()


def run():
    set_all_seeds(BASE_SEED)
    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    print(f"e005b — SWA Mask Ablation v2 (per-token NLL at positions > {SWA_WINDOW})")
    print(f"Global layers: {global_layers}")
    print(f"Target lengths: {TARGET_LENGTHS}  (all > SWA_WINDOW={SWA_WINDOW})")
    print(f"N_SAMPLES per length: {N_SAMPLES}\n")

    model, tokenizer = load_model_and_tokenizer()

    results = []
    beyond_deltas = []  # per-sequence mean delta_nll at positions > SWA_WINDOW

    for tlen in TARGET_LENGTHS:
        print(f"\n{'='*55}\nLength = {tlen}  [beyond SWA window — positions {SWA_WINDOW}–{tlen}]")

        for sidx in range(N_SAMPLES):
            print(f"  sample {sidx+1}/{N_SAMPLES}", end='  ')
            try:
                ids, corpus = build_input(tokenizer, tlen, sidx)
            except RuntimeError as e:
                print(f"corpus fail: {e}")
                continue

            ids = ids.to(model.device)
            print(f"corpus={corpus}  baseline...", end='  ', flush=True)

            handles = []
            try:
                # Baseline pass
                nll_base_tok = compute_per_token_nll_tensor(model, ids)

                # Ablated pass (SWA mask injected into Global layers)
                print("ablated...", end='  ', flush=True)
                handles = register_swa_mask_hooks(model, global_layers, tlen)
                nll_abl_tok = compute_per_token_nll_tensor(model, ids)
                for h in handles: h.remove()
                handles = []

            except RuntimeError as e:
                for h in handles: h.remove()
                torch.cuda.empty_cache()
                if 'out of memory' in str(e).lower():
                    print(f"OOM — skip remaining at len={tlen}")
                    break
                raise

            # Per-token delta
            delta_tok = nll_abl_tok - nll_base_tok  # (seq-1,)

            # Split by token position (token t corresponds to prediction position t+1)
            # Positions 0..SWA_WINDOW-1 are within-window; SWA_WINDOW.. are beyond
            within_mask = torch.arange(len(delta_tok)) < SWA_WINDOW
            beyond_mask = ~within_mask

            delta_within = delta_tok[within_mask].mean().item() if within_mask.any() else float('nan')
            delta_beyond = delta_tok[beyond_mask].mean().item() if beyond_mask.any() else float('nan')
            n_beyond_tokens = int(beyond_mask.sum().item())

            print(f"Δ_within={delta_within:+.4f}  Δ_beyond={delta_beyond:+.4f}  "
                  f"(n_beyond_tokens={n_beyond_tokens})")

            rec = {
                'seq_len': tlen,
                'sample_idx': sidx,
                'corpus': corpus,
                'delta_nll_within': delta_within,
                'delta_nll_beyond': delta_beyond,
                'n_beyond_tokens': n_beyond_tokens,
                'nll_base_mean': float(nll_base_tok.mean().item()),
                'nll_abl_mean':  float(nll_abl_tok.mean().item()),
            }
            results.append(rec)

            if not np.isnan(delta_beyond) and n_beyond_tokens > 0:
                beyond_deltas.append(delta_beyond)

            del ids, nll_base_tok, nll_abl_tok, delta_tok
            torch.cuda.empty_cache()

    # Save
    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\n=== e005b Summary ===")
    print(f"  Total sequences: {len(results)}")
    print(f"  Beyond-window delta_nll values: n={len(beyond_deltas)}")
    if beyond_deltas:
        print(f"  Mean Δ_beyond={np.mean(beyond_deltas):+.4f}  std={np.std(beyond_deltas, ddof=1):.4f}")

    within_all = [r['delta_nll_within'] for r in results if not np.isnan(r['delta_nll_within'])]
    if within_all:
        print(f"  Mean Δ_within={np.mean(within_all):+.4f}  (sanity check; expected ≈ 0)")

    summary = {
        'n_sequences': len(results),
        'n_beyond_sequences': len(beyond_deltas),
        'mean_delta_beyond': float(np.mean(beyond_deltas)) if beyond_deltas else None,
        'std_delta_beyond':  float(np.std(beyond_deltas, ddof=1)) if len(beyond_deltas) > 1 else None,
        'mean_delta_within': float(np.mean(within_all)) if within_all else None,
        'corpus_mix': list(set(r['corpus'] for r in results)),
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    plot_results(results)

    # auto_validate — one-sample t on beyond-window delta_nll
    validate_experiment(
        experiment_id="e005b",
        comparisons={"delta_nll_beyond_window": (beyond_deltas if beyond_deltas else [0.0], [])},
        output_dir=OUT_DIR,
        label_a="delta_nll_beyond",
        label_b="null_mu=0",
    )

    print(f"\nOutputs: {OUT_DIR}")


def plot_results(results):
    lengths = sorted(set(r['seq_len'] for r in results))

    beyond_means = [np.nanmean([r['delta_nll_beyond'] for r in results if r['seq_len'] == l])
                    for l in lengths]
    within_means = [np.nanmean([r['delta_nll_within'] for r in results if r['seq_len'] == l])
                    for l in lengths]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(lengths))
    w = 0.35
    ax.bar(x - w/2, beyond_means, w, label=f'Beyond window (pos>{SWA_WINDOW})', color='tomato', alpha=0.8)
    ax.bar(x + w/2, within_means, w, label=f'Within window (pos≤{SWA_WINDOW})', color='steelblue', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lengths])
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Mean ΔNLL (SWA-masked − baseline)')
    ax.set_title(f'e005b: SWA Mask Ablation — Per-Token ΔNLL\n'
                 f'(positive = NoPE Global contributes at those positions)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'swa_mask_delta_nll.png', dpi=150)
    plt.close(fig)
    print(f"Plot saved.")


if __name__ == '__main__':
    run()
