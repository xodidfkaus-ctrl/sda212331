"""
e008: Delta-Attribution Probe

RQ1: Does NoPE Global actively encode positional information (H1_alt),
or does it merely propagate SWA's signal (H1_null)?

Method: For each layer i, train a linear probe on h_in[i] (residual stream
entering the layer) and h_out[i] (residual stream exiting the layer).
delta_acc[i] = test_acc(h_out[i]) - test_acc(h_in[i])

H1_alt: delta_acc significantly > 0 at >= 1 Global layer (Global adds positional info)
H1_null: delta_acc ≈ 0 at all Global layers (Global only passes through SWA's signal)
"""

import sys
sys.path.insert(0, '/home/elicer/sda212331')

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from nope_analysis.loader import load_model_and_tokenizer, get_global_layer_indices, load_config
from nope_analysis.corpus.downloader import get_text_sample
from nope_analysis.seeds import EXPERIMENT_SEEDS
from nope_analysis.analysis.auto_validate import validate_experiment

OUT_DIR = Path('/home/elicer/sda212331/outputs/exp2c_delta_probe')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Probe config
N_PROMPTS = 300          # per PLAN.md min_n=300
SEQ_LENGTHS = [64, 128, 256]
N_POSITION_BINS = 10     # 10 position classes
PROBE_LR = 1e-3
PROBE_EPOCHS = 100
PROBE_BATCH = 64
K_FOLDS = 5
SEED = EXPERIMENT_SEEDS.get('e008', 42)


# ── Positional bin assignment ──────────────────────────────────────────────────

def assign_position_bins(seq_len, n_bins=N_POSITION_BINS):
    """Assign each token position to a bin index 0..n_bins-1."""
    bins = torch.zeros(seq_len, dtype=torch.long)
    for t in range(seq_len):
        bins[t] = min(int(t * n_bins / seq_len), n_bins - 1)
    return bins


# ── Linear probe (PyTorch, GPU) ───────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, hidden_size, n_classes):
        super().__init__()
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_probe(X_train, y_train, hidden_size, device, n_classes=N_POSITION_BINS):
    probe = LinearProbe(hidden_size, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    probe.train()
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(len(X_t))
        for start in range(0, len(X_t), PROBE_BATCH):
            idx = perm[start:start + PROBE_BATCH]
            opt.zero_grad()
            loss = loss_fn(probe(X_t[idx]), y_t[idx])
            loss.backward()
            opt.step()

    return probe


def eval_probe(probe, X_test, y_test, device):
    probe.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = probe(X_t).argmax(dim=1).cpu().numpy()
    return float((preds == y_test).mean())


# ── Hidden state extraction ───────────────────────────────────────────────────

def extract_hidden_states(model, tokenizer, texts, seq_len, global_set, n_layers, device):
    """
    Returns:
        h_in:  dict[layer_idx] -> (n_tokens_total, hidden_size)  — residual BEFORE attention
        h_out: dict[layer_idx] -> (n_tokens_total, hidden_size)  — residual AFTER attention + residual add
        labels: (n_tokens_total,)  — position bin for each token
    """
    lm_layers = model.model.language_model.layers

    h_in_store = {i: [] for i in range(n_layers)}
    h_out_store = {i: [] for i in range(n_layers)}

    def make_pre_hook(li):
        def hook(module, args):
            if args:
                h_in_store[li].append(args[0].detach().cpu().float())
        return hook

    def make_post_hook(li):
        def hook(module, input, output):
            # Decoder layer output: first element is hidden state
            out = output[0] if isinstance(output, tuple) else output
            h_out_store[li].append(out.detach().cpu().float())
        return hook

    handles = []
    for li in range(n_layers):
        handles.append(lm_layers[li].register_forward_pre_hook(make_pre_hook(li)))
        handles.append(lm_layers[li].register_forward_hook(make_post_hook(li)))

    all_labels = []

    try:
        for text in texts:
            ids = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=seq_len)['input_ids'].to(device)
            actual_len = ids.shape[1]
            pos_bins = assign_position_bins(actual_len).numpy()
            all_labels.append(pos_bins)

            with torch.no_grad():
                model(ids)

            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    labels = np.concatenate(all_labels)  # (n_tokens_total,)

    h_in_flat = {}
    h_out_flat = {}
    for li in range(n_layers):
        if h_in_store[li]:
            # Each entry: (1, seq_len, hidden) → flatten to (seq_len, hidden)
            h_in_flat[li] = torch.cat([x.squeeze(0) for x in h_in_store[li]], dim=0).numpy()
        if h_out_store[li]:
            h_out_flat[li] = torch.cat([x.squeeze(0) for x in h_out_store[li]], dim=0).numpy()

    return h_in_flat, h_out_flat, labels


# ── K-fold probe evaluation ───────────────────────────────────────────────────

def kfold_probe_acc(X, y, device, n_splits=K_FOLDS):
    """Return mean test accuracy over k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs = []
    hidden_size = X.shape[1]
    for train_idx, test_idx in skf.split(X, y):
        probe = train_probe(X[train_idx], y[train_idx], hidden_size, device)
        acc = eval_probe(probe, X[test_idx], y[test_idx], device)
        accs.append(acc)
        del probe
        torch.cuda.empty_cache()
    return float(np.mean(accs))


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("e008 — Delta-Attribution Probe")
    print(f"N_PROMPTS={N_PROMPTS}, SEQ_LENGTHS={SEQ_LENGTHS}, K_FOLDS={K_FOLDS}\n")

    cfg = load_config()
    global_layers = get_global_layer_indices(cfg)
    global_set = set(global_layers)
    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    n_layers = len(model.model.language_model.layers)
    probe_device = torch.device('cuda:0')
    hidden_size = cfg.hidden_size if hasattr(cfg, 'hidden_size') else 5120

    print(f"Global layers: {global_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"Probe device: {probe_device}\n")

    rng = np.random.default_rng(SEED)
    results = []

    for seq_len in SEQ_LENGTHS:
        print(f"\n{'='*60}")
        print(f"seq_len = {seq_len}")

        # Collect texts
        texts = []
        for i in range(N_PROMPTS):
            lang = 'en_edgar' if i < N_PROMPTS // 2 else 'en'
            try:
                t = get_text_sample(lang, min_tokens=seq_len + 64,
                                    tokenizer=tokenizer, seed=SEED + i)
                texts.append(t)
            except Exception:
                try:
                    t = get_text_sample('en', min_tokens=seq_len + 64,
                                        tokenizer=tokenizer, seed=SEED + i)
                    texts.append(t)
                except Exception as e:
                    print(f"  [corpus] skip prompt {i}: {e}")

        print(f"  Collected {len(texts)} prompts")
        if len(texts) < 50:
            print("  ⚠️  Fewer than 50 prompts — result will be EXPLORATORY")

        # Extract hidden states (batch by 50 to avoid VRAM pressure)
        h_in_all = {li: [] for li in range(n_layers)}
        h_out_all = {li: [] for li in range(n_layers)}
        labels_all = []

        batch_size = 50
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            print(f"  Extracting batch {batch_start//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...",
                  flush=True)
            h_in, h_out, labels = extract_hidden_states(
                model, tokenizer, batch, seq_len, global_set, n_layers, model.device)
            for li in range(n_layers):
                if li in h_in:
                    h_in_all[li].append(h_in[li])
                if li in h_out:
                    h_out_all[li].append(h_out[li])
            labels_all.append(labels)

        labels_concat = np.concatenate(labels_all)

        for li in range(n_layers):
            if li not in h_in_all or not h_in_all[li]:
                continue
            X_in = np.concatenate(h_in_all[li], axis=0)
            X_out = np.concatenate(h_out_all[li], axis=0)

            layer_type = 'global_nope' if li in global_set else 'swa'

            # Evaluate probes
            acc_in = kfold_probe_acc(X_in, labels_concat, probe_device)
            acc_out = kfold_probe_acc(X_out, labels_concat, probe_device)
            delta = acc_out - acc_in

            print(f"  layer {li:2d} ({layer_type:12s})  "
                  f"acc_in={acc_in:.4f}  acc_out={acc_out:.4f}  Δ={delta:+.4f}")

            results.append({
                'seq_len': seq_len,
                'layer_idx': li,
                'layer_type': layer_type,
                'acc_in': acc_in,
                'acc_out': acc_out,
                'delta_acc': delta,
                'n_tokens': len(labels_concat),
            })

            del X_in, X_out
            torch.cuda.empty_cache()

    # Save results
    with open(OUT_DIR / 'results.jsonl', 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\nResults saved: {OUT_DIR}/results.jsonl")

    # Build comparisons for auto_validate
    # For each Global layer: compare delta_acc vs zero (H1_alt: delta > 0)
    global_deltas = {}
    for li in global_layers:
        layer_results = [r for r in results if r['layer_idx'] == li]
        if layer_results:
            global_deltas[li] = [r['delta_acc'] for r in layer_results]

    comparisons = {
        f"delta_acc_global_layer_{li}": (vals, [0.0] * len(vals))
        for li, vals in global_deltas.items()
    }

    if comparisons:
        validate_experiment(
            experiment_id="e008",
            comparisons=comparisons,
            output_dir=OUT_DIR,
            label_a="h_out",
            label_b="h_in_baseline",
            repo_root=Path('/home/elicer/sda212331'),
        )
    else:
        print("[auto_validate] Skipped — no Global layer data")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Mean delta_acc by layer type")
    for ltype in ['global_nope', 'swa']:
        vals = [r['delta_acc'] for r in results if r['layer_type'] == ltype]
        if vals:
            print(f"  {ltype:12s}: mean Δ={np.mean(vals):+.4f}  std={np.std(vals):.4f}  n={len(vals)}")

    print(f"\nOutputs: {OUT_DIR}")


if __name__ == '__main__':
    run()
