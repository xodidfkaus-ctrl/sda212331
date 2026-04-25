"""
Visualization for EXAONE 4.5 weight analysis results.
Uses Agg backend (no display needed on headless GPU server).
"""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(jsonl_path: str) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_std_by_layer(records: list[dict], out_dir: str):
    """Weight std per layer, colored by SWA vs Global attention."""
    layers: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["layer_idx"] is None:
            continue
        cat = r["category"]
        if "attn" in cat:
            attn_type = "global" if "global" in cat else "swa"
            layers[r["layer_idx"]][attn_type].append(r["std"])

    if not layers:
        return

    layer_ids = sorted(layers.keys())
    swa_stds = [np.mean(layers[l]["swa"]) if layers[l]["swa"] else np.nan for l in layer_ids]
    global_stds = [np.mean(layers[l]["global"]) if layers[l]["global"] else np.nan for l in layer_ids]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(layer_ids, swa_stds, label="Sliding Window Attn", color="steelblue", linewidth=1.5)
    ax.plot(layer_ids, global_stds, label="Global Attn (NoPE)", color="coral", linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Weight Std")
    ax.set_title("EXAONE 4.5 — Attention Weight Std by Layer (SWA vs Global)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "attn_std_by_layer.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_effective_rank_by_layer(records: list[dict], out_dir: str):
    """Effective rank per layer for Q/K/V projections."""
    data: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["layer_idx"] is None or r["effective_rank"] is None:
            continue
        cat = r["category"]
        if "proj" in cat and "attn" in cat:
            data[cat][r["layer_idx"]].append(r["effective_rank"])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.tab10.colors
    for i, (cat, layer_map) in enumerate(sorted(data.items())):
        xs = sorted(layer_map.keys())
        ys = [np.mean(layer_map[x]) for x in xs]
        ax.plot(xs, ys, label=cat, color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Effective Rank")
    ax.set_title("EXAONE 4.5 — Effective Rank by Layer & Projection Type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "effective_rank_by_layer.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_category_summary(summary: dict, out_dir: str):
    """Bar chart of param count and avg std by category."""
    cats = list(summary.keys())
    params = [summary[c]["total_params"] / 1e6 for c in cats]
    stds = [summary[c]["std_mean"] for c in cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.barh(cats, params, color="steelblue")
    ax1.set_xlabel("Parameters (M)")
    ax1.set_title("Parameter Count by Category")
    ax1.bar_label(bars1, fmt="%.0f", padding=3, fontsize=8)

    bars2 = ax2.barh(cats, stds, color="coral")
    ax2.set_xlabel("Mean Weight Std")
    ax2.set_title("Weight Std by Category")
    ax2.bar_label(bars2, fmt="%.4f", padding=3, fontsize=8)

    fig.tight_layout()
    out = Path(out_dir) / "category_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_norm_weights(records: list[dict], out_dir: str):
    """Norm layer weight distributions (Reordered Norm check)."""
    norm_records = [r for r in records if r["category"] == "norm" and r["layer_idx"] is not None]
    if not norm_records:
        return

    layers = [r["layer_idx"] for r in norm_records]
    means = [r["abs_mean"] for r in norm_records]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(layers, means, s=20, alpha=0.7, color="purple")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("|mean| of Norm Weights")
    ax.set_title("EXAONE 4.5 — Reordered Norm Weight Magnitudes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_dir) / "norm_weights.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def generate_all_plots(jsonl_path: str, summary: dict, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    records = load_records(jsonl_path)
    print("Generating plots...")
    plot_std_by_layer(records, out_dir)
    plot_effective_rank_by_layer(records, out_dir)
    plot_category_summary(summary, out_dir)
    plot_norm_weights(records, out_dir)
