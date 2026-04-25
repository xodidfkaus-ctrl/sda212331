"""
Markdown report generator for EXAONE 4.5 weight analysis.
"""
import json
from datetime import datetime
from pathlib import Path


EXAONE45_ARCH_NOTE = """
## EXAONE 4.5 Architecture Reference
- **Total params**: 33B (LM 31.7B + Vision 1.29B)
- **Layers**: 64 main + 1 MTP
- **Hybrid Attention**: 16 × (3 SWA + 1 Global/NoPE)
- **GQA**: 40 Q-heads / 8 KV-heads
- **Reordered Norm**: after Attn/MLP, before residual
- **Vocab**: 153,600 | **Context**: 262,144 tokens
"""


def render_table(headers: list, rows: list) -> str:
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    header = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return f"{header}\n{sep}\n{body}"


def generate_report(summary: dict, jsonl_path: str, out_path: str, run_config: dict):
    lines = [
        f"# EXAONE 4.5 Weight Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        EXAONE45_ARCH_NOTE,
        "",
        "## Run Configuration",
        "```json",
        json.dumps(run_config, indent=2),
        "```",
        "",
        "## Category Summary",
        "",
    ]

    headers = ["Category", "Tensors", "Params (M)", "Std (mean)", "EffRank (mean)", "Sparsity"]
    rows = []
    for cat, s in sorted(summary.items()):
        rows.append([
            cat,
            s["count"],
            f"{s['total_params']/1e6:.1f}",
            f"{s['std_mean']:.5f}",
            f"{s['effective_rank_mean']:.1f}" if s["effective_rank_mean"] else "N/A",
            f"{s['sparsity_mean']:.4f}",
        ])
    lines.append(render_table(headers, rows))
    lines.append("")

    lines += [
        "## Key Observations",
        "",
        "### SWA vs Global Attention",
        "- Check `attn_std_by_layer.png` — Global (NoPE) layers typically show different weight scale",
        "- NoPE global layers encode position implicitly via attention patterns, not weights",
        "",
        "### Reordered Norm",
        "- Norm weights after Attn/MLP (pre-residual) — different from standard Pre/Post norm",
        "- Check `norm_weights.png` for magnitude consistency across layers",
        "",
        "### MTP Layer",
        "- Extra layer beyond 64 for multi-token prediction",
        "- Compare its weight statistics to main layers",
        "",
        "### Vision Encoder",
        "- 1.29B params (~4% of total) with 2D RoPE",
        "- Separate from LM — weight scale likely different from language layers",
        "",
        "## Output Files",
        "- `stats.jsonl` — raw per-tensor statistics (all runs appended)",
        "- `category_summary.png` — param count & std by category",
        "- `attn_std_by_layer.png` — SWA vs Global attention weight std",
        "- `effective_rank_by_layer.png` — effective rank per layer",
        "- `norm_weights.png` — Reordered Norm weight magnitudes",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Report saved: {out}")
