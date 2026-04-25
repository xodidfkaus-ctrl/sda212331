"""
EXAONE 4.5 weight analysis — main entry point.
Usage:
    python analyze.py                          # full analysis
    python analyze.py --no-spectral            # skip SVD (faster)
    python analyze.py --model-path ./cache     # use already-downloaded model
    python analyze.py --spectral-max-dim 1024  # limit SVD size
"""
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from src.loader import iter_tensors, download_model
from src.stats import compute_stats, StatsAggregator
from src.viz import generate_all_plots
from src.report import generate_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=None, help="Local path to model (skip download)")
    p.add_argument("--repo-id", default="LGAI-EXAONE/EXAONE-4.5-33B")
    p.add_argument("--cache-dir", default="./model_cache")
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--no-spectral", action="store_true", help="Skip SVD computation")
    p.add_argument("--spectral-max-dim", type=int, default=2048,
                   help="Max tensor dim for SVD (smaller = faster)")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = str(out_dir / "stats.jsonl")

    run_config = {
        "repo_id": args.repo_id,
        "model_path": args.model_path,
        "spectral": not args.no_spectral,
        "spectral_max_dim": args.spectral_max_dim,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Download or use local
    if args.model_path:
        model_path = args.model_path
        print(f"Using local model: {model_path}")
    else:
        model_path = download_model(args.repo_id, args.cache_dir)

    print("Starting weight analysis (streaming)...")
    aggregator = StatsAggregator(jsonl_path)

    spectral_max_dim = 0 if args.no_spectral else args.spectral_max_dim

    for name, tensor, category, layer_idx in tqdm(iter_tensors(model_path), desc="Tensors"):
        stats = compute_stats(name, tensor, category, layer_idx, spectral_max_dim)
        aggregator.add(stats)

    aggregator.close()

    summary = aggregator.summary_by_category()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    generate_all_plots(jsonl_path, summary, str(out_dir))
    generate_report(summary, jsonl_path, str(out_dir / "report.md"), run_config)

    print("\nDone. Results in:", out_dir)
    print("  stats.jsonl      — raw tensor stats")
    print("  summary.json     — category aggregates")
    print("  report.md        — analysis report")
    print("  *.png            — charts")


if __name__ == "__main__":
    main()
