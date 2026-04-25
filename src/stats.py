"""
Per-tensor and per-category statistics for EXAONE 4.5 weight analysis.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class TensorStats:
    name: str
    category: str
    layer_idx: Optional[int]
    shape: list
    dtype: str
    numel: int
    mean: float
    std: float
    abs_mean: float
    min: float
    max: float
    l2_norm: float
    sparsity: float        # fraction of near-zero weights (|w| < 1e-6)
    effective_rank: Optional[float]   # sum(s)/max(s) for 2D tensors
    stable_rank: Optional[float]      # ||W||_F^2 / ||W||_2^2


def compute_stats(name: str, tensor: torch.Tensor, category: str,
                  layer_idx: Optional[int], spectral_max_dim: int = 2048) -> TensorStats:
    t = tensor.float()
    flat = t.reshape(-1)

    mean = flat.mean().item()
    std = flat.std().item()
    abs_mean = flat.abs().mean().item()
    tmin = flat.min().item()
    tmax = flat.max().item()
    l2 = flat.norm(2).item()
    sparsity = (flat.abs() < 1e-6).float().mean().item()

    effective_rank = None
    stable_rank = None

    if t.dim() == 2 and min(t.shape) <= spectral_max_dim:
        try:
            s = torch.linalg.svdvals(t)
            s_sum = s.sum().item()
            s_max = s[0].item()
            if s_max > 0:
                effective_rank = s_sum / s_max
            frob_sq = (t ** 2).sum().item()
            spec_sq = s_max ** 2
            if spec_sq > 0:
                stable_rank = frob_sq / spec_sq
        except Exception:
            pass

    return TensorStats(
        name=name,
        category=category,
        layer_idx=layer_idx,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        numel=tensor.numel(),
        mean=mean,
        std=std,
        abs_mean=abs_mean,
        min=tmin,
        max=tmax,
        l2_norm=l2,
        sparsity=sparsity,
        effective_rank=effective_rank,
        stable_rank=stable_rank,
    )


class StatsAggregator:
    """Collects per-tensor stats and aggregates by category."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a")
        self.records: list[TensorStats] = []

    def add(self, stats: TensorStats):
        self.records.append(stats)
        self._file.write(json.dumps(asdict(stats)) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()

    def summary_by_category(self) -> dict:
        from collections import defaultdict
        buckets: dict[str, list] = defaultdict(list)
        for r in self.records:
            buckets[r.category].append(r)

        summary = {}
        for cat, items in buckets.items():
            stds = [i.std for i in items]
            abs_means = [i.abs_mean for i in items]
            e_ranks = [i.effective_rank for i in items if i.effective_rank is not None]
            s_ranks = [i.stable_rank for i in items if i.stable_rank is not None]
            summary[cat] = {
                "count": len(items),
                "total_params": sum(i.numel for i in items),
                "std_mean": float(np.mean(stds)),
                "std_std": float(np.std(stds)),
                "abs_mean_mean": float(np.mean(abs_means)),
                "effective_rank_mean": float(np.mean(e_ranks)) if e_ranks else None,
                "stable_rank_mean": float(np.mean(s_ranks)) if s_ranks else None,
                "sparsity_mean": float(np.mean([i.sparsity for i in items])),
            }
        return summary
