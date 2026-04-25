"""
EXAONE 4.5 model weight streaming loader.
Streams tensors one at a time to avoid OOM on large models.
"""
import json
from pathlib import Path
from typing import Generator, Tuple

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


EXAONE45_CONFIG = {
    "hidden_size": 5120,
    "intermediate_size": 27392,
    "num_layers": 64,
    "mtp_layers": 1,
    "num_q_heads": 40,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 153600,
    "context_length": 262144,
    "sliding_window": 4096,
    # Hybrid pattern: 16 blocks of (3 SWA + 1 Global)
    # Global attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
    "global_attn_layers": [3 + 4*i for i in range(16)],
}


def is_global_attention(layer_idx: int) -> bool:
    return layer_idx in EXAONE45_CONFIG["global_attn_layers"]


def classify_tensor(name: str, layer_idx: int | None) -> str:
    """Classify tensor into analysis category."""
    if "embed_tokens" in name or "embed" in name.split(".")[-2:]:
        return "embedding"
    if "lm_head" in name or "output" in name.split(".")[-2:]:
        return "output_head"
    if layer_idx is not None:
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            attn_type = "global_attn" if is_global_attention(layer_idx) else "swa_attn"
            proj = "q" if "q_proj" in name else ("k" if "k_proj" in name else "v")
            return f"{attn_type}_{proj}_proj"
        if "o_proj" in name:
            attn_type = "global_attn" if is_global_attention(layer_idx) else "swa_attn"
            return f"{attn_type}_o_proj"
        if "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            return "mlp"
        if "norm" in name:
            return "norm"
    if "vision" in name or "visual" in name or "patch_embed" in name:
        return "vision_encoder"
    if "mtp" in name:
        return "mtp_layer"
    return "other"


def extract_layer_idx(name: str) -> int | None:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "layer") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def iter_tensors(model_path: str) -> Generator[Tuple[str, torch.Tensor, str, int | None], None, None]:
    """Yield (name, tensor, category, layer_idx) one at a time."""
    path = Path(model_path)
    shard_files = sorted(path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    for shard in shard_files:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                layer_idx = extract_layer_idx(name)
                category = classify_tensor(name, layer_idx)
                yield name, tensor, category, layer_idx
                del tensor


def download_model(repo_id: str = "LGAI-EXAONE/EXAONE-4.5-33B", cache_dir: str = "./model_cache") -> str:
    print(f"Downloading {repo_id} ...")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.bin", "*.pt", "original/*"],
    )
    return path
