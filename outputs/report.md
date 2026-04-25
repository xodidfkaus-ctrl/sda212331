# EXAONE 4.5 Weight Analysis Report
Generated: 2026-04-25 03:18:05


## EXAONE 4.5 Architecture Reference
- **Total params**: 33B (LM 31.7B + Vision 1.29B)
- **Layers**: 64 main + 1 MTP
- **Hybrid Attention**: 16 × (3 SWA + 1 Global/NoPE)
- **GQA**: 40 Q-heads / 8 KV-heads
- **Reordered Norm**: after Attn/MLP, before residual
- **Vocab**: 153,600 | **Context**: 262,144 tokens


## Run Configuration
```json
{
  "repo_id": "LGAI-EXAONE/EXAONE-4.5-33B",
  "model_path": null,
  "spectral": true,
  "spectral_max_dim": 2048,
  "timestamp": "2026-04-25 03:10:09"
}
```

## Category Summary

| Category           | Tensors | Params (M) | Std (mean) | EffRank (mean) | Sparsity |
| ------------------ | ------- | ---------- | ---------- | -------------- | -------- |
| embedding          | 1       | 786.4      | 0.00729    | N/A            | 0.0001   |
| global_attn_k_proj | 16      | 83.9       | 0.00890    | 179.1          | 0.0001   |
| global_attn_o_proj | 16      | 419.4      | 0.00910    | N/A            | 0.0001   |
| global_attn_q_proj | 16      | 419.4      | 0.00884    | N/A            | 0.0001   |
| global_attn_v_proj | 16      | 83.9       | 0.00953    | 418.2          | 0.0001   |
| mlp                | 195     | 27348.2    | 0.00946    | N/A            | 0.0001   |
| mtp_layer          | 4       | 52.4       | 0.17212    | N/A            | 0.0000   |
| norm               | 260     | 0.7        | 0.10409    | N/A            | 0.0000   |
| other              | 1       | 0.0        | 0.23359    | N/A            | 0.0000   |
| output_head        | 1       | 786.4      | 0.01352    | N/A            | 0.0001   |
| swa_attn_k_proj    | 49      | 256.9      | 0.00895    | 172.8          | 0.0001   |
| swa_attn_o_proj    | 49      | 1284.5     | 0.00921    | N/A            | 0.0001   |
| swa_attn_q_proj    | 49      | 1284.5     | 0.00886    | N/A            | 0.0001   |
| swa_attn_v_proj    | 49      | 256.9      | 0.00963    | 409.7          | 0.0001   |
| vision_encoder     | 342     | 1286.5     | 0.03777    | 320.5          | 0.0002   |

## Key Observations

### SWA vs Global Attention
- Check `attn_std_by_layer.png` — Global (NoPE) layers typically show different weight scale
- NoPE global layers encode position implicitly via attention patterns, not weights

### Reordered Norm
- Norm weights after Attn/MLP (pre-residual) — different from standard Pre/Post norm
- Check `norm_weights.png` for magnitude consistency across layers

### MTP Layer
- Extra layer beyond 64 for multi-token prediction
- Compare its weight statistics to main layers

### Vision Encoder
- 1.29B params (~4% of total) with 2D RoPE
- Separate from LM — weight scale likely different from language layers

## Output Files
- `stats.jsonl` — raw per-tensor statistics (all runs appended)
- `category_summary.png` — param count & std by category
- `attn_std_by_layer.png` — SWA vs Global attention weight std
- `effective_rank_by_layer.png` — effective rank per layer
- `norm_weights.png` — Reordered Norm weight magnitudes