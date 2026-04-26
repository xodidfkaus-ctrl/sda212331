"""
EXAONE 4.5 model loader for NoPE analysis.
Patches CONFIG_MAPPING and loads model with attention output enabled.
"""
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.exaone4.configuration_exaone4 import Exaone4Config
CONFIG_MAPPING.register('exaone4_5_text', Exaone4Config)

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.exaone4_5.modeling_exaone4_5 import Exaone4_5_ForConditionalGeneration

# ⚠️ 클라우드 환경이 바뀌면 이 경로를 수정해야 함 (홈 디렉토리가 /home/elicer/ 가 아닐 수 있음)
MODEL_PATH = '/home/elicer/sda212331/model_cache/models--LGAI-EXAONE--EXAONE-4.5-33B/snapshots/58d6616991a60a67f84be82ad241d5bc9668a55c'


def get_layer_types(cfg) -> list[str]:
    """Returns list of 'sliding_attention' or 'full_attention' per layer."""
    return cfg.text_config.layer_types


def get_global_layer_indices(cfg) -> list[int]:
    return [i for i, t in enumerate(get_layer_types(cfg)) if t == 'full_attention']


def get_swa_layer_indices(cfg) -> list[int]:
    return [i for i, t in enumerate(get_layer_types(cfg)) if t == 'sliding_attention']


def load_model_and_tokenizer(device_map='auto', dtype=torch.bfloat16):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model (this may take a few minutes)...")
    model = Exaone4_5_ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        device_map=device_map,
        attn_implementation='eager',  # flash_attn은 attention weights 미반환
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def load_config():
    return AutoConfig.from_pretrained(MODEL_PATH)
