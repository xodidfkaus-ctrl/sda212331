"""
seeds.py — Single source of truth for all random seeds.

Import and call set_all_seeds() at the top of every experiment script.
Never set seeds inline; always go through this module so reproduction
instructions are unambiguous.

Usage:
    from nope_analysis.seeds import set_all_seeds, EXPERIMENT_SEEDS
    set_all_seeds(EXPERIMENT_SEEDS["e001"])
"""

import random

import numpy as np

# Project-wide default seed (used when no experiment-specific override exists)
GLOBAL_SEED: int = 42

# Per-experiment seed overrides.
# All use GLOBAL_SEED by default; override here only if an experiment
# requires a different seed for a documented reason.
EXPERIMENT_SEEDS: dict[str, int] = {
    "e001": 42,
    "e002": 42,
    "e003": 42,
    "e004": 42,
    "e005": 42,
    "e006": 42,
    "e007": 42,
    "e008": 42,
    "e005b": 42,
    "e006b": 42,
    "e007b": 42,
    "e009": 42,
    "e010": 42,
}


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """
    Set seeds for random, numpy, torch, and transformers in one call.

    Call this once at the top of every experiment script, before any
    data loading or model initialization.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic mode — note: may reduce performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


def get_seed(experiment_id: str) -> int:
    """Return the registered seed for an experiment ID."""
    if experiment_id not in EXPERIMENT_SEEDS:
        raise KeyError(
            f"No seed registered for '{experiment_id}'. "
            f"Add it to EXPERIMENT_SEEDS in nope_analysis/seeds.py."
        )
    return EXPERIMENT_SEEDS[experiment_id]
