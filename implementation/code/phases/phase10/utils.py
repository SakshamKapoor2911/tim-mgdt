"""
Utility functions for Phase 10
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import torch


def save_json(filepath: str, data: Any, indent: int = 2) -> None:
    """Save data to JSON file with directory creation."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    print(f"✓ Saved: {filepath}")


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_model_n_layers(model_name: str) -> int:
    """Get number of transformer layers for a model."""
    model_configs = {
        'gpt2': 12,
        'gpt2-medium': 24,
        'gpt2-large': 36,
        'gpt2-xl': 48,
        'mistral-7b': 32,
        'llama-2-7b': 32,
        'llama-3-8b': 32,
        'llama-3-70b': 80,
    }
    
    model_lower = model_name.lower()
    if model_lower in model_configs:
        return model_configs[model_lower]
    else:
        raise ValueError(f"Unknown model: {model_name}. Add to model_configs.")


def rank_layers_by_importance(
    stability: Dict[str, float],
    ascending: bool = True
) -> List[str]:
    """
    Rank layers from least to most important.
    
    Args:
        stability: Dict mapping layer names to stability scores
        ascending: If True, return least-to-most important (for pruning)
    
    Returns:
        Sorted list of layer names
    """
    sorted_layers = sorted(
        stability.items(),
        key=lambda x: x[1],
        reverse=not ascending
    )
    return [layer for layer, score in sorted_layers]


def get_device() -> str:
    """Get appropriate device (cuda or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def cleanup_pruned_models() -> None:
    """Clear GPU memory after model experiments."""
    torch.cuda.empty_cache()
