"""
Utility functions for Phase 11
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import torch

def save_json(filepath: str, data: Any, indent: int = 2) -> None:
    """Save data to JSON file with directory creation."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    print(f"[OK] Saved: {filepath}")

def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_device() -> str:
    """Get appropriate device (cuda or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_model_n_layers(model_name: str) -> int:
    """Get number of transformer layers for a model."""
    model_configs = {
        'gpt2': 12,
        'gpt2-medium': 24,
        'gpt2-large': 36,
        'gpt2-xl': 48,
        'mistral-7b': 32,
        'llama-2-7b': 32,
        'llama-2-13b': 40,
        'llama-3-8b': 32,
        'llama-3-70b': 80,
    }
    
    model_lower = model_name.lower()
    for k, v in model_configs.items():
        if k in model_lower:
            return v
    return 32  # Default fallback
