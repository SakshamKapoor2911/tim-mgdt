from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelCapabilities:
    supports_hooks: bool
    supports_8bit: bool
    supports_4bit: bool
    min_vram_gb: float
    max_seq_len: int
    architecture_family: str

MODEL_REGISTRY: Dict[str, ModelCapabilities] = {
    "mistral-7b-v0.1": ModelCapabilities(
        supports_hooks=True,
        supports_8bit=True,
        supports_4bit=True,
        min_vram_gb=16.0,
        max_seq_len=32768,
        architecture_family="llama"  # Mistral uses llama-like architecture
    ),
    "llama-3-8b": ModelCapabilities(
        supports_hooks=True,
        supports_8bit=True,
        supports_4bit=True,
        min_vram_gb=18.0,
        max_seq_len=8192,
        architecture_family="llama"
    )
}

def get_model_capabilities(model_id: str) -> Optional[ModelCapabilities]:
    # Try exact match
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]
    
    # Try fuzzy match for HF names
    for key, caps in MODEL_REGISTRY.items():
        if key in model_id.lower():
            return caps
            
    return None

def is_hardware_sufficient(model_id: str, available_vram_gb: float) -> bool:
    caps = get_model_capabilities(model_id)
    if not caps:
        return False
    return available_vram_gb >= caps.min_vram_gb
