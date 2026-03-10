"""
Dataset Manager: Load & cache calibration data from HuggingFace.
Implements local caching to avoid re-downloading for reproducibility.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

CACHE_DIR = Path(__file__).parent.parent / "data" / "calibration_cache"


def _cache_path(num_samples):
    """Return sample-count-specific cache file path."""
    return CACHE_DIR / f"wikitext2_{num_samples}samples.json"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_calibration_data(num_samples=100, force_reload=False):
    """
    Load WikiText-2 calibration samples.
    
    Args:
        num_samples: Number of random samples to use (default: 100)
        force_reload: If True, re-download from HuggingFace instead of using cache
    
    Returns:
        List of text samples (strings)
    """
    ensure_cache_dir()
    cache_file = _cache_path(num_samples)
    
    # Check if cached data exists for this sample count
    if cache_file.exists() and not force_reload:
        print(f"Loading calibration data from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            return cached["samples"]
    
    # Download from HuggingFace
    print(f"Downloading WikiText-2 calibration data (this happens only once)...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        print(f"Downloaded {len(dataset)} texts from WikiText-2")
        
        # Filter out empty/very short texts and sample
        valid_texts = [
            text.strip() 
            for text in dataset["text"] 
            if len(text.strip()) > 50  # At least 50 chars for meaningful activations
        ]
        
        # Take random sample if we have too many
        if len(valid_texts) > num_samples:
            import random
            samples = random.sample(valid_texts, num_samples)
        else:
            samples = valid_texts
        
        print(f"Selected {len(samples)} valid samples for calibration")
        
        # Cache locally
        cache_data = {"samples": samples, "num_samples": len(samples)}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cached calibration data to: {cache_file}")
        
        return samples
    
    except Exception as e:
        print(f"Warning: Could not load WikiText-2 ({e})")
        print("Falling back to synthetic calibration texts...")
        return get_fallback_calibration()

def get_fallback_calibration():
    """Fallback calibration data if dataset loading fails."""
    return [
        "The rapid advancement of artificial intelligence has transformed numerous industries, "
        "from healthcare diagnostics to financial forecasting. Machine learning models now power "
        "recommendation systems that billions of people rely on daily.",
        
        "Einstein's theory of relativity revolutionized our understanding of space and time. "
        "The relationship between energy and mass, expressed in the famous equation E=mc², "
        "provided the theoretical foundation for nuclear physics.",
        
        "Climate change represents one of the most pressing challenges of our era. Rising global "
        "temperatures are causing sea levels to rise, threatening coastal ecosystems and communities "
        "worldwide.",
        
        "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic "
        "connections. This intricate network enables complex cognition, emotion, and motor control.",
        
        "Quantum computing harnesses the principles of quantum mechanics to perform computations far "
        "beyond the reach of classical computers. Superposition and entanglement allow quantum bits to "
        "explore vast solution spaces simultaneously.",
    ]

def reset_cache():
    """Delete all cached calibration data (for debugging or refresh)."""
    import glob
    for f in CACHE_DIR.glob("wikitext2_*.json"):
        f.unlink()
        print(f"Deleted cache file: {f}")
