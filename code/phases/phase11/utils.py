"""
Utility functions for Phase 11 — Real Pruning & Measurement Pipeline

Contains the shared infrastructure used by all 4 stages:
  - compute_perplexity():  Wikitext-2 based perplexity measurement
  - get_model_config():    Architecture introspection (handles GPT-2 / Llama / Mistral)
  - apply_global_pruning(): Unstructured L1 magnitude pruning via torch.nn.utils.prune
  - remove_global_pruning(): Restore original weight buffers after a pruning sweep
  - time_forward_pass():    Measure median latency of a single forward pass
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Get appropriate device (cuda or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Model architecture introspection
# ---------------------------------------------------------------------------

def get_model_config(model: nn.Module) -> Dict[str, Any]:
    """
    Introspect a loaded HuggingFace causal-LM model and return architecture
    metadata.  Handles GPT-2 (model.transformer.h) and Llama/Mistral
    (model.model.layers) families.

    Returns dict with keys:
        num_layers, num_heads, num_kv_heads, head_dim, hidden_size,
        num_params, arch_family, layers_accessor
    """
    num_params = sum(p.numel() for p in model.parameters())
    config = model.config

    # Determine architecture family
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 family
        layers = model.transformer.h
        num_layers = config.n_layer
        num_heads = config.n_head
        num_kv_heads = config.n_head  # GPT-2 uses standard MHA
        hidden_size = config.n_embd
        head_dim = hidden_size // num_heads
        arch = "gpt2"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama / Mistral family
        layers = model.model.layers
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        arch = "llama"
    else:
        raise ValueError(
            f"Unsupported architecture: {type(model).__name__}. "
            "Expected GPT-2 (model.transformer.h) or Llama/Mistral (model.model.layers)."
        )

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "num_params": num_params,
        "arch_family": arch,
        "layers": layers,  # direct reference for hook registration
    }


def get_model_n_layers(model_name: str) -> int:
    """Get number of transformer layers for a model by name (static lookup)."""
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
    return 32


# ---------------------------------------------------------------------------
# Perplexity measurement  (Wikitext-2)
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    device: str,
    max_samples: int = 256,
    seq_len: int = 2048,
    batch_size: int = 1,
) -> float:
    """
    Compute perplexity on Wikitext-2 test split.

    This is the standard evaluation used in SparseGPT, Wanda, GPTQ, and
    virtually all LLM pruning papers.

    Args:
        model:       A HuggingFace causal LM (already on device).
        tokenizer:   Matching tokenizer.
        device:      'cuda' or 'cpu'.
        max_samples: Maximum number of sequence chunks to evaluate.
        seq_len:     Sequence length per chunk (2048 is standard for 7B models,
                     1024 works well for GPT-2).
        batch_size:  Evaluation batch size (1 is safest for VRAM).

    Returns:
        Perplexity as a float (e.g. 5.32 for Mistral-7B baseline).
    """
    from datasets import load_dataset

    # Load Wikitext-2 test set
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([s for s in dataset["text"] if s.strip()])

    # Tokenize the entire test set
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # shape: (total_tokens,)

    # Adjust seq_len for small models (GPT-2 max position = 1024)
    model_max_len = getattr(model.config, "max_position_embeddings",
                            getattr(model.config, "n_positions", seq_len))
    seq_len = min(seq_len, model_max_len)

    total_tokens = input_ids.size(0)
    num_chunks = min(max_samples, total_tokens // seq_len)

    if num_chunks == 0:
        raise ValueError(
            f"Wikitext-2 test set too short ({total_tokens} tokens) for "
            f"seq_len={seq_len}. Try a smaller seq_len."
        )

    model.eval()
    total_loss = 0.0
    total_count = 0

    print(f"    [PPL] Evaluating {num_chunks} chunks of {seq_len} tokens on {device} (batch_size={batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, num_chunks, batch_size), desc="    [PPL] batches", leave=False):
            batch_chunks = []
            for j in range(i, min(i + batch_size, num_chunks)):
                start = j * seq_len
                end = start + seq_len
                batch_chunks.append(input_ids[start:end])
                
            chunk_tensor = torch.stack(batch_chunks).to(device)
            labels = chunk_tensor.clone()

            outputs = model(input_ids=chunk_tensor, labels=labels)
            loss = outputs.loss
            
            batch_tokens = chunk_tensor.numel()
            total_loss += loss.item() * batch_tokens
            total_count += batch_tokens

    avg_loss = total_loss / total_count
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    print(f"    [PPL] Result: {ppl:.4f}  (avg CE loss: {avg_loss:.4f}, tokens: {total_count})")
    return ppl


# ---------------------------------------------------------------------------
# Global magnitude pruning helpers
# ---------------------------------------------------------------------------

def apply_global_pruning(model: nn.Module, sparsity: float) -> int:
    """
    Apply uniform unstructured L1 magnitude pruning to all Linear layers manually.
    
    This avoids torch.nn.utils.prune, which duplicates weights and causes CUDA OOM 
    by creating weight_orig and weight_mask tensors for every single layer.
    """
    modules_pruned = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                nparams_toprune = int(w.numel() * sparsity)
                if nparams_toprune == 0:
                    continue
                # Memory efficient layer-wise thresholding
                threshold = torch.kthvalue(w.abs().view(-1), nparams_toprune).values.item()
                w.mul_(w.abs() >= threshold)
                modules_pruned += 1
    return modules_pruned


def remove_global_pruning(model: nn.Module) -> int:
    """
    No-op. Since we apply masks directly to weight.data without PyTorch's hooks,
    there are no reparameterizations to remove. State is fully restored via load_state_dict.
    """
    return 0


def count_nonzero_params(model: nn.Module) -> int:
    """Count the number of non-zero parameters in the model."""
    total = 0
    for p in model.parameters():
        total += (p != 0).sum().item()
    return int(total)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def time_forward_pass(
    model: nn.Module,
    tokenizer: Any,
    device: str,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    n_trials: int = 10,
    warmup: int = 3,
) -> float:
    """
    Measure median forward-pass latency in milliseconds.

    Runs `warmup` passes first, then times `n_trials` passes and returns
    the median.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed trials
    latencies = []
    with torch.no_grad():
        for _ in range(n_trials):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    return median_latency
