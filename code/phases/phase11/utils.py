"""
Utility functions for Phase 11 — Real Pruning & Measurement Pipeline

Contains the shared infrastructure used by all 4 stages:
  - compute_perplexity():  Wikitext-2 based perplexity measurement
  - get_model_config():    Architecture introspection for Llama/Mistral-style decoders
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
    metadata for Llama/Mistral-style HuggingFace decoder families.

    Returns dict with keys:
        num_layers, num_heads, num_kv_heads, head_dim, hidden_size,
        num_params, arch_family, layers_accessor
    """
    num_params = sum(p.numel() for p in model.parameters())
    config = model.config

    if hasattr(model, "model") and hasattr(model.model, "layers"):
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
            "Expected a Llama/Mistral-style model with model.model.layers."
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
# Perplexity measurement  (Wikitext-2 and held-out corpora)
# ---------------------------------------------------------------------------

def _eval_ppl_on_ids(model, input_ids, device, seq_len, max_samples, batch_size, label):
    """Shared PPL evaluation loop over pre-tokenized input_ids."""
    model_max_len = getattr(model.config, "max_position_embeddings",
                            getattr(model.config, "n_positions", seq_len))
    seq_len = min(seq_len, model_max_len)

    total_tokens = input_ids.size(0)
    num_chunks = min(max_samples, total_tokens // seq_len)

    if num_chunks == 0:
        raise ValueError(
            f"{label} too short ({total_tokens} tokens) for "
            f"seq_len={seq_len}. Try a smaller seq_len."
        )

    model.eval()
    total_loss = 0.0
    total_count = 0

    print(f"    [{label}] Evaluating {num_chunks} chunks of {seq_len} tokens on {device} (batch_size={batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, num_chunks, batch_size), desc=f"    [{label}] batches", leave=False):
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

    print(f"    [{label}] Result: {ppl:.4f}  (avg CE loss: {avg_loss:.4f}, tokens: {total_count})")
    return ppl


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
    """
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([s for s in dataset["text"] if s.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    return _eval_ppl_on_ids(model, input_ids, device, seq_len, max_samples, batch_size, "PPL")


def compute_perplexity_c4(
    model: nn.Module,
    tokenizer: Any,
    device: str,
    max_samples: int = 128,
    seq_len: int = 2048,
    batch_size: int = 1,
) -> float:
    """
    Compute perplexity on C4 validation split (held-out corpus).
    
    This provides a generalization check that is distribution-independent
    from the WikiText-2 training data used for LoRA fine-tuning.
    """
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    # Collect enough text from the streaming dataset
    texts = []
    for i, example in enumerate(dataset):
        if i >= max_samples * 2:  # grab extra to ensure enough tokens
            break
        if example["text"].strip():
            texts.append(example["text"])
    
    text = "\n\n".join(texts)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    return _eval_ppl_on_ids(model, input_ids, device, seq_len, max_samples, batch_size, "C4-PPL")


# ---------------------------------------------------------------------------
# Global magnitude pruning helpers
# ---------------------------------------------------------------------------

def apply_layerwise_pruning(model: nn.Module, sparsity: float) -> int:
    """
    Apply layer-uniform unstructured L1 magnitude pruning to all Linear layers.
    
    NOTE: This is *layer-uniform* pruning (each layer independently loses the
    same fraction of weights), NOT true global pruning (single threshold across
    all parameters). We use this terminology deliberately — see Phase 13 audit.
    
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


# Backward-compatible alias
apply_global_pruning = apply_layerwise_pruning


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
