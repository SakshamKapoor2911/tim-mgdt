"""
Phase 11 Stage 3: Causal Activation Patching (REAL)

Implements actual activation patching via PyTorch forward hooks:
  1. Run a clean prompt and cache per-layer activations
  2. Run a corrupted prompt to get baseline corrupted logits
  3. For each layer: run corrupted prompt but replace that layer's output
     with the cached clean activation, measure logit recovery via KL-divergence

This is the mechanistic interpretability technique used in:
  - Conmy et al. (2023) "Towards Automated Circuit Discovery"
  - Meng et al. (2022) "Locating and Editing Factual Knowledge in GPT"
"""

import time
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.phases.phase11.utils import get_model_config


def run_stage3(
    model: nn.Module = None,
    tokenizer: Any = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Stage 3: Causal Activation Patching (REAL)

    For each transformer layer, measures how much patching the clean
    activation into a corrupted forward pass recovers the clean output
    logits.  Recovery is measured via KL-divergence:

        recovery = 1 - KL(patched || clean) / KL(corrupted || clean)

    A recovery of 1.0 means that layer fully restores the clean output.
    A recovery of 0.0 means patching that layer has no effect.

    Raises ValueError if no model is loaded.
    """
    if model is None or tokenizer is None:
        raise ValueError(
            "Stage 3 requires a loaded model and tokenizer for real "
            "causal activation patching. Cannot proceed without a model."
        )

    print("\n" + "-" * 70)
    print("STAGE 3: Causal Activation Patching (REAL)")
    print("-" * 70)

    cfg = get_model_config(model)
    arch = cfg["arch_family"]
    layers = cfg["layers"]
    num_layers = cfg["num_layers"]

    clean_prompt = "The prime minister arrived in London today."
    corrupted_prompt = "The chef cooked a delicious pasta today."
    print(f"  Clean prompt:     '{clean_prompt}'")
    print(f"  Corrupted prompt: '{corrupted_prompt}'")
    print(f"  Layers to patch:  {num_layers}")

    model.eval()
    patching_start = time.time()

    # --- Step 1: Cache clean activations ---
    print("  [1/4] Caching clean activations across all layers...")
    clean_cache = {}
    hooks = []

    def make_cache_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple for most transformer layers; first element is hidden state
            if isinstance(output, tuple):
                clean_cache[layer_idx] = output[0].detach().clone()
            else:
                clean_cache[layer_idx] = output.detach().clone()
        return hook_fn

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_cache_hook(i))
        hooks.append(h)

    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**clean_inputs)

    for h in hooks:
        h.remove()

    print(f"    Cached {len(clean_cache)} layer activations.")

    # --- Step 2: Get clean logits and corrupted logits ---
    print("  [2/4] Computing clean and corrupted output logits...")
    with torch.no_grad():
        clean_logits = model(**clean_inputs).logits[:, -1, :]  # last token logits
        corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(device)
        corrupted_logits = model(**corrupted_inputs).logits[:, -1, :]

    # Compute baseline KL(corrupted || clean)
    clean_probs = F.softmax(clean_logits, dim=-1)
    corrupted_log_probs = F.log_softmax(corrupted_logits, dim=-1)
    kl_baseline = F.kl_div(corrupted_log_probs, clean_probs, reduction='batchmean').item()

    print(f"    KL(corrupted || clean) = {kl_baseline:.4f}")

    if kl_baseline < 1e-6:
        print("  [WARN] Clean and corrupted logits are nearly identical! "
              "Recovery scores will be unreliable. Try more divergent prompts.")

    # --- Step 3: Patch each layer and measure recovery ---
    print(f"  [3/4] Patching each layer and measuring KL-divergence recovery...")
    causal_scores = {}

    for target_layer_idx in range(num_layers):
        cached_activation = clean_cache[target_layer_idx]

        def make_patch_hook(cached):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    # Replace hidden state, keep other outputs (e.g. attention weights)
                    # Handle shape mismatch: corrupted sequence may differ in length
                    patched = cached[:, :output[0].shape[1], :] if cached.shape[1] >= output[0].shape[1] else cached
                    return (patched,) + output[1:]
                else:
                    return cached[:, :output.shape[1], :] if cached.shape[1] >= output.shape[1] else cached
            return hook_fn

        hook = layers[target_layer_idx].register_forward_hook(
            make_patch_hook(cached_activation)
        )

        with torch.no_grad():
            patched_logits = model(**corrupted_inputs).logits[:, -1, :]

        hook.remove()

        # KL(patched || clean)
        patched_log_probs = F.log_softmax(patched_logits, dim=-1)
        kl_patched = F.kl_div(patched_log_probs, clean_probs, reduction='batchmean').item()

        # Recovery: 1 means full restoration, 0 means no effect
        if kl_baseline > 1e-6:
            recovery = max(0.0, 1.0 - (kl_patched / kl_baseline))
        else:
            recovery = 0.0

        causal_scores[f"layer_{target_layer_idx}"] = {
            "recovery_score": recovery,
            "kl_patched": kl_patched,
            "kl_baseline": kl_baseline,
        }

    patching_time = time.time() - patching_start

    # --- Step 4: Rank layers ---
    print(f"  [4/4] Ranking layers by causal importance...")
    sorted_layers = sorted(
        causal_scores.items(),
        key=lambda x: x[1]["recovery_score"],
        reverse=True,
    )

    critical_layers = [l[0] for l in sorted_layers[:5]]
    safe_layers = [l[0] for l in sorted_layers[-5:]]

    print(f"  [OK] Patching completed in {patching_time:.2f}s")
    print(f"    Critical layers (highest recovery):  {critical_layers}")
    for l in sorted_layers[:5]:
        print(f"      {l[0]}: recovery={l[1]['recovery_score']:.4f}")
    print(f"    Safe-to-prune layers (lowest recovery): {safe_layers}")
    for l in sorted_layers[-5:]:
        print(f"      {l[0]}: recovery={l[1]['recovery_score']:.4f}")

    return {
        "stage": "3_causal_patching",
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "kl_baseline": kl_baseline,
        "patching_time_seconds": patching_time,
        "causal_scores": {k: v for k, v in causal_scores.items()},
        "critical_layers": critical_layers,
        "safe_to_prune_layers": safe_layers,
        "metric_type": "MEASURED",
        "status": "complete",
    }
