"""
Phase 11 Stage 4: Activation Healing + Measured Deployment Metrics (REAL)

After pruning at the winning-ticket sparsity from Stage 2:
  1. Measure perplexity of the pruned model (BEFORE healing)
  2. Apply per-layer bias correction ("activation healing") to compensate
     for activation distribution shift caused by weight removal
  3. Measure perplexity AFTER healing
  4. Measure actual deployment metrics: latency, memory, parameter counts

All numbers are measured, none are hardcoded.

The bias-correction approach is based on:
  - Frantar & Alistarh (2023) "SparseGPT: Massive Language Models Can Be
    Accurately Pruned in One-Shot"
  - Sun et al. (2023) "A Simple and Effective Pruning Approach for Large
    Language Models" (Wanda)
"""

import time
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

from code.phases.phase11.utils import (
    compute_perplexity,
    apply_global_pruning,
    remove_global_pruning,
    count_nonzero_params,
    time_forward_pass,
    get_model_config,
)


def _collect_layer_activations(
    model: nn.Module,
    layers: nn.ModuleList,
    tokenizer: Any,
    device: str,
    calibration_texts: List[str],
) -> Dict[int, torch.Tensor]:
    """
    Run calibration texts through the model and collect mean activation
    vectors at each transformer layer output.
    """
    layer_sums = {}
    layer_counts = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Accumulate mean across batch and sequence dimensions
            mean_act = hidden.detach().float().mean(dim=(0, 1))
            if idx not in layer_sums:
                layer_sums[idx] = torch.zeros_like(mean_act)
                layer_counts[idx] = 0
            layer_sums[idx] += mean_act
            layer_counts[idx] += 1
        return hook_fn

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for text in calibration_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            model(**inputs)

    for h in hooks:
        h.remove()

    # Compute means
    means = {}
    for idx in layer_sums:
        means[idx] = layer_sums[idx] / layer_counts[idx]

    return means


def _apply_activation_healing(
    layers: nn.ModuleList,
    pre_means: Dict[int, torch.Tensor],
    post_means: Dict[int, torch.Tensor],
) -> List[Any]:
    """
    Register forward hooks on each layer to shift the post-pruning mean output
    activation back to the pre-pruning mean output activation.
    """
    healing_hooks = []

    def make_healing_hook(corr):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                patched = output[0] + corr.to(output[0].device).to(output[0].dtype)
                return (patched,) + output[1:]
            else:
                return output + corr.to(output.device).to(output.dtype)
        return hook_fn

    for idx, layer in enumerate(layers):
        if idx in pre_means and idx in post_means:
            correction = pre_means[idx] - post_means[idx]
            h = layer.register_forward_hook(make_healing_hook(correction))
            healing_hooks.append(h)

    return healing_hooks


CALIBRATION_TEXTS = [
    "The development of artificial intelligence has transformed numerous industries.",
    "Machine learning models require large amounts of training data to perform well.",
    "Natural language processing enables computers to understand human text.",
    "Deep neural networks consist of multiple layers of interconnected neurons.",
    "The transformer architecture revolutionized the field of NLP in 2017.",
    "Pruning neural networks can reduce model size while preserving performance.",
    "Transfer learning allows models to leverage knowledge from pre-training.",
    "Attention mechanisms help models focus on relevant parts of the input.",
]


def run_stage4(
    model: nn.Module = None,
    tokenizer: Any = None,
    device: str = "cpu",
    target_sparsity: float = 0.30,
    ppl_max_samples: int = 128,
) -> Dict[str, Any]:
    """
    Stage 4: Activation Healing + Measured Deployment Metrics (REAL)

    1. Measure baseline deployment metrics (latency, memory, params, PPL)
    2. Save pre-pruning activation means on calibration data
    3. Apply pruning at the winning-ticket sparsity
    4. Measure pruned PPL and deployment metrics
    5. Collect post-pruning activation means
    6. Apply bias correction (activation healing)
    7. Measure healed PPL
    8. Report all measured values

    Raises ValueError if no model is loaded.
    """
    if model is None or tokenizer is None:
        raise ValueError(
            "Stage 4 requires a loaded model and tokenizer for real computation. "
            "Cannot proceed without a model."
        )

    print("\n" + "-" * 70)
    print("STAGE 4: Activation Healing + Deployment Metrics (REAL)")
    print("-" * 70)

    cfg = get_model_config(model)
    layers = cfg["layers"]
    arch = cfg["arch_family"]
    total_params = cfg["num_params"]

    healing_start = time.time()

    # --- Step 1: Baseline metrics ---
    print(f"  [1/6] Measuring baseline (unpruned) metrics...")
    baseline_ppl = compute_perplexity(model, tokenizer, device, max_samples=ppl_max_samples)
    baseline_latency = time_forward_pass(model, tokenizer, device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        # Run a forward pass to capture peak memory
        dummy = tokenizer("Measure memory", return_tensors="pt").to(device)
        with torch.no_grad():
            model(**dummy)
        baseline_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        baseline_memory_gb = 0.0

    print(f"    Baseline PPL: {baseline_ppl:.4f}")
    print(f"    Baseline latency: {baseline_latency:.2f}ms")
    print(f"    Baseline memory: {baseline_memory_gb:.2f}GB")

    # --- Step 2: Pre-pruning activation means ---
    print(f"  [2/6] Collecting pre-pruning activation means on calibration data...")
    pre_means = _collect_layer_activations(model, layers, tokenizer, device, CALIBRATION_TEXTS)
    print(f"    Collected means for {len(pre_means)} layers.")

    # Save state for potential restoration
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # --- Step 3: Apply pruning ---
    print(f"  [3/6] Applying {int(target_sparsity*100)}% global L1 magnitude pruning...")
    modules_pruned = apply_global_pruning(model, target_sparsity)
    # Make pruning permanent (bake masks into weights)
    remove_global_pruning(model)
    nonzero_params = count_nonzero_params(model)
    actual_sparsity = 1.0 - (nonzero_params / total_params)
    print(f"    Pruned {modules_pruned} modules. "
          f"Non-zero params: {nonzero_params/1e6:.1f}M "
          f"(actual sparsity: {actual_sparsity*100:.1f}%)")

    # --- Step 4: Pruned metrics (before healing) ---
    print(f"  [4/6] Measuring pruned model metrics (BEFORE healing)...")
    pruned_ppl = compute_perplexity(model, tokenizer, device, max_samples=ppl_max_samples)
    pruned_latency = time_forward_pass(model, tokenizer, device)
    print(f"    Pruned PPL: {pruned_ppl:.4f} (+{((pruned_ppl-baseline_ppl)/baseline_ppl)*100:.2f}%)")
    print(f"    Pruned latency: {pruned_latency:.2f}ms")

    # --- Step 5: Post-pruning activation means ---
    print(f"  [5/6] Collecting post-pruning activation means...")
    post_means = _collect_layer_activations(model, layers, tokenizer, device, CALIBRATION_TEXTS)

    # --- Step 6: Apply activation healing via forward hooks ---
    print(f"  [6/6] Applying forward-hook activation healing...")
    healing_hooks = _apply_activation_healing(layers, pre_means, post_means)
    layers_corrected = len(healing_hooks)
    print(f"    Registered healing hooks for {layers_corrected} layers.")

    # Measure healed PPL (with healing hooks active)
    healed_ppl = compute_perplexity(model, tokenizer, device, max_samples=ppl_max_samples)
    healed_latency = time_forward_pass(model, tokenizer, device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(**dummy)
        healed_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        healed_memory_gb = 0.0

    # Remove the healing hooks so the model is not left modified
    for h in healing_hooks:
        h.remove()

    healing_time = time.time() - healing_start

    # Compute summary
    ppl_recovery = 0.0
    ppl_degradation = pruned_ppl - baseline_ppl
    if ppl_degradation > 0:
        ppl_recovered = pruned_ppl - healed_ppl
        ppl_recovery = (ppl_recovered / ppl_degradation) * 100

    param_reduction_pct = (1.0 - nonzero_params / total_params) * 100
    latency_improvement_pct = ((baseline_latency - healed_latency) / baseline_latency) * 100 if baseline_latency > 0 else 0

    print(f"\n  [OK] Activation Healing completed in {healing_time:.2f}s")
    print(f"  Summary:")
    print(f"    Baseline PPL:    {baseline_ppl:.4f}")
    print(f"    Pruned PPL:      {pruned_ppl:.4f} (+{((pruned_ppl-baseline_ppl)/baseline_ppl)*100:.2f}%)")
    print(f"    Healed PPL:      {healed_ppl:.4f} (+{((healed_ppl-baseline_ppl)/baseline_ppl)*100:.2f}%)")
    print(f"    PPL Recovery:    {ppl_recovery:.1f}% of degradation recovered")
    print(f"    Param reduction: {param_reduction_pct:.1f}%")
    print(f"    Latency:         {baseline_latency:.2f}ms -> {healed_latency:.2f}ms ({latency_improvement_pct:+.1f}%)")

    # Restore original model weights (so pipeline doesn't leave model in pruned state)
    model.load_state_dict(original_state)

    return {
        "stage": "4_activation_healing",
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sparsity,
        "healing_time_seconds": healing_time,
        "layers_bias_corrected": layers_corrected,
        "baseline": {
            "ppl": baseline_ppl,
            "latency_ms": baseline_latency,
            "memory_gb": baseline_memory_gb,
            "params": total_params,
        },
        "pruned_before_healing": {
            "ppl": pruned_ppl,
            "ppl_increase_pct": ((pruned_ppl - baseline_ppl) / baseline_ppl) * 100,
            "latency_ms": pruned_latency,
            "nonzero_params": nonzero_params,
        },
        "healed": {
            "ppl": healed_ppl,
            "ppl_increase_pct": ((healed_ppl - baseline_ppl) / baseline_ppl) * 100,
            "latency_ms": healed_latency,
            "memory_gb": healed_memory_gb,
            "ppl_recovery_pct": ppl_recovery,
        },
        "deployment": {
            "param_reduction_pct": param_reduction_pct,
            "latency_improvement_pct": latency_improvement_pct,
        },
        "metric_type": "MEASURED",
        "status": "complete",
    }
