"""
Phase 20: Mask-guided structured width pruning.

Phase 19 showed that dense-shaped unstructured magnitude masks can preserve
GSM8K reasoning at sparsities where uniform structural MLP-width pruning fails.
This phase asks the next hardware-relevant question: can the successful mask
signal rank whole MLP intermediate neurons, then delete low-signal neurons while
keeping Tensor Core-aligned dimensions?
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase11.utils import compute_perplexity_c4, get_device


def tag_from_sparsity(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        tag = text[2:]
        while len(tag) < 2:
            tag += "0"
        return f"s{tag}"
    return "s" + text.replace(".", "p")


def aligned_keep_size(intermediate_size: int, sparsity: float, strategy: str) -> int:
    target_size = intermediate_size * (1.0 - sparsity)
    if strategy == "ceil":
        keep_size = math.ceil(target_size / 64.0) * 64
    elif strategy == "floor":
        keep_size = math.floor(target_size / 64.0) * 64
    else:
        keep_size = int(target_size)
    min_keep = 64 if intermediate_size >= 64 else 1
    return min(intermediate_size, max(min_keep, keep_size))


def masked_axis_score(weight: torch.Tensor, axis: int, mask_sparsity: float) -> torch.Tensor:
    """Sum surviving magnitude along one axis after a layerwise magnitude mask."""
    abs_weight = weight.detach().abs()
    if mask_sparsity <= 0:
        return abs_weight.sum(dim=axis)

    flat = abs_weight.flatten()
    prune_count = int(flat.numel() * mask_sparsity)
    if prune_count <= 0:
        return abs_weight.sum(dim=axis)
    if prune_count >= flat.numel():
        return torch.zeros(abs_weight.shape[1 - axis], device=weight.device, dtype=torch.float32)

    threshold = torch.kthvalue(flat, prune_count).values
    kept = abs_weight > threshold
    return (abs_weight * kept).float().sum(dim=axis)


def compute_mask_guided_scores(model: nn.Module, mask_sparsity: float) -> dict[int, torch.Tensor]:
    layers = model.model.layers
    scores = {}
    print(f"[Phase20] Computing mask-guided MLP neuron scores with mask_sparsity={mask_sparsity:.4f}")
    for layer_idx, layer in enumerate(layers):
        gate_score = masked_axis_score(layer.mlp.gate_proj.weight, axis=1, mask_sparsity=mask_sparsity)
        up_score = masked_axis_score(layer.mlp.up_proj.weight, axis=1, mask_sparsity=mask_sparsity)
        down_score = masked_axis_score(layer.mlp.down_proj.weight, axis=0, mask_sparsity=mask_sparsity)
        scores[layer_idx] = gate_score + up_score + down_score
    return scores


def count_mlp_params(model: nn.Module) -> int:
    total = 0
    for layer in model.model.layers:
        total += layer.mlp.gate_proj.weight.numel()
        total += layer.mlp.up_proj.weight.numel()
        total += layer.mlp.down_proj.weight.numel()
    return total


def apply_mask_guided_width_pruning(
    model: nn.Module,
    scores: dict[int, torch.Tensor],
    sparsity: float,
    strategy: str,
) -> dict:
    layers = model.model.layers
    modules_pruned = 0
    before_params = count_mlp_params(model)
    before_size = layers[0].mlp.up_proj.weight.shape[0]
    keep_sizes = []

    print(
        "[Phase20] Applying mask-guided structured pruning "
        f"(target_sparsity={sparsity:.4f}, strategy={strategy})"
    )

    for layer_idx, layer in enumerate(layers):
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj

        intermediate_size = up_proj.weight.shape[0]
        hidden_size = up_proj.weight.shape[1]
        keep_size = aligned_keep_size(intermediate_size, sparsity, strategy)
        keep_sizes.append(keep_size)

        if keep_size == intermediate_size:
            continue

        top_indices = torch.topk(scores[layer_idx], keep_size).indices
        top_indices, _ = torch.sort(top_indices)
        top_indices = top_indices.to(gate_proj.weight.device)
        layer_device = gate_proj.weight.device

        new_gate = nn.Linear(
            hidden_size,
            keep_size,
            bias=False,
            device=layer_device,
            dtype=gate_proj.weight.dtype,
        )
        new_up = nn.Linear(
            hidden_size,
            keep_size,
            bias=False,
            device=layer_device,
            dtype=up_proj.weight.dtype,
        )
        new_down = nn.Linear(
            keep_size,
            hidden_size,
            bias=False,
            device=layer_device,
            dtype=down_proj.weight.dtype,
        )

        with torch.no_grad():
            new_gate.weight.copy_(gate_proj.weight.index_select(0, top_indices))
            new_up.weight.copy_(up_proj.weight.index_select(0, top_indices))
            new_down.weight.copy_(down_proj.weight.index_select(1, top_indices))

        layer.mlp.gate_proj = new_gate
        layer.mlp.up_proj = new_up
        layer.mlp.down_proj = new_down
        modules_pruned += 1

    after_params = count_mlp_params(model)
    if len(set(keep_sizes)) == 1:
        model.config.intermediate_size = keep_sizes[0]

    return {
        "modules_pruned": modules_pruned,
        "intermediate_size_before": before_size,
        "intermediate_size_after_min": min(keep_sizes),
        "intermediate_size_after_max": max(keep_sizes),
        "mlp_params_before": before_params,
        "mlp_params_after": after_params,
        "mlp_param_sparsity": 1.0 - (after_params / before_params),
    }


def measure_generation_tps(model, tokenizer, device: str, trials: int = 10) -> float:
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)

    for _ in range(3):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)

    latencies = []
    for _ in range(trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.time() - start)
    return 20.0 / (sum(latencies) / len(latencies))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--structured_sparsity", type=float, default=0.01)
    parser.add_argument("--mask_sparsity", type=float, default=0.20)
    parser.add_argument("--strategy", choices=["ceil", "floor", "none"], default="ceil")
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--output_dir", default="experiments/phase20_mask_guided")
    parser.add_argument("--output_suffix", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("Measuring dense C4 PPL and generation throughput...")
    baseline_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    baseline_tps = measure_generation_tps(model, tokenizer, device)

    scores = compute_mask_guided_scores(model, args.mask_sparsity)
    pruning_stats = apply_mask_guided_width_pruning(
        model,
        scores,
        sparsity=args.structured_sparsity,
        strategy=args.strategy,
    )

    print("Measuring mask-guided structured C4 PPL and generation throughput...")
    pruned_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    pruned_tps = measure_generation_tps(model, tokenizer, device)

    model_short = args.model.split("/")[-1]
    tag = args.output_suffix or tag_from_sparsity(args.structured_sparsity)
    ckpt_dir = os.path.join(args.output_dir, f"{model_short}_mask_guided_width_{args.strategy}_{tag}")

    print(f"Saving pruned checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    results = {
        "model": args.model,
        "method": "mask_guided_structured_mlp_width_pruning",
        "target_structured_sparsity": args.structured_sparsity,
        "scoring_mask_sparsity": args.mask_sparsity,
        "strategy": args.strategy,
        "eval_samples": args.eval_samples,
        "baseline_c4_ppl": baseline_c4,
        "pruned_c4_ppl": pruned_c4,
        "baseline_tps": baseline_tps,
        "pruned_tps": pruned_tps,
        "checkpoint_dir": ckpt_dir,
        **pruning_stats,
    }

    out_file = os.path.join(args.output_dir, f"{model_short}_mask_guided_width_{args.strategy}_{tag}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
