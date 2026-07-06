"""
Phase 19: Mask pruning control for structural-circuit fragility.

This phase keeps the model shape fixed and applies layerwise unstructured
magnitude masking. It is the paired control for Phase 18B structural width
pruning: if a dense-shaped 20% mask preserves reasoning while 1% structural
width pruning hurts it, the result isolates structural geometry as the failure
mode rather than ordinary C4 perplexity or parameter sparsity.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase11.utils import (
    apply_layerwise_pruning,
    compute_perplexity_c4,
    get_device,
)


def tag_from_sparsity(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        tag = text[2:]
        while len(tag) < 2:
            tag += "0"
        return f"s{tag}"
    return "s" + text.replace(".", "p")


def count_linear_sparsity(model: nn.Module) -> dict:
    total = 0
    zero = 0
    modules = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            weights = module.weight.data
            total += weights.numel()
            zero += int((weights == 0).sum().item())
            modules += 1

    return {
        "linear_modules": modules,
        "linear_params": total,
        "zero_linear_params": zero,
        "linear_sparsity": zero / total if total else 0.0,
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
        t0 = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.time() - t0)

    return 20.0 / (sum(latencies) / len(latencies))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--sparsity", type=float, default=0.20)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="experiments/phase19")
    parser.add_argument("--output_suffix", type=str, default="")
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

    print("Measuring baseline C4 PPL...")
    baseline_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    baseline_tps = measure_generation_tps(model, tokenizer, device)
    print(f"Baseline C4 PPL: {baseline_c4:.4f}")
    print(f"Baseline generation: {baseline_tps:.2f} tok/s")

    print(f"Applying layerwise magnitude mask at sparsity={args.sparsity:.4f}...")
    modules_pruned = apply_layerwise_pruning(model, args.sparsity)
    sparsity_stats = count_linear_sparsity(model)
    print(
        "Actual linear sparsity: "
        f"{sparsity_stats['linear_sparsity'] * 100:.2f}% "
        f"across {sparsity_stats['linear_modules']} Linear modules"
    )

    print("Measuring masked C4 PPL...")
    masked_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    masked_tps = measure_generation_tps(model, tokenizer, device)
    print(f"Masked C4 PPL: {masked_c4:.4f}")
    print(f"Masked generation: {masked_tps:.2f} tok/s")

    model_short = args.model.split("/")[-1]
    tag = args.output_suffix or tag_from_sparsity(args.sparsity)
    ckpt_dir = os.path.join(args.output_dir, f"{model_short}_mask_pruned_{tag}")

    print(f"Saving masked checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    results = {
        "model": args.model,
        "method": "layerwise_unstructured_magnitude_mask",
        "target_sparsity": args.sparsity,
        "eval_samples": args.eval_samples,
        "modules_pruned": modules_pruned,
        "baseline_c4_ppl": baseline_c4,
        "masked_c4_ppl": masked_c4,
        "baseline_tps": baseline_tps,
        "masked_tps": masked_tps,
        "checkpoint_dir": ckpt_dir,
        **sparsity_stats,
    }

    out_file = os.path.join(args.output_dir, f"{model_short}_mask_pruning_{tag}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
