"""Build grouped dense-revert ablations for the Phase 19 mask boundary.

The base mask is the Phase 19 layerwise unstructured magnitude mask. A
``revert_*`` ablation keeps selected layers or module families dense while
applying the same layerwise mask to the rest of the Linear modules.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase11.utils import compute_perplexity_c4, get_device
from code.phases.phase19.mask_pruning import count_linear_sparsity


ABLATIONS = ("s25", "revert_early", "revert_mid", "revert_late", "revert_mlp", "revert_attention")


def layer_block(layer_idx: int, num_layers: int) -> str:
    """Map a transformer layer index to an approximately equal third."""
    early_end = (num_layers + 2) // 3
    mid_end = (2 * num_layers + 2) // 3
    if layer_idx < early_end:
        return "early"
    if layer_idx < mid_end:
        return "mid"
    return "late"


def module_family(module_name: str) -> str:
    if "self_attn" in module_name:
        return "attention"
    if "mlp" in module_name:
        return "mlp"
    return "other"


def module_metadata(name: str, num_layers: int) -> dict:
    parts = name.split(".")
    layer_idx = None
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            break

    family = module_family(name)
    block = layer_block(layer_idx, num_layers) if layer_idx is not None else "non_layer"
    return {"layer_idx": layer_idx, "layer_block": block, "module_family": family}


def should_prune(ablation: str, metadata: dict) -> bool:
    if ablation == "s25":
        return True
    if ablation == "revert_early":
        return metadata["layer_block"] != "early"
    if ablation == "revert_mid":
        return metadata["layer_block"] != "mid"
    if ablation == "revert_late":
        return metadata["layer_block"] != "late"
    if ablation == "revert_mlp":
        return metadata["module_family"] != "mlp"
    if ablation == "revert_attention":
        return metadata["module_family"] != "attention"
    raise ValueError(f"Unknown ablation: {ablation}")


def _blank_counter() -> dict:
    return {"linear_modules": 0, "linear_params": 0, "zero_linear_params": 0}


def apply_grouped_layerwise_mask(model: nn.Module, sparsity: float, ablation: str) -> dict:
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    modules_pruned = 0
    modules_reverted = 0
    stats = {
        "ablation": ablation,
        "target_sparsity": sparsity,
        "modules_pruned": 0,
        "modules_reverted_to_dense": 0,
        "by_layer_block": defaultdict(_blank_counter),
        "by_module_family": defaultdict(_blank_counter),
    }

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            metadata = module_metadata(name, num_layers)
            prune_module = should_prune(ablation, metadata)
            weight = module.weight.data

            if prune_module:
                nparams_toprune = int(weight.numel() * sparsity)
                if nparams_toprune > 0:
                    threshold = torch.kthvalue(weight.abs().view(-1), nparams_toprune).values.item()
                    weight.mul_(weight.abs() >= threshold)
                    modules_pruned += 1
            else:
                modules_reverted += 1

            zero_count = int((weight == 0).sum().item())
            for key, value in (
                ("by_layer_block", metadata["layer_block"]),
                ("by_module_family", metadata["module_family"]),
            ):
                bucket = stats[key][value]
                bucket["linear_modules"] += 1
                bucket["linear_params"] += int(weight.numel())
                bucket["zero_linear_params"] += zero_count

    stats["modules_pruned"] = modules_pruned
    stats["modules_reverted_to_dense"] = modules_reverted
    for group_name in ("by_layer_block", "by_module_family"):
        serializable = {}
        for key, bucket in stats[group_name].items():
            total = bucket["linear_params"]
            bucket["linear_sparsity"] = bucket["zero_linear_params"] / total if total else 0.0
            serializable[key] = dict(bucket)
        stats[group_name] = dict(sorted(serializable.items()))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--sparsity", type=float, default=0.25)
    parser.add_argument("--ablation", choices=ABLATIONS, required=True)
    parser.add_argument("--eval_samples", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="experiments/phase24_grouped_ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Applying grouped layerwise mask: ablation={args.ablation}, sparsity={args.sparsity:.4f}")
    grouped_stats = apply_grouped_layerwise_mask(model, args.sparsity, args.ablation)
    sparsity_stats = count_linear_sparsity(model)
    print(
        "Actual linear sparsity: "
        f"{sparsity_stats['linear_sparsity'] * 100:.2f}% "
        f"across {sparsity_stats['linear_modules']} Linear modules"
    )

    masked_c4 = None
    if args.eval_samples > 0:
        print(f"Measuring C4 PPL on {args.eval_samples} chunks...")
        masked_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
        print(f"Masked C4 PPL: {masked_c4:.4f}")

    model_short = args.model.split("/")[-1]
    ckpt_dir = output_dir / f"{model_short}_mask_pruned_{args.ablation}"
    print(f"Saving checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    results = {
        "model": args.model,
        "method": "layerwise_unstructured_magnitude_mask_grouped_dense_revert",
        "base_mask_scope": "layerwise",
        "target_sparsity": args.sparsity,
        "ablation": args.ablation,
        "eval_samples": args.eval_samples,
        "masked_c4_ppl": masked_c4,
        "checkpoint_dir": str(ckpt_dir),
        **sparsity_stats,
        "grouped_stats": grouped_stats,
    }

    out_file = output_dir / f"{model_short}_grouped_mask_{args.ablation}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()

