"""
Phase 14: Bulletproof Evaluation
Script 4: High-sample-count evaluation of baseline and pruned PPL.

This script does NOT retrain. It re-evaluates baseline and pruned
perplexity at eval_samples=256 to provide statistically robust
anchor points for the ablation table.

The 'recovered' PPL rows require retraining (LoRA weights were not saved).
Run this to validate the baseline/pruned rows with high confidence.

Usage:
    python -m code.phases.phase14_final.4_bulletproof_eval \
        --models mistralai/Mistral-7B-v0.1 meta-llama/Meta-Llama-3-8B \
        --eval_samples 256 \
        --output_file experiments/phase14_final/bulletproof_eval.json
"""

import os
import gc
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase11.utils import (
    get_device,
    compute_perplexity_c4,
    apply_layerwise_pruning,
)


def evaluate_model(model_name: str, sparsity: float, eval_samples: int, device: str):
    """
    Load model, evaluate baseline C4 PPL, apply pruning, evaluate pruned C4 PPL.
    Returns a dict with both measurements.
    """
    print(f"\n{'='*65}")
    print(f"  Model: {model_name}")
    print(f"  Sparsity: {sparsity*100:.0f}%  |  eval_samples: {eval_samples}")
    print(f"{'='*65}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n[1/2] Evaluating BASELINE C4 PPL ...")
    baseline_c4 = compute_perplexity_c4(
        model, tokenizer, device, max_samples=eval_samples
    )
    print(f"  Baseline C4 PPL: {baseline_c4:.4f}")

    # ── Pruning ───────────────────────────────────────────────────────────────
    print(f"\n[2/2] Applying {sparsity*100:.0f}% pruning → evaluating PRUNED C4 PPL ...")
    n_pruned = apply_layerwise_pruning(model, sparsity)
    print(f"  Modules pruned: {n_pruned}")

    model.eval()
    pruned_c4 = compute_perplexity_c4(
        model, tokenizer, device, max_samples=eval_samples
    )
    print(f"  Pruned C4 PPL:  {pruned_c4:.4f}")
    print(f"  Δ PPL (pruned vs baseline): {pruned_c4 - baseline_c4:+.4f}")

    result = {
        "model": model_name,
        "sparsity": sparsity,
        "eval_samples": eval_samples,
        "baseline_c4_ppl": baseline_c4,
        "pruned_c4_ppl": pruned_c4,
        "pruned_delta_abs": pruned_c4 - baseline_c4,
        "pruned_delta_pct": ((pruned_c4 - baseline_c4) / baseline_c4) * 100,
    }

    # Explicit VRAM flush
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bulletproof baseline/pruned C4 PPL evaluation (no retraining)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mistralai/Mistral-7B-v0.1", "meta-llama/Meta-Llama-3-8B"],
        help="List of HuggingFace model identifiers to evaluate.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.20,
        help="Unstructured magnitude pruning sparsity (default: 0.20 = 20%%).",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=256,
        help="Number of 2048-token C4 chunks to evaluate on (default: 256).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="experiments/phase14_final/bulletproof_eval.json",
        help="Path to write results JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load existing results so we can resume if interrupted
    all_results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            all_results = json.load(f)
        print(f"[resume] Loaded {len(all_results)} existing results from {args.output_file}")

    for model_name in args.models:
        model_short = model_name.split("/")[-1]
        key = f"{model_short}_sparsity{int(args.sparsity*100)}_samples{args.eval_samples}"

        if key in all_results:
            print(f"\n[skip] {key} already evaluated.")
            continue

        result = evaluate_model(model_name, args.sparsity, args.eval_samples, device)
        all_results[key] = result

        # Incremental save
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[saved] {args.output_file}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print(f"{'BULLETPROOF EVAL SUMMARY':^75}")
    print("=" * 75)
    print(f"  eval_samples = {args.eval_samples}  |  sparsity = {args.sparsity*100:.0f}%")
    print("-" * 75)
    print(f"  {'Model':<35} {'Baseline C4':>12} {'Pruned C4':>12} {'Δ%':>8}")
    print("-" * 75)
    for key, r in all_results.items():
        name = r["model"].split("/")[-1]
        print(
            f"  {name:<35} {r['baseline_c4_ppl']:>12.4f} {r['pruned_c4_ppl']:>12.4f} "
            f"{r['pruned_delta_pct']:>+8.2f}%"
        )
    print("=" * 75)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
