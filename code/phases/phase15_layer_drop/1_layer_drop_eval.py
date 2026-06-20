"""
Phase 15: Oracle-Guided Layer Dropping
Script 1: Evaluate structural damage and latency of dropped layers.
"""
import os
import json
import time
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from code.phases.phase11.utils import get_device, compute_perplexity_c4, time_forward_pass

def find_best_drop_window(map_file: str, n_drop: int):
    with open(map_file, "r") as f:
        data = json.load(f)
    sens = data["sensitivities"]
    n_layers = len(sens)
    
    # Extract scores in order
    scores = [sens[f"layer_{i}"] for i in range(n_layers)]
    
    # Sliding window
    min_sum = float('inf')
    best_start = 0
    
    for i in range(n_layers - n_drop + 1):
        window_sum = sum(scores[i:i+n_drop])
        if window_sum < min_sum:
            min_sum = window_sum
            best_start = i
            
    return list(range(best_start, best_start + n_drop)), min_sum

def drop_layers(model, drop_indices):
    """Physically remove layers from the model"""
    old_layers = model.model.layers
    new_layers = nn.ModuleList([l for i, l in enumerate(old_layers) if i not in drop_indices])
    
    # Fix internal layer indices for HF cache
    for new_idx, layer in enumerate(new_layers):
        if hasattr(layer, 'layer_idx'):
            layer.layer_idx = new_idx
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_idx'):
            layer.self_attn.layer_idx = new_idx
            
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--n_drop", type=int, default=4)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    out_dir = "experiments/phase15_layer_drop"
    os.makedirs(out_dir, exist_ok=True)
    
    model_short = args.model.split("/")[-1]
    map_file = f"experiments/phase14_final/{model_short}_sensitivity_map.json"
    
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Missing {map_file}. Run Phase 14 oracle first.")
        
    drop_indices, min_sens_sum = find_best_drop_window(map_file, args.n_drop)
    print(f"\n[{model_short}] Best {args.n_drop} layers to drop: {drop_indices} (Total Sens: {min_sens_sum:.4f})")

    # 1. Baseline eval
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    
    print("\nMeasuring BASELINE latency...")
    baseline_latency = time_forward_pass(model, tokenizer, device, n_trials=20)
    print(f"  Baseline latency: {baseline_latency:.2f} ms")
    
    print("\nMeasuring BASELINE C4 PPL...")
    baseline_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    print(f"  Baseline C4 PPL: {baseline_c4:.4f}")

    # 2. Drop layers
    print(f"\nDropping layers {drop_indices}...")
    model = drop_layers(model, drop_indices)
    
    print("\nMeasuring DROPPED latency...")
    dropped_latency = time_forward_pass(model, tokenizer, device, n_trials=20)
    latency_reduction = ((baseline_latency - dropped_latency) / baseline_latency) * 100
    print(f"  Dropped latency: {dropped_latency:.2f} ms (-{latency_reduction:.1f}%)")
    
    print("\nMeasuring DROPPED C4 PPL...")
    dropped_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    print(f"  Dropped C4 PPL: {dropped_c4:.4f}")
    
    results = {
        "model": args.model,
        "n_drop": args.n_drop,
        "drop_indices": drop_indices,
        "eval_samples": args.eval_samples,
        "baseline_latency_ms": baseline_latency,
        "dropped_latency_ms": dropped_latency,
        "latency_reduction_pct": latency_reduction,
        "baseline_c4_ppl": baseline_c4,
        "dropped_c4_ppl": dropped_c4
    }
    
    out_file = os.path.join(out_dir, f"{model_short}_drop{args.n_drop}_eval.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw drop evaluation to {out_file}")

if __name__ == "__main__":
    main()
