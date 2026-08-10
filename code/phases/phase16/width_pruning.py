"""
Phase 16: Hardware-Aware Width Pruning
Implements width pruning of MLP internal dimensions based on empirical FIM,
with rounding constraints (ceil vs floor) for Tensor Core hardware alignment.
"""
import os
import math
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from code.phases.phase11.utils import get_device, compute_perplexity_c4
from code.utils.dataset_manager import load_calibration_data

def compute_neuron_fim(model, tokenizer, num_samples=50):
    """
    Compute empirical FIM for each neuron in the intermediate_size dimension
    across all MLP layers.
    """
    device = model.device
    calibration_texts = load_calibration_data(num_samples=num_samples)
    
    model.eval()
    
    # Identify layer architecture
    if hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Unsupported model architecture")
        
    n_layers = len(layers)
    
    # We will accumulate FIM for each layer's intermediate dimension
    # shape of FIM per layer: (intermediate_size,)
    # up_proj, gate_proj are (intermediate_size, hidden_size)
    # down_proj is (hidden_size, intermediate_size)
    
    fim_accumulators = {}
    
    print("[Width Pruning] Computing empirical FIM for MLP neurons...")
    
    for text_idx, text in enumerate(tqdm(calibration_texts)):
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if tokens.shape[1] < 2:
            continue
            
        model.zero_grad()
        try:
            with torch.enable_grad():
                logits = model(tokens).logits
                
                # Sample labels from the model's own output distribution
                with torch.no_grad():
                    probs = F.softmax(logits[:, :-1, :].detach(), dim=-1)
                    sampled = torch.multinomial(
                        probs.reshape(-1, probs.shape[-1]), 1
                    ).reshape(1, -1)
                    
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                selected = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                loss = selected.sum()
                loss.backward()
                
            for layer_idx, layer in enumerate(layers):
                if layer_idx not in fim_accumulators:
                    intermediate_size = layer.mlp.up_proj.weight.shape[0]
                    layer_device = layer.mlp.gate_proj.weight.device
                    fim_accumulators[layer_idx] = torch.zeros(intermediate_size, device=layer_device)
                    
                # sum of squared gradients across hidden_size dimension
                g_gate = layer.mlp.gate_proj.weight.grad
                g_up = layer.mlp.up_proj.weight.grad
                g_down = layer.mlp.down_proj.weight.grad
                
                fim_gate = (g_gate ** 2).sum(dim=1)
                fim_up = (g_up ** 2).sum(dim=1)
                fim_down = (g_down ** 2).sum(dim=0)
                
                fim_accumulators[layer_idx] += (fim_gate + fim_up + fim_down)
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            else:
                print(f"Error on sample {text_idx}: {e}")
        finally:
            model.zero_grad()
            
    return fim_accumulators

def apply_width_pruning(model, fim_scores, sparsity=0.20, strategy="ceil"):
    """
    Prune MLP width using FIM scores.
    strategy: "ceil" rounds up to nearest 64, "floor" rounds down.
    """
    layers = model.model.layers
    modules_pruned = 0
    total_params_removed = 0
    
    print(f"Applying Width Pruning (Sparsity: {sparsity*100}%, Strategy: {strategy})")
    
    for layer_idx, layer in enumerate(layers):
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj
        
        intermediate_size = up_proj.weight.shape[0]
        hidden_size = up_proj.weight.shape[1]
        
        target_size = intermediate_size * (1.0 - sparsity)
        
        if strategy == "ceil":
            aligned_size = math.ceil(target_size / 64.0) * 64
        elif strategy == "floor":
            aligned_size = math.floor(target_size / 64.0) * 64
        else:
            aligned_size = int(target_size)
            
        aligned_size = min(aligned_size, intermediate_size)
        
        if aligned_size == intermediate_size:
            continue
            
        scores = fim_scores[layer_idx]
        # Get top-k indices to keep
        _, top_indices = torch.topk(scores, aligned_size)
        top_indices, _ = torch.sort(top_indices)
        
        layer_device = gate_proj.weight.device
        # Create new smaller linear layers
        new_gate = nn.Linear(hidden_size, aligned_size, bias=False, device=layer_device, dtype=gate_proj.weight.dtype)
        new_up = nn.Linear(hidden_size, aligned_size, bias=False, device=layer_device, dtype=up_proj.weight.dtype)
        new_down = nn.Linear(aligned_size, hidden_size, bias=False, device=layer_device, dtype=down_proj.weight.dtype)
        
        # Slice weights
        with torch.no_grad():
            new_gate.weight.data = gate_proj.weight.data[top_indices, :]
            new_up.weight.data = up_proj.weight.data[top_indices, :]
            new_down.weight.data = down_proj.weight.data[:, top_indices]
            
        # Replace
        layer.mlp.gate_proj = new_gate
        layer.mlp.up_proj = new_up
        layer.mlp.down_proj = new_down
        
        total_params_removed += 2 * (hidden_size * (intermediate_size - aligned_size)) # gate and up
        total_params_removed += (hidden_size * (intermediate_size - aligned_size)) # down
        modules_pruned += 1
        
        # We assume all layers are pruned to the same aligned_size
        model.config.intermediate_size = aligned_size
        
    print(f"Realigned intermediate sizes. Removed {total_params_removed/1e6:.2f}M params.")
    return modules_pruned

def measure_latency(model, tokenizer, device, num_trials=10):
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)
            
    latencies = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)
        torch.cuda.synchronize()
        t1 = time.time()
        latencies.append(t1 - t0)
        
    # Return avg tokens per second (20 tokens generated)
    return 20.0 / (sum(latencies)/len(latencies))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--sparsity", type=float, default=0.20)
    parser.add_argument("--strategy", type=str, choices=["ceil", "floor"], default="ceil")
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = get_device()
    out_dir = "experiments/phase16"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    print("Measuring baseline C4 PPL...")
    baseline_ppl = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    baseline_tps = measure_latency(model, tokenizer, device)
    print(f"Baseline C4 PPL: {baseline_ppl:.4f}")
    print(f"Baseline Latency: {baseline_tps:.2f} t/s")
    
    fim_scores = compute_neuron_fim(model, tokenizer, num_samples=20)
    
    apply_width_pruning(model, fim_scores, sparsity=args.sparsity, strategy=args.strategy)
    
    print(f"Measuring Pruned ({args.strategy}) C4 PPL...")
    pruned_ppl = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    pruned_tps = measure_latency(model, tokenizer, device)
    
    print(f"Pruned C4 PPL: {pruned_ppl:.4f}")
    print(f"Pruned Latency: {pruned_tps:.2f} t/s")
    
    results = {
        "model": args.model,
        "sparsity": args.sparsity,
        "strategy": args.strategy,
        "eval_samples": args.eval_samples,
        "baseline_c4_ppl": baseline_ppl,
        "pruned_c4_ppl": pruned_ppl,
        "baseline_tps": baseline_tps,
        "pruned_tps": pruned_tps
    }
    
    model_short = args.model.split("/")[-1]
    
    ckpt_dir = os.path.join(out_dir, f"{model_short}_pruned_{args.strategy}")
    print(f"Saving pruned checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    
    out_file = os.path.join(out_dir, f"{model_short}_width_pruning_{args.strategy}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
