"""
Phase 18: Sensitivity-Guided Non-Uniform Width Pruning
Implements width pruning by using Delta CE Sensitivity (Phase 14) to dynamically
allocate sparsity. Highly sensitive layers are completely spared, while redundant
layers are heavily compressed. This respects the structural inequality of the model.
"""
import os
import math
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from code.phases.phase11.utils import get_device, compute_perplexity_c4
from code.phases.phase16.width_pruning import compute_neuron_fim

def apply_non_uniform_pruning(model, fim_scores, sensitivity_map, global_sparsity=0.20, strategy="ceil", max_layer_sparsity=0.35):
    layers = model.model.layers
    n_layers = len(layers)
    
    # Sample dimensions from first layer
    intermediate_size = layers[0].mlp.up_proj.weight.shape[0]
    hidden_size = layers[0].mlp.up_proj.weight.shape[1]
    
    # 1. Dynamic Sparsity Allocation
    global_target_neurons_to_remove = int(n_layers * intermediate_size * global_sparsity)
    neurons_removed = 0
    layer_sparsities = {i: 0.0 for i in range(n_layers)}
    
    # Sort layers by sensitivity, ascending (least sensitive first)
    sensitivities = sensitivity_map["sensitivities"]
    sorted_layers = sorted(range(n_layers), key=lambda i: sensitivities[f"layer_{i}"])
    
    print(f"\n[Phase 18] Dynamic Sparsity Allocation (Target Global Sparsity: {global_sparsity*100}%):")
    for layer_idx in sorted_layers:
        if neurons_removed >= global_target_neurons_to_remove:
            break
            
        can_remove = int(intermediate_size * max_layer_sparsity)
        needed = global_target_neurons_to_remove - neurons_removed
        
        remove_this_layer = min(can_remove, needed)
        
        # We must also align the remove_this_layer so that the *remaining* size aligns with strategy
        # Let's enforce it cleanly during the actual pruning step, so here we just set the exact target
        layer_sparsities[layer_idx] = remove_this_layer / intermediate_size
        neurons_removed += remove_this_layer

    for i in range(n_layers):
        print(f"  Layer {i:02d} (Sens: {sensitivities[f'layer_{i}']:.4f}) -> Sparsity: {layer_sparsities[i]*100:.2f}%")
        
    # 2. Prune
    total_params_removed = 0
    modules_pruned = 0
    
    for layer_idx, layer in enumerate(layers):
        sparsity = layer_sparsities[layer_idx]
        if sparsity <= 0.0:
            continue
            
        gate_proj = layer.mlp.gate_proj
        up_proj = layer.mlp.up_proj
        down_proj = layer.mlp.down_proj
        
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
        
        # We need to drop (intermediate_size - aligned_size) neurons
        neurons_to_drop = intermediate_size - aligned_size
        
        # Get bottom-k indices to drop
        _, bottom_indices = torch.topk(scores, neurons_to_drop, largest=False)
        
        with torch.no_grad():
            gate_proj.weight.data[bottom_indices, :] = 0.0
            up_proj.weight.data[bottom_indices, :] = 0.0
            down_proj.weight.data[:, bottom_indices] = 0.0
            
        total_params_removed += 2 * (hidden_size * neurons_to_drop)
        total_params_removed += (hidden_size * neurons_to_drop)
        modules_pruned += 1
        
    print(f"\n[Phase 18] Pruned {modules_pruned} layers. Removed {total_params_removed/1e6:.2f}M params.")
    
def compute_neuron_magnitude(model):
    magnitudes = {}
    for layer_idx, layer in enumerate(model.model.layers):
        w_gate = layer.mlp.gate_proj.weight.data
        w_up = layer.mlp.up_proj.weight.data
        w_down = layer.mlp.down_proj.weight.data
        
        mag_gate = (w_gate ** 2).sum(dim=1)
        mag_up = (w_up ** 2).sum(dim=1)
        mag_down = (w_down ** 2).sum(dim=0)
        
        magnitudes[layer_idx] = mag_gate + mag_up + mag_down
    return magnitudes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--global_sparsity", type=float, default=0.20)
    parser.add_argument("--strategy", type=str, choices=["ceil", "floor"], default="ceil")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = get_device()
    out_dir = "experiments/phase18"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    # 1. Load Sensitivity Map
    map_file = "experiments/phase14_final/Mistral-7B-v0.1_sensitivity_map.json"
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Missing {map_file}. Please run Phase 14 first.")
    with open(map_file, "r") as f:
        sensitivity_map = json.load(f)
        
    # 2. Compute Magnitude for all layers
    fim_scores = compute_neuron_magnitude(model)
    
    # 3. Apply Non-Uniform Pruning
    apply_non_uniform_pruning(model, fim_scores, sensitivity_map, global_sparsity=args.global_sparsity, strategy=args.strategy)
    
    # 4. Save Pruned Model
    model_short = args.model.split("/")[-1]
    ckpt_dir = os.path.join(out_dir, f"{model_short}_non_uniform_pruned_{args.strategy}")
    print(f"Saving non-uniform pruned checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print("Done!")

if __name__ == "__main__":
    main()
