import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from code.phases.phase11.utils import get_device, compute_perplexity, apply_global_pruning

def build_svd_lora(model, sensitivities, target_sparsity=0.20, rank=16):
    """
    Builds the LoRA config for the top sensitive layers, applies it, 
    and then customizes the initialization of the LoRA A and B matrices 
    using the SVD of the exact pruning damage (delta W).
    """
    # 1. We must first get the original weights before pruning
    # Since we need delta_w = w_orig - w_pruned, we will extract original weights
    # for the targeted modules.
    
    sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    top_layers = [layer for layer, score in sorted_layers[:6]]
    
    target_modules = []
    for layer_name in top_layers:
        idx = layer_name.split("_")[1]
        target_modules.extend([
            f"model.layers.{idx}.self_attn.q_proj",
            f"model.layers.{idx}.self_attn.v_proj",
            f"model.layers.{idx}.mlp.gate_proj",
            f"model.layers.{idx}.mlp.up_proj",
            f"model.layers.{idx}.mlp.down_proj"
        ])
    
    # Extract original weights for target modules
    w_origs = {}
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, nn.Linear):
            w_origs[name] = module.weight.data.clone().float()

    print(f"Applying {target_sparsity*100}% global pruning to base model...")
    # This modifies model weights in place
    apply_global_pruning(model, target_sparsity)

    # Now wrap with PEFT
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2, # standard is alpha = 2*r
        target_modules=target_modules,
        lora_dropout=0.0, # Disable dropout for SVD consistency
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=False # We will initialize manually
    )
    
    model = get_peft_model(model, config)
    
    # Now manually initialize the LoRA weights using SVD on the pruning delta
    print(f"Initializing LoRA adapters with SVD of pruning damage...")
    for name, w_orig in w_origs.items():
        # PEFT wraps the module. The base_layer is the original Linear.
        peft_module_name = name + ".lora_A.default"
        
        # Get the pruned weight
        # Find the module in the PEFT wrapped model
        peft_module_name = "base_model.model." + name
        peft_module = dict(model.named_modules())[peft_module_name]
        w_pruned = peft_module.base_layer.weight.data.float()
        
        # Compute delta W
        delta_w = w_orig - w_pruned
        
        if delta_w.abs().sum() == 0:
            continue
            
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
        r = min(rank, S.size(0))
        
        # lora_A: in_features -> r
        # lora_B: r -> out_features
        # W_approx = B @ A. delta_w is shape (out, in).
        # U @ S @ Vh = (out, r) @ (r, r) @ (r, in)
        # So lora_B = U[:, :r] @ S_r
        # lora_A = Vh[:r, :]
        # But wait! PEFT defines lora_A shape as (r, in) and lora_B as (out, r).
        
        S_r = torch.diag(S[:r])
        
        lora_B_weight = U[:, :r] @ S_r
        lora_A_weight = Vh[:r, :]
        
        # Also need to account for scaling: PEFT applies scaling = lora_alpha / r
        # We need (B @ A) * scaling = delta_w_approx
        # So we divide by scaling
        scaling = config.lora_alpha / config.r
        lora_B_weight = lora_B_weight / scaling
        
        # Set weights
        peft_module.lora_A['default'].weight.data = lora_A_weight.to(device=peft_module.lora_A['default'].weight.device, dtype=peft_module.lora_A['default'].weight.dtype)
        peft_module.lora_B['default'].weight.data = lora_B_weight.to(device=peft_module.lora_B['default'].weight.device, dtype=peft_module.lora_B['default'].weight.dtype)
        
    return model

def train_lora(model, tokenizer, device, samples=256, batch_size=1):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([s for s in dataset["text"] if s.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    seq_len = 2048
    total_tokens = input_ids.size(0)
    num_chunks = min(samples, total_tokens // seq_len)
    
    chunks = []
    for i in range(num_chunks):
        chunks.append(input_ids[i*seq_len : (i+1)*seq_len])
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Lower LR since it's already well-initialized
    model.train()
    
    print(f"Fine-tuning SVD-initialized LoRA on {num_chunks} sequences...")
    for chunk in tqdm(chunks):
        chunk = chunk.unsqueeze(0).to(device)
        outputs = model(input_ids=chunk, labels=chunk)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=0.20)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    
    val_data = json.load(open('experiments/phase12/proxy_validation.json'))
    sensitivities = {k: v['ground_truth_delta_ce'] for k, v in val_data['metrics'].items()}
    
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print("Measuring baseline...")
    baseline_ppl = compute_perplexity(model, tokenizer, device, max_samples=args.samples)
    
    start_time = time.time()
    model = build_svd_lora(model, sensitivities, target_sparsity=args.sparsity, rank=args.rank)
    model.print_trainable_parameters()
    
    # Evaluate PPL *before* fine-tuning but *after* SVD initialization
    print("Measuring SVD-initialized PPL (before fine-tuning)...")
    model.eval()
    init_ppl = compute_perplexity(model, tokenizer, device, max_samples=args.samples)
    
    train_lora(model, tokenizer, device, samples=args.train_samples)
    recovery_time = time.time() - start_time
    
    model.eval()
    print("Measuring recovered PPL (after fine-tuning)...")
    recovered_ppl = compute_perplexity(model, tokenizer, device, max_samples=args.samples)
    
    results = {
        "method": "method3_svd_lora",
        "rank": args.rank,
        "train_samples": args.train_samples,
        "recovery_time": recovery_time,
        "baseline_ppl": baseline_ppl,
        "init_ppl": init_ppl,
        "recovered_ppl": recovered_ppl,
        "ppl_increase_pct": ((recovered_ppl - baseline_ppl) / baseline_ppl) * 100
    }
    
    print("\n--- Method 3 Results ---")
    print(f"Time: {recovery_time:.2f}s")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"SVD Init PPL: {init_ppl:.4f}")
    print(f"Recovered PPL: {recovered_ppl:.4f} (+{results['ppl_increase_pct']:.2f}%)")
    
    os.makedirs("experiments/phase12", exist_ok=True)
    with open(f"experiments/phase12/method3_svd_lora_rank{args.rank}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
