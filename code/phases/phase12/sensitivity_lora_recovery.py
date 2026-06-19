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

def build_sensitivity_lora_config(model, sensitivities, max_rank=16, min_rank=2):
    """
    Builds a list of target_modules (layer specific) for PEFT based on sensitivity.
    We map high sensitivity (ΔCE) to max_rank, low to min_rank, and negative/zero to skip.
    PEFT's LoraConfig currently supports a single rank (r) for all target_modules if using
    the high-level API. To vary rank *per layer*, we would need advanced PEFT configuration 
    or manual injection. For Phase 12, we will select ONLY the top N most sensitive layers 
    to apply LoRA to, proving the concept of "surgical" recovery.
    """
    # Sort layers by sensitivity (descending)
    sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    
    # We will target the top 6 most sensitive layers
    top_layers = [layer for layer, score in sorted_layers[:6]]
    print(f"Top sensitive layers selected for LoRA: {top_layers}")
    
    target_modules = []
    # Convert "layer_X" to the actual module name pattern in Llama/Mistral
    for layer_name in top_layers:
        idx = layer_name.split("_")[1]
        target_modules.append(f"model.layers.{idx}.self_attn.q_proj")
        target_modules.append(f"model.layers.{idx}.self_attn.v_proj")
        # We also add the MLP up/down projections for these critical layers to maximize recovery
        target_modules.append(f"model.layers.{idx}.mlp.gate_proj")
        target_modules.append(f"model.layers.{idx}.mlp.up_proj")
        target_modules.append(f"model.layers.{idx}.mlp.down_proj")
        
    config = LoraConfig(
        r=max_rank,
        lora_alpha=max_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return config

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
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    
    print(f"Training LoRA on {num_chunks} sequences...")
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
    
    # 1. Load Sensitivity Data
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
    
    print(f"Applying {args.sparsity*100}% global pruning...")
    apply_global_pruning(model, args.sparsity)
    
    print("Measuring pruned PPL...")
    pruned_ppl = compute_perplexity(model, tokenizer, device, max_samples=args.samples)
    
    start_time = time.time()
    lora_config = build_sensitivity_lora_config(model, sensitivities, max_rank=args.rank)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    train_lora(model, tokenizer, device, samples=args.train_samples)
    recovery_time = time.time() - start_time
    
    model.eval()
    print("Measuring recovered PPL...")
    recovered_ppl = compute_perplexity(model, tokenizer, device, max_samples=args.samples)
    
    results = {
        "method": "method2_lora",
        "rank": args.rank,
        "train_samples": args.train_samples,
        "recovery_time": recovery_time,
        "baseline_ppl": baseline_ppl,
        "pruned_ppl": pruned_ppl,
        "recovered_ppl": recovered_ppl,
        "ppl_increase_pct": ((recovered_ppl - baseline_ppl) / baseline_ppl) * 100
    }
    
    print("\n--- Method 2 Results ---")
    print(f"Time: {recovery_time:.2f}s")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Pruned PPL:   {pruned_ppl:.4f}")
    print(f"Recovered PPL: {recovered_ppl:.4f} (+{results['ppl_increase_pct']:.2f}%)")
    
    os.makedirs("experiments/phase12", exist_ok=True)
    with open(f"experiments/phase12/method2_lora_rank{args.rank}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
