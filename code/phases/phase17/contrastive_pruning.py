"""
Phase 17: Contrastive Circuit Sparing (Semantic Lottery Ticket)
Implements width pruning by explicitly calculating the contrastive activation
difference between mathematical reasoning (GSM8K) and general text (Wikitext).
Neurons that highly activate on Math but not Text are "spared", while the lowest
contrastive neurons are pruned to meet the 20% sparsity target aligned to Tensor Cores.
"""
import os
import math
import json
import time
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from code.phases.phase11.utils import get_device, compute_perplexity_c4

def load_calibration_datasets(tokenizer, num_samples=50, seq_len=512):
    """Loads text samples from GSM8K (Math) and Wikitext (General Text)."""
    # 1. GSM8K
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
    gsm_texts = []
    for i in range(num_samples):
        # Format typical of reasoning prompts
        text = f"Question: {gsm_ds[i]['question']}\nAnswer: {gsm_ds[i]['answer']}"
        gsm_texts.append(text)
        
    # 2. Wikitext
    wiki_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    wiki_texts = []
    text_corpus = "\n".join([s["text"] for s in wiki_ds if len(s["text"].strip()) > 50])
    
    encodings = tokenizer(text_corpus, return_tensors="pt").input_ids[0]
    
    # Take contiguous chunks
    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len
        chunk = encodings[start:end]
        wiki_texts.append(tokenizer.decode(chunk))
        
    return gsm_texts, wiki_texts

def compute_contrastive_activations(model, tokenizer, num_samples=50):
    """
    Computes (GSM8K_Activation - Wikitext_Activation) for each neuron in each layer.
    """
    device = model.device
    gsm_texts, wiki_texts = load_calibration_datasets(tokenizer, num_samples)
    
    model.eval()
    layers = model.model.layers
    
    # Store activations per layer
    # activation shape per layer: (intermediate_size,)
    gsm_activations = {i: 0 for i in range(len(layers))}
    wiki_activations = {i: 0 for i in range(len(layers))}
    
    # Define a hook factory
    def get_hook(layer_idx, storage_dict):
        def hook(module, input, output):
            # input[0] is the tensor going into down_proj
            # shape: (batch_size, seq_len, intermediate_size)
            # We take mean absolute activation over batch and seq_len
            mean_act = input[0].abs().mean(dim=(0, 1)).detach()
            storage_dict[layer_idx] += mean_act
        return hook
        
    print("[Circuit Sparing] Measuring GSM8K (Math) Activations...")
    hooks = []
    for i, layer in enumerate(layers):
        h = layer.mlp.down_proj.register_forward_hook(get_hook(i, gsm_activations))
        hooks.append(h)
        
    for text in tqdm(gsm_texts):
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(tokens)
            
    for h in hooks:
        h.remove()
        
    print("[Circuit Sparing] Measuring Wikitext (General) Activations...")
    hooks = []
    for i, layer in enumerate(layers):
        h = layer.mlp.down_proj.register_forward_hook(get_hook(i, wiki_activations))
        hooks.append(h)
        
    for text in tqdm(wiki_texts):
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model(tokens)
            
    for h in hooks:
        h.remove()
        
    # Compute contrastive scores
    contrastive_scores = {}
    for i in range(len(layers)):
        # Normalize by num_samples to get average per prompt
        gsm_activations[i] /= num_samples
        wiki_activations[i] /= num_samples
        # Score = Math - General
        contrastive_scores[i] = gsm_activations[i] - wiki_activations[i]
        
    return contrastive_scores

def apply_contrastive_width_pruning(model, contrastive_scores, sparsity=0.20, strategy="ceil"):
    """
    Prune MLP width using contrastive scores.
    Neurons with high scores (math-specific) are kept. Lowest are pruned.
    """
    layers = model.model.layers
    total_params_removed = 0
    
    print(f"Applying Contrastive Circuit Sparing (Sparsity: {sparsity*100}%, Strategy: {strategy})")
    
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
            
        scores = contrastive_scores[layer_idx]
        # Get top-k indices to keep (highest math-specific activations)
        _, top_indices = torch.topk(scores, aligned_size)
        top_indices, _ = torch.sort(top_indices)
        
        layer_device = gate_proj.weight.device
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
        
        total_params_removed += 2 * (hidden_size * (intermediate_size - aligned_size))
        total_params_removed += (hidden_size * (intermediate_size - aligned_size))
        
        model.config.intermediate_size = aligned_size
        
    print(f"Removed {total_params_removed/1e6:.2f}M parameters of general/noise neurons.")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--sparsity", type=float, default=0.20)
    parser.add_argument("--strategy", type=str, choices=["ceil", "floor"], default="ceil")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = get_device()
    out_dir = "experiments/phase17"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    # 1. Compute Semantic Ticket
    contrastive_scores = compute_contrastive_activations(model, tokenizer, num_samples=100)
    
    # 2. Prune Model
    apply_contrastive_width_pruning(model, contrastive_scores, sparsity=args.sparsity, strategy=args.strategy)
    
    # 3. Save Pruned Model
    model_short = args.model.split("/")[-1]
    ckpt_dir = os.path.join(out_dir, f"{model_short}_contrastive_pruned_{args.strategy}")
    print(f"Saving contrastively pruned checkpoint to {ckpt_dir}...")
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print("Done!")

if __name__ == "__main__":
    main()
