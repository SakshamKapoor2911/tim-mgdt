# NOTE: Quick validation script evaluated on WikiText-2.
# Canonical paper perplexities are evaluated on C4 validation per the paper's Section 4.1 protocol.
import os
import sys
import json
import torch
import torch.nn as nn
import gc
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from tqdm import tqdm

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
sys.modules.pop('code', None)

from code.phases.phase11.utils import get_device, compute_perplexity_c4

# Helper for layer drop
def drop_layers_with_mapping(model, drop_indices):
    old_layers = model.model.layers
    new_layers = []
    old_to_new_map = {}
    curr_idx = 0
    for i, l in enumerate(old_layers):
        if i not in drop_indices:
            new_layers.append(l)
            old_to_new_map[i] = curr_idx
            curr_idx += 1
            
    # Fix internal layer indices for HF cache
    for new_idx, layer in enumerate(new_layers):
        if hasattr(layer, 'layer_idx'):
            layer.layer_idx = new_idx
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_idx'):
            layer.self_attn.layer_idx = new_idx
            
    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    return model, old_to_new_map

def build_surgical_lora_config(old_to_new_map, rank=16, boundary_thickness=2, dropped_indices=[]):
    d_start = min(dropped_indices)
    d_end = max(dropped_indices)
    
    target_layer_indices = []
    # Left boundary
    for offset in range(1, boundary_thickness + 1):
        idx = d_start - offset
        if idx in old_to_new_map:
            target_layer_indices.append(idx)
    # Right boundary
    for offset in range(1, boundary_thickness + 1):
        idx = d_end + offset
        if idx in old_to_new_map:
            target_layer_indices.append(idx)
            
    target_modules = []
    for old_idx in target_layer_indices:
        new_idx = old_to_new_map[old_idx]
        target_modules.extend([
            f"model.layers.{new_idx}.self_attn.q_proj",
            f"model.layers.{new_idx}.self_attn.v_proj",
            f"model.layers.{new_idx}.mlp.gate_proj",
            f"model.layers.{new_idx}.mlp.up_proj",
            f"model.layers.{new_idx}.mlp.down_proj"
        ])
    
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True
    )
    return config

def train_lora_with_logging(model, tokenizer, device, samples=150, lr=1e-5, warmup_ratio=0.10):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([s for s in dataset["text"] if s.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    seq_len = 2048
    num_chunks = min(samples, input_ids.size(0) // seq_len)
    chunks = [input_ids[i*seq_len:(i+1)*seq_len] for i in range(num_chunks)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = int(warmup_ratio * len(chunks))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(chunks))
    
    model.train()
    for i, chunk in enumerate(tqdm(chunks, desc="    Training Healing LoRA")):
        chunk = chunk.unsqueeze(0).to(device)
        outputs = model(input_ids=chunk, labels=chunk)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    model.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, help="Skip depth k")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    model_id = "mistralai/Mistral-7B-v0.1"
    b_adapter = "checkpoints/Mistral-7B-v0.1_boundary_healed_t2_lora"
    device = args.device
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    k = args.k
    print(f"\n==================== skip depth k={k} ====================")
    drop_indices = list(range(11, 11 + k))
    
    # 1. Unhealed Model PPL
    print("Evaluating Unhealed Model PPL...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
    )
    if k > 0:
        model, _ = drop_layers_with_mapping(model, drop_indices)
    
    ppl_unhealed = compute_perplexity_c4(model, tokenizer, device, max_samples=100)
    print(f"  Unhealed C4 PPL: {ppl_unhealed:.4f}")
    
    # 2. Healed Model PPL
    if k == 0:
        ppl_healed = ppl_unhealed
    else:
        print("Preparing Healed Model...")
        if k == 4:
            # Load saved checkpoint
            model_healed = PeftModel.from_pretrained(model, b_adapter)
        else:
            # Train a quick surgical LoRA adapter
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            
            _, old_to_new_map = drop_layers_with_mapping(
                AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
                ), drop_indices
            )
            
            # Move model to device map for PEFT training
            model = model.to(device)
            config = build_surgical_lora_config(old_to_new_map, rank=16, boundary_thickness=2, dropped_indices=drop_indices)
            model = get_peft_model(model, config)
            train_lora_with_logging(model, tokenizer, device, samples=150)
            model_healed = model
        
        ppl_healed = compute_perplexity_c4(model_healed, tokenizer, device, max_samples=100)
        print(f"  Healed C4 PPL: {ppl_healed:.4f}")
        
    out_file = f"experiments/phase25_causal_interventions/temp_ppl_k{k}.json"
    with open(out_file, "w") as f:
        json.dump({
            "unhealed_ppl": ppl_unhealed,
            "healed_ppl": ppl_healed
        }, f, indent=2)
    print(f"Saved temp results for k={k} to {out_file}")

if __name__ == "__main__":
    main()
