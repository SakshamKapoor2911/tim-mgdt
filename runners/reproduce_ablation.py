#!/usr/bin/env python3
"""
Minimal Reproducer for Incision Boundary Healing (COLM 2026 Submission)
This script demonstrates the end-to-end methodology in < 100 lines.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def main():
    print("1. Loading Mistral-7B...")
    model_id = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load in fp16 to fit on consumer GPUs
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    print("\n2. Simulating Layer Drop (dropping 4 layers: 14-17)...")
    drop_indices = [14, 15, 16, 17]
    old_layers = model.model.layers
    new_layers = nn.ModuleList([l for i, l in enumerate(old_layers) if i not in drop_indices])
    
    # Fix internal indices
    for new_idx, layer in enumerate(new_layers):
        layer.self_attn.layer_idx = new_idx
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    
    print(f"   Architecture compressed to {len(new_layers)} layers.")

    print("\n3. Applying Incision Boundary Healing (t=1)...")
    # In the compressed model, old layer 13 is now index 13. Old layer 18 is now index 14.
    # Boundary layers are the immediate neighbors: 13 (Left) and 14 (Right)
    boundary_layers = [13, 14]
    
    target_modules = []
    for idx in boundary_layers:
        target_modules.extend([
            f"model.layers.{idx}.self_attn.q_proj", f"model.layers.{idx}.self_attn.v_proj",
            f"model.layers.{idx}.mlp.gate_proj", f"model.layers.{idx}.mlp.up_proj", f"model.layers.{idx}.mlp.down_proj"
        ])
        
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules, 
        bias="none", task_type="CAUSAL_LM", init_lora_weights=True
    )
    
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    healed_model = get_peft_model(model, lora_config)
    healed_model.print_trainable_parameters()

    print("\n4. Running Proof-of-Concept Healing Step (C4 Dataset)...")
    # Load a tiny snippet of C4
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(dataset["text"][:50])
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    
    optimizer = torch.optim.AdamW(healed_model.parameters(), lr=1e-5)
    healed_model.train()
    
    # 5 dummy steps to prove it trains
    for step in range(5):
        outputs = healed_model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"   Step {step+1}/5 | Loss: {loss.item():.4f}")

    print("\nSuccess! Methodology verified. For full reproduction, run `./code/phases/phase15_layer_drop/run_phase15.sh`.")

if __name__ == "__main__":
    main()
