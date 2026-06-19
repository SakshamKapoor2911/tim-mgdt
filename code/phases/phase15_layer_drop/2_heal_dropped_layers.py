"""
Phase 15: Oracle-Guided Layer Dropping
Script 2: Heal dropped layers using Surgical LoRA on boundary/sensitive layers.
"""
import os
import json
import time
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase11.utils import get_device, compute_perplexity_c4

def find_best_drop_window(map_file: str, n_drop: int):
    with open(map_file, "r") as f:
        data = json.load(f)
    sens = data["sensitivities"]
    n_layers = len(sens)
    
    scores = [sens[f"layer_{i}"] for i in range(n_layers)]
    min_sum = float('inf')
    best_start = 0
    for i in range(n_layers - n_drop + 1):
        window_sum = sum(scores[i:i+n_drop])
        if window_sum < min_sum:
            min_sum = window_sum
            best_start = i
    return list(range(best_start, best_start + n_drop))

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

def build_surgical_lora_config(map_file, old_to_new_map, rank):
    with open(map_file, "r") as f:
        data = json.load(f)
    sens = data["sensitivities"]
    
    sorted_layers = sorted(sens.items(), key=lambda x: x[1], reverse=True)
    top_layers = [layer for layer, score in sorted_layers[:6]]
    
    target_modules = []
    for layer_name in top_layers:
        old_idx = int(layer_name.split("_")[1])
        if old_idx not in old_to_new_map:
            raise ValueError(f"High sensitivity layer {old_idx} was incorrectly dropped!")
            
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

def train_lora_with_logging(model, tokenizer, device, samples, lr, warmup_ratio):
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
    losses = []
    for chunk in tqdm(chunks, desc="    Training Healing LoRA"):
        chunk = chunk.unsqueeze(0).to(device)
        outputs = model(input_ids=chunk, labels=chunk)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--n_drop", type=int, default=4)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
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
    drop_indices = find_best_drop_window(map_file, args.n_drop)
    
    print(f"\n[{model_short}] Loading and dropping layers {drop_indices}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    model, old_to_new_map = drop_layers_with_mapping(model, drop_indices)
    
    print("Attaching Surgical LoRA for healing...")
    lora_config = build_surgical_lora_config(map_file, old_to_new_map, args.rank)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    start_time = time.time()
    losses = train_lora_with_logging(model, tokenizer, device, args.train_samples, args.lr, args.warmup_ratio)
    recovery_time = time.time() - start_time
    
    print("\nMeasuring HEALED C4 PPL...")
    model.eval()
    healed_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=args.eval_samples)
    print(f"  Healed C4 PPL: {healed_c4:.4f}")
    
    # Load previously saved baseline/drop metrics to compute complete picture
    eval_file = os.path.join(out_dir, f"{model_short}_drop{args.n_drop}_eval.json")
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            prev_results = json.load(f)
    else:
        prev_results = {}
        
    results = {
        "model": args.model,
        "n_drop": args.n_drop,
        "drop_indices": drop_indices,
        "train_samples": args.train_samples,
        "rank": args.rank,
        "recovery_time_s": recovery_time,
        "baseline_c4_ppl": prev_results.get("baseline_c4_ppl"),
        "dropped_c4_ppl": prev_results.get("dropped_c4_ppl"),
        "healed_c4_ppl": healed_c4,
        "loss_curve": losses
    }
    
    out_file = os.path.join(out_dir, f"{model_short}_drop{args.n_drop}_healed.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved healed evaluation to {out_file}")

if __name__ == "__main__":
    main()
