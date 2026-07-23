"""
Phase 15: Gemma-2-2B Calibration & Global Sensitivity Healing Run
GPU Task 0:
  1. Dense calibrated baseline (1,000 samples WikiText-2 train, seed=42, no layers dropped)
  2. Global sensitivity LoRA healing (--healing_mode global, n_drop=4)
"""

import os
import json
import time
import importlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from code.phases.phase11.utils import get_device, compute_perplexity_c4

# Import module with leading digit safely
heal_module = importlib.import_module("code.phases.phase15_layer_drop.2_heal_dropped_layers")
drop_layers_with_mapping = heal_module.drop_layers_with_mapping
find_best_drop_window = heal_module.find_best_drop_window
train_lora_with_logging = heal_module.train_lora_with_logging

def run_dense_calibration(model_name="google/gemma-2-2b", train_samples=1000, eval_samples=128, rank=16, lr=1e-5, seed=42):
    print(f"\n[Gemma Calibration] Training 1,000-step dense LoRA calibration on {model_name}...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )

    # Attach dense LoRA across transformer layers
    target_modules = []
    for i in range(model.config.num_hidden_layers):
        target_modules.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.mlp.gate_proj",
            f"model.layers.{i}.mlp.up_proj",
            f"model.layers.{i}.mlp.down_proj"
        ])

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    losses, eval_ppls = train_lora_with_logging(
        model, tokenizer, device, train_samples, lr, warmup_ratio=0.10
    )

    model.eval()
    print("Measuring CALIBRATED DENSE C4 PPL...")
    calibrated_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=eval_samples)
    print(f" -> Calibrated Dense C4 PPL: {calibrated_c4:.4f}")

    res = {
        "model": model_name,
        "calibrated_c4_ppl": calibrated_c4,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "seed": seed
    }
    out_file = "experiments/phase15_layer_drop/gemma-2-2b_dense_calibrated_eval.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    return calibrated_c4

def run_global_healing(model_name="google/gemma-2-2b", n_drop=4, train_samples=1000, eval_samples=128, rank=16, lr=1e-5, seed=42):
    print(f"\n[Gemma Global Healing] Running --healing_mode global for {model_name} (drop={n_drop})...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    out_dir = "experiments/phase15_layer_drop"
    model_short = model_name.split("/")[-1]
    map_file = f"experiments/phase14_final/{model_short}_sensitivity_map.json"

    # Fallback map file if missing
    if not os.path.exists(map_file):
        print(f"Warning: {map_file} missing. Using middle drop window.")
        drop_indices = [8, 9, 10, 11]
    else:
        drop_indices = find_best_drop_window(map_file, n_drop)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model, old_to_new_map = drop_layers_with_mapping(model, drop_indices)

    # Top 6 sensitive layers
    if os.path.exists(map_file):
        with open(map_file) as f:
            data = json.load(f)
        sens = data.get("sensitivities", {})
        sorted_layers = sorted(sens.items(), key=lambda x: x[1], reverse=True)
        top_layers = [l[0] for l in sorted_layers[:6]]
        target_indices = [int(l.split("_")[1]) for l in top_layers if int(l.split("_")[1]) in old_to_new_map]
    else:
        target_indices = list(old_to_new_map.keys())[:6]

    target_modules = []
    for old_idx in target_indices:
        new_idx = old_to_new_map[old_idx]
        target_modules.extend([
            f"model.layers.{new_idx}.self_attn.q_proj",
            f"model.layers.{new_idx}.self_attn.v_proj",
            f"model.layers.{new_idx}.mlp.gate_proj",
            f"model.layers.{new_idx}.mlp.up_proj",
            f"model.layers.{new_idx}.mlp.down_proj"
        ])

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    losses, eval_ppls = train_lora_with_logging(model, tokenizer, device, train_samples, lr, warmup_ratio=0.10)
    model.eval()
    healed_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=eval_samples)
    print(f" -> Gemma Global Healed C4 PPL: {healed_c4:.4f}")

    results = {
        "model": model_name,
        "n_drop": n_drop,
        "drop_indices": drop_indices,
        "train_samples": train_samples,
        "rank": rank,
        "healing_mode": "global",
        "healed_c4_ppl": healed_c4
    }
    out_file = os.path.join(out_dir, f"{model_short}_drop{n_drop}_healed.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    return healed_c4

def main():
    print("="*60)
    print("GPU 0: Running Gemma-2-2B Calibration & Global Healing")
    print("="*60)
    calibrated_ppl = run_dense_calibration()
    global_ppl = run_global_healing()
    print("\n[Gemma Complete]")
    print(f"  Calibrated Dense Baseline PPL: {calibrated_ppl:.4f}")
    print(f"  Global Sensitivity Healed PPL: {global_ppl:.4f}")

if __name__ == "__main__":
    main()
