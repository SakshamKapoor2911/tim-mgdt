"""
Phase 14: Final Actionable Interpretability Pipeline
Script 2: Surgical Recovery Ablations

This script runs the core experiments to prove "Sensitivity-Guided Surgical Allocation":
1. Uniform Allocation Baseline
2. "Ends-Only" Heuristic Baseline (Control for top/bottom layer heuristics)
3. Surgical Allocation (Guided by exact Delta CE Bathtub Curve)
4. SVD-Init Baseline (Comparative baseline to demonstrate dense noise pollution failure)
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase11.utils import get_device, compute_perplexity_c4, apply_layerwise_pruning

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def build_lora_config(sensitivities, rank, mode="surgical"):
    """
    mode="surgical": top-6 sensitive layers from exact delta CE map.
    mode="uniform": all layers.
    mode="ends-only": exactly 3 top layers and 3 bottom layers.
    """
    if mode == "surgical":
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
            
    elif mode == "ends-only":
        n_layers = len(sensitivities)
        # 3 from start, 3 from end
        target_layers = [0, 1, 2, n_layers-3, n_layers-2, n_layers-1]
        target_modules = []
        for idx in target_layers:
            target_modules.extend([
                f"model.layers.{idx}.self_attn.q_proj",
                f"model.layers.{idx}.self_attn.v_proj",
                f"model.layers.{idx}.mlp.gate_proj",
                f"model.layers.{idx}.mlp.up_proj",
                f"model.layers.{idx}.mlp.down_proj"
            ])
            
    elif mode == "uniform":
        target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True  # Random init
    )
    return config

def train_lora_with_logging(model, tokenizer, device, samples=500, lr=1e-5, warmup_ratio=0.10):
    """Train LoRA and return per-step loss curve. Increased to 500 steps."""
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

    for chunk in tqdm(chunks, desc="    Training"):
        chunk = chunk.unsqueeze(0).to(device)
        outputs = model(input_ids=chunk, labels=chunk)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return losses

def run_ablation(model_name, map_file, sparsity, rank, train_samples, eval_samples, mode, lr, device, warmup_ratio):
    print(f"\n{'='*60}")
    print(f"  mode={mode}, sparsity={sparsity}, rank={rank}, lr={lr}")
    print(f"{'='*60}")

    with open(map_file, "r") as f:
        sensitivities = json.load(f)["sensitivities"]

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # We evaluate exclusively on C4 to avoid training data contamination
    print("Measuring baseline C4 PPL...")
    baseline_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=eval_samples)

    start_time = time.time()

    # Prune
    apply_layerwise_pruning(model, sparsity)
    
    # Measure pruned PPL
    model.eval()
    pruned_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=eval_samples)

    # Attach LoRA and Train
    lora_config = build_lora_config(sensitivities, rank, mode=mode)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    losses = train_lora_with_logging(model, tokenizer, device, samples=train_samples, lr=lr, warmup_ratio=warmup_ratio)
    recovery_time = time.time() - start_time

    # Evaluate recovered PPL
    model.eval()
    recovered_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=eval_samples)

    results = {
        "mode": mode,
        "sparsity": sparsity,
        "rank": rank,
        "train_samples": train_samples,
        "recovery_time_s": recovery_time,
        "baseline_c4_ppl": baseline_c4,
        "pruned_c4_ppl": pruned_c4,
        "recovered_c4_ppl": recovered_c4,
        "c4_delta_pct": ((recovered_c4 - baseline_c4) / baseline_c4) * 100,
        "loss_curve": losses,
    }

    print(f"\n--- Results ---")
    print(f"  Baseline C4:  {baseline_c4:.4f}")
    print(f"  Pruned C4:    {pruned_c4:.4f}")
    print(f"  Recovered C4: {recovered_c4:.4f} ({results['c4_delta_pct']:+.2f}%)")

    # Critical memory flush
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--eval_samples", type=int, default=30)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    out_dir = "experiments/phase14_final"
    os.makedirs(out_dir, exist_ok=True)
    
    model_short = args.model.split("/")[-1]
    map_file = os.path.join(out_dir, f"{model_short}_sensitivity_map.json")
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Must run 1_measure_sensitivity.py first to generate {map_file}")

    results_file = os.path.join(out_dir, f"{model_short}_ablation_results.json")
    all_results = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)

    def run_and_save(name, kwargs):
        if name in all_results:
            return
        r = run_ablation(**kwargs)
        all_results[name] = r
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Control 1: Uniform Allocation
    run_and_save("uniform_20pct", dict(
        model_name=args.model, map_file=map_file, sparsity=0.20, rank=args.rank, 
        train_samples=args.train_samples, eval_samples=args.eval_samples, 
        mode="uniform", lr=args.lr, device=device, warmup_ratio=args.warmup_ratio
    ))

    # Control 2: Ends-Only Heuristic
    run_and_save("ends_only_20pct", dict(
        model_name=args.model, map_file=map_file, sparsity=0.20, rank=args.rank, 
        train_samples=args.train_samples, eval_samples=args.eval_samples, 
        mode="ends-only", lr=args.lr, device=device, warmup_ratio=args.warmup_ratio
    ))

    # Actual Method: Surgical Delta CE Allocation
    run_and_save("surgical_20pct", dict(
        model_name=args.model, map_file=map_file, sparsity=0.20, rank=args.rank, 
        train_samples=args.train_samples, eval_samples=args.eval_samples, 
        mode="surgical", lr=args.lr, device=device, warmup_ratio=args.warmup_ratio
    ))

if __name__ == "__main__":
    main()
