"""
Phase 13: Ablation & Hardening Sweep

Runs all ablations needed to make the Phase 12 results publishable:
  1. Held-out corpus (C4) evaluation for all methods
  2. Uniform LoRA control (all layers) vs surgical (top-6 sensitive)
  3. Fair comparison: Method 2 vs Method 3 with identical hyperparams
  4. Training loss curve logging for convergence comparison
  5. Sparsity sweep: 10%, 20%, 30%, 40%
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase11.utils import (
    get_device, compute_perplexity, compute_perplexity_c4,
    apply_layerwise_pruning
)


def load_model_and_tokenizer(model_name, device):
    """Load model fresh (no side effects from previous runs)."""
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


def get_sensitivities():
    """Load ground-truth layer sensitivities from proxy validation."""
    val_data = json.load(open('experiments/phase12/proxy_validation.json'))
    return {k: v['ground_truth_delta_ce'] for k, v in val_data['metrics'].items()}


def build_lora_config(sensitivities, rank, mode="surgical"):
    """
    Build LoRA config.
    mode="surgical": top-6 sensitive layers only
    mode="uniform": ALL layers (control experiment)
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
    elif mode == "uniform":
        # Target ALL layers — the control experiment
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
        init_lora_weights=True  # Standard random init
    )
    return config


def build_svd_lora_config(model, sensitivities, sparsity, rank):
    """Build LoRA config with SVD initialization of pruning damage."""
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

    # Save original weights before pruning
    w_origs = {}
    for name, module in model.named_modules():
        if name in target_modules and isinstance(module, nn.Linear):
            w_origs[name] = module.weight.data.clone().float()

    # Prune
    apply_layerwise_pruning(model, sparsity)

    # Wrap with PEFT
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=False
    )
    model = get_peft_model(model, config)

    # SVD initialize
    scaling = config.lora_alpha / config.r
    for name, w_orig in w_origs.items():
        peft_name = "base_model.model." + name
        peft_module = dict(model.named_modules())[peft_name]
        w_pruned = peft_module.base_layer.weight.data.float()
        delta_w = w_orig.to(w_pruned.device) - w_pruned

        if delta_w.abs().sum() == 0:
            continue

        U, S, Vh = torch.linalg.svd(delta_w.cpu(), full_matrices=False)
        r = min(rank, S.size(0))
        S_r = torch.diag(S[:r])

        lora_B_weight = (U[:, :r] @ S_r) / scaling
        lora_A_weight = Vh[:r, :]

        peft_module.lora_A['default'].weight.data = lora_A_weight.to(
            device=peft_module.lora_A['default'].weight.device,
            dtype=peft_module.lora_A['default'].weight.dtype)
        peft_module.lora_B['default'].weight.data = lora_B_weight.to(
            device=peft_module.lora_B['default'].weight.device,
            dtype=peft_module.lora_B['default'].weight.dtype)

    return model


def train_lora_with_logging(model, tokenizer, device, samples=100, lr=2e-4):
    """Train LoRA and return per-step loss curve."""
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([s for s in dataset["text"] if s.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    seq_len = 2048
    num_chunks = min(samples, input_ids.size(0) // seq_len)
    chunks = [input_ids[i*seq_len:(i+1)*seq_len] for i in range(num_chunks)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []

    for chunk in tqdm(chunks, desc="    Training"):
        chunk = chunk.unsqueeze(0).to(device)
        outputs = model(input_ids=chunk, labels=chunk)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return losses


def run_ablation(model_name, sparsity, rank, train_samples, eval_samples,
                 mode, svd_init, lr, device):
    """Run a single ablation experiment and return results dict."""
    sensitivities = get_sensitivities()

    print(f"\n{'='*60}")
    print(f"  mode={mode}, svd_init={svd_init}, sparsity={sparsity}, rank={rank}, lr={lr}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Baseline PPL (both corpora)
    print("Measuring baseline PPL...")
    baseline_wt2 = compute_perplexity(model, tokenizer, device, max_samples=eval_samples)
    print("Measuring baseline C4 PPL...")
    baseline_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=eval_samples)

    start_time = time.time()

    if svd_init:
        # Method 3: SVD-init LoRA (handles pruning internally)
        model = build_svd_lora_config(model, sensitivities, sparsity, rank)
    else:
        # Method 2 / 2b: Random-init LoRA
        apply_layerwise_pruning(model, sparsity)
        lora_config = build_lora_config(sensitivities, rank, mode=mode)
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Measure pruned PPL before training
    model.eval()
    pruned_wt2 = compute_perplexity(model, tokenizer, device, max_samples=eval_samples)

    # Train
    losses = train_lora_with_logging(model, tokenizer, device, samples=train_samples, lr=lr)
    recovery_time = time.time() - start_time

    # Evaluate recovered PPL (both corpora)
    model.eval()
    recovered_wt2 = compute_perplexity(model, tokenizer, device, max_samples=eval_samples)
    recovered_c4 = compute_perplexity_c4(model, tokenizer, device, max_samples=eval_samples)

    results = {
        "mode": mode,
        "svd_init": svd_init,
        "sparsity": sparsity,
        "rank": rank,
        "lr": lr,
        "train_samples": train_samples,
        "recovery_time_s": recovery_time,
        "baseline_wt2_ppl": baseline_wt2,
        "baseline_c4_ppl": baseline_c4,
        "pruned_wt2_ppl": pruned_wt2,
        "recovered_wt2_ppl": recovered_wt2,
        "recovered_c4_ppl": recovered_c4,
        "wt2_delta_pct": ((recovered_wt2 - baseline_wt2) / baseline_wt2) * 100,
        "c4_delta_pct": ((recovered_c4 - baseline_c4) / baseline_c4) * 100,
        "loss_curve": losses,
    }

    print(f"\n--- Results ---")
    print(f"  Baseline WT2:  {baseline_wt2:.4f}  |  C4: {baseline_c4:.4f}")
    print(f"  Pruned WT2:    {pruned_wt2:.4f}")
    print(f"  Recovered WT2: {recovered_wt2:.4f} ({results['wt2_delta_pct']:+.2f}%)")
    print(f"  Recovered C4:  {recovered_c4:.4f} ({results['c4_delta_pct']:+.2f}%)")
    print(f"  Time: {recovery_time:.1f}s")

    # Critical fix: Free GPU memory to prevent OOM on subsequent ablations
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
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = get_device()
    os.makedirs("experiments/phase13", exist_ok=True)

    results_file = "experiments/phase13/ablation_results.json"
    curves_file = "experiments/phase13/loss_curves.json"

    # Load existing to resume
    all_results = {}
    curves = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
    if os.path.exists(curves_file):
        with open(curves_file, "r") as f:
            curves = json.load(f)

    def run_and_save(name, kwargs):
        if name in all_results:
            print(f"\nSkipping {name} (already completed)")
            return
        
        r = run_ablation(**kwargs)
        curves[name] = r.pop("loss_curve")
        all_results[name] = r
        
        # Save incrementally
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        with open(curves_file, "w") as f:
            json.dump(curves, f)

    # ─── Ablation 1: Surgical vs Uniform LoRA (at 20% sparsity) ───
    print("\n" + "="*70)
    print("ABLATION 1: Surgical (top-6) vs Uniform (all layers) LoRA")
    print("="*70)

    run_and_save("surgical_random_20pct", dict(
        model_name=args.model, sparsity=0.20, rank=args.rank, train_samples=args.train_samples,
        eval_samples=args.eval_samples, mode="surgical", svd_init=False, lr=args.lr, device=device
    ))

    run_and_save("uniform_random_20pct", dict(
        model_name=args.model, sparsity=0.20, rank=args.rank, train_samples=args.train_samples,
        eval_samples=args.eval_samples, mode="uniform", svd_init=False, lr=args.lr, device=device
    ))

    # ─── Ablation 2: Random-init vs SVD-init (fair comparison, same hyperparams) ───
    print("\n" + "="*70)
    print("ABLATION 2: Random-init vs SVD-init (identical hyperparams)")
    print("="*70)

    run_and_save("surgical_svd_20pct", dict(
        model_name=args.model, sparsity=0.20, rank=args.rank, train_samples=args.train_samples,
        eval_samples=args.eval_samples, mode="surgical", svd_init=True, lr=args.lr, device=device
    ))

    # ─── Ablation 3: Sparsity sweep (SVD-init surgical only) ───
    print("\n" + "="*70)
    print("ABLATION 3: Sparsity sweep (10%, 30%, 40%)")
    print("="*70)

    for sp in [0.10, 0.30, 0.40]:
        name = f"surgical_svd_{int(sp*100)}pct"
        run_and_save(name, dict(
            model_name=args.model, sparsity=sp, rank=args.rank, train_samples=args.train_samples,
            eval_samples=args.eval_samples, mode="surgical", svd_init=True, lr=args.lr, device=device
        ))

    # ─── Print summary table ───
    print("\n\n" + "="*90)
    print("PHASE 13 ABLATION SUMMARY")
    print("="*90)
    print(f"{'Experiment':<30} {'WT2 PPL':>10} {'WT2 Δ%':>10} {'C4 PPL':>10} {'C4 Δ%':>10} {'Time':>8}")
    print("-"*90)
    for name, r in all_results.items():
        print(f"{name:<30} {r['recovered_wt2_ppl']:>10.4f} {r['wt2_delta_pct']:>+10.2f} "
              f"{r['recovered_c4_ppl']:>10.4f} {r['c4_delta_pct']:>+10.2f} {r['recovery_time_s']:>7.0f}s")

    print(f"\nResults saved to experiments/phase13/ablation_results.json")


if __name__ == "__main__":
    main()
