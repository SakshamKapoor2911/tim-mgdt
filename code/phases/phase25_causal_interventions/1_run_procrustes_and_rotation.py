"""
P1: Direct Causal Interventions
  Exp 1A: Orthogonal Procrustes realignment (R*) on pruned model -> GSM8K recovery
  Exp 1B: Artificial rotation in intact model at multiple angles -> reasoning collapse

Saves results to experiments/phase25_causal_interventions/
"""

import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase25_causal_interventions.causal_hooks import (
    compute_procrustes_rotation,
    register_collector,
    register_intervention,
)
from code.phases.phase11.utils import get_device, save_json
from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k



def compute_mcnemar(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return {"p_value": None, "b": 0, "c": 0}
    from scipy.stats import binom
    b = sum(1 for x, y in zip(vec_a, vec_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(vec_a, vec_b) if x == 0 and y == 1)
    a = sum(1 for x, y in zip(vec_a, vec_b) if x == 1 and y == 1)
    d = sum(1 for x, y in zip(vec_a, vec_b) if x == 0 and y == 0)
    n = b + c
    if n > 0:
        p_val = float(2.0 * binom.cdf(min(b, c), n, 0.5))
        p_val = min(p_val, 1.0)
    else:
        p_val = 1.0
    return {
        "p_value": p_val,
        "both_correct": a,
        "a_only": b,
        "b_only": c,
        "both_incorrect": d,
    }


def collect_hidden_at_layer(
    model_name, drop_indices, layer_idx, tokenizer, num_prompts=50
):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    if drop_indices:
        _drop_layers(model, drop_indices)
    model.config.use_cache = False
    collector, handle = register_collector(model, layer_idx)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= num_prompts:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)
    all_hidden = []
    for text in tqdm(prompts, desc="Collecting hidden states", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        all_hidden.append(collector.hidden)
    handle.remove()
    del model
    torch.cuda.empty_cache()
    return torch.cat(all_hidden, dim=0)


def _drop_layers(model, drop_indices):
    remaining = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)


def run_procrustes_recovery(args):
    print("=" * 60)
    print("EXP 1A: Procrustes Realignment (R*) on Pruned Model")
    print("=" * 60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    intact_layer = args.post_incision_layer
    drop_set = list(range(args.incision_start, args.incision_start + args.skip_depth))
    pruned_post_layer = intact_layer - args.skip_depth

    print(f"Collecting intact hidden states at Layer {intact_layer}...")
    intact_hidden = collect_hidden_at_layer(
        args.model_name, None, intact_layer, tokenizer, args.num_calibration
    )
    print(f"  Collected intact hidden: {intact_hidden.shape}")

    print(f"Collecting pruned hidden states (drop {drop_set}) at Layer {pruned_post_layer}...")
    pruned_hidden = collect_hidden_at_layer(
        args.model_name, drop_set, pruned_post_layer, tokenizer, args.num_calibration
    )
    print(f"  Collected pruned hidden: {pruned_hidden.shape}")

    print("Computing Procrustes rotation R*...")
    R_star = compute_procrustes_rotation(intact_hidden, pruned_hidden)
    save_json(os.path.join(args.output_dir, "R_star_shape.json"), {"shape": list(R_star.shape)})
    torch.save(R_star, os.path.join(args.output_dir, "R_star.pt"))
    print(f"  R* shape: {R_star.shape}")

    print("Loading pruned model for Procrustes-intervened GSM8K eval...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    _drop_layers(model, drop_set)
    hook, handle = register_intervention(model, pruned_post_layer)
    hook.set_rotation(R_star)

    acc_procrustes, vec_procrustes, _, _, _, _ = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples, num_shots=8)
    print(f"  Procrustes R* GSM8K accuracy: {acc_procrustes:.2f}%")

    handle.remove()
    del model, hook
    torch.cuda.empty_cache()

    print("Loading pruned model WITHOUT intervention for baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    _drop_layers(model, drop_set)
    acc_pruned_baseline, vec_pruned_baseline, _, _, _, _ = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples, num_shots=8)
    print(f"  Pruned unhealed GSM8K accuracy: {acc_pruned_baseline:.2f}%")

    del model
    torch.cuda.empty_cache()

    result = {
        "experiment": "1A_procrustes_realignment",
        "model": args.model_name,
        "drop_indices": drop_set,
        "intact_hidden_shape": list(intact_hidden.shape),
        "pruned_hidden_shape": list(pruned_hidden.shape),
        "R_star_shape": list(R_star.shape),
        "acc_procrustes_intervened": acc_procrustes,
        "acc_pruned_baseline": acc_pruned_baseline,
        "vec_procrustes_intervened": vec_procrustes,
        "vec_pruned_baseline": vec_pruned_baseline,
        "num_eval_samples": args.num_eval_samples,
        "num_calibration_prompts": args.num_calibration,
    }
    save_json(os.path.join(args.output_dir, "exp_1a_procrustes.json"), result)
    return result
def run_sufficiency_intervention(args):
    print("\n" + "=" * 60)
    print("EXP 1B: Causal Sufficiency Intervention (R_inv = (R*)^T) on Intact Model")
    print("=" * 60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    r_star_path = os.path.join(args.output_dir, "R_star.pt")
    if not os.path.exists(r_star_path):
        print(f"Error: R_star.pt not found at {r_star_path}. Run Exp 1A first.")
        return {}

    print(f"Loading R_star from {r_star_path}...")
    R_star = torch.load(r_star_path)
    R_inv = R_star.T
    print(f"  R_inv shape: {R_inv.shape}")

    # --- Phase 1: Dense Baseline & Random SO(d) Control ---
    print("\n--- Phase 1: Dense Baseline & Random SO(d) Control on Intact Model ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    rotation_layer = args.post_incision_layer
    hook, handle = register_intervention(model, rotation_layer)

    print("Evaluating dense intact model WITHOUT rotation baseline...")
    dense_baseline_acc, vec_dense_baseline, _, _, _, _ = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples, num_shots=8)
    print(f"  Dense baseline (no rotation): {dense_baseline_acc:.2f}%")

    from code.phases.phase25_causal_interventions.causal_hooks import random_orthogonal_matrix
    hidden_dim = R_star.shape[0]
    
    accs_random = []
    vecs_random = []
    for seed in [42, 43, 44]:
        R_rand = random_orthogonal_matrix(hidden_dim, seed=seed)
        print(f"Applying random SO(d) placebo rotation at Layer {rotation_layer} (seed {seed})...")
        hook.set_rotation(R_rand)
        acc_r, vec_r, _, _, _, _ = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples, num_shots=8)
        accs_random.append(acc_r)
        vecs_random.append(vec_r)
    
    acc_random_rotated = float(np.mean(accs_random))
    vec_random_rotated = vecs_random[0]
    print(f"  Intact model with random SO(d) rotation GSM8K accuracy (mean of 3 seeds): {acc_random_rotated:.2f}%")

    handle.remove()
    del model, hook
    torch.cuda.empty_cache()

    # --- Phase 2: Causal Sufficiency (R*)^T Rotation ---
    print("\n--- Phase 2: Causal Sufficiency (R*)^T Rotation on Intact Model ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    hook, handle = register_intervention(model, rotation_layer)

    print(f"Applying (R*)^T rotation at Layer {rotation_layer}...")
    hook.set_rotation(R_inv)

    acc_rotated, vec_rotated, _, _, _, _ = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples, num_shots=8)
    print(f"  Intact model with (R*)^T rotation GSM8K accuracy: {acc_rotated:.2f}%")

    handle.remove()
    del model, hook
    torch.cuda.empty_cache()

    results = {
        "experiment": "1B_causal_sufficiency",
        "model": args.model_name,
        "rotation_layer": rotation_layer,
        "dense_baseline": dense_baseline_acc,
        "acc_random_rotated_control": acc_random_rotated,
        "acc_rotated_sufficiency": acc_rotated,
        "vec_dense_baseline": vec_dense_baseline,
        "vec_random_rotated_control": vec_random_rotated,
        "vec_rotated_sufficiency": vec_rotated,
        "num_eval_samples": args.num_eval_samples,
    }
    save_json(os.path.join(args.output_dir, "exp_1b_sufficiency.json"), results)
    save_json(os.path.join(args.output_dir, "exp_1b_rotation_sweep.json"), results)
    return results


def main():
    parser = argparse.ArgumentParser(description="P1: Causal Intervention Experiments")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/phase25_causal_interventions")
    parser.add_argument("--incision_start", type=int, default=11)
    parser.add_argument("--skip_depth", type=int, default=4)
    parser.add_argument("--post_incision_layer", type=int, default=15)
    parser.add_argument("--rotation_layer", type=int, default=-1)
    parser.add_argument("--rotation_angles", type=float, nargs="+",
                        default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    parser.add_argument("--num_calibration", type=int, default=50)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--skip_exp_1a", action="store_true")
    parser.add_argument("--skip_exp_1b", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    overall = {"model": args.model_name, "drop_indices": args.incision_start}
    save_json(os.path.join(args.output_dir, "_checkpoint_1_started.json"),
              {"status": "started", "args": vars(args)})

    result_1a = {}
    result_1b = {}

    if args.skip_exp_1a:
        print("Skipping Exp 1A (Procrustes).")
    else:
        result_1a = run_procrustes_recovery(args)
        overall["exp_1a"] = result_1a

    if args.skip_exp_1b:
        print("Skipping Exp 1B (Artificial Rotation).")
    else:
        result_1b = run_sufficiency_intervention(args)
        overall["exp_1b"] = result_1b

    mcnemar_out = {}
    if "vec_dense_baseline" in result_1b and "vec_rotated_sufficiency" in result_1b:
        mcnemar_dense_vs_rotated = compute_mcnemar(
            result_1b["vec_dense_baseline"], result_1b["vec_rotated_sufficiency"]
        )
        overall["mcnemar_dense_vs_rotated"] = mcnemar_dense_vs_rotated
        mcnemar_out["mcnemar_dense_vs_rotated"] = mcnemar_dense_vs_rotated

    if "vec_dense_baseline" in result_1b and "vec_pruned_baseline" in result_1a:
        mcnemar_dense_vs_pruned = compute_mcnemar(
            result_1b["vec_dense_baseline"], result_1a["vec_pruned_baseline"]
        )
        overall["mcnemar_dense_vs_pruned"] = mcnemar_dense_vs_pruned
        mcnemar_out["mcnemar_dense_vs_pruned"] = mcnemar_dense_vs_pruned

    if "vec_dense_baseline" in result_1b and "vec_random_rotated_control" in result_1b:
        mcnemar_dense_vs_random = compute_mcnemar(
            result_1b["vec_dense_baseline"], result_1b["vec_random_rotated_control"]
        )
        overall["mcnemar_dense_vs_random"] = mcnemar_dense_vs_random
        mcnemar_out["mcnemar_dense_vs_random"] = mcnemar_dense_vs_random

    if "vec_random_rotated_control" in result_1b and "vec_rotated_sufficiency" in result_1b:
        mcnemar_random_vs_rotated = compute_mcnemar(
            result_1b["vec_random_rotated_control"], result_1b["vec_rotated_sufficiency"]
        )
        overall["mcnemar_random_vs_rotated"] = mcnemar_random_vs_rotated
        mcnemar_out["mcnemar_random_vs_rotated"] = mcnemar_random_vs_rotated

    if mcnemar_out:
        save_json(os.path.join(args.output_dir, "exp_mcnemar.json"), mcnemar_out)

    save_json(os.path.join(args.output_dir, "_checkpoint_1_complete.json"),
              {"status": "complete", "overall": overall})
    print("\n" + "=" * 60)
    print("P1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
