import torch
import numpy as np
import os
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import sem, t

# Import hook definitions
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../code/phases/phase25_causal_interventions")))
from causal_hooks import register_collector

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _drop_layers(model, drop_indices):
    remaining = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)])
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)

def collect_pruned_hidden_list(model_name, drop_set, layer_idx, tokenizer, num_prompts=50):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    if drop_set:
        _drop_layers(model, drop_set)
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
    
    hidden_list = []
    for text in tqdm(prompts, desc=f"Collecting hidden states ({model_name})", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        hidden_list.append(collector.hidden.clone())
    handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hidden_list

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n <= 1:
        return 0.0
    mean_val = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2.0, n - 1)
    return float(h)

def main():
    parser = argparse.ArgumentParser(description="Compute true PC1 direction and energy collapse with per-passage CI.")
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="experiments/phase25_causal_interventions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = [
        {
            "name": "mistralai/Mistral-7B-v0.1",
            "drop_set": list(range(11, 15)),
            "post_incision_layer": 15,
            "pruned_post_layer": 11
        },
        {
            "name": "meta-llama/Meta-Llama-3-8B",
            "drop_set": list(range(8, 12)),
            "post_incision_layer": 12,
            "pruned_post_layer": 8
        },
        {
            "name": "google/gemma-2-2b",
            "drop_set": list(range(12, 16)),
            "post_incision_layer": 16,
            "pruned_post_layer": 12
        }
    ]

    all_results = {}

    for cfg in configs:
        model_name = cfg["name"]
        print(f"\n==================================================")
        print(f"Evaluating {model_name}...")
        print(f"==================================================")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        drop_set = cfg["drop_set"]
        post_incision_layer = cfg["post_incision_layer"]
        pruned_post_layer = cfg["pruned_post_layer"]

        print("Collecting intact hidden states...")
        intact_list = collect_pruned_hidden_list(model_name, None, post_incision_layer, tokenizer, args.num_prompts)
        
        print("Collecting pruned hidden states...")
        pruned_list = collect_pruned_hidden_list(model_name, drop_set, pruned_post_layer, tokenizer, args.num_prompts)

        # 1. Compute concatenated statistics
        intact_cat = torch.cat(intact_list, dim=0).detach().float().cpu().numpy()
        pruned_cat = torch.cat(pruned_list, dim=0).detach().float().cpu().numpy()

        intact_centered_cat = intact_cat - np.mean(intact_cat, axis=0, keepdims=True)
        pruned_centered_cat = pruned_cat - np.mean(pruned_cat, axis=0, keepdims=True)

        _, S_i_cat, Vt_i_cat = np.linalg.svd(intact_centered_cat, full_matrices=False)
        _, S_p_cat, Vt_p_cat = np.linalg.svd(pruned_centered_cat, full_matrices=False)

        var_i_cat = (S_i_cat[0] ** 2) / intact_centered_cat.shape[0]
        var_p_cat = (S_p_cat[0] ** 2) / pruned_centered_cat.shape[0]
        collapse_cat = 1.0 - (var_p_cat / var_i_cat)

        proj_raw_cat = np.dot(Vt_p_cat[0], Vt_i_cat[0])
        proj_sync_cat = -proj_raw_cat if proj_raw_cat < 0 else proj_raw_cat
        theta_cat = np.degrees(np.arccos(np.clip(proj_sync_cat, -1.0, 1.0)))

        # 2. Compute per-passage statistics
        per_passage_angles = []
        per_passage_collapses = []

        for h_i_tensor, h_p_tensor in zip(intact_list, pruned_list):
            h_i = h_i_tensor.detach().float().cpu().numpy()  # [512, d]
            h_p = h_p_tensor.detach().float().cpu().numpy()  # [512, d]

            h_i_c = h_i - np.mean(h_i, axis=0, keepdims=True)
            h_p_c = h_p - np.mean(h_p, axis=0, keepdims=True)

            _, S_i, Vt_i = np.linalg.svd(h_i_c, full_matrices=False)
            _, S_p, Vt_p = np.linalg.svd(h_p_c, full_matrices=False)

            v_i = Vt_i.T[:, 0]
            v_p = Vt_p.T[:, 0]

            proj_raw = np.dot(v_p, v_i)
            v_p_sync = -v_p if proj_raw < 0 else v_p
            proj_sync = np.dot(v_p_sync, v_i)
            
            theta = np.degrees(np.arccos(np.clip(proj_sync, -1.0, 1.0)))
            per_passage_angles.append(theta)

            var_i = (S_i[0] ** 2) / h_i.shape[0]
            var_p = (S_p[0] ** 2) / h_p.shape[0]
            collapse = 1.0 - (var_p / var_i)
            per_passage_collapses.append(collapse * 100.0)

        mean_angle = float(np.mean(per_passage_angles))
        ci_angle = compute_confidence_interval(per_passage_angles)

        mean_collapse = float(np.mean(per_passage_collapses))
        ci_collapse = compute_confidence_interval(per_passage_collapses)

        print(f"Results for {model_name}:")
        print(f"  Concatenated PC1 Energy Collapse: {collapse_cat * 100:.2f}%")
        print(f"  Concatenated PC1 Theta: {theta_cat:.4f}°")
        print(f"  Per-passage PC1 Energy Collapse: {mean_collapse:.2f}% ± {ci_collapse:.2f}%")
        print(f"  Per-passage PC1 Theta: {mean_angle:.4f}° ± {ci_angle:.4f}°")

        all_results[model_name] = {
            "concatenated": {
                "energy_collapse_percent": float(collapse_cat * 100.0),
                "theta_deg": float(theta_cat),
                "raw_projection": float(proj_raw_cat),
                "synced_projection": float(proj_sync_cat)
            },
            "per_passage": {
                "mean_energy_collapse_percent": mean_collapse,
                "ci_energy_collapse_percent": ci_collapse,
                "mean_theta_deg": mean_angle,
                "ci_theta_deg": ci_angle,
                "raw_values_energy_collapse": [float(x) for x in per_passage_collapses],
                "raw_values_theta": [float(x) for x in per_passage_angles]
            }
        }

    with open(os.path.join(args.output_dir, "energy_collapse.json"), "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
