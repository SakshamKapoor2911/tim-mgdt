import torch
import numpy as np
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from tqdm import tqdm

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

def collect_pruned_hidden(model_name, drop_set, layer_idx, tokenizer, num_prompts=50):
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
    all_hidden = []
    for text in tqdm(prompts, desc="Collecting hidden states", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        all_hidden.append(collector.hidden.clone())
    handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Keep it as list of tensors, each of shape [512, d]
    return all_hidden

def main():
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

    for cfg in configs:
        model_name = cfg["name"]
        print(f"\n==================================================")
        print(f"Evaluating {model_name} (per-passage)...")
        print(f"==================================================")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        drop_set = cfg["drop_set"]
        post_incision_layer = cfg["post_incision_layer"]
        pruned_post_layer = cfg["pruned_post_layer"]

        intact_hidden_list = collect_pruned_hidden(model_name, None, post_incision_layer, tokenizer, 50)
        pruned_hidden_list = collect_pruned_hidden(model_name, drop_set, pruned_post_layer, tokenizer, 50)

        collapses_centered = []
        collapses_uncentered = []
        acute_angles = []

        for h_i_tensor, h_p_tensor in zip(intact_hidden_list, pruned_hidden_list):
            h_i = h_i_tensor.detach().float().cpu().numpy()  # [512, d]
            h_p = h_p_tensor.detach().float().cpu().numpy()  # [512, d]

            # 1. Centered SVD
            h_i_c = h_i - np.mean(h_i, axis=0, keepdims=True)
            h_p_c = h_p - np.mean(h_p, axis=0, keepdims=True)

            _, S_i_c, Vt_i_c = np.linalg.svd(h_i_c, full_matrices=False)
            _, S_p_c, Vt_p_c = np.linalg.svd(h_p_c, full_matrices=False)

            v_i_c = Vt_i_c.T[:, 0]
            v_p_c = Vt_p_c.T[:, 0]

            # Sign sync for centered PC1
            proj_c = np.dot(v_p_c, v_i_c)
            v_p_c_sync = -v_p_c if proj_c < 0 else v_p_c
            proj_c_sync = np.dot(v_p_c_sync, v_i_c)
            theta_rad = np.arccos(np.clip(proj_c_sync, -1.0, 1.0))
            theta_deg = np.degrees(theta_rad)
            acute_angles.append(theta_deg)

            var_i_c = (S_i_c[0] ** 2) / 512
            var_p_c = (S_p_c[0] ** 2) / 512
            collapse_c = 1.0 - (var_p_c / var_i_c)
            collapses_centered.append(collapse_c)

            # 2. Uncentered SVD
            _, S_i_un, Vt_i_un = np.linalg.svd(h_i, full_matrices=False)
            _, S_p_un, Vt_p_un = np.linalg.svd(h_p, full_matrices=False)

            var_i_un = (S_i_un[0] ** 2) / 512
            var_p_un = (S_p_un[0] ** 2) / 512
            collapse_un = 1.0 - (var_p_un / var_i_un)
            collapses_uncentered.append(collapse_un)

        # Compute stats
        mean_angle = np.mean(acute_angles)
        std_angle = np.std(acute_angles)
        ci_angle = 1.96 * (std_angle / np.sqrt(50))

        mean_c = np.mean(collapses_centered)
        std_c = np.std(collapses_centered)
        ci_c = 1.96 * (std_c / np.sqrt(50))

        mean_un = np.mean(collapses_uncentered)
        std_un = np.std(collapses_uncentered)
        ci_un = 1.96 * (std_un / np.sqrt(50))

        print(f"Results for {model_name} (over 50 passages):")
        print(f"  PC1 Acute Angle: {mean_angle:.4f}° ± {ci_angle:.4f}°")
        print(f"  Centered PC1 Energy Collapse: {mean_c * 100:.2f}% ± {ci_c * 100:.2f}%")
        print(f"  Uncentered PC1 Energy Collapse: {mean_un * 100:.2f}% ± {ci_un * 100:.2f}%")

if __name__ == "__main__":
    main()
