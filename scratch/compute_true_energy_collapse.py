import torch
import numpy as np
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes

# Import hook definitions
import sys
import os
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
    for text in tqdm(prompts, desc="Collecting hidden states"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        all_hidden.append(collector.hidden)
    handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(all_hidden, dim=0)

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
        print(f"Evaluating {model_name}...")
        print(f"==================================================")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        drop_set = cfg["drop_set"]
        post_incision_layer = cfg["post_incision_layer"]
        pruned_post_layer = cfg["pruned_post_layer"]

        print("Collecting intact hidden states...")
        intact_hidden = collect_pruned_hidden(model_name, None, post_incision_layer, tokenizer, 50)
        print("Collecting pruned hidden states...")
        pruned_hidden = collect_pruned_hidden(model_name, drop_set, pruned_post_layer, tokenizer, 50)

        # Move to numpy
        intact_np = intact_hidden.detach().float().cpu().numpy()
        pruned_np = pruned_hidden.detach().float().cpu().numpy()

        # Center hidden states
        intact_centered = intact_np - np.mean(intact_np, axis=0, keepdims=True)
        pruned_centered = pruned_np - np.mean(pruned_np, axis=0, keepdims=True)

        print("Computing SVD on centered representations...")
        _, S_i, Vt_i = np.linalg.svd(intact_centered, full_matrices=False)
        _, S_p, Vt_p = np.linalg.svd(pruned_centered, full_matrices=False)

        V_i = Vt_i.T
        V_p = Vt_p.T

        v_i = V_i[:, 0]
        v_p = V_p[:, 0]

        # Sign synchronization
        proj_raw = np.dot(v_p, v_i)
        if proj_raw < 0:
            v_p_sync = -v_p
        else:
            v_p_sync = v_p

        proj_sync = np.dot(v_p_sync, v_i)
        theta_rad = np.arccos(np.clip(proj_sync, -1.0, 1.0))
        theta_deg = np.degrees(theta_rad)

        # Variance along PC1
        var_i = (S_i[0] ** 2) / intact_np.shape[0]
        var_p = (S_p[0] ** 2) / pruned_np.shape[0]
        energy_collapse = 1.0 - (var_p / var_i)

        print(f"Results for {model_name} at post-incision layer:")
        print(f"  Raw projection: {proj_raw:.6f}")
        print(f"  Sign-synchronized projection: {proj_sync:.6f}")
        print(f"  Acute Angle: {theta_deg:.6f} degrees")
        print(f"  Intact PC1 Variance: {var_i:.6f}")
        print(f"  Pruned PC1 Variance: {var_p:.6f}")
        print(f"  PC1 Energy Collapse: {energy_collapse * 100:.2f}%")

if __name__ == "__main__":
    main()
