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
        }
    ]

    for cfg in configs:
        model_name = cfg["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        drop_set = cfg["drop_set"]
        post_incision_layer = cfg["post_incision_layer"]
        pruned_post_layer = cfg["pruned_post_layer"]

        intact_hidden = collect_pruned_hidden(model_name, None, post_incision_layer, tokenizer, 50)
        pruned_hidden = collect_pruned_hidden(model_name, drop_set, pruned_post_layer, tokenizer, 50)

        intact_np = intact_hidden.detach().float().cpu().numpy()
        pruned_np = pruned_hidden.detach().float().cpu().numpy()

        # Uncentered SVD
        _, S_i_un, Vt_i_un = np.linalg.svd(intact_np, full_matrices=False)
        _, S_p_un, Vt_p_un = np.linalg.svd(pruned_np, full_matrices=False)

        var_i_un = (S_i_un[0] ** 2) / intact_np.shape[0]
        var_p_un = (S_p_un[0] ** 2) / pruned_np.shape[0]
        collapse_un = 1.0 - (var_p_un / var_i_un)

        # Centered SVD
        intact_centered = intact_np - np.mean(intact_np, axis=0, keepdims=True)
        pruned_centered = pruned_np - np.mean(pruned_np, axis=0, keepdims=True)
        _, S_i_c, Vt_i_c = np.linalg.svd(intact_centered, full_matrices=False)
        _, S_p_c, Vt_p_c = np.linalg.svd(pruned_centered, full_matrices=False)

        var_i_c = (S_i_c[0] ** 2) / intact_np.shape[0]
        var_p_c = (S_p_c[0] ** 2) / pruned_np.shape[0]
        collapse_c = 1.0 - (var_p_c / var_i_c)

        print(f"\nMistral-7B Results:")
        print(f"  Uncentered PC1 Energy Collapse: {collapse_un * 100:.2f}%")
        print(f"  Centered PC1 Energy Collapse: {collapse_c * 100:.2f}%")

if __name__ == "__main__":
    main()
