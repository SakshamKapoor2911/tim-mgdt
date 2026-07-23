"""
Phase 17: Mechanistic BOS Attention Sink Heatmap
Plots attention mass to the BOS token across depth for 4 key conditions:
  1. Dense Baseline
  2. Dropped Unhealed (t=0)
  3. Global Sensitivity LoRA Healed
  4. Incision Boundary LoRA Healed (t=2)

Saves figure to paper/figures/bos_attention_sink_comparison.png
"""

import os
import sys
import json
import importlib

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from code.phases.phase11.utils import get_device

heal_module = importlib.import_module("code.phases.phase15_layer_drop.2_heal_dropped_layers")
drop_layers_with_mapping = heal_module.drop_layers_with_mapping

def extract_attention_to_sink(model, input_ids):
    """
    Extracts the average attention mass directed to the BOS (sink) token for each layer.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    attentions = outputs.attentions  # Tuple of (batch, num_heads, seq_len, seq_len) per layer
    if not attentions:
        raise ValueError("Model did not return output_attentions. Ensure attn_implementation='eager' is used.")

    sink_mass_per_layer = []
    for layer_attn in attentions:
        # layer_attn: [1, num_heads, seq, seq]
        avg_attn = layer_attn[0].mean(dim=0)  # [seq, seq]
        # Attention to BOS (token 0) across all query tokens (1..seq_len)
        sink_mass = avg_attn[1:, 0].mean().item()
        sink_mass_per_layer.append(sink_mass)

    return sink_mass_per_layer

def align_layers_to_full_depth(sink_mass, drop_indices, total_layers=32):
    aligned = []
    curr_idx = 0
    for i in range(total_layers):
        if i in drop_indices:
            aligned.append(float('nan'))
        else:
            aligned.append(sink_mass[curr_idx])
            curr_idx += 1
    return aligned

def main():
    model_id = "mistralai/Mistral-7B-v0.1"
    drop_indices = [11, 12, 13, 14]
    b_adapter = "checkpoints/Mistral-7B-v0.1_boundary_healed_t2_lora"
    g_adapter = "checkpoints/Mistral-7B-v0.1_global_healed_lora"
    device = get_device()

    print("Loading tokenizer and sequence...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(dataset["text"][:200])
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    # 1. Dense Baseline
    print("Measuring 1/4: Dense Baseline attention sink...")
    model_dense = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    dense_sink = extract_attention_to_sink(model_dense, inputs.input_ids)
    del model_dense
    torch.cuda.empty_cache()

    # 2. Dropped Unhealed (t=0)
    print("Measuring 2/4: Dropped Unhealed (t=0) attention sink...")
    model_drop = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    model_drop, _ = drop_layers_with_mapping(model_drop, drop_indices)
    dropped_sink = align_layers_to_full_depth(extract_attention_to_sink(model_drop, inputs.input_ids), drop_indices)
    del model_drop
    torch.cuda.empty_cache()

    # 3. Global Sensitivity LoRA Healed
    print("Measuring 3/4: Global LoRA Healed attention sink...")
    if os.path.exists(g_adapter):
        model_g = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
        )
        model_g, _ = drop_layers_with_mapping(model_g, drop_indices)
        model_g = PeftModel.from_pretrained(model_g, g_adapter)
        global_sink = align_layers_to_full_depth(extract_attention_to_sink(model_g, inputs.input_ids), drop_indices)
        del model_g
        torch.cuda.empty_cache()
    else:
        print(" Global adapter not found, skipping global curve.")
        global_sink = None

    # 4. Boundary LoRA Healed (t=2)
    print("Measuring 4/4: Boundary LoRA Healed (t=2) attention sink...")
    model_b = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    model_b, _ = drop_layers_with_mapping(model_b, drop_indices)
    model_b = PeftModel.from_pretrained(model_b, b_adapter)
    boundary_sink = align_layers_to_full_depth(extract_attention_to_sink(model_b, inputs.input_ids), drop_indices)
    del model_b
    torch.cuda.empty_cache()

    # Plotting
    print("Generating 4-condition BOS Attention Sink Figure...")
    plt.figure(figsize=(11, 6))
    layers = list(range(32))

    plt.plot(layers, dense_sink, marker='o', label='Dense Baseline', color='#2b5c8f', linewidth=2.5)
    plt.plot(layers, dropped_sink, marker='x', label='Dropped Unhealed (t=0)', color='#d9534f', linewidth=2, linestyle='--')
    if global_sink:
        plt.plot(layers, global_sink, marker='s', label='Global Sensitivity LoRA', color='#f0ad4e', linewidth=2, linestyle='-.')
    plt.plot(layers, boundary_sink, marker='^', label='Incision Boundary LoRA (t=2)', color='#5cb85c', linewidth=2.5)

    # Highlight incision zone
    d_min, d_max = min(drop_indices), max(drop_indices)
    plt.axvspan(d_min - 0.5, d_max + 0.5, color='#e2e2e2', alpha=0.6, label='Incision Site (Dropped Layers 11-14)')

    plt.title('Attention Mass Allocated to BOS Token Across Layer Depth (Mistral-7B-v0.1)', fontsize=13, fontweight='bold')
    plt.xlabel('Transformer Layer Depth', fontsize=11)
    plt.ylabel('Mean Attention to BOS Token', fontsize=11)
    plt.xticks(layers[::2])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='upper left')

    os.makedirs('paper/figures', exist_ok=True)
    plt.tight_layout()
    out_img = 'paper/figures/bos_attention_sink_comparison.png'
    plt.savefig(out_img, dpi=300)
    print(f"Saved figure to {out_img}")

if __name__ == "__main__":
    main()
