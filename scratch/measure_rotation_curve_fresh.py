"""Fresh per-passage rotation curve k=1..4 for Mistral-7B (drop 11..11+k-1),
theta at pruned index 11 vs intact layer 11+k. Same protocol as measure_healed_geometry.py.
"""
import argparse
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from scratch.measure_healed_geometry import topk_theta, t_ci

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
N_PROMPTS = 50
MAX_LEN = 512
K_TOP = 10


def _drop_layers(model, drop_indices):
    import torch.nn as nn
    remaining = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)


def collect_hidden(model, layer_idx, tokenizer, device, out):
    model.config.use_cache = False
    holder = {}
    def hook(module, args, output):
        holder["h"] = output[0].detach().float().clone()
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts = []
    for example in dataset:
        if len(prompts) >= N_PROMPTS:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)
    for i, text in enumerate(prompts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=MAX_LEN).to(device)
        with torch.no_grad():
            model(**inputs)
        out.append(holder["h"].clone())
    handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="experiments/phase25_causal_interventions/exp_7_rotation_curve_fresh.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    intact = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                  device_map="auto", low_cpu_mem_usage=True)
    intact_hidden = {L: [] for L in range(12, 16)}
    for L in range(12, 16):
        print(f"Collecting INTACT L{L}...")
        collect_hidden(intact, L, tokenizer, device, intact_hidden[L])
    del intact
    torch.cuda.empty_cache()

    results = {"model": MODEL_NAME, "num_prompts": N_PROMPTS, "max_len": MAX_LEN, "k_top": K_TOP}
    for k in range(1, 5):
        drop_set = list(range(11, 11 + k))
        pruned = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                                      device_map="auto", low_cpu_mem_usage=True)
        _drop_layers(pruned, drop_set)
        print(f"Collecting PRUNED index 11 (drop {drop_set})...")
        pruned_hidden = []
        collect_hidden(pruned, 11, tokenizer, device, pruned_hidden)
        del pruned
        torch.cuda.empty_cache()
        intact_list = intact_hidden[11 + k]
        pp = [topk_theta(h_t, h_s)[0] for h_t, h_s in zip(intact_list, pruned_hidden)]
        results[f"k{k}"] = {
            "drop_set": drop_set,
            "theta_per_passage_deg": round(float(np.mean(pp)), 4),
            "theta_per_passage_ci_deg": round(t_ci(pp), 4),
            "per_passage_values": [round(v, 4) for v in pp],
        }
        print(f"k={k}: per-passage {results[f'k{k}']['theta_per_passage_deg']} "
              f"+- {results[f'k{k}']['theta_per_passage_ci_deg']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()