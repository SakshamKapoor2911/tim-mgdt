"""
Healed-Geometry Measurement (P9): principal subspace angle theta at the incision
interface for intact vs. pruned-unhealed vs. boundary-healed (t=2) Mistral-7B-v0.1.

Protocol matches exp_7_rotation_curve.json / energy_collapse.json:
  - N=50 C4 validation passages, tokenized to 512 tokens
  - Hidden states at the post-incision layer (intact L15 == pruned index 11)
  - Per-passage: top-10 principal angles (Eqs. 2-4), mean theta + t-CI
  - Concatenated: theta of the pooled 50-passage SVD (Table 2 protocol)
  - Paired per-passage test (Wilcoxon) for healed vs. unhealed Delta-theta

Boundary t=2 adapter: checkpoints/Mistral-7B-v0.1_boundary_healed_t2_lora
(adapter targets pruned indices 9-12 = original boundary layers 9,10,15,16).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.stats import sem, t, wilcoxon
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "code/phases/phase25_causal_interventions"))
from causal_hooks import register_collector

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DROP_SET = list(range(11, 15))
POST_INCISION = 15          # intact layer index
PRUNED_POST = 11            # same physical layer after drop (original 15)
ADAPTER = "checkpoints/Mistral-7B-v0.1_boundary_healed_t2_lora"

CONFIGS = {
    "mistral": dict(model=MODEL_NAME, drop_set=list(range(11, 15)),
                    post_incision=15, pruned_post=11,
                    adapter=ADAPTER),
    "llama": dict(model="meta-llama/Meta-Llama-3-8B", drop_set=list(range(8, 12)),
                  post_incision=12, pruned_post=8, adapter=None),
    "gemma": dict(model="google/gemma-2-2b", drop_set=list(range(12, 16)),
                  post_incision=16, pruned_post=12, adapter=None),
}
N_PROMPTS = 50
MAX_LEN = 512
K = 10


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _drop_layers(model, drop_indices):
    remaining = nn.ModuleList([layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)])
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)


def _get_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    return output


class HiddenStateCollector:
    def __init__(self):
        self.hidden = None

    def __call__(self, module, input, output):
        h = _get_hidden(output).detach().cpu()
        self.hidden = h.reshape(-1, h.shape[-1])
        return output


def collect_hidden(model, layer_idx, tokenizer, device):
    model.config.use_cache = False
    base = getattr(model, "base_model", model)
    while not hasattr(base, "layers") and hasattr(base, "model"):
        base = base.model
    layers = base.layers
    collector = HiddenStateCollector()
    handle = layers[layer_idx].register_forward_hook(collector)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts = []
    for example in dataset:
        if len(prompts) >= N_PROMPTS:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)
    hidden_list = []
    for text in tqdm(prompts, leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=MAX_LEN).to(device)
        with torch.no_grad():
            model(**inputs)
        hidden_list.append(collector.hidden.float().cpu())
    handle.remove()
    return hidden_list


def topk_theta(h_t, h_s, k=K):
    """Per-passage: mean of top-k principal angles between centered hidden states."""
    h_t_c = h_t - h_t.mean(dim=0, keepdim=True)
    h_s_c = h_s - h_s.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(h_t_c, full_matrices=False)
    _, _, vs = torch.linalg.svd(h_s_c, full_matrices=False)
    M = vs[:k] @ vt[:k].T
    sig = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(sig)).mean().item()), sig.cpu().numpy()


def concat_theta(h_t_list, h_s_list, k=K):
    h_t = torch.cat(h_t_list, dim=0)
    h_s = torch.cat(h_s_list, dim=0)
    return topk_theta(h_t, h_s, k)[0]


def t_ci(data, confidence=0.95):
    if len(data) <= 1:
        return 0.0
    return float(sem(data) * t.ppf((1 + confidence) / 2.0, len(data) - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mistral", choices=list(CONFIGS))
    parser.add_argument("--num_prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--output", type=str,
                        default="experiments/phase25_causal_interventions/healed_geometry.json")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    model_name = cfg["model"]
    drop_set = cfg["drop_set"]
    post_incision = cfg["post_incision"]
    pruned_post = cfg["pruned_post"]
    adapter = args.adapter if args.adapter else cfg["adapter"]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Collecting INTACT (L{post_incision})...")
    intact = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                  device_map="auto", low_cpu_mem_usage=True)
    intact_list = collect_hidden(intact, post_incision, tokenizer, device)
    del intact
    torch.cuda.empty_cache()

    print(f"Collecting PRUNED UNHEALED (index {pruned_post})...")
    pruned = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                  device_map="auto", low_cpu_mem_usage=True)
    _drop_layers(pruned, drop_set)
    pruned_list = collect_hidden(pruned, pruned_post, tokenizer, device)
    del pruned
    torch.cuda.empty_cache()

    healed_list = None
    if adapter is not None:
        print(f"Collecting BOUNDARY-HEALED t=2 (index {pruned_post})...")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                    device_map="auto", low_cpu_mem_usage=True)
        _drop_layers(base, drop_set)
        healed = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        healed_list = collect_hidden(healed, pruned_post, tokenizer, device)
        del healed, base
        torch.cuda.empty_cache()

    # Per-passage thetas
    dro_pp = [topk_theta(h_t, h_s)[0] for h_t, h_s in zip(intact_list, pruned_list)]
    hea_pp = [topk_theta(h_t, h_s)[0] for h_t, h_s in zip(intact_list, healed_list)] if healed_list else None

    # Concatenated thetas (Table 2 protocol)
    theta_dro_cat = concat_theta(intact_list, pruned_list)
    theta_hea_cat = concat_theta(intact_list, healed_list) if healed_list else None

    delta_mean = delta_ci = w_stat = w_p = None
    if hea_pp is not None:
        deltas = [h - d for h, d in zip(hea_pp, dro_pp)]
        try:
            w_stat, w_p = wilcoxon(deltas)
        except ValueError:
            w_stat, w_p = None, None
        delta_mean = float(np.mean(deltas))
        delta_ci = t_ci(deltas)

    result = {
        "experiment": "9_healed_geometry",
        "model": model_name,
        "drop_indices": drop_set,
        "intact_layer": post_incision,
        "pruned_post_incision_layer": pruned_post,
        "adapter": adapter,
        "num_prompts": len(intact_list),
        "max_len": MAX_LEN,
        "k": K,
        "theta_dropped_concat_deg": round(theta_dro_cat, 4),
        "theta_healed_concat_deg": round(theta_hea_cat, 4) if theta_hea_cat else None,
        "delta_theta_concat_deg": round(theta_hea_cat - theta_dro_cat, 4) if theta_hea_cat else None,
        "theta_dropped_per_passage": {
            "mean_deg": round(float(np.mean(dro_pp)), 4),
            "ci_deg": round(t_ci(dro_pp), 4),
            "values": [round(v, 4) for v in dro_pp],
        },
        "theta_healed_per_passage": None if hea_pp is None else {
            "mean_deg": round(float(np.mean(hea_pp)), 4),
            "ci_deg": round(t_ci(hea_pp), 4),
            "values": [round(v, 4) for v in hea_pp],
        },
        "delta_theta_per_passage": None if delta_mean is None else {
            "mean_deg": round(delta_mean, 4),
            "ci_deg": round(delta_ci, 4),
            "wilcoxon_p": w_p,
            "wilcoxon_stat": w_stat,
        },
    }

    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()