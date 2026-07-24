"""
P3: Fixed-Depth Competing Metrics Matrix
  At fixed skip-depth k=4, compute per-passage:
    1. Subspace rotation angle θ
    2. Residual vector norm ||h||_2
    3. LayerNorm mean/variance drift (μ, σ)
    4. Attention entropy H(A)
    5. Effective rank
  Then compute inter-metric Pearson/Spearman correlation matrix.

Saves results to experiments/phase25_causal_interventions/
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase25_causal_interventions.causal_hooks import (
    HiddenStateCollector,
)
from code.phases.phase11.utils import get_device, save_json


def _drop_layers(model, drop_indices):
    model.model.layers = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    model.config.num_hidden_layers = len(model.model.layers)


def compute_principal_angles(H_teacher, H_student, top_m=10):
    Bs, Ss, Ds = H_teacher.shape
    H_t = H_teacher.detach().float().reshape(-1, Ds).cpu().numpy()
    H_s = H_student.detach().float().reshape(-1, Ds).cpu().numpy()
    _, _, Vt_t = np.linalg.svd(H_t, full_matrices=False)
    _, _, Vt_s = np.linalg.svd(H_s, full_matrices=False)
    V_t = Vt_t[:top_m].T
    V_s = Vt_s[:top_m].T
    M = V_t.T @ V_s
    _, sigma, _ = np.linalg.svd(M)
    sigma = np.clip(sigma, -1.0, 1.0)
    angles = np.arccos(sigma)
    theta = np.degrees(np.mean(angles))
    return theta


def compute_residual_norm(hidden):
    return torch.norm(hidden, dim=-1).mean().item()


def compute_layernorm_drift(hidden_teacher, hidden_student):
    mu_t = hidden_teacher.mean(dim=-1).mean().item()
    mu_s = hidden_student.mean(dim=-1).mean().item()
    std_t = hidden_teacher.std(dim=-1).mean().item()
    std_s = hidden_student.std(dim=-1).mean().item()
    return {"mu_teacher": mu_t, "mu_student": mu_s, "mu_drift": abs(mu_t - mu_s),
            "std_teacher": std_t, "std_student": std_s, "std_drift": abs(std_t - std_s)}


def compute_attention_entropy(model, tokenizer, text, device, layer_idx):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    attention_outputs = {}

    def hook_fn(module, input, output):
        attn_weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if attn_weights is not None:
            attention_outputs["weights"] = attn_weights.detach().cpu()

    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs, output_attentions=True)
    handle.remove()

    if "weights" in attention_outputs:
        A = attention_outputs["weights"]
        A = A.clamp(min=1e-10)
        entropy = -(A * A.log()).sum(dim=-1).mean().item()
        return entropy
    return float("nan")


def compute_effective_rank(hidden):
    if isinstance(hidden, torch.Tensor):
        S = torch.linalg.svdvals(hidden.reshape(-1, hidden.shape[-1]))
    else:
        _, S, _ = np.linalg.svd(hidden.reshape(-1, hidden.shape[-1]), full_matrices=False)
        S = torch.from_numpy(S)
    p = S / S.sum()
    H = -(p * p.clamp(min=1e-10).log()).sum().item()
    eff_rank = np.exp(H)
    return eff_rank


def main():
    parser = argparse.ArgumentParser(description="P3: Competing Metrics Matrix")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="experiments/phase25_causal_interventions")
    parser.add_argument("--incision_start", type=int, default=11)
    parser.add_argument("--skip_depth", type=int, default=4)
    parser.add_argument("--post_incision_layer", type=int, default=15)
    parser.add_argument("--num_calibration", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    drop_set = list(range(args.incision_start, args.incision_start + args.skip_depth))
    pruned_post_layer = args.post_incision_layer - args.skip_depth
    pre_incision_layer = args.incision_start - 1

    dataset = load_dataset("c4", "en", split="validation", streaming=True)
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= args.num_calibration:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 25:
            prompts.append(text)

    print("Loading intact model...")
    intact_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )

    collector_post = HiddenStateCollector()
    handle_post = intact_model.model.layers[args.post_incision_layer].register_forward_hook(collector_post)

    collector_pre = HiddenStateCollector()
    handle_pre = intact_model.model.layers[pre_incision_layer].register_forward_hook(collector_pre)

    print("Collecting intact hidden states per passage...")
    intact_hidden_post_list = []
    intact_hidden_pre_list = []
    for text in tqdm(prompts, desc="Intact"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len).to(device)
        with torch.no_grad():
            intact_model(**inputs)
        intact_hidden_post_list.append(collector_post.hidden)
        intact_hidden_pre_list.append(collector_pre.hidden)

    handle_post.remove()
    handle_pre.remove()

    intact_hidden_post = torch.cat(intact_hidden_post_list, dim=0)
    intact_hidden_pre = torch.cat(intact_hidden_pre_list, dim=0)

    del intact_model
    torch.cuda.empty_cache()

    print("Loading pruned model...")
    pruned_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    _drop_layers(pruned_model, drop_set)

    collector_post_s = HiddenStateCollector()
    handle_post_s = pruned_model.model.layers[pruned_post_layer].register_forward_hook(collector_post_s)

    collector_pre_s = HiddenStateCollector()
    handle_pre_s = pruned_model.model.layers[pre_incision_layer].register_forward_hook(collector_pre_s)

    print("Collecting pruned hidden states per passage...")
    pruned_hidden_post_list = []
    pruned_hidden_pre_list = []
    for text in tqdm(prompts, desc="Pruned"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len).to(device)
        with torch.no_grad():
            pruned_model(**inputs)
        pruned_hidden_post_list.append(collector_post_s.hidden)
        pruned_hidden_pre_list.append(collector_pre_s.hidden)

    handle_post_s.remove()
    handle_pre_s.remove()

    pruned_hidden_post = torch.cat(pruned_hidden_post_list, dim=0)
    pruned_hidden_pre = torch.cat(pruned_hidden_pre_list, dim=0)

    print("Computing per-passage metrics...")
    B = args.num_calibration
    metrics = {
        "rotation_theta_deg": [],
        "residual_norm": [],
        "mu_drift": [],
        "std_drift": [],
        "attention_entropy": [],
        "effective_rank": [],
        "pruned_effective_rank": [],
    }

    for i in tqdm(range(B), desc="Metrics"):
        h_teacher = intact_hidden_post_list[i]
        h_student = pruned_hidden_post_list[i]

        metrics["rotation_theta_deg"].append(compute_principal_angles(h_teacher, h_student, top_m=10))

        metrics["residual_norm"].append(compute_residual_norm(h_student))

        ln = compute_layernorm_drift(h_teacher, h_student)
        metrics["mu_drift"].append(ln["mu_drift"])
        metrics["std_drift"].append(ln["std_drift"])

        text = prompts[i]
        ent = compute_attention_entropy(pruned_model, tokenizer, text, device, pruned_post_layer)
        metrics["attention_entropy"].append(ent)

        metrics["effective_rank"].append(compute_effective_rank(h_teacher))
        metrics["pruned_effective_rank"].append(compute_effective_rank(h_student))

    del pruned_model
    torch.cuda.empty_cache()

    metric_names = ["rotation_theta_deg", "residual_norm", "mu_drift", "std_drift",
                    "attention_entropy", "effective_rank", "pruned_effective_rank"]

    pearson_matrix = {}
    spearman_matrix = {}
    for i, name_i in enumerate(metric_names):
        for j, name_j in enumerate(metric_names):
            if j <= i:
                continue
            vals_i = np.array(metrics[name_i])
            vals_j = np.array(metrics[name_j])
            mask = ~(np.isnan(vals_i) | np.isnan(vals_j))
            if mask.sum() > 2:
                r_p, p_p = pearsonr(vals_i[mask], vals_j[mask])
                r_s, p_s = spearmanr(vals_i[mask], vals_j[mask])
            else:
                r_p, p_p = float("nan"), float("nan")
                r_s, p_s = float("nan"), float("nan")
            pearson_matrix[f"{name_i}_vs_{name_j}"] = {"r": r_p, "p": p_p}
            spearman_matrix[f"{name_i}_vs_{name_j}"] = {"rho": r_s, "p": p_s}

    summary = {
        "experiment": "P3_competing_metrics_fixed_depth",
        "model": args.model_name,
        "drop_indices": drop_set,
        "num_prompts": args.num_calibration,
        "per_passage_metrics": {k: v for k, v in metrics.items()},
        "pearson_correlation_matrix": pearson_matrix,
        "spearman_correlation_matrix": spearman_matrix,
    }

    save_json(os.path.join(args.output_dir, "exp_3_competing_metrics.json"), summary)
    print("\n" + "=" * 50)
    print("P3 COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
