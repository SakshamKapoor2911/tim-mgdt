"""
P7: Skip-Depth Rotation Curve at 8-shot N=500 (Mistral-7B)
Re-runs the paper's Table "rotation_curve" GSM8K column under the unified
8-shot protocol with N=500 for drop depths k = 0..4 (drop layers 11..11+k-1).

For each k, theta is measured at the post-incision layer (pruned model layer
index 11, aligned to intact model layer 11+k) using top-10 canonical angles.

Saves results to experiments/phase25_causal_interventions/exp_7_rotation_curve.json
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from code.phases.phase25_causal_interventions.causal_hooks import HiddenStateCollector
from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k
from code.phases.phase11.utils import get_device, save_json


def _drop_layers(model, drop_indices):
    remaining = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)


def compute_principal_angles(H_teacher, H_student, top_m=10):
    Ds = H_teacher.shape[-1]
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
    return float(np.degrees(np.mean(angles)))


def collect_hidden_at_layer(model, layer_idx, tokenizer, num_prompts=50, seq_len=512):
    device = get_device()
    model.config.use_cache = False
    collector = HiddenStateCollector()
    handle = model.model.layers[layer_idx].register_forward_hook(collector)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= num_prompts:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)
    all_hidden = []
    for text in tqdm(prompts, desc=f"Collecting L{layer_idx} hidden", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=seq_len).to(device)
        with torch.no_grad():
            model(**inputs)
        all_hidden.append(collector.hidden.clone())
    handle.remove()
    return torch.cat(all_hidden, dim=0)


def main():
    parser = argparse.ArgumentParser(description="P7: Skip-Depth Rotation Curve")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="experiments/phase25_causal_interventions")
    parser.add_argument("--incision_start", type=int, default=11)
    parser.add_argument("--max_skip_depth", type=int, default=4)
    parser.add_argument("--num_calibration", type=int, default=50)
    parser.add_argument("--num_eval_samples", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ks = list(range(0, args.max_skip_depth + 1))
    intact_layers = [args.incision_start + k for k in ks]  # 11, 12, 13, 14, 15
    pruned_collect_layer = args.incision_start - 1  # pruned model layer index 11

    # Collect intact hidden at each post-incision layer in one pass
    print("Loading intact model and collecting post-incision hidden states...")
    intact_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    intact_hidden = {}
    for layer in intact_layers:
        intact_hidden[layer] = collect_hidden_at_layer(
            intact_model, layer, tokenizer, args.num_calibration)
    del intact_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Dense baseline (k=0) GSM8K
    print("\n[k=0] Dense baseline GSM8K (8-shot, N=%d)..." % args.num_eval_samples)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    acc, vec, lo, hi, correct, total = evaluate_gsm8k(
        model, tokenizer, device, args.num_eval_samples, num_shots=8)
    print(f"  GSM8K: {acc:.2f}% [{lo:.1f}, {hi:.1f}]")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "experiment": "P7_rotation_curve",
        "model": args.model_name,
        "incision_start": args.incision_start,
        "num_eval_samples": args.num_eval_samples,
        "k0_dense": {"theta_deg": 0.0, "gsm8k": acc, "ci_95": [lo, hi],
                     "correct": correct, "total": total,
                     "per_prompt": vec},
    }

    for k in ks[1:]:
        drop_set = list(range(args.incision_start, args.incision_start + k))
        print(f"\n[k={k}] Dropping layers {drop_set}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True
        )
        _drop_layers(model, drop_set)

        print(f"  Collecting pruned hidden at layer {pruned_collect_layer}...")
        pruned_hidden = collect_hidden_at_layer(
            model, pruned_collect_layer, tokenizer, args.num_calibration)
        theta = compute_principal_angles(intact_hidden[args.incision_start + k], pruned_hidden)
        print(f"  theta = {theta:.2f} deg")

        print("  GSM8K (8-shot, N=%d)..." % args.num_eval_samples)
        acc, vec, lo, hi, correct, total = evaluate_gsm8k(
            model, tokenizer, device, args.num_eval_samples, num_shots=8)
        print(f"  GSM8K: {acc:.2f}% [{lo:.1f}, {hi:.1f}]")

        results[f"k{k}"] = {"drop_set": drop_set, "theta_deg": theta,
                            "gsm8k": acc, "ci_95": [lo, hi],
                            "correct": correct, "total": total,
                            "per_prompt": vec}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(os.path.join(args.output_dir, "exp_7_rotation_curve.json"), results)
    print("\n" + "=" * 50)
    print("P7 COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
