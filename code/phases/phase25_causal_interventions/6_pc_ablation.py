"""
P6: Causal PC Ablation at Layer 10 (Intact Model)
Re-runs the paper's Table "causal_pc" under the unified 8-shot GSM8K protocol
at N=500 so the baseline matches the N=500 dense baseline (39.6%).

For each ablated component set, the projection of every hidden state at Layer 10
onto the specified principal directions is removed (v_{10,i} -> 0).

Saves results to experiments/phase25_causal_interventions/exp_6_pc_ablation.json
"""

import os
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from code.phases.phase25_causal_interventions.causal_hooks import HiddenStateCollector
from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k
from code.phases.phase11.utils import get_device, save_json


class PCAblationHook:
    """Removes the projection of hidden states onto a set of principal directions."""

    def __init__(self, directions=None):
        self.directions = directions  # torch.Tensor [d, m] or None

    def set_directions(self, directions):
        self.directions = directions

    def __call__(self, module, input, output):
        if self.directions is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        v = self.directions.to(h.device, dtype=h.dtype)
        proj = h @ v  # [b, s, m]
        h_ablated = h - (proj @ v.transpose(0, 1))
        if isinstance(output, tuple):
            return (h_ablated,) + output[1:]
        return h_ablated


def compute_pc_directions(hidden, top_m=10):
    H = hidden.detach().float().reshape(-1, hidden.shape[-1]).cpu().numpy()
    H = H - np.mean(H, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return torch.from_numpy(Vt[:top_m].T).float()  # [d, m]


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
    parser = argparse.ArgumentParser(description="P6: Causal PC Ablation at Layer 10")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="experiments/phase25_causal_interventions")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--num_calibration", type=int, default=50)
    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )

    print("Computing PC directions at Layer", args.layer, "...")
    hidden = collect_hidden_at_layer(model, args.layer, tokenizer, args.num_calibration)
    directions = compute_pc_directions(hidden, top_m=10)
    print("  PC directions shape:", directions.shape)

    hook = PCAblationHook()
    handle = model.model.layers[args.layer].register_forward_hook(hook)

    ablation_sets = {
        "intact_baseline": [],
        "ablate_pc1": [0],
        "ablate_pc2": [1],
        "ablate_pc8": [7],
        "ablate_pc3_pc5": [2, 4],
        "ablate_pc4_6_7_9_10": [3, 5, 6, 8, 9],
    }

    results = {"experiment": "P6_pc_ablation", "model": args.model_name,
               "layer": args.layer, "num_eval_samples": args.num_eval_samples}
    per_condition_vectors = {}

    for name, idxs in ablation_sets.items():
        if name == "intact_baseline" and args.skip_baseline:
            continue
        print(f"\n[{name}]")
        if idxs:
            hook.set_directions(directions[:, idxs])
        else:
            hook.set_directions(None)
        acc, vec, lo, hi, correct, total = evaluate_gsm8k(
            model, tokenizer, device, args.num_eval_samples, num_shots=8)
        print(f"  GSM8K: {acc:.2f}% [{lo:.1f}, {hi:.1f}] ({correct}/{total})")
        results[name] = {"accuracy": acc, "ci_95": [lo, hi], "correct": correct, "total": total}
        per_condition_vectors[name] = vec

    handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results["per_prompt_vectors"] = per_condition_vectors
    save_json(os.path.join(args.output_dir, "exp_6_pc_ablation.json"), results)
    print("\n" + "=" * 50)
    print("P6 COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
