"""
P4: Multi-Location & Seed Robustness Sweep
  Evaluates GSM8K accuracy after 4-layer block drop at:
    - Early:   layers 3-6
    - Middle:  layers 11-14
    - Late:    layers 20-23
  Each with 3 random seeds (controlling calibration data sampling).

Saves results to experiments/phase25_causal_interventions/
"""

import os
import re
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import beta
from code.phases.phase11.utils import get_device, save_json
from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k



def _drop_layers(model, drop_indices):
    remaining = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)


def compute_clopper_pearson_ci(k, n, alpha=0.05):
    """Compute Clopper-Pearson 95% confidence interval for accuracy proportion."""
    if n == 0:
        return 0.0, 0.0
    lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lower * 100.0), float(upper * 100.0)


def run_location_sweep(args):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_locations = {
        "early": {"name": "Early (3-6)", "drop": [3, 4, 5, 6]},
        "middle": {"name": "Middle (11-14)", "drop": [11, 12, 13, 14]},
        "late": {"name": "Late (20-23)", "drop": [20, 21, 22, 23]},
    }
    location_keys = args.locations if args.locations else [k for k in all_locations]
    locations = {k: all_locations[k] for k in location_keys if k in all_locations}
    if not locations:
        locations = all_locations

    all_results = {}
    for loc_key, loc_info in locations.items():
        print(f"\n{'=' * 50}")
        print(f"Location: {loc_info['name']}")
        print(f"{'=' * 50}")
        loc_results = {"location": loc_info["name"], "drop_indices": loc_info["drop"], "seeds": {}}

        total_correct = 0
        total_evaluated = 0
        for seed_idx in range(args.num_seeds):
            seed = args.seed_base + seed_idx
            print(f"\n  Seed {seed} ...")

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True
            )
            _drop_layers(model, loc_info["drop"])

            _, _, _, _, correct, total = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples, num_shots=8, seed=seed)
            acc = (correct / total) * 100.0
            print(f"    GSM8K accuracy: {acc:.2f}% ({correct}/{total})")
            loc_results["seeds"][str(seed)] = acc
            total_correct += correct
            total_evaluated += total

            del model
            torch.cuda.empty_cache()

        mean_acc = (total_correct / total_evaluated) * 100.0
        ci_low, ci_high = compute_clopper_pearson_ci(total_correct, total_evaluated)
        loc_results["mean"] = mean_acc
        loc_results["total_correct"] = total_correct
        loc_results["total_evaluated"] = total_evaluated
        loc_results["clopper_pearson_ci_95"] = [ci_low, ci_high]
        all_results[loc_key] = loc_results

    out = {
        "experiment": "P4_multi_location_seed_sweep",
        "model": args.model_name,
        "num_seeds": args.num_seeds,
        "seed_base": args.seed_base,
        "num_eval_samples": args.num_eval_samples,
        "locations": all_results,
    }

    out_filename = f"exp_4_location_sweep{args.output_suffix}.json"
    save_json(os.path.join(args.output_dir, out_filename), out)
    print("\n" + "=" * 60)
    print("P4 LOCATION SWEEP COMPLETE. SUMMARY:")
    print("=" * 60)
    for loc_key, loc_res in all_results.items():
        print(f"  {loc_res['location']}: {loc_res['mean']:.2f}% "
              f"[95% CI: {loc_res['clopper_pearson_ci_95'][0]:.2f}, {loc_res['clopper_pearson_ci_95'][1]:.2f}]")
    return out


def main():
    parser = argparse.ArgumentParser(description="P4: Multi-Location & Seed Robustness Sweep")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="experiments/phase25_causal_interventions")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--locations", type=str, nargs="+", default=None,
                        help="Location subset: early middle late (default: all)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for result file to avoid conflicts in parallel runs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(os.path.join(args.output_dir, f"_checkpoint_4_started{args.output_suffix}.json"),
              {"status": "started", "args": vars(args)})

    run_location_sweep(args)

    if args.output_suffix:
        save_json(os.path.join(args.output_dir, f"_checkpoint_4_complete{args.output_suffix}.json"),
                  {"status": "complete"})
    else:
        save_json(os.path.join(args.output_dir, "_checkpoint_4_complete.json"),
                  {"status": "complete"})
    print("\nP4 COMPLETE")


if __name__ == "__main__":
    main()
