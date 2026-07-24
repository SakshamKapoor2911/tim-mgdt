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
from code.phases.phase11.utils import get_device, save_json

GSM8K_PROMPT_TEMPLATE = """Q: There are 15 trees in the park. Park workers will plant trees in the park today. After they are done, there will be 21 trees in the park. How many trees did the park workers plant today?
A: There are 15 trees originally. Then more trees were planted. It total there are 21 trees. So the number of trees planted must be 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he has 12 now, he must have given Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn has 5 toys. He got 2 from mom and 2 from dad. 5 + 2 + 2 = 9. The answer is 9.

Q: {question}
A:"""


def extract_strict_answer(text):
    text = text.split("Q:")[0].strip()
    match = re.search(r"[Tt]he answer is\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return ""


def evaluate_gsm8k(model, tokenizer, device, num_samples=100):
    dataset = load_dataset("gsm8k", "main", split="test")
    model.eval()
    correct = 0
    total = min(num_samples, len(dataset))

    for i in tqdm(range(total), desc="GSM8K", leave=False):
        item = dataset[i]
        question = item["question"]
        gold_text = item["answer"]
        gold_ans = extract_strict_answer(gold_text)
        prompt = GSM8K_PROMPT_TEMPLATE.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred_ans = extract_strict_answer(gen_text)

        if pred_ans.strip() == gold_ans.strip() and gold_ans.strip() != "":
            correct += 1

    return (correct / total) * 100.0


def _drop_layers(model, drop_indices):
    model.model.layers = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    model.config.num_hidden_layers = len(model.model.layers)


def compute_bootstrap_ci(values, n_iterations=2000, seed=42):
    rng = np.random.RandomState(seed)
    means = []
    n = len(values)
    for _ in range(n_iterations):
        sample = rng.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    return lower, upper


def run_location_sweep(args):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    locations = {
        "early": {"name": "Early (3-6)", "drop": [3, 4, 5, 6]},
        "middle": {"name": "Middle (11-14)", "drop": [11, 12, 13, 14]},
        "late": {"name": "Late (20-23)", "drop": [20, 21, 22, 23]},
    }

    all_results = {}
    for loc_key, loc_info in locations.items():
        print(f"\n{'=' * 50}")
        print(f"Location: {loc_info['name']}")
        print(f"{'=' * 50}")
        loc_results = {"location": loc_info["name"], "drop_indices": loc_info["drop"], "seeds": {}}

        for seed_idx in range(args.num_seeds):
            seed = args.seed_base + seed_idx
            print(f"\n  Seed {seed} ...")

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True
            )
            _drop_layers(model, loc_info["drop"])

            acc = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
            print(f"    GSM8K accuracy: {acc:.2f}%")
            loc_results["seeds"][str(seed)] = acc

            del model
            torch.cuda.empty_cache()

        acc_values = list(loc_results["seeds"].values())
        mean_acc = np.mean(acc_values)
        std_acc = np.std(acc_values, ddof=1) if len(acc_values) > 1 else 0.0
        ci_low, ci_high = compute_bootstrap_ci(acc_values, seed=args.seed_base)
        loc_results["mean"] = mean_acc
        loc_results["std"] = std_acc
        loc_results["bootstrap_ci_95"] = [float(ci_low), float(ci_high)]
        all_results[loc_key] = loc_results

    out = {
        "experiment": "P4_multi_location_seed_sweep",
        "model": args.model_name,
        "num_seeds": args.num_seeds,
        "seed_base": args.seed_base,
        "num_eval_samples": args.num_eval_samples,
        "locations": all_results,
    }

    save_json(os.path.join(args.output_dir, "exp_4_location_sweep.json"), out)
    print("\n" + "=" * 60)
    print("P4 LOCATION SWEEP COMPLETE. SUMMARY:")
    print("=" * 60)
    for loc_key, loc_res in all_results.items():
        print(f"  {loc_res['location']}: {loc_res['mean']:.2f}% ± {loc_res['std']:.2f}% "
              f"[95% CI: {loc_res['bootstrap_ci_95'][0]:.2f}, {loc_res['bootstrap_ci_95'][1]:.2f}]")
    return out


def main():
    parser = argparse.ArgumentParser(description="P4: Multi-Location & Seed Robustness Sweep")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="experiments/phase25_causal_interventions")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(os.path.join(args.output_dir, "_checkpoint_4_started.json"),
              {"status": "started", "args": vars(args)})

    run_location_sweep(args)

    save_json(os.path.join(args.output_dir, "_checkpoint_4_complete.json"),
              {"status": "complete"})
    print("\nP4 COMPLETE")


if __name__ == "__main__":
    main()
