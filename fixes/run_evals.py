"""
Fix #3: Cross-validate pruned baseline with 8-shot GSM8K
Fix #1: Re-run headline conditions at N=500 for tighter CIs
Fix #7: Run Llama-3-8B boundary-healed GSM8K

Usage:
    python fixes/run_evals.py --task cross_validate     # Fix #3: N=100, 5-shot vs 8-shot
    python fixes/run_evals.py --task n500_mistral       # Fix #1: N=500, 8-shot, Mistral headline conditions
    python fixes/run_evals.py --task llama3_boundary    # Fix #7: N=100, Llama-3-8B boundary healed
    python fixes/run_evals.py --task all                # Run everything
"""

import os
import sys
import re
import json
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import binom

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 8-shot GSM8K prompt (standard CoT exemplars)
GSM8K_PROMPT_8SHOT = """Q: There are 15 trees in the park. Park workers will plant trees in the park today. After they are done, there will be 21 trees in the park. How many trees did the park workers plant today?
A: There are 15 trees originally. Then more trees were planted. In total there are 21 trees. So the number of trees planted must be 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he has 12 now, he must have given Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn has 5 toys. He got 2 from mom and 2 from dad. 5 + 2 + 2 = 9. The answer is 9.

Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
A: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 on Wednesday, he had 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. She bought 5 bagels at 3 dollars each. 5 * 3 = 15 dollars spent. 23 - 15 = 8 dollars left. The answer is 8.

Q: {question}
A:"""

GSM8K_PROMPT_5SHOT = """Q: There are 15 trees in the park. Park workers will plant trees in the park today. After they are done, there will be 21 trees in the park. How many trees did the park workers plant today?
A: There are 15 trees originally. Then more trees were planted. In total there are 21 trees. So the number of trees planted must be 21 - 15 = 6. The answer is 6.

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


def clopper_pearson_ci(correct, total, alpha=0.05):
    """Two-sided Clopper-Pearson 95% CI."""
    low = binom.ppf(alpha / 2, total, correct / total) / total if correct > 0 else 0.0
    high = binom.ppf(1 - alpha / 2, total, correct / total) / total if correct < total else 1.0
    return low * 100, high * 100


def evaluate_gsm8k(model, tokenizer, device, num_samples=100, num_shots=8):
    prompt_template = GSM8K_PROMPT_8SHOT if num_shots == 8 else GSM8K_PROMPT_5SHOT
    dataset = load_dataset("gsm8k", "main", split="test")
    model.eval()
    correct = 0
    total = min(num_samples, len(dataset))

    for i in tqdm(range(total), desc=f"GSM8K {num_shots}-shot", leave=False):
        item = dataset[i]
        question = item["question"]
        gold_text = item["answer"]
        gold_ans = extract_strict_answer(gold_text)

        prompt = prompt_template.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred_ans = extract_strict_answer(gen_text)

        if pred_ans.strip() == gold_ans.strip() and gold_ans.strip() != "":
            correct += 1

    acc = (correct / total) * 100.0
    ci_low, ci_high = clopper_pearson_ci(correct, total)
    return acc, ci_low, ci_high, correct, total


def _drop_layers(model, drop_indices):
    remaining = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name, drop_indices=None, lora_dir=None):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    if drop_indices is not None:
        _drop_layers(model, drop_indices)

    if lora_dir is not None:
        model = PeftModel.from_pretrained(model, lora_dir)
        # Don't merge_and_unload — keep PeftModel for eval to avoid hanging

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def format_result(name, acc, ci_low, ci_high, correct, total, num_shots):
    return {
        "name": name,
        "num_shots": num_shots,
        "accuracy": round(acc, 2),
        "ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "correct": correct,
        "total": total,
    }


def print_result(r):
    print(f"  {r['name']} ({r['num_shots']}-shot): "
          f"{r['accuracy']:.2f}% [{r['ci_95'][0]:.1f}%, {r['ci_95'][1]:.1f}%] "
          f"({r['correct']}/{r['total']})")


# ──────────────────────── Tasks ────────────────────────

def task_cross_validate(num_samples=100):
    """Fix #3: Cross-validate pruned baseline with 8-shot vs 5-shot."""
    model_name = "mistralai/Mistral-7B-v0.1"
    drop_set = [11, 12, 13, 14]
    results = []

    print("\n" + "=" * 60, flush=True)
    print("FIX #3: CROSS-VALIDATE PRUNED BASELINE (shot-count hypothesis)", flush=True)
    print(f"N={num_samples}, comparing 5-shot vs 8-shot", flush=True)
    print("=" * 60, flush=True)

    # Pruned unhealed 5-shot (matches Phase 25)
    print("\n[Pruned unhealed, 5-shot]", flush=True)
    model, tok, dev = load_model(model_name, drop_indices=drop_set)
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=5)
    r = format_result("pruned_unhealed", acc, lo, hi, c, t, 5)
    results.append(r); print_result(r)
    print(flush=True)
    del model; torch.cuda.empty_cache()

    # Pruned unhealed 8-shot (matches paper)
    print("\n[Pruned unhealed, 8-shot]", flush=True)
    model, tok, dev = load_model(model_name, drop_indices=drop_set)
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("pruned_unhealed", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    print(flush=True)
    del model; torch.cuda.empty_cache()

    # Dense 8-shot (sanity check against paper's 34.0%)
    print("\n[Dense, 8-shot]", flush=True)
    model, tok, dev = load_model(model_name)
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("dense_baseline", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    print(flush=True)
    del model; torch.cuda.empty_cache()

    out_file = "fixes/cross_validate_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_file}", flush=True)
    return results


def task_n500_mistral(num_samples=500):
    """Fix #1: Re-run headline conditions at N=500, 8-shot."""
    model_name = "mistralai/Mistral-7B-v0.1"
    drop_set = [11, 12, 13, 14]
    results = []

    print("\n" + "=" * 60)
    print("FIX #1: N=500 HEADLINE CONDITIONS (Mistral-7B, 8-shot)")
    print("=" * 60)

    # Dense baseline
    print("\n[Dense baseline, N=500, 8-shot]")
    t0 = time.time()
    model, tok, dev = load_model(model_name)
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("dense_baseline", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    print(f"  Wall time: {time.time() - t0:.0f}s")
    del model; torch.cuda.empty_cache()

    # Pruned unhealed
    print("\n[Pruned unhealed, N=500, 8-shot]")
    t0 = time.time()
    model, tok, dev = load_model(model_name, drop_indices=drop_set)
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("pruned_unhealed", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    print(f"  Wall time: {time.time() - t0:.0f}s")
    del model; torch.cuda.empty_cache()

    # Boundary healed t=2
    print("\n[Boundary healed t=2, N=500, 8-shot]")
    t0 = time.time()
    model, tok, dev = load_model(
        model_name, drop_indices=drop_set,
        lora_dir="checkpoints/Mistral-7B-v0.1_boundary_healed_t2_lora"
    )
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("boundary_healed_t2", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    print(f"  Wall time: {time.time() - t0:.0f}s")
    del model; torch.cuda.empty_cache()

    out_file = "fixes/n500_mistral_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_file}")
    return results


def task_llama3_boundary(num_samples=100):
    """Fix #7: Llama-3-8B boundary healed GSM8K."""
    model_name = "meta-llama/Meta-Llama-3-8B"
    drop_set = [8, 9, 10, 11]
    results = []

    print("\n" + "=" * 60)
    print("FIX #7: LLAMA-3-8B BOUNDARY HEALED GSM8K")
    print("=" * 60)

    # Dense 8-shot
    print("\n[Llama-3-8B Dense, 8-shot]")
    model, tok, dev = load_model(model_name)
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("llama3_dense_baseline", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    del model; torch.cuda.empty_cache()

    # Boundary healed t=2 8-shot
    print("\n[Llama-3-8B Boundary healed t=2, 8-shot]")
    model, tok, dev = load_model(
        model_name, drop_indices=drop_set,
        lora_dir="checkpoints/Meta-Llama-3-8B_boundary_healed_t2_lora"
    )
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("llama3_boundary_healed_t2", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    del model; torch.cuda.empty_cache()

    # Pruned unhealed 8-shot
    print("\n[Llama-3-8B Pruned unhealed, 8-shot]")
    model, tok, dev = load_model(model_name, drop_indices=drop_set)
    acc, lo, hi, c, t = evaluate_gsm8k(model, tok, dev, num_samples, num_shots=8)
    r = format_result("llama3_pruned_unhealed", acc, lo, hi, c, t, 8)
    results.append(r); print_result(r)
    del model; torch.cuda.empty_cache()

    out_file = "fixes/llama3_boundary_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_file}")
    return results


def main():
    # Unbuffered output for live log viewing
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cross_validate", "n500_mistral", "llama3_boundary", "all"],
                        default="cross_validate")
    parser.add_argument("--num_samples", type=int, default=None, help="Override N")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[GPU {args.gpu}] Running task: {args.task}")

    all_results = {}

    if args.task in ("cross_validate", "all"):
        n = args.num_samples if args.num_samples else 100
        all_results["cross_validate"] = task_cross_validate(num_samples=n)

    if args.task in ("n500_mistral", "all"):
        n = args.num_samples if args.num_samples else 500
        all_results["n500_mistral"] = task_n500_mistral(num_samples=n)

    if args.task in ("llama3_boundary", "all"):
        n = args.num_samples if args.num_samples else 100
        all_results["llama3_boundary"] = task_llama3_boundary(num_samples=n)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("ALL TASKS COMPLETE. SUMMARY:", flush=True)
    print("=" * 60, flush=True)
    for task_name, results in all_results.items():
        print(f"\n[{task_name}]", flush=True)
        for r in results:
            print_result(r)
            print(flush=True)


if __name__ == "__main__":
    main()
