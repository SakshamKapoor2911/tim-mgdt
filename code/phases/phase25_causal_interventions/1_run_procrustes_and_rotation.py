"""
P1: Direct Causal Interventions
  Exp 1A: Orthogonal Procrustes realignment (R*) on pruned model -> GSM8K recovery
  Exp 1B: Artificial rotation in intact model at multiple angles -> reasoning collapse

Saves results to experiments/phase25_causal_interventions/
"""

import os
import re
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase25_causal_interventions.causal_hooks import (
    compute_procrustes_rotation,
    synthetic_rotation_matrix,
    register_collector,
    register_intervention,
)
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

    accuracy = (correct / total) * 100.0
    return accuracy


def collect_hidden_at_layer(
    model_name, drop_indices, layer_idx, tokenizer, num_prompts=50
):
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    if drop_indices:
        _drop_layers(model, drop_indices)
    collector, handle = register_collector(model, layer_idx)
    dataset = load_dataset("c4", "en", split="validation", streaming=True)
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= num_prompts:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)
    all_hidden = []
    for text in tqdm(prompts, desc="Collecting hidden states", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        all_hidden.append(collector.hidden)
    handle.remove()
    del model
    torch.cuda.empty_cache()
    return torch.cat(all_hidden, dim=1)


def _drop_layers(model, drop_indices):
    model.model.layers = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    model.config.num_hidden_layers = len(model.model.layers)


def run_procrustes_recovery(args):
    print("=" * 60)
    print("EXP 1A: Procrustes Realignment (R*) on Pruned Model")
    print("=" * 60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    intact_layer = args.post_incision_layer
    drop_set = list(range(args.incision_start, args.incision_start + args.skip_depth))
    pruned_post_layer = intact_layer - args.skip_depth

    print(f"Collecting intact hidden states at Layer {intact_layer}...")
    intact_hidden = collect_hidden_at_layer(
        args.model_name, None, intact_layer, tokenizer, args.num_calibration
    )
    print(f"  Collected intact hidden: {intact_hidden.shape}")

    print(f"Collecting pruned hidden states (drop {drop_set}) at Layer {pruned_post_layer}...")
    pruned_hidden = collect_hidden_at_layer(
        args.model_name, drop_set, pruned_post_layer, tokenizer, args.num_calibration
    )
    print(f"  Collected pruned hidden: {pruned_hidden.shape}")

    print("Computing Procrustes rotation R*...")
    R_star = compute_procrustes_rotation(intact_hidden, pruned_hidden)
    save_json(os.path.join(args.output_dir, "R_star_shape.json"), {"shape": list(R_star.shape)})
    torch.save(R_star, os.path.join(args.output_dir, "R_star.pt"))
    print(f"  R* shape: {R_star.shape}")

    print("Loading pruned model for Procrustes-intervened GSM8K eval...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    _drop_layers(model, drop_set)
    hook, handle = register_intervention(model, pruned_post_layer)
    hook.set_rotation(R_star)

    acc_procrustes = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
    print(f"  Procrustes R* GSM8K accuracy: {acc_procrustes:.2f}%")

    handle.remove()
    del model, hook
    torch.cuda.empty_cache()

    print("Loading pruned model WITHOUT intervention for baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    _drop_layers(model, drop_set)
    acc_pruned_baseline = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
    print(f"  Pruned unhealed GSM8K accuracy: {acc_pruned_baseline:.2f}%")

    del model
    torch.cuda.empty_cache()

    result = {
        "experiment": "1A_procrustes_realignment",
        "model": args.model_name,
        "drop_indices": drop_set,
        "intact_hidden_shape": list(intact_hidden.shape),
        "pruned_hidden_shape": list(pruned_hidden.shape),
        "R_star_shape": list(R_star.shape),
        "acc_procrustes_intervened": acc_procrustes,
        "acc_pruned_baseline": acc_pruned_baseline,
        "num_eval_samples": args.num_eval_samples,
        "num_calibration_prompts": args.num_calibration,
    }
    save_json(os.path.join(args.output_dir, "exp_1a_procrustes.json"), result)
    return result


def run_artificial_rotation_sweep(args):
    print("\n" + "=" * 60)
    print("EXP 1B: Artificial Rotation Sweep on Intact Model")
    print("=" * 60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    angles = args.rotation_angles

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )

    if args.rotation_layer < 0:
        args.rotation_layer = model.config.num_hidden_layers + args.rotation_layer

    hidden_dim = model.config.hidden_size
    hook, handle = register_intervention(model, args.rotation_layer)

    dense_baseline_acc = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
    print(f"  Dense baseline (no rotation): {dense_baseline_acc:.2f}%")

    results = {"dense_baseline": dense_baseline_acc, "angles": {}}
    results["num_eval_samples"] = args.num_eval_samples
    results["rotation_layer"] = args.rotation_layer

    for angle in angles:
        print(f"\n  Testing rotation θ = {angle}° ...")
        R = synthetic_rotation_matrix(angle, hidden_dim)
        hook.set_rotation(R)
        acc = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
        print(f"    GSM8K accuracy at {angle}°: {acc:.2f}%")
        results["angles"][str(angle)] = acc

    handle.remove()
    del model, hook
    torch.cuda.empty_cache()

    save_json(os.path.join(args.output_dir, "exp_1b_rotation_sweep.json"), results)
    return results


def main():
    parser = argparse.ArgumentParser(description="P1: Causal Intervention Experiments")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/phase25_causal_interventions")
    parser.add_argument("--incision_start", type=int, default=11)
    parser.add_argument("--skip_depth", type=int, default=4)
    parser.add_argument("--post_incision_layer", type=int, default=15)
    parser.add_argument("--rotation_layer", type=int, default=-1)
    parser.add_argument("--rotation_angles", type=float, nargs="+",
                        default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    parser.add_argument("--num_calibration", type=int, default=50)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--skip_exp_1a", action="store_true")
    parser.add_argument("--skip_exp_1b", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    overall = {"model": args.model_name, "drop_indices": args.incision_start}
    save_json(os.path.join(args.output_dir, "_checkpoint_1_started.json"),
              {"status": "started", "args": vars(args)})

    if args.skip_exp_1a:
        print("Skipping Exp 1A (Procrustes).")
    else:
        result_1a = run_procrustes_recovery(args)
        overall["exp_1a"] = result_1a

    if args.skip_exp_1b:
        print("Skipping Exp 1B (Artificial Rotation).")
    else:
        result_1b = run_artificial_rotation_sweep(args)
        overall["exp_1b"] = result_1b

    save_json(os.path.join(args.output_dir, "_checkpoint_1_complete.json"),
              {"status": "complete", "overall": overall})
    print("\n" + "=" * 60)
    print("P1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
