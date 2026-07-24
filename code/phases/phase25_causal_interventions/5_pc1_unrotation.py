"""
P5: PC₁-Specific Subspace Un-Rotation

The paper claims PC₁ is causally responsible for reasoning (Table 3: 34% -> 11%).
Full Procrustes R* (P1 Exp 1A) operates on all 4096 dimensions, diluting the PC₁
correction across the full space. This script performs PRECISE PC₁-only realignment:

  1. Compute SVD of intact hidden states at post-incision layer
  2. Compute SVD of pruned hidden states at post-incision layer
  3. Extract PC₁ direction from each
  4. Build a Givens rotation that rotates the pruned PC₁ to align with intact PC₁
     while leaving all other dimensions unchanged
  5. Apply this rotation via forward hook on pruned model
  6. Evaluate GSM8K recovery

Expected outcome: PC₁ un-rotation should recover reasoning significantly better
than full Procrustes (~3%), potentially approaching the dense baseline (~34%).
"""

import os
import re
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import orthogonal_procrustes
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase25_causal_interventions.causal_hooks import (
    HiddenStateCollector,
    SubspaceInterventionHook,
    register_collector,
    register_intervention,
    compute_procrustes_rotation,
    random_orthogonal_matrix,
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
    return (correct / total) * 100.0


def _drop_layers(model, drop_indices):
    remaining = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    for idx, layer in enumerate(remaining):
        layer.self_attn.layer_idx = idx
    model.model.layers = remaining
    model.config.num_hidden_layers = len(remaining)


def collect_pruned_hidden(model_name, drop_set, layer_idx, tokenizer, num_prompts=50):
    device = get_device()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    if drop_set:
        _drop_layers(model, drop_set)
    model.config.use_cache = False
    collector, handle = register_collector(model, layer_idx)
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= num_prompts:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)
    all_hidden = []
    for text in tqdm(prompts, desc="Collecting hidden states", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        all_hidden.append(collector.hidden)
    handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(all_hidden, dim=0)


def compute_pc1_rotation(intact_hidden, pruned_hidden):
    intact_np = intact_hidden.detach().float().cpu().numpy()
    pruned_np = pruned_hidden.detach().float().cpu().numpy()

    _, _, Vt_i = np.linalg.svd(intact_np, full_matrices=False)
    _, _, Vt_p = np.linalg.svd(pruned_np, full_matrices=False)

    V_i = Vt_i.T
    V_p = Vt_p.T

    d = V_i.shape[0]
    v_i = V_i[:, 0]
    v_p = V_p[:, 0]

    u_a = v_p / np.linalg.norm(v_p)
    proj = np.dot(u_a, v_i)
    perp = v_i - proj * u_a
    norm_perp = np.linalg.norm(perp)

    if norm_perp < 1e-12:
        return torch.eye(d).float(), 0.0, 0.0

    u_b = perp / norm_perp
    cos_theta = np.clip(proj, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    sin_theta = np.sin(theta_rad)

    P_plane = np.outer(u_a, u_a) + np.outer(u_b, u_b)
    Q_rot = np.outer(u_b, u_a) - np.outer(u_a, u_b)
    R = np.eye(d) + (cos_theta - 1.0) * P_plane + sin_theta * Q_rot

    return torch.from_numpy(R).float(), theta_deg, proj


def main():
    parser = argparse.ArgumentParser(
        description="P5: PC1-Specific Subspace Un-Rotation"
    )
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="experiments/phase25_causal_interventions")
    parser.add_argument("--incision_start", type=int, default=11)
    parser.add_argument("--skip_depth", type=int, default=4)
    parser.add_argument("--post_incision_layer", type=int, default=15)
    parser.add_argument("--num_calibration", type=int, default=50)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--skip_pc1", action="store_true")
    parser.add_argument("--skip_full_procrustes", action="store_true")
    parser.add_argument("--skip_random_rotation", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    drop_set = list(range(args.incision_start, args.incision_start + args.skip_depth))
    pruned_post_layer = args.post_incision_layer - args.skip_depth

    results = {
        "experiment": "P5_PC1_unrotation",
        "model": args.model_name,
        "drop_indices": drop_set,
        "num_eval_samples": args.num_eval_samples,
        "num_calibration_prompts": args.num_calibration,
    }

    # --- 1. Baselines ---
    if not args.skip_baselines:
        print("\n[Baseline 1] Dense intact model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True
        )
        acc_dense = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
        print(f"  Dense baseline: {acc_dense:.2f}%")
        results["baseline_dense"] = acc_dense
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n[Baseline 2] Pruned unhealed model...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True
        )
        _drop_layers(model, drop_set)
        acc_pruned = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
        print(f"  Pruned unhealed: {acc_pruned:.2f}%")
        results["baseline_pruned"] = acc_pruned
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("[SKIP] Baselines.")

    # --- 2. Compute PC1 rotation ---
    if not args.skip_pc1:
        print("\n[PC₁ Rotation] Collecting hidden states...")
        intact_hidden = collect_pruned_hidden(
            args.model_name, None, args.post_incision_layer,
            tokenizer, args.num_calibration
        )
        pruned_hidden = collect_pruned_hidden(
            args.model_name, drop_set, pruned_post_layer,
            tokenizer, args.num_calibration
        )

        print("Computing PC₁-specific Givens rotation...")
        R_pc1, theta_deg, proj = compute_pc1_rotation(intact_hidden, pruned_hidden)
        print(f"  PC₁ alignment score (projection): {proj:.4f}")
        print(f"  PC₁ angular deviation θ: {theta_deg:.2f}°")
        results["pc1_alignment_projection"] = float(proj)
        results["pc1_theta_degrees"] = float(theta_deg)

        print("\n[PC₁ Un-Rotation] Evaluating pruned model with PC₁ realignment...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True
        )
        _drop_layers(model, drop_set)
        hook, handle = register_intervention(model, pruned_post_layer)
        hook.set_rotation(R_pc1)
        acc_pc1 = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
        print(f"  PC₁ realigned GSM8K: {acc_pc1:.2f}%")
        results["pc1_unrotation_gsm8k"] = acc_pc1
        handle.remove()
        del model, hook
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.save(R_pc1, os.path.join(args.output_dir, "R_pc1.pt"))
        save_json(os.path.join(args.output_dir, "R_pc1_theta.json"),
                  {"theta_deg": theta_deg, "projection": float(proj)})
    else:
        print("[SKIP] PC₁ rotation.")

    # --- 3. Full Procrustes R* comparison ---
    if not args.skip_full_procrustes:
        r_star_path = os.path.join(args.output_dir, "R_star.pt")
        if os.path.exists(r_star_path):
            print("\n[Full Procrustes R*] Evaluating pruned model with R*...")
            R_star = torch.load(r_star_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True
            )
            _drop_layers(model, drop_set)
            hook, handle = register_intervention(model, pruned_post_layer)
            hook.set_rotation(R_star)
            acc_rstar = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
            print(f"  Full R* GSM8K: {acc_rstar:.2f}%")
            results["full_procrustes_gsm8k"] = acc_rstar
            handle.remove()
            del model, hook
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("\n[Full Procrustes R*] R_star.pt not found at {r_star_path}. Skip.")
    else:
        print("[SKIP] Full Procrustes.")

    # --- 4. Random rotation comparison ---
    if not args.skip_random_rotation:
        print("\n[Random Rotation] Evaluating pruned with random SO(d)...")
        hidden_dim = 4096
        all_rand = []
        for seed in [0, 1, 2]:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, torch_dtype=torch.float16, device_map="auto",
                low_cpu_mem_usage=True
            )
            _drop_layers(model, drop_set)
            R_rand = random_orthogonal_matrix(hidden_dim, seed=seed)
            hook, handle = register_intervention(model, pruned_post_layer)
            hook.set_rotation(R_rand)
            acc = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
            print(f"  Random seed {seed}: {acc:.2f}%")
            all_rand.append(acc)
            handle.remove()
            del model, hook
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["random_rotation_seeds"] = all_rand
        results["random_rotation_mean"] = float(np.mean(all_rand))
    else:
        print("[SKIP] Random rotation.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("P5 PC₁ UN-ROTATION COMPLETE")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        elif isinstance(v, list) and all(isinstance(x, float) for x in v):
            print(f"  {k}: {[f'{x:.2f}' for x in v]}")
        else:
            print(f"  {k}: {v}")

    save_json(os.path.join(args.output_dir, "exp_5_pc1_unrotation.json"), results)


if __name__ == "__main__":
    main()
