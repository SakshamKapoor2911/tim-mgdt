"""
P2: Mechanism Negative Controls
  Control 2A: Random orthogonal rotation (placebo) on pruned model
  Control 2B: Wrong-layer repair (apply teacher-aligned R* at unperturbed layer)
  Control 2C: Wrong-PC repair (realign PC2 instead of PC1)

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
from code.phases.phase25_causal_interventions.causal_hooks import (
    compute_procrustes_rotation,
    random_orthogonal_matrix,
    register_collector,
    register_intervention,
    HiddenStateCollector,
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
    model.model.layers = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i not in set(drop_indices)]
    )
    model.config.num_hidden_layers = len(model.model.layers)


def load_model_with_drop(model_name, drop_indices):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    if drop_indices:
        _drop_layers(model, drop_indices)
    return model


def collect_hidden_states(model, tokenizer, layer_idx, num_prompts=50):
    device = get_device()
    collector = HiddenStateCollector()
    handle = model.model.layers[layer_idx].register_forward_hook(collector)
    dataset = load_dataset("c4", "en", split="validation", streaming=True)
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= num_prompts:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)
    all_hidden = []
    for text in tqdm(prompts, desc="Collecting hidden", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        all_hidden.append(collector.hidden)
    handle.remove()
    return torch.cat(all_hidden, dim=1)


def run_random_rotation_control(args):
    print("\n[Control 2A] Random Orthogonal Rotation (Placebo)")
    print("-" * 40)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    drop_set = list(range(args.incision_start, args.incision_start + args.skip_depth))
    pruned_post_layer = args.post_incision_layer - args.skip_depth
    hidden_dim = None

    print("  Generating random rotations at 3 seeds...")
    seeds = [0, 1, 2]
    results = {}
    for seed in seeds:
        print(f"    Seed {seed}...")
        model = load_model_with_drop(args.model_name, drop_set)
        if hidden_dim is None:
            hidden_dim = model.config.hidden_size
        R_rand = random_orthogonal_matrix(hidden_dim, seed=seed)
        hook, handle = register_intervention(model, pruned_post_layer)
        hook.set_rotation(R_rand)
        acc = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
        print(f"      Random rotation (seed={seed}) GSM8K: {acc:.2f}%")
        results[str(seed)] = acc
        handle.remove()
        del model, hook
        torch.cuda.empty_cache()

    out = {
        "experiment": "2A_random_rotation_placebo",
        "model": args.model_name,
        "drop_indices": drop_set,
        "rotation_layer": pruned_post_layer,
        "results_per_seed": results,
    }
    save_json(os.path.join(args.output_dir, "exp_2a_random_rotation.json"), out)
    return out


def run_wrong_layer_repair(args):
    print("\n[Control 2B] Wrong-Layer Repair")
    print("-" * 40)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    drop_set = list(range(args.incision_start, args.incision_start + args.skip_depth))
    pruned_post_layer = args.post_incision_layer - args.skip_depth
    n_layers = 32

    dataset = load_dataset("c4", "en", split="validation", streaming=True)
    prompts = []
    for i, example in enumerate(dataset):
        if len(prompts) >= args.num_calibration:
            break
        text = example["text"].strip()
        if text and len(text.split()) > 50:
            prompts.append(text)

    # 1. Collect Intact hidden states
    print("  Collecting intact model hidden states...")
    intact_collector = HiddenStateCollector()
    intact_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    handle_intact = intact_model.model.layers[args.post_incision_layer].register_forward_hook(intact_collector)
    intact_hidden_list = []
    for text in tqdm(prompts, desc="Intact (Wrong-Layer Collect)", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            intact_model(**inputs)
        intact_hidden_list.append(intact_collector.hidden)
    handle_intact.remove()
    del intact_model, intact_collector
    torch.cuda.empty_cache()

    # 2. Collect Pruned hidden states
    print("  Collecting pruned model hidden states...")
    pruned_collector = HiddenStateCollector()
    pruned_model = load_model_with_drop(args.model_name, drop_set)
    handle_pruned = pruned_model.model.layers[pruned_post_layer].register_forward_hook(pruned_collector)
    pruned_hidden_list = []
    for text in tqdm(prompts, desc="Pruned (Wrong-Layer Collect)", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            pruned_model(**inputs)
        pruned_hidden_list.append(pruned_collector.hidden)
    handle_pruned.remove()
    del pruned_model, pruned_collector
    torch.cuda.empty_cache()

    intact_hidden = torch.cat(intact_hidden_list, dim=1)
    pruned_hidden = torch.cat(pruned_hidden_list, dim=1)

    R_star = compute_procrustes_rotation(intact_hidden, pruned_hidden)
    torch.save(R_star, os.path.join(args.output_dir, "R_star_for_wrong_layer.pt"))

    wrong_layers = args.wrong_layers
    results = {}
    for wrong_lyr in wrong_layers:
        print(f"  Applying R* at wrong layer {wrong_lyr}...")
        model = load_model_with_drop(args.model_name, drop_set)
        hook, handle = register_intervention(model, wrong_lyr)
        hook.set_rotation(R_star)
        acc = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
        print(f"    Wrong-layer ({wrong_lyr}) GSM8K: {acc:.2f}%")
        results[str(wrong_lyr)] = acc
        handle.remove()
        del model, hook
        torch.cuda.empty_cache()

    model = load_model_with_drop(args.model_name, drop_set)
    hook, handle = register_intervention(model, pruned_post_layer)
    hook.set_rotation(R_star)
    acc_correct = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
    print(f"    Correct-layer ({pruned_post_layer}) GSM8K: {acc_correct:.2f}%")
    results[f"correct_layer_{pruned_post_layer}"] = acc_correct
    handle.remove()
    del model, hook
    torch.cuda.empty_cache()

    out = {
        "experiment": "2B_wrong_layer_repair",
        "model": args.model_name,
        "drop_indices": drop_set,
        "correct_layer": pruned_post_layer,
        "wrong_layers_tested": wrong_layers,
        "results": results,
    }
    save_json(os.path.join(args.output_dir, "exp_2b_wrong_layer_repair.json"), out)
    return out


def run_wrong_pc_repair(args):
    print("\n[Control 2C] Wrong-PC Repair (realign PC2 instead of PC1)")
    print("-" * 40)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    drop_set = list(range(args.incision_start, args.incision_start + args.skip_depth))
    pruned_post_layer = args.post_incision_layer - args.skip_depth

    print("  Collecting intact model hidden states...")
    intact_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True
    )
    intact_hidden = collect_hidden_states(
        intact_model, tokenizer, args.post_incision_layer, args.num_calibration
    )
    del intact_model
    torch.cuda.empty_cache()

    print("  Collecting pruned model hidden states...")
    pruned_model = load_model_with_drop(args.model_name, drop_set)
    pruned_hidden = collect_hidden_states(
        pruned_model, tokenizer, pruned_post_layer, args.num_calibration
    )
    del pruned_model
    torch.cuda.empty_cache()

    intact_np = intact_hidden.detach().float().reshape(-1, intact_hidden.shape[-1]).cpu().numpy()
    pruned_np = pruned_hidden.detach().float().reshape(-1, pruned_hidden.shape[-1]).cpu().numpy()

    U_i, S_i, Vt_i = np.linalg.svd(intact_np, full_matrices=False)
    U_p, S_p, Vt_p = np.linalg.svd(pruned_np, full_matrices=False)

    V_i = Vt_i.T
    V_p = Vt_p.T

    def realign_component(k):
        d = V_i.shape[0]
        v_i = V_i[:, k]
        v_p = V_p[:, k]
        u_a = v_p / np.linalg.norm(v_p)
        proj = np.dot(u_a, v_i)
        perp = v_i - proj * u_a
        norm_perp = np.linalg.norm(perp)
        if norm_perp < 1e-12:
            return torch.eye(d).float()
        u_b = perp / norm_perp
        cos_theta = np.clip(proj, -1.0, 1.0)
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))

        P_plane = np.outer(u_a, u_a) + np.outer(u_b, u_b)
        Q_rot = np.outer(u_b, u_a) - np.outer(u_a, u_b)

        R = np.eye(d) + (cos_theta - 1.0) * P_plane + sin_theta * Q_rot
        return torch.from_numpy(R).float()

    pc_indices = args.wrong_pc_indices
    results = {}
    for k in pc_indices:
        print(f"  Realigning PC{k}...")
        R_k = realign_component(k)
        model = load_model_with_drop(args.model_name, drop_set)
        hook, handle = register_intervention(model, pruned_post_layer)
        hook.set_rotation(R_k)
        acc = evaluate_gsm8k(model, tokenizer, device, args.num_eval_samples)
        print(f"    PC{k} realign GSM8K: {acc:.2f}%")
        results[f"PC_{k}"] = acc
        handle.remove()
        del model, hook
        torch.cuda.empty_cache()

    out = {
        "experiment": "2C_wrong_pc_repair",
        "model": args.model_name,
        "drop_indices": drop_set,
        "pc_indices_tested": pc_indices,
        "results": results,
    }
    save_json(os.path.join(args.output_dir, "exp_2c_wrong_pc_repair.json"), out)
    return out


def main():
    parser = argparse.ArgumentParser(description="P2: Negative Controls")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="experiments/phase25_causal_interventions")
    parser.add_argument("--incision_start", type=int, default=11)
    parser.add_argument("--skip_depth", type=int, default=4)
    parser.add_argument("--post_incision_layer", type=int, default=15)
    parser.add_argument("--num_calibration", type=int, default=50)
    parser.add_argument("--num_eval_samples", type=int, default=100)
    parser.add_argument("--wrong_layers", type=int, nargs="+", default=[0, 5, 25])
    parser.add_argument("--wrong_pc_indices", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--skip_2a", action="store_true")
    parser.add_argument("--skip_2b", action="store_true")
    parser.add_argument("--skip_2c", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(os.path.join(args.output_dir, "_checkpoint_2_started.json"),
              {"status": "started", "args": vars(args)})

    if not args.skip_2a:
        run_random_rotation_control(args)
    if not args.skip_2b:
        run_wrong_layer_repair(args)
    if not args.skip_2c:
        run_wrong_pc_repair(args)

    save_json(os.path.join(args.output_dir, "_checkpoint_2_complete.json"),
              {"status": "complete"})
    print("\nP2 NEGATIVE CONTROLS COMPLETE")


if __name__ == "__main__":
    main()
