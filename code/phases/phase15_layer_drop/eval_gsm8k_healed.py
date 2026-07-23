"""
GSM8K Downstream Reasoning Evaluation Post-Healing
Evaluates GSM8K 5-shot strict exact match accuracy across 4 key conditions:
  1. Dense Baseline
  2. Dropped Unhealed (t=0)
  3. Global Sensitivity LoRA Healed
  4. Incision Boundary LoRA Healed (t=2)

Targets: Mistral-7B-v0.1 and Meta-Llama-3-8B
"""

import os
import re
import json
import importlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from tqdm import tqdm
from code.phases.phase11.utils import get_device

heal_module = importlib.import_module("code.phases.phase15_layer_drop.2_heal_dropped_layers")
drop_layers_with_mapping = heal_module.drop_layers_with_mapping
train_lora_with_logging = heal_module.train_lora_with_logging

# Standard 5-shot GSM8K Prompt template
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

    for i in tqdm(range(total), desc="Evaluating GSM8K", leave=False):
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
                pad_token_id=tokenizer.pad_token_id
            )

        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_ans = extract_strict_answer(gen_text)

        if pred_ans.strip() == gold_ans.strip() and gold_ans.strip() != "":
            correct += 1

    accuracy = (correct / total) * 100.0
    print(f"  -> GSM8K Accuracy (n={total}): {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def ensure_global_lora_checkpoint(model_name, drop_indices, map_file, ckpt_dir, train_samples=500, rank=16, lr=1e-5, seed=42):
    if os.path.exists(os.path.join(ckpt_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(ckpt_dir, "adapter_model.bin")):
        print(f"Global LoRA checkpoint already exists at {ckpt_dir}.")
        return

    print(f"Global LoRA checkpoint missing. Training and saving to {ckpt_dir}...")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    model, old_to_new_map = drop_layers_with_mapping(model, drop_indices)

    with open(map_file) as f:
        data = json.load(f)
    sens = data.get("sensitivities", {})
    sorted_layers = sorted(sens.items(), key=lambda x: x[1], reverse=True)
    top_layers = [l[0] for l in sorted_layers[:6]]
    target_indices = [int(l.split("_")[1]) for l in top_layers if int(l.split("_")[1]) in old_to_new_map]

    target_modules = []
    for old_idx in target_indices:
        new_idx = old_to_new_map[old_idx]
        target_modules.extend([
            f"model.layers.{new_idx}.self_attn.q_proj",
            f"model.layers.{new_idx}.self_attn.v_proj",
            f"model.layers.{new_idx}.mlp.gate_proj",
            f"model.layers.{new_idx}.mlp.up_proj",
            f"model.layers.{new_idx}.mlp.down_proj"
        ])

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_lora_with_logging(model, tokenizer, device, train_samples, lr, warmup_ratio=0.10)
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Saved Global LoRA checkpoint to {ckpt_dir}")

def evaluate_model_conditions(model_name, drop_indices, b_adapter_dir, g_adapter_dir, map_file, num_samples=100):
    device = get_device()
    model_short = model_name.split("/")[-1]
    print(f"\n=========================================")
    print(f"Evaluating GSM8K for {model_short}")
    print(f"=========================================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # 1. Dense Baseline
    print(f"\n[Condition 1/4] Dense Baseline: {model_short}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    results["dense"] = evaluate_gsm8k(model, tokenizer, device, num_samples)
    del model
    torch.cuda.empty_cache()

    # 2. Dropped Unhealed (t=0)
    print(f"\n[Condition 2/4] Dropped Unhealed (t=0): {model_short}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model, _ = drop_layers_with_mapping(model, drop_indices)
    results["dropped_unhealed"] = evaluate_gsm8k(model, tokenizer, device, num_samples)
    del model
    torch.cuda.empty_cache()

    # Ensure global checkpoint exists
    ensure_global_lora_checkpoint(model_name, drop_indices, map_file, g_adapter_dir)

    # 3. Global Sensitivity LoRA Healed
    print(f"\n[Condition 3/4] Global Sensitivity LoRA Healed: {model_short}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    base_model, _ = drop_layers_with_mapping(base_model, drop_indices)
    healed_global = PeftModel.from_pretrained(base_model, g_adapter_dir)
    results["global_healed"] = evaluate_gsm8k(healed_global, tokenizer, device, num_samples)
    del base_model, healed_global
    torch.cuda.empty_cache()

    # 4. Boundary LoRA Healed (t=2)
    print(f"\n[Condition 4/4] Incision Boundary LoRA Healed (t=2): {model_short}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    base_model, _ = drop_layers_with_mapping(base_model, drop_indices)
    healed_boundary = PeftModel.from_pretrained(base_model, b_adapter_dir)
    results["boundary_healed_t2"] = evaluate_gsm8k(healed_boundary, tokenizer, device, num_samples)
    del base_model, healed_boundary
    torch.cuda.empty_cache()

    return results

def main():
    num_samples = 100  # 100 GSM8K test samples per condition for fast, clean comparison
    out_dir = "experiments/phase15_layer_drop"
    all_results = {}

    # Mistral-7B
    m_name = "mistralai/Mistral-7B-v0.1"
    m_drop = [11, 12, 13, 14]
    m_b_adapter = "checkpoints/Mistral-7B-v0.1_boundary_healed_t2_lora"
    m_g_adapter = "checkpoints/Mistral-7B-v0.1_global_healed_lora"
    m_map = "experiments/phase14_final/Mistral-7B-v0.1_sensitivity_map.json"
    all_results["Mistral-7B-v0.1"] = evaluate_model_conditions(m_name, m_drop, m_b_adapter, m_g_adapter, m_map, num_samples)

    # Llama-3-8B
    l_name = "meta-llama/Meta-Llama-3-8B"
    l_drop = [8, 9, 10, 11]
    l_b_adapter = "checkpoints/Meta-Llama-3-8B_boundary_healed_t2_lora"
    l_g_adapter = "checkpoints/Meta-Llama-3-8B_global_healed_lora"
    l_map = "experiments/phase14_final/Meta-Llama-3-8B_sensitivity_map.json"
    all_results["Meta-Llama-3-8B"] = evaluate_model_conditions(l_name, l_drop, l_b_adapter, l_g_adapter, l_map, num_samples)

    out_file = os.path.join(out_dir, "gsm8k_healed_eval_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("GSM8K EVALUATION COMPLETE. RESULTS:")
    print("="*60)
    for model, res in all_results.items():
        print(f"\n{model}:")
        print(f"  Dense Baseline:            {res['dense']:.2f}%")
        print(f"  Dropped Unhealed (t=0):    {res['dropped_unhealed']:.2f}%")
        print(f"  Global Sensitivity LoRA:   {res['global_healed']:.2f}%")
        print(f"  Incision Boundary (t=2):   {res['boundary_healed_t2']:.2f}%")

if __name__ == "__main__":
    main()
