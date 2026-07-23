"""
Parallel GSM8K evaluation for Meta-Llama-3-8B Boundary Healed (t=2) on GPU 0
"""

import os
import sys
import json
import importlib

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from code.phases.phase11.utils import get_device

heal_module = importlib.import_module("code.phases.phase15_layer_drop.2_heal_dropped_layers")
drop_layers_with_mapping = heal_module.drop_layers_with_mapping

gsm8k_eval_module = importlib.import_module("code.phases.phase15_layer_drop.eval_gsm8k_healed")
evaluate_gsm8k = gsm8k_eval_module.evaluate_gsm8k

def main():
    model_name = "meta-llama/Meta-Llama-3-8B"
    drop_indices = [8, 9, 10, 11]
    b_adapter_dir = "checkpoints/Meta-Llama-3-8B_boundary_healed_t2_lora"
    device = get_device()

    print(f"\n[GPU 0 Parallel Eval] Evaluating GSM8K for Llama-3-8B Boundary Healed (t=2)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    base_model, _ = drop_layers_with_mapping(base_model, drop_indices)
    healed_boundary = PeftModel.from_pretrained(base_model, b_adapter_dir)

    accuracy = evaluate_gsm8k(healed_boundary, tokenizer, device, num_samples=100)
    print(f"\n -> Llama-3-8B Boundary Healed (t=2) GSM8K Accuracy: {accuracy:.2f}%")

    out_file = "experiments/phase15_layer_drop/llama3_boundary_t2_gsm8k.json"
    with open(out_file, "w") as f:
        json.dump({"model": model_name, "condition": "boundary_healed_t2", "gsm8k_accuracy": accuracy}, f, indent=2)

if __name__ == "__main__":
    main()
