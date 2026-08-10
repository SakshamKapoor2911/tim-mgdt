import torch
import glob
import json
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file
from peft import PeftModel
import lm_eval
from lm_eval.models.huggingface import HFLM

def load_pruned_model(model_id, pruned_dir, lora_dir=None):
    print("\nLoading Pruned Subnetwork...")
    config = AutoConfig.from_pretrained(model_id)
    pruned_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    
    st_files = glob.glob(f"{pruned_dir}/*.safetensors")
    state_dict = {}
    for st_file in st_files:
        state_dict.update(load_file(st_file))
        
    head_dim = config.hidden_size // config.num_attention_heads
    
    for key, tensor in state_dict.items():
        if "weight" in key and ("proj" in key or "embed" in key or "lm_head" in key):
            parts = key.split(".")
            if len(parts) < 2: continue
            out_f, in_f = tensor.shape if len(tensor.shape) == 2 else (tensor.shape[0], tensor.shape[0])
            
            curr = pruned_model
            for p in parts[:-2]:
                curr = getattr(curr, p)
            proj_name = parts[-2]
            
            if hasattr(curr, proj_name) and isinstance(getattr(curr, proj_name), torch.nn.Linear):
                setattr(curr, proj_name, torch.nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16))
                
                if "q_proj" in key:
                    curr.num_heads = out_f // head_dim
                if "k_proj" in key:
                    curr.num_key_value_heads = out_f // head_dim

    pruned_model.load_state_dict(state_dict, strict=False)
    
    if lora_dir and os.path.exists(lora_dir):
        print(f"\nAttaching LoRA from {lora_dir}...")
        pruned_model = PeftModel.from_pretrained(pruned_model, lora_dir)
        pruned_model = pruned_model.merge_and_unload()
        
    pruned_model = pruned_model.to("cuda:0")
    pruned_model.eval()
    return pruned_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu,asdiv")
    parser.add_argument("--lora_dir", type=str, default="./checkpoints/Meta-Llama-3-8B_sparse_lora_healed")
    parser.add_argument("--pruned_dir", type=str, default="./checkpoints/Meta-Llama-3-8B_subnetwork_pruned")
    parser.add_argument("--output_file", type=str, default="./results/Meta-Llama-3-8B_sparse_healed_eval.json")
    args = parser.parse_args()

    model_id = "meta-llama/Meta-Llama-3-8B"
    pruned_dir = args.pruned_dir
    
    model = load_pruned_model(model_id, pruned_dir, lora_dir=args.lora_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("\nWrapping in lm_eval HFLM...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")
    
    print(f"\nEvaluating tasks: {args.tasks}...")
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=5,
    )
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Extract only the exact metrics we need to prevent serialization errors
    cleaned_results = {}
    if "results" in results:
        for task, metrics in results["results"].items():
            cleaned_results[task] = metrics

    with open(args.output_file, "w") as f:
        json.dump(cleaned_results, f, indent=4)
        
    print(f"\nResults saved to {args.output_file}")
    for task, res in cleaned_results.items():
        print(f"{task}: {res}")

if __name__ == "__main__":
    main()
