import os
import sys
import json
import subprocess
import shutil
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Helper for layer drop
def drop_layers(model, drop_indices):
    old_layers = model.model.layers
    new_layers = nn.ModuleList([l for i, l in enumerate(old_layers) if i not in drop_indices])
    
    # Fix internal layer indices for HF cache
    for new_idx, layer in enumerate(new_layers):
        if hasattr(layer, 'layer_idx'):
            layer.layer_idx = new_idx
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'layer_idx'):
            layer.self_attn.layer_idx = new_idx
            
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model

def main():
    device = "cuda:1" # Use cuda:1 which has 20GB free memory!
    print(f"Running orchestrator on {device}...")
    
    ks = [0, 1, 2, 3, 4]
    
    # Part 1: Perplexity Evaluation
    out_ppl = "experiments/phase25_causal_interventions/exp_rotation_curve_perplexity.json"
    if os.path.exists(out_ppl):
        print(f"Perplexity results already exist at {out_ppl}. Skipping Part 1.")
    else:
        perplexity_results = {}
        for k in ks:
            print(f"\n>>> Running Perplexity Evaluation for k={k} on {device}...")
            cmd = [
                "venv/bin/python", "scripts/recompute_ppl_single.py",
                "--k", str(k),
                "--device", device
            ]
            subprocess.run(cmd, check=True)
            
            # Load the temp file
            temp_file = f"experiments/phase25_causal_interventions/temp_ppl_k{k}.json"
            with open(temp_file, "r") as f:
                temp_data = json.load(f)
            perplexity_results[f"k{k}"] = temp_data
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        # Save the consolidated perplexity results
        with open(out_ppl, "w") as f:
            json.dump(perplexity_results, f, indent=2)
        print(f"\nSaved consolidated perplexity curve to {out_ppl}")
    
    # Part 2: BBH Evaluation
    model_id = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bbh_scores = {}
    
    for k in ks:
        print(f"\n>>> Running BBH Evaluation for k={k} on {device}...")
        if k == 0:
            model_path = model_id
        else:
            model_path = f"checkpoints/temp_k{k}"
            if not os.path.exists(model_path):
                print(f"Creating dropped model checkpoint for k={k}...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
                drop_indices = list(range(11, 11 + k))
                model = drop_layers(model, drop_indices)
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                del model
                torch.cuda.empty_cache()
                
        output_json = f"experiments/phase25_causal_interventions/temp_bbh_k{k}.json"
        
        # Run lm_eval using venv_eval on device
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        cmd = [
            "venv_eval/bin/lm_eval", "run",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", "bbh_cot_fewshot_multistep_arithmetic_two",
            "--batch_size", "4",
            "--limit", "100",
            "--output_path", output_json
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        
        # Read score using glob to handle timestamp suffix added by lm_eval
        import glob
        prefix = output_json.replace(".json", "")
        matching_files = glob.glob(f"{prefix}*.json")
        if not matching_files:
            raise FileNotFoundError(f"No output file found matching pattern {prefix}*.json")
        
        actual_output_json = matching_files[0]
        with open(actual_output_json, "r") as f:
            data = json.load(f)
        score = data["results"]["bbh_cot_fewshot_multistep_arithmetic_two"]["exact_match,get-answer"]
        print(f"Result for k={k}: {score * 100:.2f}%")
        bbh_scores[f"k{k}"] = float(score)
        
        # Clean up
        if k > 0 and os.path.exists(model_path):
            shutil.rmtree(model_path)
        for fpath in matching_files:
            if os.path.exists(fpath):
                os.remove(fpath)
            
    out_bbh = "experiments/phase25_causal_interventions/exp_rotation_curve_bbh.json"
    with open(out_bbh, "w") as f:
        json.dump({"experiment": "rotation_curve_bbh", "results": bbh_scores}, f, indent=2)
    print(f"\nSaved consolidated BBH curve to {out_bbh}")
    
    print("\nAll recomputations completed successfully!")

if __name__ == "__main__":
    main()
