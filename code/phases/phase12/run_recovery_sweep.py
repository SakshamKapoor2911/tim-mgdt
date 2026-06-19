import os
import json
import subprocess

def run_script(script_name, args):
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    cmd = ["python", "-m", f"code.phases.phase12.{script_name}"] + args
    subprocess.run(cmd, check=True)

def main():
    model_args = ["--model", "mistralai/Mistral-7B-v0.1", "--samples", "30"]
    
    # 1. Method 1: SVD Recovery
    # Use smaller samples for rapid iteration
    run_script("closed_form_svd_recovery", model_args + ["--rank", "8"])
    
    # 2. Method 2: Sensitivity LoRA
    run_script("sensitivity_lora_recovery", model_args + ["--rank", "16", "--train_samples", "100"])
    
    # 3. Method 3: SVD-Initialized LoRA
    run_script("svd_initialized_lora", model_args + ["--rank", "16", "--train_samples", "100"])
    
    # Aggregate results
    print("\n\n" + "="*60)
    print("RECOVERY SWEEP RESULTS")
    print("="*60)
    
    results = {}
    for m in [("method1_svd_rank8.json", "Method 1: SVD Recovery (Rank 8)"), 
              ("method2_lora_rank16.json", "Method 2: Sensitivity LoRA (Rank 16)"), 
              ("method3_svd_lora_rank16.json", "Method 3: SVD-Init LoRA (Rank 16)")]:
        file_path = f"experiments/phase12/{m[0]}"
        if os.path.exists(file_path):
            data = json.load(open(file_path))
            results[m[1]] = data
            
            print(f"{m[1]}:")
            print(f"  Recovery Time: {data['recovery_time']:.1f}s")
            print(f"  Final PPL:     {data['recovered_ppl']:.4f} (+{data['ppl_increase_pct']:.2f}%)")
            if 'init_ppl' in data:
                print(f"  (SVD Init PPL: {data['init_ppl']:.4f})")
            print()
            
    with open("experiments/phase12/recovery_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
