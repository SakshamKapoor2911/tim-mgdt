import os
import time
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from code.phases.phase11.utils import get_device, compute_perplexity, time_forward_pass

def apply_svd_recovery(model: nn.Module, target_sparsity: float, rank: int = 8):
    """
    Method 1: Closed-Form SVD Recovery.
    Prunes the model to target_sparsity, but instead of just discarding the weights,
    it computes a low-rank SVD approximation of the discarded weights and adds it back.
    """
    modules_pruned = 0
    total_params_recovered = 0
    
    spectra = {}

    print(f"Applying SVD Recovery (rank={rank}) at {target_sparsity*100:.1f}% sparsity...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 1. Get original weights
                w_orig = module.weight.data.clone().float() # keep in float32 for SVD
                
                # 2. Determine mask
                nparams_toprune = int(w_orig.numel() * target_sparsity)
                if nparams_toprune == 0:
                    continue
                    
                threshold = torch.kthvalue(w_orig.abs().view(-1), nparams_toprune).values.item()
                mask = w_orig.abs() >= threshold
                
                # 3. Prune weight in-place (fp16)
                module.weight.data.mul_(mask.to(module.weight.dtype))
                
                # 4. Compute ΔW = discarded weights
                delta_w = w_orig * (~mask)
                
                # Skip if effectively no pruning happened (e.g. all zeros)
                if delta_w.abs().sum() == 0:
                    continue
                
                # 5. Compute SVD on CPU
                delta_w_cpu = delta_w.cpu()
                U, S, Vh = torch.linalg.svd(delta_w_cpu, full_matrices=False)
                
                # Record singular values for analysis
                spectra[name] = S.cpu().tolist()
                
                # 6. Low-rank approximation
                r = min(rank, S.size(0))
                # W_recovered = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]
                S_r = torch.diag(S[:r])
                delta_w_approx = U[:, :r] @ S_r @ Vh[:r, :]
                
                # 7. Add back to pruned weights
                # NOTE: The low-rank matrix is dense! Adding it back destroys 
                # the *exact* element-wise sparsity of the weight matrix. 
                # This tests whether the *mathematical* rank is more important than literal zeros.
                module.weight.data.add_(delta_w_approx.to(device=module.weight.device, dtype=module.weight.dtype))
                
                modules_pruned += 1
                total_params_recovered += (U.shape[0]*r + r + Vh.shape[1]*r) # approx parameters used
                
    return modules_pruned, spectra

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.20)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    print(f"Loading {args.model} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    print("Measuring baseline...")
    baseline_ppl = compute_perplexity(model, tokenizer, device, max_samples=args.samples)
    
    start_time = time.time()
    modules_pruned, spectra = apply_svd_recovery(model, args.sparsity, args.rank)
    recovery_time = time.time() - start_time
    
    print("Measuring recovered PPL...")
    recovered_ppl = compute_perplexity(model, tokenizer, device, max_samples=args.samples)
    
    results = {
        "method": "method1_svd",
        "rank": args.rank,
        "recovery_time": recovery_time,
        "baseline_ppl": baseline_ppl,
        "recovered_ppl": recovered_ppl,
        "ppl_increase_pct": ((recovered_ppl - baseline_ppl) / baseline_ppl) * 100
    }
    
    print("\n--- Method 1 Results ---")
    print(f"Time: {recovery_time:.2f}s")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Recovered PPL: {recovered_ppl:.4f} (+{results['ppl_increase_pct']:.2f}%)")
    
    os.makedirs("experiments/phase12", exist_ok=True)
    with open(f"experiments/phase12/method1_svd_rank{args.rank}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"experiments/phase12/svd_spectra.json", "w") as f:
        json.dump(spectra, f)

if __name__ == "__main__":
    main()
