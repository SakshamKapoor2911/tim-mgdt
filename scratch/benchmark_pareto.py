import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_macs(model):
    total_macs = 0
    
    for layer in model.model.layers:
        # MLP MACs (per token): gate + up + down
        h = model.config.hidden_size
        i = layer.mlp.down_proj.weight.shape[1] # intermediate size
        mlp_macs = 3 * h * i
        
        # Attention MACs (per token): q + k + v + o
        q_heads = layer.self_attn.num_heads
        k_heads = layer.self_attn.num_key_value_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        
        q_macs = h * (q_heads * head_dim)
        k_macs = h * (k_heads * head_dim)
        v_macs = h * (k_heads * head_dim)
        o_macs = (q_heads * head_dim) * h
        attn_macs = q_macs + k_macs + v_macs + o_macs
        
        total_macs += (mlp_macs + attn_macs)
        
    return total_macs

def benchmark_latency(model, tokenizer, num_prompts=10, input_len=256, output_len=128):
    model.eval()
    
    # Create dummy input of fixed length
    input_ids = torch.randint(100, 10000, (1, input_len), device=model.device)
    
    # Warmup
    print("  Warming up GPU...")
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=10, min_new_tokens=10, use_cache=True)
    
    total_tokens_generated = num_prompts * output_len
    start_time = time.time()
    
    print("  Running latency benchmark...")
    with torch.no_grad():
        for _ in range(num_prompts):
            model.generate(input_ids, max_new_tokens=output_len, min_new_tokens=output_len, use_cache=True)
            
    end_time = time.time()
    elapsed = end_time - start_time
    tps = total_tokens_generated / elapsed
    return tps

def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    pruned_dir = "./checkpoints/Meta-Llama-3-8B_subnetwork_pruned"
    
    print("Loading Baseline Model...")
    baseline_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("\n--- Theoretical MAC Calculation ---")
    base_macs = calculate_macs(baseline_model)
    print(f"Baseline MACs per token: {base_macs / 1e9:.2f} G-MACs")
    
    print("\n--- Baseline Latency Benchmark ---")
    base_tps = benchmark_latency(baseline_model, tokenizer, num_prompts=10)
    print(f"Baseline TPS: {base_tps:.2f} tokens/sec")
    
    # Free VRAM
    del baseline_model
    torch.cuda.empty_cache()
    
    print("\nLoading Pruned Subnetwork...")
    pruned_model = AutoModelForCausalLM.from_pretrained(pruned_dir, torch_dtype=torch.bfloat16, device_map="cuda:0")
    
    print("\n--- Theoretical MAC Calculation ---")
    pruned_macs = calculate_macs(pruned_model)
    mac_reduction = 100 * (1 - pruned_macs / base_macs)
    print(f"Pruned MACs per token: {pruned_macs / 1e9:.2f} G-MACs")
    print(f"Theoretical Compute Reduction: {mac_reduction:.2f}%")
    
    print("\n--- Pruned Latency Benchmark ---")
    pruned_tps = benchmark_latency(pruned_model, tokenizer, num_prompts=10)
    speedup = 100 * (pruned_tps / base_tps - 1)
    print(f"Pruned TPS: {pruned_tps:.2f} tokens/sec")
    print(f"End-to-End Speedup: +{speedup:.2f}%")

if __name__ == "__main__":
    main()
