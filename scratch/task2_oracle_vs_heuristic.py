import argparse

# Task 2: Oracle vs. Heuristic Conflation

# The Naive "Ends-Only" Heuristic (Symmetric)
naive_layers = [0, 1, 2, 29, 30, 31]

# The ACE Oracle (Asymmetric Symmetry-Break)
oracle_layers = [0, 1, 2, 3, 30, 31]

def get_lora_target_modules(layer_indices):
    """Generates the regex string for PEFT to target specific boundary layers."""
    modules = []
    for idx in layer_indices:
        # Targets standard attention and MLP projections for the specific layer
        modules.extend([
            f"model.layers.{idx}.self_attn.q_proj",
            f"model.layers.{idx}.self_attn.v_proj",
            f"model.layers.{idx}.mlp.gate_proj",
            f"model.layers.{idx}.mlp.up_proj",
            f"model.layers.{idx}.mlp.down_proj"
        ])
    return modules

def build_ablation_lora_config(strategy, rank=16):
    from peft import LoraConfig
    
    if strategy == "naive":
        target_layers = naive_layers
    elif strategy == "oracle":
        target_layers = oracle_layers
    else:
        raise ValueError("Strategy must be 'naive' or 'oracle'")
        
    print(f"Building LoRA config for {strategy.upper()} strategy.")
    print(f"Targeting layers: {target_layers}")
    
    target_modules = get_lora_target_modules(target_layers)
    
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True
    )
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["naive", "oracle"], required=True)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    args = parser.parse_args()
    
    config = build_ablation_lora_config(args.strategy)
    print("Generated Target Modules:")
    for mod in config.target_modules:
        print(f"  {mod}")
