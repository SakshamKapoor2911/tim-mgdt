import os
from safetensors import safe_open
import glob

def main():
    print("Loading Pruned Checkpoint Metadata via Safetensors...")
    save_dir = "./checkpoints/Meta-Llama-3-8B_subnetwork_pruned"
    
    safetensor_files = glob.glob(os.path.join(save_dir, "*.safetensors"))
    if not safetensor_files:
        print("No safetensors found!")
        return

    print("\n--- Structural Pruning Diagnostic ---")
    layer_sizes = {}
    
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mlp.down_proj.weight" in key:
                    # Key format: model.layers.X.mlp.down_proj.weight
                    layer_idx = int(key.split("layers.")[1].split(".")[0])
                    # shape is [hidden_size, intermediate_size]
                    inter_size = f.get_tensor(key).shape[1]
                    layer_sizes[layer_idx] = inter_size
                    
    baseline_size = 14336
    
    print("\n[Layer-by-Layer MLP Surviving Dimensions]")
    for i in sorted(layer_sizes.keys()):
        pruned_pct = 100 * (1.0 - (layer_sizes[i] / baseline_size))
        print(f"Layer {i:02d}: {layer_sizes[i]} / {baseline_size} neurons (Pruned: {pruned_pct:.1f}%)")
        
    sorted_layers = sorted(layer_sizes.items(), key=lambda x: x[1])
    
    print("\n[Top 6 Most Damaged Layers for Sparse LoRA Targeting]")
    top_6 = []
    for i in range(min(6, len(sorted_layers))):
        idx, size = sorted_layers[i]
        pruned_pct = 100 * (1.0 - (size / baseline_size))
        top_6.append(idx)
        print(f"Target {i+1} -> Layer {idx:02d} (Pruned: {pruned_pct:.1f}%)")
        
    print(f"\nRecommended damaged_layers config: {top_6}")

if __name__ == "__main__":
    main()
