import os
import glob
from safetensors import safe_open
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("Loading Pruned Checkpoint Metadata for Visualization...")
    save_dir = "./checkpoints/Meta-Llama-3-8B_subnetwork_pruned"
    
    safetensor_files = glob.glob(os.path.join(save_dir, "*.safetensors"))
    if not safetensor_files:
        print("No safetensors found! Run pruning script first.")
        return

    layer_sizes = {}
    
    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "mlp.down_proj.weight" in key:
                    layer_idx = int(key.split("layers.")[1].split(".")[0])
                    inter_size = f.get_tensor(key).shape[1]
                    layer_sizes[layer_idx] = inter_size
                    
    baseline_size = 14336
    
    layers = sorted(layer_sizes.keys())
    pruned_pcts = [100 * (1.0 - (layer_sizes[i] / baseline_size)) for i in layers]
    
    # Set seaborn style for publication quality
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    plt.figure(figsize=(12, 6))
    
    # Create the bar plot
    bars = sns.barplot(x=layers, y=pruned_pcts, color="#2b5c8f", edgecolor="black", linewidth=1.2)
    
    # Highlight the 6 most damaged layers (targets for LoRA)
    sorted_indices = np.argsort(pruned_pcts)[::-1]
    top_6_indices = sorted_indices[:6]
    
    for idx in top_6_indices:
        bars.patches[idx].set_color("#c44e52")
        bars.patches[idx].set_edgecolor("black")
        bars.patches[idx].set_linewidth(1.2)
        
    plt.axhline(y=0, color='black', linewidth=1.5)
    
    plt.title("Structural Pruning Density across Llama-3-8B Layers\n(FIM Saliency Target: 20% Global Sparsity)", 
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Transformer Layer Depth", fontsize=14, labelpad=10)
    plt.ylabel("Parameters Pruned (%)", fontsize=14, labelpad=10)
    
    plt.ylim(0, 85)
    plt.xticks(rotation=0, fontsize=10)
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2b5c8f', edgecolor='black', label='Base Sparsity'),
        Patch(facecolor='#c44e52', edgecolor='black', label='Targeted LoRA Healing')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, shadow=True)
    
    plt.tight_layout()
    
    os.makedirs("./paper/figures", exist_ok=True)
    plt.savefig("./paper/figures/pruning_density.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("./paper/figures/pruning_density.png", format='png', dpi=300, bbox_inches='tight')
    
    print("Saved publication figures to ./paper/figures/pruning_density.pdf and .png")

if __name__ == "__main__":
    main()
