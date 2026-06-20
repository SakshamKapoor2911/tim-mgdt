import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    if not os.path.exists("experiments/phase12/svd_spectra.json"):
        print("No SVD spectra found. Run closed_form_svd_recovery.py first.")
        return

    data = json.load(open("experiments/phase12/svd_spectra.json"))
    
    # We will plot the top 50 singular values for a few layers
    layers_to_plot = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.15.mlp.gate_proj",
        "model.layers.31.self_attn.v_proj"
    ]
    
    plt.figure(figsize=(10, 6))
    
    for layer in layers_to_plot:
        if layer in data:
            s = np.array(data[layer])
            # Plot the cumulative explained variance ratio
            explained_variance = np.cumsum(s**2) / np.sum(s**2)
            plt.plot(explained_variance[:50], marker='o', label=layer)
            
            # Print text stats
            print(f"Layer: {layer}")
            print(f"  Rank 8 explains {explained_variance[7]*100:.1f}% of pruning damage")
            print(f"  Rank 16 explains {explained_variance[15]*100:.1f}% of pruning damage")
            print()
            
    plt.axhline(0.8, color='r', linestyle='--', label='80% Variance')
    plt.axhline(0.9, color='g', linestyle='--', label='90% Variance')
    plt.title("Cumulative Explained Variance of Pruning Damage (ΔW)")
    plt.xlabel("Rank")
    plt.ylabel("Explained Variance Ratio")
    plt.legend()
    plt.grid(True)
    
    os.makedirs("visualizations/phase12", exist_ok=True)
    plt.savefig("visualizations/phase12/svd_spectra_variance.png", dpi=300)
    print("Saved plot to visualizations/phase12/svd_spectra_variance.png")

if __name__ == "__main__":
    main()
