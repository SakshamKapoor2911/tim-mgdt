import matplotlib.pyplot as plt
import os
import numpy as np

def generate_pareto_plot():
    # Data from Mistral-7B Phase 16B Sweep
    layers_dropped = [0, 2, 4, 6, 8]
    latency_reduction = [0, 10.36, 15.62, 19.79, 24.57]
    baseline_ppl = 7.855
    
    # Dropped PPL without healing (just for context, though we focus on healed)
    dropped_ppl = [baseline_ppl, 8.733, 9.930, 12.734, 20.822]
    
    # Healed PPL using Boundary Healing
    healed_ppl = [baseline_ppl, 8.378, 9.073, 10.866, 11.517]
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Baseline PPL reference line
    ax.axhline(y=baseline_ppl, color='black', linestyle='--', alpha=0.6, label='Baseline (7.855 PPL)')
    
    # Plot Dropped (Unhealed) PPL
    ax.plot(latency_reduction, dropped_ppl, marker='o', linestyle='-', color='red', alpha=0.5, label='Dropped (Unhealed) PPL')
    
    # Plot Healed PPL
    ax.plot(latency_reduction, healed_ppl, marker='s', linestyle='-', color='blue', linewidth=2.5, markersize=8, label='Boundary Healed PPL')
    
    # Highlight the 4-layer drop (Pareto Knee)
    ax.plot(15.62, 9.073, marker='*', color='gold', markersize=15, markeredgecolor='black', label='Pareto Knee (4 Drops)')
    
    # Add annotations
    for i, drop in enumerate(layers_dropped):
        if drop > 0:
            ax.annotate(f'{drop} Drops', 
                        (latency_reduction[i], healed_ppl[i]), 
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=9,
                        fontweight='bold')
    
    # Set labels and title
    ax.set_title('Mistral-7B Phase 16B: Latency vs. Perplexity Pareto Frontier', fontsize=14, pad=15)
    ax.set_xlabel('Inference Latency Reduction (%)', fontsize=12)
    ax.set_ylabel('C4 Perplexity (Lower is Better)', fontsize=12)
    
    # Format axes
    ax.set_xticks(latency_reduction)
    ax.set_xticklabels([f'-{x}%' if x > 0 else '0%' for x in latency_reduction])
    
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    
    # Save the figure
    os.makedirs('dashboard/figures', exist_ok=True)
    os.makedirs('paper/figures', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig('dashboard/figures/pareto_frontier.png', dpi=300)
    plt.savefig('paper/figures/pareto_frontier.png', dpi=300)
    print("Saved Pareto frontier plots")

def generate_thickness_plot():
    thickness = [1, 2, 3]
    healed_ppl = [9.163, 9.121, 9.120]
    baseline_global = 9.730
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(thickness, healed_ppl, color=['#3498db', '#2980b9', '#1f618d'], width=0.6)
    ax.axhline(y=baseline_global, color='red', linestyle='--', alpha=0.8, label='Global Healing (9.730 PPL)')
    
    ax.set_ylim(8.8, 9.9)
    ax.set_xticks(thickness)
    ax.set_xticklabels(['t=1\n(2 layers)', 't=2\n(4 layers)', 't=3\n(6 layers)'])
    
    ax.set_ylabel('C4 Perplexity (Lower is Better)', fontsize=12)
    ax.set_title('Mistral-7B 4-Drop: Boundary Thickness Ablation', fontsize=14, pad=15)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
                
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('dashboard/figures/thickness_ablation.png', dpi=300)
    plt.savefig('paper/figures/thickness_ablation.png', dpi=300)
    print("Saved Thickness Ablation plots")

def generate_bathtub_plot():
    import json
    map_file = "experiments/phase14_final/Mistral-7B-v0.1_sensitivity_map.json"
    if not os.path.exists(map_file):
        print(f"Cannot find {map_file}, skipping bathtub plot.")
        return
        
    with open(map_file, "r") as f:
        data = json.load(f)
    
    sens = data["sensitivities"]
    layers = list(range(len(sens)))
    scores = [sens[f"layer_{i}"] for i in layers]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(layers, scores, marker='o', linestyle='-', color='#8e44ad', linewidth=2, markersize=6)
    
    ax.set_title('The Bathtub Curve: Causal Sensitivity (Mistral-7B)', fontsize=14, pad=15)
    ax.set_xlabel('Transformer Layer Depth', fontsize=12)
    ax.set_ylabel('Causal $\Delta$CE (Sensitivity)', fontsize=12)
    
    ax.set_xticks(range(0, len(layers), 2))
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('dashboard/figures/bathtub_curve.png', dpi=300)
    plt.savefig('paper/figures/bathtub_curve.png', dpi=300)
    print("Saved Bathtub Curve plots")

if __name__ == '__main__':
    os.makedirs('dashboard/figures', exist_ok=True)
    os.makedirs('paper/figures', exist_ok=True)
    generate_pareto_plot()
    generate_thickness_plot()
    generate_bathtub_plot()
