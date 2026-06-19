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
    print("Saved Pareto frontier plots to dashboard/figures and paper/figures")

if __name__ == '__main__':
    generate_pareto_plot()
