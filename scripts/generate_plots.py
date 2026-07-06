import matplotlib.pyplot as plt
import os
import glob
import json
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

def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def _gsm8k_metric(path):
    data = _load_json(path)
    if not data:
        return {}
    gsm8k = data.get("results", {}).get("gsm8k", {})
    samples = data.get("n-samples", {}).get("gsm8k", {})
    return {
        "n": samples.get("effective", gsm8k.get("sample_len")) or 0,
        "strict": gsm8k.get("exact_match,strict-match"),
        "path": path,
    }

def _best_gsm8k(pattern):
    metrics = [_gsm8k_metric(p) for p in glob.glob(pattern)]
    metrics = [m for m in metrics if m.get("strict") is not None]
    if not metrics:
        return {}
    return sorted(metrics, key=lambda m: (m["n"], os.path.getmtime(m["path"])))[-1]

def generate_phase19_reasoning_plot():
    """Plot the mask-vs-structural dissociation behind the Phase 19 claim."""
    points = []

    dense_metric = _best_gsm8k("experiments/phase19/gsm8k_eval_baseline_full*.json")
    first_mask = _load_json("experiments/phase19/Mistral-7B-v0.1_mask_pruning_s20.json") or {}
    if dense_metric:
        points.append({
            "label": "Dense",
            "family": "Dense baseline",
            "c4": first_mask.get("baseline_c4_ppl"),
            "strict": dense_metric.get("strict"),
            "n": dense_metric.get("n"),
            "sparsity": 0.0,
        })

    for path in sorted(glob.glob("experiments/phase19/Mistral-7B-v0.1_mask_pruning_s*.json")):
        data = _load_json(path) or {}
        tag = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        metric = _best_gsm8k(f"experiments/phase19/gsm8k_eval_mask_{tag}*.json")
        if not metric:
            continue
        points.append({
            "label": f"Mask {data.get('target_sparsity', 0) * 100:.0f}%",
            "family": "Dense-shaped mask",
            "c4": data.get("masked_c4_ppl"),
            "strict": metric.get("strict"),
            "n": metric.get("n"),
            "sparsity": data.get("target_sparsity", 0),
        })

    for path in sorted(glob.glob("experiments/phase18b/Mistral-7B-v0.1_width_pruning_ceil_s*.json")):
        data = _load_json(path) or {}
        tag = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        metric = _best_gsm8k(f"experiments/phase18b/gsm8k_eval_ceil_{tag}*.json")
        if not metric:
            continue
        points.append({
            "label": f"Width {data.get('sparsity', 0) * 100:.1f}%",
            "family": "Structural width",
            "c4": data.get("pruned_c4_ppl"),
            "strict": metric.get("strict"),
            "n": metric.get("n"),
            "sparsity": data.get("sparsity", 0),
        })

    phase16 = _load_json("experiments/phase16/Mistral-7B-v0.1_width_pruning_ceil.json") or {}
    phase16_metric = _best_gsm8k("experiments/phase16/gsm8k_eval_ceil_*.json")
    if phase16 and phase16_metric:
        points.append({
            "label": "Width 20% full",
            "family": "Structural width",
            "c4": phase16.get("pruned_c4_ppl"),
            "strict": phase16_metric.get("strict"),
            "n": phase16_metric.get("n"),
            "sparsity": phase16.get("sparsity", 0.2),
        })

    points = [p for p in points if p.get("c4") is not None and p.get("strict") is not None]
    if not points:
        print("No Phase 19 points found, skipping reasoning dissociation plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))

    styles = {
        "Dense baseline": {"marker": "*", "color": "#111827", "size": 220},
        "Dense-shaped mask": {"marker": "o", "color": "#2563eb", "size": 100},
        "Structural width": {"marker": "s", "color": "#dc2626", "size": 85},
    }

    for family, style in styles.items():
        family_points = [p for p in points if p["family"] == family]
        if not family_points:
            continue
        ax.scatter(
            [p["c4"] for p in family_points],
            [p["strict"] * 100 for p in family_points],
            s=style["size"],
            marker=style["marker"],
            color=style["color"],
            label=family,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
            zorder=3,
        )

    label_offsets = {
        "Dense": (8, 10),
        "Mask 20%": (8, 10),
        "Mask 30% smoke": (8, 10),
        "Mask 40% smoke": (8, 10),
        "Width 1.0% smoke": (-42, 10),
        "Width 2.5% smoke": (8, 10),
        "Width 5.0% smoke": (8, 10),
        "Width 10.0% smoke": (8, 10),
        "Width 15.0% smoke": (8, 10),
        "Width 20.0% smoke": (8, -2),
        "Width 20% full": (8, 10),
    }

    for point in points:
        is_full = point.get("n", 0) and point["n"] > 100
        alpha = 0.95 if is_full else 0.65
        label = point["label"] if is_full else f"{point['label']} smoke"
        xytext = label_offsets.get(label, (6, 6))
        ax.annotate(
            label,
            (point["c4"], point["strict"] * 100),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
            alpha=alpha,
        )

    ax.set_title("Phase 19: Reasoning Retention Separates Masking from Structural Pruning", fontsize=13, pad=12)
    ax.set_xlabel("C4 perplexity (lower is better)", fontsize=11)
    ax.set_ylabel("GSM8K strict exact match (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(frameon=True, fontsize=10)

    note = "Full GSM8K points use n=1319; smoke points use n=50."
    ax.text(0.01, -0.14, note, transform=ax.transAxes, fontsize=9, color="#4b5563")

    os.makedirs('dashboard/figures', exist_ok=True)
    os.makedirs('paper/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('dashboard/figures/phase19_reasoning_dissociation.png', dpi=300)
    plt.savefig('paper/figures/phase19_reasoning_dissociation.png', dpi=300)
    print("Saved Phase 19 reasoning dissociation plots")

if __name__ == '__main__':
    os.makedirs('dashboard/figures', exist_ok=True)
    os.makedirs('paper/figures', exist_ok=True)
    generate_pareto_plot()
    generate_thickness_plot()
    generate_bathtub_plot()
    generate_phase19_reasoning_plot()
