import json
import os

print("# 1. Mistral-7B Pareto Sweep (Phase 15)")
results = []
for drop in [2, 4, 6, 8]:
    eval_file = f"experiments/phase15_layer_drop/Mistral-7B-v0.1_drop{drop}_eval.json"
    heal_file = f"experiments/phase15_layer_drop/Mistral-7B-v0.1_drop{drop}_boundary_healed.json"
    
    if os.path.exists(eval_file) and os.path.exists(heal_file):
        with open(eval_file) as f:
            e = json.load(f)
        with open(heal_file) as f:
            h = json.load(f)
        results.append({
            "drop": drop,
            "lat_pct": e["latency_reduction_pct"],
            "base_ppl": e["baseline_c4_ppl"],
            "drop_ppl": e["dropped_c4_ppl"],
            "heal_ppl": h["healed_c4_ppl"]
        })

print("| Layers Dropped | Latency Reduction | Baseline PPL | Dropped PPL | Healed PPL |")
print("|----------------|-------------------|--------------|-------------|------------|")
for r in results:
    print(f"| {r['drop']} ({r['drop']}/32) | -{r['lat_pct']:.2f}% | {r['base_ppl']:.3f} | {r['drop_ppl']:.3f} | {r['heal_ppl']:.3f} |")

print("\n# 2. Boundary Thickness Ablations (Phase 16C, 4-Layer Drop)")
models = ["Mistral-7B-v0.1", "Meta-Llama-3-8B", "gemma-2-2b"]
thicknesses = [1, 2, 3]

for model in models:
    print(f"\n### {model}")
    print("| Thickness ($t$) | Layers Healed | Healed C4 PPL |")
    print("|-----------------|---------------|---------------|")
    for t in thicknesses:
        path = f"experiments/phase15_layer_drop/{model}_drop4_boundary_healed_t{t}.json"
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            layers_healed = t * 2
            print(f"| $t={t}$ | {layers_healed} | {d['healed_c4_ppl']:.3f} |")
        else:
            print(f"| $t={t}$ | {t*2} | [Running...] |")

