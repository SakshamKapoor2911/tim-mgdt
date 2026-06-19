import json
import os

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
