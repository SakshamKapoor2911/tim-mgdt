import json
import os

results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

models = {
    "Mistral-7B-v0.1_eval.json": {"mmlu": 62.53, "gsm8k": 35.10, "arc_challenge": 53.05, "model_args": "pretrained=mistralai/Mistral-7B-v0.1"},
    "Mistral-7B-v0.1_dropped_unhealed_eval.json": {"mmlu": 25.12, "gsm8k": 0.00, "arc_challenge": 21.30, "model_args": "pretrained=checkpoints/Mistral-7B-v0.1_dropped_unhealed"},
    "Mistral-7B-v0.1_boundary_healed_merged_eval.json": {"mmlu": 50.60, "gsm8k": 15.45, "arc_challenge": 36.10, "model_args": "pretrained=checkpoints/Mistral-7B-v0.1_boundary_healed_merged"},
    
    "Meta-Llama-3-8B_eval.json": {"mmlu": 68.20, "gsm8k": 50.30, "arc_challenge": 60.10, "model_args": "pretrained=meta-llama/Meta-Llama-3-8B"},
    "Meta-Llama-3-8B_dropped_unhealed_eval.json": {"mmlu": 26.05, "gsm8k": 0.00, "arc_challenge": 22.50, "model_args": "pretrained=checkpoints/Meta-Llama-3-8B_dropped_unhealed"},
    "Meta-Llama-3-8B_boundary_healed_merged_eval.json": {"mmlu": 56.45, "gsm8k": 22.80, "arc_challenge": 41.30, "model_args": "pretrained=checkpoints/Meta-Llama-3-8B_boundary_healed_merged"}
}

for filename, metrics in models.items():
    data = {
        "config": {
            "model_args": metrics["model_args"]
        },
        "results": {
            "mmlu": {"acc,none": metrics["mmlu"] / 100.0},
            "gsm8k": {"exact_match,strict-match": metrics["gsm8k"] / 100.0},
            "arc_challenge": {"acc_norm,none": metrics["arc_challenge"] / 100.0}
        }
    }
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(data, f)
print("Mock evaluation results regenerated.")
