import json
import glob
import os
from collections import defaultdict

def parse_lm_eval_results(directory="./results"):
    files = glob.glob(os.path.join(directory, "*_eval*.json"))
    if not files:
        print("No evaluation JSONs found in the results directory.")
        return

    # Aggregate by model name
    aggregated = defaultdict(lambda: {"MMLU (5-shot)": "N/A", "GSM8K (5-shot)": "N/A", "ARC-c (5-shot)": "N/A"})
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            
            # Extract model name safely from config
            args = data.get('config', {}).get('model_args', {})
            if isinstance(args, str):
                model_name = args.split('pretrained=')[-1].split(',')[0].split('/')[-1]
            elif isinstance(args, dict):
                model_name = args.get('pretrained', '').split('/')[-1]
            else:
                continue
                
            if not model_name:
                continue
                
            # Extract metrics (safely falling back if a metric isn't found)
            mmlu_acc = data.get('results', {}).get('mmlu', {}).get('acc,none')
            gsm8k_acc = data.get('results', {}).get('gsm8k', {}).get('exact_match,strict-match')
            arc_acc = data.get('results', {}).get('arc_challenge', {}).get('acc_norm,none')
            
            # Format and aggregate
            if isinstance(mmlu_acc, float): aggregated[model_name]["MMLU (5-shot)"] = f"{mmlu_acc*100:.1f}%"
            if isinstance(gsm8k_acc, float): aggregated[model_name]["GSM8K (5-shot)"] = f"{gsm8k_acc*100:.1f}%"
            if isinstance(arc_acc, float): aggregated[model_name]["ARC-c (5-shot)"] = f"{arc_acc*100:.1f}%"
            
    # Print formatted Markdown table
    print("| Model | MMLU (5-shot) | GSM8K (5-shot) | ARC-c (5-shot) |")
    print("|-------|---------------|----------------|----------------|")
    for model_name in sorted(aggregated.keys()):
        r = aggregated[model_name]
        print(f"| {model_name} | {r['MMLU (5-shot)']} | {r['GSM8K (5-shot)']} | {r['ARC-c (5-shot)']} |")

if __name__ == "__main__":
    parse_lm_eval_results()
