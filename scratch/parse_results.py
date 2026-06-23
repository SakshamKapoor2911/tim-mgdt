import json
import glob
import os

def parse_lm_eval_results(directory="./results"):
    files = glob.glob(os.path.join(directory, "*_eval.json"))
    if not files:
        print("No evaluation JSONs found in the results directory.")
        return

    results = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            model_name = data['config']['model_args'].split('/')[-1]
            
            # Extract metrics (safely falling back if a metric isn't found)
            mmlu_acc = data['results'].get('mmlu', {}).get('acc,none', 'N/A')
            gsm8k_acc = data['results'].get('gsm8k', {}).get('exact_match,strict-match', 'N/A')
            arc_acc = data['results'].get('arc_challenge', {}).get('acc_norm,none', 'N/A')
            
            # Format nicely
            if isinstance(mmlu_acc, float): mmlu_acc = f"{mmlu_acc*100:.1f}%"
            if isinstance(gsm8k_acc, float): gsm8k_acc = f"{gsm8k_acc*100:.1f}%"
            if isinstance(arc_acc, float): arc_acc = f"{arc_acc*100:.1f}%"
            
            results.append({
                "Model": model_name,
                "MMLU (5-shot)": mmlu_acc,
                "GSM8K (5-shot)": gsm8k_acc,
                "ARC-c (5-shot)": arc_acc
            })
            
    # Print formatted Markdown table
    print("| Model | MMLU (5-shot) | GSM8K (5-shot) | ARC-c (5-shot) |")
    print("|-------|---------------|----------------|----------------|")
    for r in sorted(results, key=lambda x: x['Model']):
        print(f"| {r['Model']} | {r['MMLU (5-shot)']} | {r['GSM8K (5-shot)']} | {r['ARC-c (5-shot)']} |")

if __name__ == "__main__":
    parse_lm_eval_results()
