import json
import os

def main():
    print("Aggregating empirical results...")
    
    # 1. Parse Pareto Benchmark log output
    # For now, we will just format the table structure. 
    # Real numbers will be plugged in or parsed from stdout logs.
    
    eval_file_a = "./results/Meta-Llama-3-8B_sparse_healed_eval_A.json"
    eval_file_b = "./results/Meta-Llama-3-8B_sparse_healed_eval_B.json"
    eval_file_c = "./results/Meta-Llama-3-8B_sparse_healed_eval_C.json"
    
    mmlu_acc = "TBD"
    gsm8k_acc = "TBD"
    asdiv_acc = "TBD"
    
    if os.path.exists(eval_file_a):
        with open(eval_file_a, "r") as f:
            res_a = json.load(f)
        mmlu_acc = res_a.get("mmlu", {}).get("acc,none", "TBD")
        
    if os.path.exists(eval_file_b):
        with open(eval_file_b, "r") as f:
            res_b = json.load(f)
        gsm8k_acc = res_b.get("gsm8k", {}).get("exact_match,strict-match", "TBD")

    if os.path.exists(eval_file_c):
        with open(eval_file_c, "r") as f:
            res_c = json.load(f)
        asdiv_acc = res_c.get("asdiv_cot_llama", {}).get("exact_match,strict-match", "TBD")
        
    if isinstance(mmlu_acc, float): mmlu_acc = f"{mmlu_acc*100:.2f}\\%"
    if isinstance(gsm8k_acc, float): gsm8k_acc = f"{gsm8k_acc*100:.2f}\\%"
    if isinstance(asdiv_acc, float): asdiv_acc = f"{asdiv_acc*100:.2f}\\%"

    eval_file_ceil = "./results/Meta-Llama-3-8B_sparse_healed_eval_ceil.json"
    eval_file_floor = "./results/Meta-Llama-3-8B_sparse_healed_eval_floor.json"
    
    ceil_gsm8k = "TBD"
    floor_gsm8k = "TBD"
    
    if os.path.exists(eval_file_ceil):
        with open(eval_file_ceil, "r") as f:
            res_ceil = json.load(f)
        ceil_gsm8k = res_ceil.get("gsm8k", {}).get("exact_match,strict-match", "TBD")
        if isinstance(ceil_gsm8k, float): ceil_gsm8k = f"{ceil_gsm8k*100:.2f}\\%"
        
    if os.path.exists(eval_file_floor):
        with open(eval_file_floor, "r") as f:
            res_floor = json.load(f)
        floor_gsm8k = res_floor.get("gsm8k", {}).get("exact_match,strict-match", "TBD")
        if isinstance(floor_gsm8k, float): floor_gsm8k = f"{floor_gsm8k*100:.2f}\\%"

    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Empirical Capability Recovery and Hardware Alignment Penalty}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Model Pipeline}} & \\textbf{{Theoretical MAC Red.}} & \\textbf{{Native PyTorch TPS}} & \\textbf{{MMLU}} & \\textbf{{GSM8K}} & \\textbf{{ASDiv}} \\\\
\\midrule
Meta-Llama-3-8B (Dense Base) & - & 40.08 \\text{{t/s}} & 65.5\\% & 50.5\\% & 54.2\\% \\\\
\\midrule
Layer-Dropped (Unhealed) & -15.84\\% & +21.4\\% & 26.5\\% & 1.1\\% & 2.5\\% \\\\
Layer-Dropped (Boundary Healed) & -15.84\\% & +21.4\\% & 31.5\\% & 0.9\\% & 3.2\\% \\\\
\\midrule
\\textbf{{Subnetwork Pruned (Sparse LoRA)}} & -20.06\\% & +0.00\\% (39.94 \\text{{t/s}}) & {mmlu_acc} & {gsm8k_acc} & {asdiv_acc} \\\\
\\textbf{{`ceil` Alignment}} & -19.8\\% & TBD & TBD & {ceil_gsm8k} & TBD \\\\
\\textbf{{`floor` Alignment}} & -20.4\\% & TBD & TBD & {floor_gsm8k} & TBD \\\\
\\bottomrule
\\end{{tabular}}
\\label{{tab:pareto_recovery}}
\\end{{table}}
"""
    
    print("\n--- Final LaTeX Table for Manuscript ---\n")
    print(latex_table)
    
    with open("./paper/final_table.tex", "w") as f:
        f.write(latex_table)

if __name__ == "__main__":
    main()
