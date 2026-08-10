import os
import glob
import torch
from safetensors.torch import load_file

def analyze_alignment_penalty(checkpoint_dir):
    print(f"📦 Loading pruned tensors from directory: {checkpoint_dir}")
    
    st_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
    if not st_files:
        print(f"❌ No .safetensors files found in {checkpoint_dir}")
        return
        
    state_dict = {}
    for st_file in st_files:
        state_dict.update(load_file(st_file))
    
    alignments = [16, 32, 64]
    results = {a: {"wasted_params": 0, "padded_dims": 0} for a in alignments}
    total_pruned_params = 0
    
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or len(tensor.shape) < 2:
            continue
            
        # Target weight matrices from MLP and Attention blocks
        if "weight" in name and any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            out_features, in_features = tensor.shape
            total_pruned_params += tensor.numel()
            
            for align in alignments:
                # Calculate required padding for output features
                rem_out = out_features % align
                pad_out = (align - rem_out) if rem_out != 0 else 0
                
                # Calculate required padding for input features
                rem_in = in_features % align
                pad_in = (align - rem_in) if rem_in != 0 else 0
                
                # Compute total parameters wasted due to padding to hit alignment
                padded_elements = ((out_features + pad_out) * (in_features + pad_in)) - (out_features * in_features)
                results[align]["wasted_params"] += padded_elements
                if pad_out > 0 or pad_in > 0:
                    results[align]["padded_dims"] += 1

    print("\n📊 --- HARDWARE ALIGNMENT PENALTY ANALYSIS ---")
    print(f"Current Raw Pruned Parameters: {total_pruned_params:,}")
    for align in alignments:
        wasted = results[align]["wasted_params"]
        pct_lost = (wasted / total_pruned_params) * 100
        print(f"\nTensor Core Alignment Base: {align}")
        print(f"  Padded Matrices: {results[align]['padded_dims']}")
        print(f"  Artificially Added Elements (Padding): {wasted:,}")
        print(f"  Theoretical Efficiency Lost to Alignment: {pct_lost:.2f}%")

if __name__ == "__main__":
    pruned_ckpt_dir = "./checkpoints/Meta-Llama-3-8B_subnetwork_pruned"
    if os.path.exists(pruned_ckpt_dir):
        analyze_alignment_penalty(pruned_ckpt_dir)
    else:
        print(f"❌ Checkpoint directory not found at {pruned_ckpt_dir}. Provide valid path.")
