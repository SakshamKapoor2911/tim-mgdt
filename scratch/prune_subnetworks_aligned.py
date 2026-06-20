import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import gc
import os
import math
import argparse

def get_calibration_data(num_samples=120):
    try:
        gsm8k = load_dataset("gsm8k", "main", split="train", streaming=True)
        texts = []
        for i, item in enumerate(gsm8k):
            if i >= num_samples: break
            texts.append(f"Question: {item['question']}\nAnswer: {item['answer']}")
    except:
        texts = ["Explain the theory of relativity." * 10 for _ in range(num_samples)]
    return texts

def setup_masks(model):
    head_masks = {}
    mlp_masks = {}
    hooks = []
    
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    intermediate_size = model.config.intermediate_size
    
    model.requires_grad_(False)
    
    def get_head_hook(name):
        def hook(module, inputs):
            output = inputs[0]
            mask = head_masks[name].to(torch.bfloat16).view(1, 1, num_heads, 1)
            batch, seq, hidden = output.shape
            reshaped = output.view(batch, seq, num_heads, head_dim)
            masked = reshaped * mask
            return (masked.view(batch, seq, hidden),)
        return hook

    def get_mlp_hook(name):
        def hook(module, inputs):
            mask = mlp_masks[name].to(torch.bfloat16).view(1, 1, intermediate_size)
            masked = inputs[0] * mask
            return (masked,)
        return hook

    for i, layer in enumerate(model.model.layers):
        layer_device = next(layer.parameters()).device
        h_mask = nn.Parameter(torch.ones(num_heads, dtype=torch.float32, device=layer_device))
        m_mask = nn.Parameter(torch.ones(intermediate_size, dtype=torch.float32, device=layer_device))
        
        head_masks[i] = h_mask
        mlp_masks[i] = m_mask
        
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(get_head_hook(i)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(get_mlp_hook(i)))
        
    return head_masks, mlp_masks, hooks

def calculate_global_fim(model, tokenizer, texts, head_masks, mlp_masks):
    fim_head = {i: torch.zeros_like(head_masks[i]).float() for i in head_masks}
    fim_mlp = {i: torch.zeros_like(mlp_masks[i]).float() for i in mlp_masks}
    
    for mask in head_masks.values():
        if mask.grad is not None: mask.grad.zero_()
    for mask in mlp_masks.values():
        if mask.grad is not None: mask.grad.zero_()

    model.eval()
    for idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        
        with torch.no_grad():
            for i in head_masks:
                if head_masks[i].grad is not None:
                    fim_head[i] += (head_masks[i].grad.float() ** 2)
                    head_masks[i].grad.zero_()
            for i in mlp_masks:
                if mlp_masks[i].grad is not None:
                    fim_mlp[i] += (mlp_masks[i].grad.float() ** 2)
                    mlp_masks[i].grad.zero_()
        
        del loss, outputs, inputs
        torch.cuda.empty_cache()
        gc.collect()

    head_saliency = {}
    mlp_saliency = {}
    
    for i in head_masks:
        head_saliency[i] = ((head_masks[i].float() ** 2) * fim_head[i]).detach().cpu().numpy()
    for i in mlp_masks:
        mlp_saliency[i] = ((mlp_masks[i].float() ** 2) * fim_mlp[i]).detach().cpu().numpy()
        
    return head_saliency, mlp_saliency

def align_dimension(target_dim: int, alignment_base: int = 64, strategy: str = "nearest") -> int:
    if target_dim % alignment_base == 0:
        return target_dim
    if strategy == "ceil":
        return int(math.ceil(target_dim / alignment_base) * alignment_base)
    elif strategy == "floor":
        return int(math.floor(target_dim / alignment_base) * alignment_base)
    elif strategy == "nearest":
        return int(round(target_dim / alignment_base) * alignment_base)
    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")

def physical_prune_aligned(model, head_saliency, mlp_saliency, prune_ratio=0.20, alignment_base=64, strategy="nearest"):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    groups = num_heads // num_kv_heads
    
    print(f"\n--- Physical Pruning Phase (Alignment Base: {alignment_base}, Strategy: {strategy}) ---")
    
    # 1. MLP Global Pruning
    all_mlp_sal = np.concatenate([mlp_saliency[i] for i in range(num_layers)])
    mlp_threshold = np.percentile(all_mlp_sal, prune_ratio * 100)
    
    # 2. GQA Group Saliency (Average of 4 query heads in each group)
    group_saliency = {}
    for i in range(num_layers):
        group_sal = []
        for g in range(num_kv_heads):
            g_sal = np.mean(head_saliency[i][g*groups : (g+1)*groups])
            group_sal.append(g_sal)
        group_saliency[i] = np.array(group_sal)
        
    all_group_sal = np.concatenate([group_saliency[i] for i in range(num_layers)])
    group_threshold = np.percentile(all_group_sal, prune_ratio * 100)
    
    total_original_params = 0
    total_pruned_params = 0
    
    for i in range(num_layers):
        layer = model.model.layers[i]
        
        # --- Slice MLPs ---
        # Get raw indices above threshold
        sorted_mlp_indices = np.argsort(mlp_saliency[i])[::-1]
        raw_keep_count = np.sum(mlp_saliency[i] > mlp_threshold)
        
        # Apply Hardware Alignment snap
        aligned_keep_count = align_dimension(raw_keep_count, alignment_base, strategy)
        aligned_keep_count = max(alignment_base, min(aligned_keep_count, model.config.intermediate_size))
        
        keep_mlp = np.sort(sorted_mlp_indices[:aligned_keep_count])
        new_inter_size = len(keep_mlp)
        
        if new_inter_size < model.config.intermediate_size:
            with torch.no_grad():
                layer.mlp.gate_proj.weight = nn.Parameter(layer.mlp.gate_proj.weight[keep_mlp, :])
                layer.mlp.up_proj.weight = nn.Parameter(layer.mlp.up_proj.weight[keep_mlp, :])
                layer.mlp.down_proj.weight = nn.Parameter(layer.mlp.down_proj.weight[:, keep_mlp])
                
        # --- Slice Attention (GQA Groups) ---
        # GQA groups automatically align to 4 * head_dim (which is 512, always multiple of 64/128)
        keep_groups = np.where(group_saliency[i] > group_threshold)[0]
        new_kv_heads = len(keep_groups)
        new_q_heads = new_kv_heads * groups
        
        if new_kv_heads < num_kv_heads:
            keep_q_heads = []
            for g in keep_groups:
                keep_q_heads.extend(list(range(g*groups, (g+1)*groups)))
            
            with torch.no_grad():
                qw = layer.self_attn.q_proj.weight.view(num_heads, head_dim, -1)
                layer.self_attn.q_proj.weight = nn.Parameter(qw[keep_q_heads].reshape(new_q_heads * head_dim, -1))
                
                kw = layer.self_attn.k_proj.weight.view(num_kv_heads, head_dim, -1)
                layer.self_attn.k_proj.weight = nn.Parameter(kw[keep_groups].reshape(new_kv_heads * head_dim, -1))
                
                vw = layer.self_attn.v_proj.weight.view(num_kv_heads, head_dim, -1)
                layer.self_attn.v_proj.weight = nn.Parameter(vw[keep_groups].reshape(new_kv_heads * head_dim, -1))
                
                ow = layer.self_attn.o_proj.weight.view(-1, num_heads, head_dim)
                layer.self_attn.o_proj.weight = nn.Parameter(ow[:, keep_q_heads, :].reshape(-1, new_q_heads * head_dim))
                
                layer.self_attn.num_heads = new_q_heads
                layer.self_attn.num_key_value_heads = new_kv_heads
                
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment_base", type=int, default=64)
    parser.add_argument("--strategy", type=str, default="nearest")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/Meta-Llama-3-8B_subnetwork_pruned_aligned")
    args = parser.parse_args()
    
    print(f"Loading Llama-3-8B for Global FIM Calibration and Aligned Structural Pruning...")
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    head_masks, mlp_masks, hooks = setup_masks(model)
    
    print("Loading 120 calibration samples...")
    texts = get_calibration_data(120)
    
    print("Calculating Global FIM Saliency...")
    h_sal, m_sal = calculate_global_fim(model, tokenizer, texts, head_masks, mlp_masks)
    
    for h in hooks: h.remove()
    
    print("Executing Physical Aligned Subnetwork Pruning (20% Target)...")
    pruned_model = physical_prune_aligned(
        model, 
        h_sal, 
        m_sal, 
        prune_ratio=0.20, 
        alignment_base=args.alignment_base, 
        strategy=args.strategy
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving aligned pruned model to {args.output_dir}...")
    pruned_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Complete!")

if __name__ == "__main__":
    main()
