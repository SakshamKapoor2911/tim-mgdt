import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
import gc

def get_calibration_data(num_samples=120):
    try:
        # Load reasoning traces
        gsm8k = load_dataset("gsm8k", "main", split="train", streaming=True)
        texts = []
        for i, item in enumerate(gsm8k):
            if i >= num_samples: break
            texts.append(f"Question: {item['question']}\nAnswer: {item['answer']}")
    except Exception as e:
        print("Failed to load HF dataset, using fallback dummy text.", e)
        texts = ["Explain the theory of relativity." * 10 for _ in range(num_samples)]
        
    return texts

def setup_masks(model):
    head_masks = {}
    mlp_masks = {}
    hooks = []
    
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    intermediate_size = model.config.intermediate_size
    
    # 1. Strict Graph Isolation
    model.requires_grad_(False)
    
    def get_head_hook(name):
        def hook(module, inputs):
            output = inputs[0]
            # output is (batch, seq, hidden_size)
            # we need to apply head_masks[name] of shape (num_heads,)
            mask = head_masks[name].to(torch.bfloat16).view(1, 1, num_heads, 1)
            # reshape output to (batch, seq, num_heads, head_dim)
            batch, seq, hidden = output.shape
            reshaped = output.view(batch, seq, num_heads, head_dim)
            masked = reshaped * mask
            return (masked.view(batch, seq, hidden),)
        return hook

    def get_mlp_hook(name):
        def hook(module, inputs):
            # output of activation function or input to down_proj is (batch, seq, intermediate_size)
            # wait, down_proj input is a tuple of args in hook? inputs[0] is the tensor
            # If we hook the down_proj forward pre-hook, we modify inputs[0]
            # Actually, standard forward hook on gate_proj/up_proj is hard. 
            # Best is a forward_pre_hook on down_proj
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

def calculate_split_saliency(model, tokenizer, texts, head_masks, mlp_masks):
    # Initialize FIM accumulators
    fim_head = {i: torch.zeros_like(head_masks[i]).float() for i in head_masks}
    fim_mlp = {i: torch.zeros_like(mlp_masks[i]).float() for i in mlp_masks}
    
    # Reset gradients
    for mask in head_masks.values():
        if mask.grad is not None:
            mask.grad.zero_()
    for mask in mlp_masks.values():
        if mask.grad is not None:
            mask.grad.zero_()

    model.eval()
    
    for idx, text in enumerate(texts):
        # 2. VRAM Fragmentation Control: micro-batch size 1
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        loss.backward()
        
        # Accumulate squared gradients for FIM
        with torch.no_grad():
            for i in head_masks:
                if head_masks[i].grad is not None:
                    fim_head[i] += (head_masks[i].grad.float() ** 2)
                    head_masks[i].grad.zero_()
            for i in mlp_masks:
                if mlp_masks[i].grad is not None:
                    fim_mlp[i] += (mlp_masks[i].grad.float() ** 2)
                    mlp_masks[i].grad.zero_()
        
        # Free graph activations immediately
        del loss
        del outputs
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate FIM-based Saliency: S = (Mask^2) * FIM
    head_saliency = {}
    mlp_saliency = {}
    
    for i in head_masks:
        head_saliency[i] = ((head_masks[i].float() ** 2) * fim_head[i]).detach().cpu().numpy()
        
    for i in mlp_masks:
        mlp_saliency[i] = ((mlp_masks[i].float() ** 2) * fim_mlp[i]).detach().cpu().numpy()
        
    return head_saliency, mlp_saliency

def main():
    print("Loading Llama-3-8B in bfloat16 for Empirical FIM calibration...")
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
    splits = [texts[0:40], texts[40:80], texts[80:120]]
    
    split_results_heads = []
    split_results_mlps = []
    
    for i, split_texts in enumerate(splits):
        print(f"Processing Split {i+1}/3 ({len(split_texts)} samples)...")
        h_sal, m_sal = calculate_split_saliency(model, tokenizer, split_texts, head_masks, mlp_masks)
        split_results_heads.append(h_sal)
        split_results_mlps.append(m_sal)
        
    # Remove hooks
    for h in hooks:
        h.remove()
        
    print("\nCalculating Spearman Rank Correlation across splits...")
    
    def check_correlation(results, name):
        # Flatten the arrays across all layers for global ranking correlation
        s1 = np.concatenate([results[0][i] for i in range(len(results[0]))])
        s2 = np.concatenate([results[1][i] for i in range(len(results[1]))])
        s3 = np.concatenate([results[2][i] for i in range(len(results[2]))])
        
        rho_12, _ = spearmanr(s1, s2)
        rho_23, _ = spearmanr(s2, s3)
        rho_13, _ = spearmanr(s1, s3)
        
        avg_rho = (rho_12 + rho_23 + rho_13) / 3.0
        print(f"[{name}] Correlation -> 1v2: {rho_12:.3f} | 2v3: {rho_23:.3f} | 1v3: {rho_13:.3f} | AVG: {avg_rho:.3f}")
        return avg_rho

    avg_h = check_correlation(split_results_heads, "Attention Heads")
    avg_m = check_correlation(split_results_mlps, "MLP Dimensions")
    
    if avg_h < 0.85 or avg_m < 0.85:
        print("\nWARNING: Variance check failed (\u03c1 < 0.85) even with FIM. Data diversity is still insufficient.")
    else:
        print("\nSUCCESS: The 120-sample Empirical FIM masks are robust (\u03c1 >= 0.85).")
        print("Proceeding to structural masking pipeline.")

if __name__ == "__main__":
    main()
