import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import torch.nn as nn

def extract_attention_to_sink(model, input_ids):
    """
    Extracts the average attention mass directed to the BOS (sink) token for each layer.
    A spike in sink attention indicates the layer is highly uncertain or structurally disrupted.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
    
    attentions = outputs.attentions # Tuple of (batch, num_heads, seq_len, seq_len) per layer
    if not attentions:
        raise ValueError("Model did not return output_attentions. Ensure attn_implementation='eager' is used.")
    
    sink_mass_per_layer = []
    for layer_attn in attentions:
        # Average across batch and heads
        # layer_attn: [1, 32, seq, seq]
        avg_attn = layer_attn[0].mean(dim=0) # [seq, seq]
        
        # Attention to BOS (token 0) across all query tokens (ignoring the BOS query itself)
        # We average how much tokens 1..N attend to token 0
        sink_mass = avg_attn[1:, 0].mean().item()
        sink_mass_per_layer.append(sink_mass)
        
    return sink_mass_per_layer

def main():
    print("Loading Mistral-7B...")
    model_id = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
    )
    
    print("Preparing standard sequence (2048 tokens)...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(dataset["text"][:200])
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(model.device)
    
    print("Measuring PRE-drop sink attention...")
    pre_sink_mass = extract_attention_to_sink(model, inputs.input_ids)
    
    print("Dropping layers 14-17...")
    drop_indices = [14, 15, 16, 17]
    old_layers = model.model.layers
    new_layers = nn.ModuleList([l for i, l in enumerate(old_layers) if i not in drop_indices])
    for new_idx, layer in enumerate(new_layers):
         layer.self_attn.layer_idx = new_idx
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)
    
    print("Measuring POST-drop sink attention...")
    post_sink_mass = extract_attention_to_sink(model, inputs.input_ids)
    
    # Align the post-drop layers with original layer indices for plotting
    post_aligned = []
    curr_idx = 0
    for i in range(32):
        if i in drop_indices:
            post_aligned.append(float('nan'))
        else:
            post_aligned.append(post_sink_mass[curr_idx])
            curr_idx += 1
            
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    
    layers = list(range(32))
    plt.plot(layers, pre_sink_mass, marker='o', label='Pre-Drop (Baseline)', color='blue', linewidth=2)
    plt.plot(layers, post_aligned, marker='x', label='Post-Drop (Unhealed)', color='red', linewidth=2, linestyle='--')
    
    # Highlight incision zone
    plt.axvspan(13.5, 17.5, color='gray', alpha=0.3, label='Incision Zone (Dropped)')
    
    # Highlight boundary layers (13 and 18, which are plotted at x=13 and x=18)
    plt.axvline(x=13, color='green', linestyle=':', linewidth=2, label='Boundary Layer (Left)')
    plt.axvline(x=18, color='green', linestyle=':', linewidth=2, label='Boundary Layer (Right)')
    
    plt.title('Mechanistic Disruption: Attention Mass Directed to Sink (BOS) Token', fontsize=14, fontweight='bold')
    plt.xlabel('Transformer Layer Depth', fontsize=12)
    plt.ylabel('Average Attention to Sink Token', fontsize=12)
    plt.xticks(layers[::2])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    
    os.makedirs('paper/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('paper/figures/attention_heatmap.png', dpi=300)
    print("Saved paper/figures/attention_heatmap.png")

if __name__ == "__main__":
    main()
