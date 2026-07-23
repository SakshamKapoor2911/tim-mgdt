import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import torch
import torch.nn as nn
import importlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from code.phases.phase11.utils import get_device, compute_perplexity_c4

class IdentityAttn(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states, None

def prune_attention_only(model, drop_indices):
    for idx in drop_indices:
        model.model.layers[idx].self_attn = IdentityAttn()
    return model

def main():
    model_name = "mistralai/Mistral-7B-v0.1"
    drop_indices = [11, 12, 13, 14]
    device = get_device()

    print("============================================================")
    print("Full 1,000-Step Attention-Only Incision Pruning Run")
    print("============================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("1. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    print(f"2. Pruning Attention-Only (replacing MHSA at layers {drop_indices})...")
    model = prune_attention_only(model, drop_indices)

    # Apply LoRA on boundary layers (9, 10, 11, 12, 13, 14, 15, 16)
    boundary_layers = [9, 10, 11, 12, 13, 14, 15, 16]
    target_modules = [f"model.layers.{i}.mlp.down_proj" for i in boundary_layers] + \
                     [f"model.layers.{i}.mlp.up_proj" for i in boundary_layers] + \
                     [f"model.layers.{i}.mlp.gate_proj" for i in boundary_layers]

    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"3. Boundary LoRA initialized. Trainable parameters: {trainable_params:,}")

    # Prepare 1,000 training steps on Wikitext-2 / C4
    print("4. Preparing 1,000-step training dataset...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings["input_ids"])
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    loader = DataLoader(SimpleDataset(enc), batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    model.train()
    print("5. Launching 1,000-step Boundary LoRA training loop...")
    step = 0
    total_steps = 1000
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step % 100 == 0 or step == total_steps - 1:
                print(f" Step {step:4d}/{total_steps} | Loss: {loss.item():.4f}")
            step += 1

    print("\n6. Evaluating final C4 Perplexity...")
    model.eval()
    ppl = compute_perplexity_c4(model, tokenizer, device)
    print(f"============================================================")
    print(f" -> Final 1,000-Step Attention-Only Incision C4 PPL: {ppl:.4f}")
    print(f"============================================================")

    out_file = "experiments/phase15_layer_drop/attn_only_1000step_result.json"
    with open(out_file, "w") as f:
        json.dump({"model": model_name, "pruned_layers": drop_indices, "kv_cache_reduction": "33.3%", "final_c4_ppl": ppl}, f, indent=2)

if __name__ == "__main__":
    main()
