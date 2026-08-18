"""Verify unmodified dense Gemma-2-2B C4 PPL under the current harness."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from code.phases.phase11.utils import get_device, compute_perplexity_c4

model_name = "google/gemma-2-2b"
device = get_device()
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
model.eval()
ppl = compute_perplexity_c4(model, tokenizer, device, max_samples=128)
print(f"Unmodified dense Gemma-2-2B C4 PPL (128 samples): {ppl:.4f}")
