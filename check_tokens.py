from transformers import AutoTokenizer

models = ["mistralai/Mistral-7B-v0.1", "meta-llama/Meta-Llama-3-8B"]
prompts = [
    "The prime minister arrived in London today.",
    "The chef cooked a delicious pasta today."
]

for m in models:
    print(f"\nModel: {m}")
    try:
        tok = AutoTokenizer.from_pretrained(m)
        for p in prompts:
            tokens = tok.encode(p)
            print(f"Length {len(tokens)}: {tokens}")
    except Exception as e:
        print(f"Skipping {m}: {e}")
