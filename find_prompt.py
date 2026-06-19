from transformers import AutoTokenizer

m1 = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
m2 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

prompts = [
    "The local baker arrived in London today.",
    "The chief executive arrived in London today.",
    "The young teacher arrived in London today.",
    "The famous actor arrived in London today.",
    "The new director arrived in London today.",
    "A tall teacher arrived in London today."
]

for p in prompts:
    t1 = m1.encode(p)
    t2 = m2.encode(p)
    if len(t1) == 9 and len(t2) == 9:
        print(f"FOUND MATCH: '{p}'")
