import sys
import os
import re
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from scipy.stats import beta as _beta_dist

# 8-shot GSM8K prompt (standard CoT exemplars)
GSM8K_PROMPT_8SHOT = """Q: There are 15 trees in the park. Park workers will plant trees in the park today. After they are done, there will be 21 trees in the park. How many trees did the park workers plant today?
A: There are 15 trees originally. Then more trees were planted. In total there are 21 trees. So the number of trees planted must be 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he has 12 now, he must have given Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn has 5 toys. He got 2 from mom and 2 from dad. 5 + 2 + 2 = 9. The answer is 9.

Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
A: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 on Wednesday, he had 35 - 2 = 33. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. She bought 5 bagels at 3 dollars each. 5 * 3 = 15 dollars spent. 23 - 15 = 8 dollars left. The answer is 8.

Q: {question}
A:"""

GSM8K_PROMPT_5SHOT = """Q: There are 15 trees in the park. Park workers will plant trees in the park today. After they are done, there will be 21 trees in the park. How many trees did the park workers plant today?
A: There are 15 trees originally. Then more trees were planted. In total there are 21 trees. So the number of trees planted must be 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he has 12 now, he must have given Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn has 5 toys. He got 2 from mom and 2 from dad. 5 + 2 + 2 = 9. The answer is 9.

Q: {question}
A:"""

def extract_strict_answer(text):
    text = text.split("Q:")[0].strip()
    match = re.search(r"[Tt]he answer is\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return ""

def clopper_pearson_ci(correct, total, alpha=0.05):
    """Two-sided Clopper-Pearson 95% CI (Beta-distribution quantiles)."""
    p = correct / total
    low = _beta_dist.ppf(alpha / 2, correct, total - correct + 1) if correct > 0 else 0.0
    high = _beta_dist.ppf(1 - alpha / 2, correct + 1, total - correct) if correct < total else 1.0
    return low * 100, high * 100

def evaluate_gsm8k(model, tokenizer, device, num_samples=100, num_shots=8, seed=None):
    prompt_template = GSM8K_PROMPT_8SHOT if num_shots == 8 else GSM8K_PROMPT_5SHOT
    dataset = load_dataset("gsm8k", "main", split="test")
    model.eval()
    correct = 0
    total = min(num_samples, len(dataset))
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(dataset), size=total, replace=False)
    else:
        indices = list(range(total))

    per_prompt_results = []
    for idx in tqdm(indices, desc=f"GSM8K {num_shots}-shot", leave=False):
        item = dataset[int(idx)]
        question = item["question"]
        gold_text = item["answer"]
        gold_ans = extract_strict_answer(gold_text)

        prompt = prompt_template.format(question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred_ans = extract_strict_answer(gen_text)

        is_correct = int(pred_ans.strip() == gold_ans.strip() and gold_ans.strip() != "")
        correct += is_correct
        per_prompt_results.append(is_correct)

    acc = (correct / total) * 100.0
    ci_low, ci_high = clopper_pearson_ci(correct, total)
    return acc, per_prompt_results, ci_low, ci_high, correct, total
