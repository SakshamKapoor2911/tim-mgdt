"""E1 at full power: ablate PC1 at L15 of the intact Mistral-7B model, N=500.

Companion to exp_6_pc_ablation.json (L10 ablation). The N=500 intact baseline
per-prompt vector is reused from exp_6_pc_ablation.json for the paired McNemar
(both harnesses use seed=None, i.e., the same first-500 GSM8K questions).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. venv/bin/python \
      scratch/pc1_l15_n500.py
"""
import importlib
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k
from code.phases.phase11.utils import get_device, save_json

_procrustes_mod = importlib.import_module(
    "code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation")
collect_hidden_at_layer = _procrustes_mod.collect_hidden_at_layer

_pc_mod = importlib.import_module("scratch.pc1_l15_interventions")
PCAblationHook = _pc_mod.PCAblationHook
pc1_direction = _pc_mod.pc1_direction

MODEL = "mistralai/Mistral-7B-v0.1"
POST_INCISION = 15
NUM_EVAL = 500


def main():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Collecting intact L15 hidden states (50 passages)...")
    intact_hidden = collect_hidden_at_layer(MODEL, None, POST_INCISION, tokenizer, 50)
    v1 = pc1_direction(intact_hidden)
    print(f"PC1 direction: {v1.shape}")

    print("E1: ablate PC1 at L15 (intact), N=500...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True)
    handle = model.model.layers[POST_INCISION].register_forward_hook(
        PCAblationHook(v1))
    acc, vec, lo, hi, correct, total = evaluate_gsm8k(
        model, tokenizer, device, NUM_EVAL, num_shots=8)
    handle.remove()
    del model
    torch.cuda.empty_cache()

    results = {
        "experiment": "6B_pc1_l15_n500",
        "model": MODEL,
        "layer": POST_INCISION,
        "num_eval_samples": NUM_EVAL,
        "ablate_pc1_l15": {"accuracy": acc, "ci_95": [lo, hi],
                           "correct": correct, "total": total, "per_prompt": vec},
    }

    base = json.load(open("experiments/phase25_causal_interventions/exp_6_pc_ablation.json"))
    base_vec = base["per_prompt_vectors"]["intact_baseline"]
    if len(base_vec) != NUM_EVAL:
        raise ValueError(f"baseline vector length {len(base_vec)} != {NUM_EVAL}")
    b = sum(1 for x, y in zip(base_vec, vec) if x and not y)
    c = sum(1 for x, y in zip(base_vec, vec) if y and not x)
    from scipy.stats import binom
    p = min(1.0, float(2.0 * binom.cdf(min(b, c), b + c, 0.5)))
    results["mcnemar_vs_intact_baseline"] = {"b": b, "c": c, "n": b + c, "p": p}
    print(f"E1: {acc:.2f}% [{lo:.1f}, {hi:.1f}]  "
          f"McNemar vs baseline: b={b} c={c} p={p:.3e}")

    save_json("experiments/phase25_causal_interventions/exp_6b_pc1_l15_n500.json",
              results)
    print("Saved.")


if __name__ == "__main__":
    main()
