"""Llama-3-8B causal sufficiency cross-family check (N=100, 8-shot).

Computes the Procrustes rotation R* between intact L12 and pruned (drop 8-11)
post-incision hidden states, applies (R*)^T to the INTACT model at L12, and
evaluates GSM8K N=100 alongside a dense baseline.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. venv/bin/python \
      scratch/llama_sufficiency_n100.py
"""
import importlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase25_causal_interventions.causal_hooks import (
    compute_procrustes_rotation,
    register_intervention,
)
from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k
from code.phases.phase11.utils import get_device, save_json

_procrustes_mod = importlib.import_module(
    "code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation")
collect_hidden_at_layer = _procrustes_mod.collect_hidden_at_layer
_drop_layers = _procrustes_mod._drop_layers

MODEL = "meta-llama/Meta-Llama-3-8B"
DROP_SET = list(range(8, 12))
POST_INCISION = 12
PRUNED_POST = 8
NUM_CALIBRATION = 50
NUM_EVAL = 100


def main():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Collecting intact L12 hidden states...")
    intact_hidden = collect_hidden_at_layer(MODEL, None, POST_INCISION, tokenizer,
                                            NUM_CALIBRATION)
    print("Collecting pruned (drop 8-11) hidden states at index 8...")
    pruned_hidden = collect_hidden_at_layer(MODEL, DROP_SET, PRUNED_POST, tokenizer,
                                            NUM_CALIBRATION)

    print("Computing Procrustes rotation R*...")
    R_star = compute_procrustes_rotation(intact_hidden, pruned_hidden)

    print("Dense baseline N=%d..." % NUM_EVAL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True)
    dense_acc, dense_vec, dense_lo, dense_hi, _, _ = evaluate_gsm8k(
        model, tokenizer, device, NUM_EVAL, num_shots=8)
    print(f"  dense: {dense_acc:.2f}% [{dense_lo:.1f}, {dense_hi:.1f}]")

    print("Applying (R*)^T at L12 of intact model...")
    hook, handle = register_intervention(model, POST_INCISION)
    hook.set_rotation(R_star.T)
    rot_acc, rot_vec, rot_lo, rot_hi, _, _ = evaluate_gsm8k(
        model, tokenizer, device, NUM_EVAL, num_shots=8)
    handle.remove()
    print(f"  (R*)^T: {rot_acc:.2f}% [{rot_lo:.1f}, {rot_hi:.1f}]")

    results = {
        "experiment": "1B_llama_sufficiency_n100",
        "model": MODEL,
        "drop_indices": DROP_SET,
        "rotation_layer": POST_INCISION,
        "num_eval_samples": NUM_EVAL,
        "num_calibration_prompts": NUM_CALIBRATION,
        "dense_baseline": dense_acc,
        "dense_ci_95": [dense_lo, dense_hi],
        "acc_rotated_sufficiency": rot_acc,
        "rotated_ci_95": [rot_lo, rot_hi],
        "dense_per_prompt": dense_vec,
        "rotated_per_prompt": rot_vec,
    }
    save_json("experiments/phase25_causal_interventions/exp_1b_llama_sufficiency_n100.json",
              results)
    print("Saved.")


if __name__ == "__main__":
    main()
