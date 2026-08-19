"""PC1 interventions at L15 (post-incision layer) on Mistral-7B-v0.1.

Closes the L10-vs-L15 layer-offset gap in the necessity arm:
  E1: ablate PC1 at L15 in the INTACT model (parallel to the L10 result).
  E3: compress PC1 variance at L15 in the INTACT model to the measured
      pruned level (the exact post-pruning PC1 energy collapse, ~4.21%).
  E2: restore PC1 variance at the pruned model's post-incision layer to the
      intact level (repair direction; null expected due to residual rotation).
Baselines: intact N=100 and pruned-unhealed N=100 (same 8-shot harness).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. venv/bin/python \
      scratch/pc1_l15_interventions.py --num_eval 100
"""
import argparse
import importlib
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k
from code.phases.phase11.utils import get_device, save_json

_procrustes_mod = importlib.import_module(
    "code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation")
collect_hidden_at_layer = _procrustes_mod.collect_hidden_at_layer
_drop_layers = _procrustes_mod._drop_layers

MODEL = "mistralai/Mistral-7B-v0.1"
DROP_SET = list(range(11, 15))
POST_INCISION = 15
PRUNED_POST = 11


class PCAblationHook:
    def __init__(self, v):
        self.v = v

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        v = self.v.to(h.device, dtype=h.dtype)
        proj = h @ v
        h_new = h - (proj @ v.transpose(0, 1))
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


class VarScaleHook:
    """Rescale hidden-state variance along direction v by factor alpha."""
    def __init__(self, v, alpha):
        self.v = v
        self.alpha = alpha

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        v = self.v.to(h.device, dtype=h.dtype)
        proj = h @ v
        h_new = h + (self.alpha - 1.0) * (proj @ v.transpose(0, 1))
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new


def pc1_direction(hidden):
    H = hidden.detach().float().reshape(-1, hidden.shape[-1]).cpu().numpy()
    H = H - np.mean(H, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return torch.from_numpy(Vt[:1].T).float()  # [d, 1]


def proj_std(hidden, v):
    H = hidden.detach().float().reshape(-1, hidden.shape[-1]).cpu().numpy()
    Hc = H - np.mean(H, axis=0, keepdims=True)
    return float(np.std(Hc @ v.cpu().numpy()))


def eval_with_hook(model, layer_idx, hook_obj, tokenizer, device, n):
    handle = model.model.layers[layer_idx].register_forward_hook(hook_obj)
    acc, vec, lo, hi, correct, total = evaluate_gsm8k(
        model, tokenizer, device, n, num_shots=8)
    handle.remove()
    return {"accuracy": acc, "ci_95": [lo, hi], "correct": correct,
            "total": total, "per_prompt": vec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eval", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="experiments/phase25_causal_interventions/exp_6b_pc1_l15.json")
    args = parser.parse_args()

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"experiment": "6B_pc1_l15", "model": MODEL,
               "drop_indices": DROP_SET, "intact_layer": POST_INCISION,
               "pruned_post_layer": PRUNED_POST, "num_eval_samples": args.num_eval}

    print("Collecting intact L15 hidden states...")
    intact_hidden = collect_hidden_at_layer(MODEL, None, POST_INCISION, tokenizer, 50)
    v1 = pc1_direction(intact_hidden)
    print("Collecting pruned (drop 11-14) hidden states at index 11...")
    pruned_hidden = collect_hidden_at_layer(MODEL, DROP_SET, PRUNED_POST, tokenizer, 50)

    std_intact = proj_std(intact_hidden, v1)
    std_pruned = proj_std(pruned_hidden, v1)
    alpha = std_pruned / std_intact
    collapse = 1.0 - alpha ** 2
    results["pc1_std_ratio_pruned_over_intact"] = round(alpha, 6)
    results["pc1_energy_collapse_from_calibration"] = round(collapse, 6)
    print(f"PC1 std ratio pruned/intact = {alpha:.5f} "
          f"(energy collapse {100 * collapse:.2f}%)")

    def load(drop=None):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True)
        if drop is not None:
            _drop_layers(model, drop)
        return model

    print("Baseline: intact N=%d..." % args.num_eval)
    model = load()
    results["intact_baseline"] = eval_with_hook(model, POST_INCISION,
                                                PCAblationHook(torch.zeros(v1.shape)),
                                                tokenizer, device, args.num_eval)
    del model
    torch.cuda.empty_cache()

    print("E1: ablate PC1 at L15 (intact)...")
    model = load()
    results["e1_ablate_pc1_l15_intact"] = eval_with_hook(
        model, POST_INCISION, PCAblationHook(v1), tokenizer, device, args.num_eval)
    del model
    torch.cuda.empty_cache()

    print(f"E3: compress PC1 at L15 to pruned level (alpha={alpha:.5f})...")
    model = load()
    results["e3_compress_pc1_l15_intact"] = eval_with_hook(
        model, POST_INCISION, VarScaleHook(v1, alpha), tokenizer, device, args.num_eval)
    del model
    torch.cuda.empty_cache()

    print("Baseline: pruned unhealed N=%d..." % args.num_eval)
    model = load(DROP_SET)
    results["pruned_baseline"] = eval_with_hook(model, PRUNED_POST,
                                                PCAblationHook(torch.zeros(v1.shape)),
                                                tokenizer, device, args.num_eval)
    del model
    torch.cuda.empty_cache()

    print(f"E2: restore PC1 at pruned index 11 (alpha={1.0 / alpha:.5f})...")
    model = load(DROP_SET)
    results["e2_restore_pc1_pruned"] = eval_with_hook(
        model, PRUNED_POST, VarScaleHook(v1, 1.0 / alpha), tokenizer, device, args.num_eval)
    del model
    torch.cuda.empty_cache()

    save_json(args.output, results)
    print("\nRESULTS")
    for k, v in results.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"  {k}: {v['accuracy']:.2f}% {v['ci_95']}")


if __name__ == "__main__":
    main()
