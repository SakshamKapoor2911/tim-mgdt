"""
P8: Rotation Dose-Response Sweep (matched-angle control for Exp 1B).

Applies a random planar rotation R(theta) in SO(d) at the post-incision layer
(Layer 15) of the INTACT Mistral-7B-v0.1 model and measures GSM8K accuracy.

  R(theta) = I + (cos(theta) - 1) * P_plane + sin(theta) * Q_rot
  P_plane = u u^T + v v^T,  Q_rot = v u^T - u v^T
  (u, v) = random orthonormal pair  ->  det(R) = +1, geodesic angle = theta

Protocol mirrors Exp 1B: N=500, 8-shot CoT, greedy decoding, hook removed after
evaluation, no weights modified. Per-question correctness vectors are saved so
paired McNemar tests against the dense baseline remain computable.

Usage (split across GPUs via CUDA_VISIBLE_DEVICES):
  CUDA_VISIBLE_DEVICES=0 python -m code.phases.phase25_causal_interventions.8_rotation_dose_response \
      --jobs 5:0 15:0 28.71:0 28.71:1
  CUDA_VISIBLE_DEVICES=1 python -m code.phases.phase25_causal_interventions.8_rotation_dose_response \
      --jobs 28.71:2 45:0 60:0 90:0
"""

import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase25_causal_interventions.causal_hooks import register_intervention
from code.phases.phase11.utils import get_device, save_json
from code.phases.phase25_causal_interventions.gsm8k_eval_shared import evaluate_gsm8k


def random_planar_rotation(dim, angle_deg, seed):
    """SO(d) rotation by angle_deg inside a random 2-plane; identity elsewhere."""
    rng = np.random.RandomState(seed)
    G = rng.randn(dim, 2)
    Q, _ = np.linalg.qr(G)
    u = Q[:, 0]
    v = Q[:, 1]
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    P_plane = np.outer(u, u) + np.outer(v, v)
    Q_rot = np.outer(v, u) - np.outer(u, v)
    R = np.eye(dim) + (c - 1.0) * P_plane + s * Q_rot
    R = torch.from_numpy(R).float()
    assert abs(torch.slogdet(R.double())[0].item() - 1.0) < 1e-4, "planar rotation must be det +1"
    return R


def parse_jobs(job_specs):
    jobs = []
    for spec in job_specs:
        angle_str, seed_str = spec.split(":")
        jobs.append((float(angle_str), int(seed_str)))
    return jobs


def run_job(args, angle_deg, seed):
    tag = f"angle_{angle_deg}_seed_{seed}"
    out_path = os.path.join(args.output_dir, f"exp_8_dose_{tag}.json")
    if os.path.exists(out_path) and not args.force:
        print(f"[{tag}] already exists; skipping (--force to rerun).")
        return json.load(open(out_path))

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{tag}] loading intact model on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True,
    )
    hook, handle = register_intervention(model, args.rotation_layer)

    R = random_planar_rotation(model.config.hidden_size, angle_deg, seed)
    hook.set_rotation(R)

    print(f"[{tag}] evaluating GSM8K N={args.num_eval_samples} 8-shot ...")
    acc, vec, ci_low, ci_high, correct, total = evaluate_gsm8k(
        model, tokenizer, device, args.num_eval_samples, num_shots=8
    )
    print(f"[{tag}] GSM8K accuracy: {acc:.2f}% [{ci_low:.1f}, {ci_high:.1f}]")

    handle.remove()
    del model, hook
    torch.cuda.empty_cache()

    result = {
        "experiment": "8_rotation_dose_response",
        "model": args.model_name,
        "rotation_layer": args.rotation_layer,
        "angle_deg": angle_deg,
        "seed": seed,
        "acc": acc,
        "ci_95": [ci_low, ci_high],
        "correct": correct,
        "total": total,
        "vec": vec,
        "num_eval_samples": args.num_eval_samples,
    }
    save_json(out_path, result)
    print(f"[{tag}] saved -> {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="P8: Rotation dose-response sweep")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/phase25_causal_interventions")
    parser.add_argument("--rotation_layer", type=int, default=15)
    parser.add_argument("--num_eval_samples", type=int, default=500)
    parser.add_argument("--jobs", type=str, nargs="+",
                        default=["5:0", "15:0", "28.71:0", "28.71:1"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = parse_jobs(args.jobs)
    results = []
    for angle_deg, seed in jobs:
        results.append(run_job(args, angle_deg, seed))

    print("\n" + "=" * 60)
    for r in results:
        print(f"  angle={r['angle_deg']:>6} seed={r['seed']} -> acc={r['acc']:.2f}% "
              f"[{r['ci_95'][0]:.1f}, {r['ci_95'][1]:.1f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
