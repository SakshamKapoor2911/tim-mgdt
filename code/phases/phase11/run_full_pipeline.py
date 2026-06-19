"""
Phase 11: Advanced Subnetwork & Compression Optimizations - Pipeline Orchestrator

Runs all 4 stages with REAL model computation:
  1. Structured GQA-Aware Head Pruning (measured PPL)
  2. Iterative Magnitude Pruning search (measured PPL at each sparsity)
  3. Causal Activation Patching (real forward-hook interventions)
  4. Activation Healing (bias correction + measured deployment metrics)

This pipeline REQUIRES a real model.  If the model fails to load, execution
stops with a clear error.  No synthetic/fake fallbacks.
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase11.stage1_structured_pruning import run_stage1
from code.phases.phase11.stage2_imp_search import run_stage2
from code.phases.phase11.stage3_causal_patching import run_stage3
from code.phases.phase11.stage4_activation_healing import run_stage4
from code.phases.phase11.utils import get_device, save_json

MODEL_REGISTRY: Dict[str, str] = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "gpt2": "gpt2",
}


def run_full_pipeline(
    model_key: str = "mistral-7b",
    num_samples: int = 128,
    output_dir: str = "experiments/phase11",
    device: Optional[str] = None,
    ppl_threshold_pct: float = 5.0,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Run all 4 stages of the Phase 11 real pruning pipeline.

    Args:
        model_key:        One of 'mistral-7b', 'llama-3-8b', 'gpt2'
        num_samples:      Max Wikitext-2 chunks for perplexity evaluation
        output_dir:       Where to save results JSON
        device:           Force device ('cuda'/'cpu'), or auto-detect
        ppl_threshold_pct: Max acceptable PPL increase for IMP winner (Stage 2)
        batch_size:       Batch size for PPL evaluation (default: 4 to saturate VRAM)

    Raises:
        RuntimeError: If model fails to load (no synthetic fallback)
    """
    if device is None:
        device = get_device()

    timestamp = datetime.now().isoformat()
    hf_id = MODEL_REGISTRY.get(model_key.lower())
    if hf_id is None:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    print("\n" + "=" * 70)
    print("PHASE 11: REAL PRUNING & MEASUREMENT PIPELINE")
    print("=" * 70)
    print(f"  Model:            {model_key} ({hf_id})")
    print(f"  Device:           {device}")
    print(f"  PPL samples:      {num_samples}")
    print(f"  PPL threshold:    {ppl_threshold_pct}%")
    print(f"  Output:           {output_dir}")
    print("=" * 70 + "\n")

    # --- Load model (fail loudly, no fallback) ---
    print(f"[LOAD] Loading {model_key} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    # Set pad token if missing (GPT-2 doesn't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

    if device == "cpu":
        model = model.to(device)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model loaded: {total_params/1e6:.1f}M parameters on {device}\n")

    # --- Run pipeline ---
    pipeline_start = time.time()

    # Stage 1: Structured head pruning
    stage1_res = run_stage1(
        model, tokenizer, device,
        model_name=model_key,
        ppl_max_samples=num_samples,
        batch_size=batch_size,
    )

    # Stage 2: IMP search
    stage2_res = run_stage2(
        model, tokenizer, device,
        stage1_results=stage1_res,
        ppl_max_samples=num_samples,
        ppl_threshold_pct=ppl_threshold_pct,
        batch_size=batch_size,
    )

    # Stage 3: Causal activation patching
    stage3_res = run_stage3(model, tokenizer, device)

    # Stage 4: Activation healing + deployment metrics
    target_sparsity = stage2_res.get("best_winner", {}).get("target_sparsity", 0.30)
    stage4_res = run_stage4(
        model, tokenizer, device,
        target_sparsity=target_sparsity,
        ppl_max_samples=num_samples,
        batch_size=batch_size,
    )

    total_elapsed = time.time() - pipeline_start

    # --- Assemble results ---
    results = {
        "timestamp": timestamp,
        "model_key": model_key,
        "hf_id": hf_id,
        "execution_type": "REAL_GPU" if device == "cuda" else "REAL_CPU",
        "device": device,
        "total_params": total_params,
        "ppl_max_samples": num_samples,
        "ppl_threshold_pct": ppl_threshold_pct,
        "total_pipeline_time_seconds": total_elapsed,
        "stages": {
            "stage1_structured_pruning": stage1_res,
            "stage2_imp_search": stage2_res,
            "stage3_causal_patching": stage3_res,
            "stage4_activation_healing": stage4_res,
        },
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{model_key.lower().replace('-', '_')}_phase11_results.json"
    save_json(str(out_file), results)

    print("\n" + "=" * 70)
    print("PHASE 11 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time:  {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results:     {out_file}")
    print(f"  All metrics: MEASURED (zero hardcoded values)")
    print("=" * 70 + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 11: Real Pruning & Measurement Pipeline"
    )
    parser.add_argument(
        "--model", type=str, default="gpt2",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to evaluate (default: gpt2 for fast testing)",
    )
    parser.add_argument(
        "--samples", type=int, default=128,
        help="Max Wikitext-2 chunks for PPL evaluation (default: 128)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/phase11",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device (cuda/cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--ppl-threshold", type=float, default=5.0,
        help="Max acceptable PPL increase %% for IMP winning ticket (default: 5.0)",
    )

    args = parser.parse_args()

    run_full_pipeline(
        model_key=args.model,
        num_samples=args.samples,
        output_dir=args.output_dir,
        device=args.device,
        ppl_threshold_pct=args.ppl_threshold,
    )


if __name__ == "__main__":
    main()
