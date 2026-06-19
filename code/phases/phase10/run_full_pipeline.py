"""Phase 10: Full Pipeline Orchestration"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from .stage1_identification import run_stage1
from .stage2_refinement import run_stage2
from .stage3_pattern_analysis import run_stage3
from .stage4_deployment_metrics import run_stage4
from .utils import get_device, save_json


def run_full_pipeline(
    model: str = "gpt2",
    num_samples: int = 500,
    output_dir: str = "experiments/phase10",
    phase5_dir: str = "experiments/phase5",
    stages: Optional[List[int]] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Run all 4 stages of Phase 10 pipeline.
    
    Args:
        model: Model name (gpt2, llama-7b, llama-3-8b, etc.)
        num_samples: Number of evaluation samples
        output_dir: Output directory for results
        phase5_dir: Directory containing Phase 5 results
        stages: List of stages to run (default: [1,2,3,4])
        device: Device to use (auto-detect if None)
    
    Returns:
        Dict with all stage results
    """
    if device is None:
        device = get_device()
    
    if stages is None:
        stages = [1, 2, 3, 4]
    
    print("\n" + "=" * 70)
    print("PHASE 10: LOTTERY TICKET HYPOTHESIS - FULL PIPELINE")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Samples: {num_samples}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Stages: {stages}")
    print("=" * 70 + "\n")
    
    results = {
        "model": model,
        "num_samples": num_samples,
        "device": device,
        "stages": {}
    }
    
    # Stage 1: Ticket Identification
    if 1 in stages:
        print("Running STAGE 1: Ticket Identification...")
        try:
            stage1_results = run_stage1()
            results['stages'][1] = stage1_results
            print("✓ Stage 1 complete\n")
        except Exception as e:
            print(f"✗ Stage 1 failed: {e}\n")
            results['stages'][1] = {"error": str(e)}
    else:
        print("Skipping Stage 1\n")
    
    # Stage 2: Iterative Refinement
    if 2 in stages:
        print("Running STAGE 2: Iterative Refinement...")
        try:
            stage2_results = run_stage2()
            results['stages'][2] = stage2_results
            print("✓ Stage 2 complete\n")
        except Exception as e:
            print(f"✗ Stage 2 failed: {e}\n")
            results['stages'][2] = {"error": str(e)}
    else:
        print("Skipping Stage 2\n")
    
    # Stage 3: Pattern Analysis
    if 3 in stages:
        print("Running STAGE 3: Pattern Analysis...")
        try:
            stage3_results = run_stage3()
            results['stages'][3] = stage3_results
            print("✓ Stage 3 complete\n")
        except Exception as e:
            print(f"✗ Stage 3 failed: {e}\n")
            results['stages'][3] = {"error": str(e)}
    else:
        print("Skipping Stage 3\n")
    
    # Stage 4: Deployment Metrics
    if 4 in stages:
        print("Running STAGE 4: Deployment Metrics...")
        try:
            stage4_results = run_stage4()
            results['stages'][4] = stage4_results
            print("✓ Stage 4 complete\n")
        except Exception as e:
            print(f"✗ Stage 4 failed: {e}\n")
            results['stages'][4] = {"error": str(e)}
    else:
        print("Skipping Stage 4\n")
    
    # Save overall results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"{model}_phase10_complete_results.json"
    save_json(str(results_file), results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 10 PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_file}\n")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 10: Lottery Ticket Hypothesis - Full Pipeline"
    )
    parser.add_argument("--model", default="gpt2", help="Model name (gpt2, llama-7b, llama-3-8b, etc.)")
    parser.add_argument("--samples", type=int, default=500, help="Number of evaluation samples")
    parser.add_argument("--output-dir", default="experiments/phase10", help="Output directory")
    parser.add_argument("--phase5-dir", default="experiments/phase5", help="Phase 5 results directory")
    parser.add_argument("--stages", nargs="+", type=int, default=[1,2,3,4], 
                        help="Stages to run (1 2 3 4)")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        model=args.model,
        num_samples=args.samples,
        output_dir=args.output_dir,
        phase5_dir=args.phase5_dir,
        stages=args.stages,
        device=args.device,
    )
    
    return results


if __name__ == "__main__":
    main()
