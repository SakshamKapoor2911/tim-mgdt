"""
Phase 11 Stage 2: Iterative Magnitude Pruning (IMP) with Real Measurement

Applies global unstructured L1 magnitude pruning at increasing sparsity levels
(10% -> 20% -> 30% -> 40% -> 50%), measures actual Wikitext-2 perplexity at
each level, and selects the highest sparsity "winning ticket" that keeps
quality degradation below a configurable threshold.
"""

import time
from typing import Dict, List, Any

import torch

from code.phases.phase11.utils import (
    compute_perplexity,
    apply_global_pruning,
    remove_global_pruning,
    count_nonzero_params,
)


def run_stage2(
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
    stage1_results: Dict[str, Any] = None,
    ppl_max_samples: int = 128,
    ppl_threshold_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    Stage 2: Iterative Magnitude Pruning (IMP) Search (REAL)

    For each sparsity level:
      1. Apply global L1 unstructured pruning to all Linear layers
      2. Measure Wikitext-2 perplexity
      3. Remove pruning reparameterization (restore original weight buffers)
      4. Record measured PPL and compute quality delta vs baseline

    The "winning ticket" is the highest sparsity where PPL increase stays
    below `ppl_threshold_pct` (default 5%).

    Raises ValueError if no model is loaded.
    """
    if model is None or tokenizer is None:
        raise ValueError(
            "Stage 2 requires a loaded model and tokenizer for real computation. "
            "Cannot proceed without a model."
        )

    print("\n" + "-" * 70)
    print("STAGE 2: Iterative Magnitude Pruning (IMP) Search (REAL)")
    print("-" * 70)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  PPL threshold: {ppl_threshold_pct}% increase")

    # --- Baseline ---
    # Use Stage 1 baseline if available, otherwise measure fresh
    baseline_ppl = None
    if stage1_results and "baseline_ppl" in stage1_results:
        baseline_ppl = stage1_results["baseline_ppl"]
        print(f"  [REUSE] Baseline PPL from Stage 1: {baseline_ppl:.4f}")
    
    if baseline_ppl is None:
        print("  [1/2] Measuring baseline perplexity...")
        baseline_ppl = compute_perplexity(model, tokenizer, device, max_samples=ppl_max_samples)

    # --- IMP sweep ---
    print(f"  [2/2] Running IMP sweep across 5 sparsity levels...")
    prune_steps = [0.10, 0.20, 0.30, 0.40, 0.50]
    search_history = []

    # Save original state for restoration between steps
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    for i, target_sparsity in enumerate(prune_steps, 1):
        step_start = time.time()

        # Apply global L1 magnitude pruning
        modules_pruned = apply_global_pruning(model, target_sparsity)

        # Count actual non-zero parameters
        nonzero = count_nonzero_params(model)
        actual_sparsity = 1.0 - (nonzero / total_params)

        # Measure real perplexity
        pruned_ppl = compute_perplexity(model, tokenizer, device, max_samples=ppl_max_samples)
        ppl_increase_pct = ((pruned_ppl - baseline_ppl) / baseline_ppl) * 100

        elapsed = time.time() - step_start

        step_result = {
            "step": i,
            "target_sparsity": target_sparsity,
            "actual_sparsity": actual_sparsity,
            "nonzero_params": nonzero,
            "modules_pruned": modules_pruned,
            "measured_ppl": pruned_ppl,
            "ppl_increase_pct": ppl_increase_pct,
            "within_threshold": ppl_increase_pct < ppl_threshold_pct,
            "step_time_seconds": elapsed,
            "metric_type": "MEASURED",
        }
        search_history.append(step_result)

        status = "PASS" if ppl_increase_pct < ppl_threshold_pct else "FAIL"
        print(f"  [OK] IMP Step {i}: {int(target_sparsity*100)}% sparsity "
              f"(actual: {actual_sparsity*100:.1f}%) | "
              f"PPL: {pruned_ppl:.2f} (+{ppl_increase_pct:.2f}%) | "
              f"{elapsed:.1f}s | {status}")

        # Restore original weights for next step (fresh pruning each time)
        model.load_state_dict(original_state)

    # --- Select winning ticket ---
    best_winner = None
    for step in reversed(search_history):
        if step["within_threshold"]:
            best_winner = step
            break

    if best_winner is None:
        # No sparsity level met the threshold; pick the least damaging
        best_winner = min(search_history, key=lambda x: x["ppl_increase_pct"])
        print(f"\n  [WARN] No sparsity level met the {ppl_threshold_pct}% threshold!")
        print(f"    Selecting least-damaging: {int(best_winner['target_sparsity']*100)}% "
              f"(PPL +{best_winner['ppl_increase_pct']:.2f}%)")
    else:
        print(f"\n  [OK] Winning Ticket: {int(best_winner['target_sparsity']*100)}% sparsity "
              f"| PPL: {best_winner['measured_ppl']:.2f} (+{best_winner['ppl_increase_pct']:.2f}%)")

    return {
        "stage": "2_imp_search",
        "baseline_ppl": baseline_ppl,
        "ppl_threshold_pct": ppl_threshold_pct,
        "steps_processed": len(search_history),
        "best_winner": best_winner,
        "search_history": search_history,
        "metric_type": "MEASURED",
        "status": "complete",
    }
