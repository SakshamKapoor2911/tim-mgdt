"""
Phase 5 with Phase 4 Integration - Complete Compression Pipeline.

This script chains Phase 4 (baseline correlation) with Phase 5 (granular compression),
enabling a full end-to-end demonstration of the compression pipeline using your
existing Phase 4 results.

Usage:
    python run_phase5_with_phase4_integration.py \
        --phase4-results experiments/sensitivity_results_500samples.json \
        --model gpt2 \
        --output-dir experiments
"""

import torch
import json
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

from code.models.model_wrapper import HookedModelWrapper
from code.advanced_proxies import HeadJacobian, CircuitAnalyzer, PruningEngine
from code.advanced_proxies import HeadJacobianConfig, TicketConfig, PruningConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("phase5_integration.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_phase4_results(results_path: str) -> Dict:
    """Load Phase 4 sensitivity results."""
    logger.info(f"Loading Phase 4 results from {results_path}...")
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        logger.info(f"[OK] Phase 4 results loaded")
        return data
    except Exception as e:
        logger.error(f"Failed to load Phase 4 results: {e}")
        raise


def integrate_phase4_phase5(
    phase4_results: Dict,
    wrapper: HookedModelWrapper,
    num_head_samples: int = 5,
) -> Dict:
    """
    Integrate Phase 4 and Phase 5:
    1. Reuse Phase 4 ED, Jacobian, nAUDC
    2. Add Phase 5 head-level granularity
    3. Compute stability and identify tickets
    4. Plan pruning
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4 + PHASE 5 Integration")
    logger.info("=" * 80)

    n_layers = wrapper.model.cfg.n_layers

    # Extract Phase 4 scores
    logger.info("\nExtracting Phase 4 scores...")
    raw_scores = phase4_results.get("raw_scores", {})
    
    phase4_jacobian = raw_scores.get("causal_jacobian", {})
    phase4_ed = raw_scores.get("geometric_ed", {})
    phase4_naudc = raw_scores.get("propagation_naudc", {})
    phase4_ablation = raw_scores.get("ground_truth_delta_ce", {})

    logger.info(f"  [OK] Retrieved Phase 4 Jacobian: {len(phase4_jacobian)} layers")
    logger.info(f"  [OK] Retrieved Phase 4 ED: {len(phase4_ed)} layers")
    logger.info(f"  [OK] Retrieved Phase 4 nAUDC: {len(phase4_naudc)} layers")
    logger.info(f"  [OK] Retrieved Phase 4 Ablation (Ground Truth): {len(phase4_ablation)} layers")

    # Phase 5: Compute head-level Jacobian
    logger.info("\nComputing Phase 5 head-level Jacobian...")
    config = HeadJacobianConfig(num_samples=num_head_samples)
    head_jacobian = HeadJacobian(config=config)
    head_scores = head_jacobian.compute_head_scores(wrapper, num_samples=num_head_samples)
    head_layer_agg = head_jacobian.aggregate_to_layer_level(head_scores, n_layers)
    logger.info(f"  [OK] Computed {len(head_scores)} head-level scores")

    # Phase 5: Lottery Ticket Identification using Phase 4 data
    logger.info("\nIdentifying lottery tickets (Phase 4 + Phase 5)...")
    config = TicketConfig(verbose=True)
    analyzer = CircuitAnalyzer(config=config)
    
    stability_scores = analyzer.find_winning_tickets(
        phase4_ed, phase4_jacobian, phase4_naudc, n_layers
    )
    logger.info(f"  [OK] Computed stability scores for {len(stability_scores)} layers")

    # Identify ticket candidates
    candidates = analyzer.identify_lottery_ticket_candidates(
        stability_scores, phase4_jacobian, phase4_ed, percentile_threshold=75.0
    )
    
    ticket_count = sum(1 for c in candidates.values() if c.get("is_lottery_ticket", False))
    logger.info(f"  [OK] Found {ticket_count} lottery ticket candidates")

    # Phase 5: Plan pruning
    logger.info("\nPlanning pruning strategy...")
    k_prune = max(1, n_layers // 4)  # Prune bottom 25%
    
    ranked = analyzer.rank_by_stability(stability_scores, ascending=True)
    pruned_layers = [int(k.split("_")[1]) for k, v in ranked[:k_prune]]
    
    logger.info(f"  Layers to prune (lowest stability): {pruned_layers}")

    # Compile integrated results
    integrated_results = {
        "phase4": {
            "model": phase4_results.get("config", {}).get("model", "unknown"),
            "num_samples": phase4_results.get("config", {}).get("num_samples", 0),
            "correlations": phase4_results.get("correlations", {}),
        },
        "phase5": {
            "head_jacobian_scores": head_scores,
            "stability_scores": stability_scores,
            "lottery_ticket_candidates": candidates,
            "pruning_plan": {
                "pruned_layers": pruned_layers,
                "k_prune": k_prune,
                "pruning_percentage": 25.0,
            },
        },
        "comparison": {
            "phase4_vs_phase5_jacobian": {
                "phase4_layer_jacobian": phase4_jacobian,
                "phase5_head_jacobian_aggregated": head_layer_agg,
            },
        },
    }

    logger.info("\n" + "=" * 80)
    logger.info("Integration Complete!")
    logger.info("=" * 80)
    logger.info(f"[PHASE 4] Correlation: rho = {phase4_results.get('correlations', {}).get('Ensemble (ED+invJac+invnAUDC)', {}).get('spearman_rho', 'N/A'):.4f}")
    logger.info(f"[PHASE 5] Head Granularity: {len(head_scores)} heads analyzed")
    logger.info(f"[TICKETS] Found: {ticket_count}/{n_layers} layers")
    logger.info(f"[PRUNING] Remove {len(pruned_layers)} layers > {((n_layers - len(pruned_layers)) / n_layers * 100):.1f}% remain")

    return integrated_results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 with Phase 4 Integration - Complete Compression Pipeline"
    )
    parser.add_argument(
        "--phase4-results",
        type=str,
        required=True,
        help="Path to Phase 4 results JSON (e.g., experiments/sensitivity_results_500samples.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (must match Phase 4 model)",
    )
    parser.add_argument(
        "--head-samples",
        type=int,
        default=5,
        help="Number of calibration texts for Phase 5 head Jacobian",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for integration results",
    )

    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info("Phase 5 with Phase 4 Integration")
    logger.info(f"{'='*80}")

    # Load Phase 4 results
    phase4_results = load_phase4_results(args.phase4_results)

    # Initialize model
    logger.info(f"\nInitializing model: {args.model}")
    try:
        wrapper = HookedModelWrapper(args.model, load_in_8bit=False)
        logger.info(f"[OK] Model loaded")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    # Integrate Phase 4 + Phase 5
    try:
        integrated = integrate_phase4_phase5(
            phase4_results, wrapper, num_head_samples=args.head_samples
        )

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = Path(args.output_dir) / "phase5_integrated_results.json"

        with open(output_path, "w") as f:
            json.dump(integrated, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Integration failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
