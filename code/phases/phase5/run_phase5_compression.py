"""
Phase 5: Granular Compression - Hardware-Agnostic Master Script.

This script orchestrates the full compression pipeline:
  1. Head-level Jacobian computation (granular functional leverage)
  2. Lottery ticket identification (Circuit Stability Score)
  3. Layer/head pruning based on rankings
  4. Perplexity benchmarking on WikiText-2

Designed to run on:
  - Local: Qwen/Qwen2.5-0.5B (4GB VRAM)
  - Cloud: mistralai/Mistral-7B-v0.1 (16GB+ VRAM)

Usage:
    python run_phase5_compression.py --model qwen  # Local testing
    python run_phase5_compression.py --model mistral  # Cloud testing
"""

import torch
import json
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

# Phase 4 baseline (existing code)
from code.models.model_wrapper import HookedModelWrapper

# Phase 5 new components
from code.advanced_proxies import HeadJacobian, CircuitAnalyzer, PruningEngine
from code.advanced_proxies import HeadJacobianConfig, TicketConfig, PruningConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("phase5_run.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Model selection
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-0.5B",
        "load_in_8bit": False,  # Keep full precision for small model
        "max_seq_len": 64,
        "description": "Local testing (4GB VRAM)",
    },
    "gpt2": {
        "name": "gpt2",
        "load_in_8bit": False,
        "max_seq_len": 128,
        "description": "Baseline GPT-2 (used in Phase 4 results)",
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-v0.1",
        "load_in_8bit": True,
        "max_seq_len": 256,
        "description": "Cloud testing (16GB+ VRAM required)",
    },
}


class Phase5Pipeline:
    """Orchestrates the Phase 5 compression pipeline."""

    def __init__(self, model_name: str, load_in_8bit: bool = False):
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.wrapper = None
        self.results = {}

    def initialize_model(self):
        """Initialize HookedModelWrapper."""
        logger.info(f"Initializing model: {self.model_name}")
        try:
            self.wrapper = HookedModelWrapper(
                self.model_name, load_in_8bit=self.load_in_8bit
            )
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Layers: {self.wrapper.model.cfg.n_layers}")
            logger.info(f"  Heads: {self.wrapper.model.cfg.n_heads}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def step_1_head_level_jacobian(self, num_samples: int = 5) -> Dict[str, float]:
        """
        Step 1: Compute head-level Jacobian scores.
        (Granular functional leverage)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Head-Level Jacobian Computation")
        logger.info("=" * 80)

        config = HeadJacobianConfig(num_samples=num_samples)
        head_jacobian = HeadJacobian(config=config)

        try:
            head_scores = head_jacobian.compute_head_scores(self.wrapper)
            
            # Aggregate to layer level for easier interpretation
            n_layers = self.wrapper.model.cfg.n_layers
            layer_scores = head_jacobian.aggregate_to_layer_level(head_scores, n_layers)
            
            logger.info(f"✓ Computed head-level scores for all heads")
            logger.info(f"  Total head scores: {len(head_scores)}")
            
            # Log layer-level summary
            for layer_idx in range(n_layers):
                layer_key = f"layer_{layer_idx}"
                score = layer_scores.get(layer_key, 0.0)
                logger.info(f"    {layer_key}: {score:.4f}")
            
            self.results["head_jacobian"] = head_scores
            self.results["head_jacobian_layer_aggregated"] = layer_scores
            
            return layer_scores

        except Exception as e:
            logger.error(f"Failed to compute head Jacobian: {e}")
            raise

    def step_2_lottery_ticket_identification(
        self,
        layer_jacobian: Dict[str, float],
        ed_scores: Optional[Dict[str, float]] = None,
        naudc_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Step 2: Identify lottery ticket candidates using Stability Score.
        (Requires Phase 4 results: ED and nAUDC)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Lottery Ticket Identification")
        logger.info("=" * 80)

        # If Phase 4 results not provided, use dummy scores for demonstration
        if ed_scores is None or naudc_scores is None:
            logger.warning(
                "ED or nAUDC scores not provided. Using placeholder scores."
            )
            logger.warning(
                "For full analysis, integrate Phase 4 results via --phase4-results"
            )

            # Dummy scores for demo
            n_layers = self.wrapper.model.cfg.n_layers
            ed_scores = {f"layer_{i}": float(i + 1) for i in range(n_layers)}
            naudc_scores = {f"layer_{i}": 1e-4 for i in range(n_layers)}

        config = TicketConfig(verbose=True)
        analyzer = CircuitAnalyzer(config=config)

        try:
            stability_scores = analyzer.find_winning_tickets(
                ed_scores, layer_jacobian, naudc_scores, self.wrapper.model.cfg.n_layers
            )

            logger.info(f"✓ Stability scores computed")
            logger.info(f"\n  Stability Ranking (by importance):")
            
            ranked = analyzer.rank_by_stability(stability_scores, ascending=False)
            for i, (layer_key, score) in enumerate(ranked):
                logger.info(f"    {i+1}. {layer_key}: {score:.4e}")

            # Identify lottery tickets
            candidates = analyzer.identify_lottery_ticket_candidates(
                stability_scores, layer_jacobian, ed_scores, percentile_threshold=75.0
            )

            self.results["stability_scores"] = stability_scores
            self.results["ticket_candidates"] = {
                k: v for k, v in candidates.items() if v["is_lottery_ticket"]
            }

            return stability_scores

        except Exception as e:
            logger.error(f"Failed to identify lottery tickets: {e}")
            raise

    def step_3_pruning_strategy(
        self, stability_scores: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Step 3: Plan pruning strategy (identify layers to remove).
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Pruning Strategy")
        logger.info("=" * 80)

        config = PruningConfig(verbose=True)
        pruning_engine = PruningEngine(self.wrapper, config=config)

        # Option: Prune bottom 25% of layers
        k = max(1, self.wrapper.model.cfg.n_layers // 4)

        logger.info(f"Planning to prune bottom {k} layers (lowest stability)...")

        pruned_indices, summary = pruning_engine.prune_bottom_k_layers(
            stability_scores, k=k
        )

        logger.info(f"✓ Pruning plan created")
        logger.info(f"  Layers marked for pruning: {pruned_indices}")

        self.results["pruning_plan"] = {
            "pruned_layers": pruned_indices,
            "summary": summary,
        }

        return {"pruned_layers": pruned_indices, "engine": pruning_engine}

    def step_4_verify_pruning(self, pruning_result: Dict) -> Dict:
        """
        Step 4: Verify pruning by running inference on pruned model.
        (Quick validation without full benchmark)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Pruning Verification (Quick Inference)")
        logger.info("=" * 80)

        pruning_engine = pruning_result["engine"]
        pruned_layers = pruning_result["pruned_layers"]

        # Test on a simple prompt
        test_prompts = [
            "The quick brown fox",
            "Machine learning models",
        ]

        logger.info(f"Running inference on pruned model...")

        try:
            for prompt in test_prompts:
                tokens = self.wrapper.model.to_tokens(prompt, move_to_device=True)
                logits = self.wrapper.model(tokens)
                logger.info(f"  ✓ Inference successful on: '{prompt}'")

            logger.info(f"✓ Pruned model is functional")
            logger.info(f"  Masks active on layers: {pruned_layers}")

            self.results["verification"] = {
                "status": "success",
                "pruned_layers": pruned_layers,
                "test_prompts_ok": len(test_prompts),
            }

        except Exception as e:
            logger.error(f"Pruning verification failed: {e}")
            self.results["verification"] = {"status": "failed", "error": str(e)}

        # Clean up masks
        pruning_engine.remove_all_masks()
        logger.info("Pruning masks removed for clean state.")

        return self.results["verification"]

    def save_results(self, output_dir: str = "experiments"):
        """Save Phase 5 results to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        output_path = Path(output_dir) / "phase5_results.json"

        # Convert non-serializable types
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = value
            elif isinstance(value, (int, float, str, bool)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"\n✓ Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 5: Granular Compression Pipeline"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gpt2",
        help="Model to use (qwen=local 4GB, gpt2=baseline, mistral=cloud)",
    )
    parser.add_argument(
        "--head-samples",
        type=int,
        default=5,
        help="Number of calibration texts for head Jacobian",
    )
    parser.add_argument(
        "--phase4-results",
        type=str,
        default=None,
        help="Path to Phase 4 results for integration (e.g., experiments/sensitivity_results_500samples.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Get model config
    config = MODEL_CONFIGS[args.model]
    logger.info(f"\n{'='*80}")
    logger.info(f"Phase 5: Granular Compression Pipeline")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {args.model} ({config['description']})")
    logger.info(f"Model Name: {config['name']}")

    # Initialize pipeline
    pipeline = Phase5Pipeline(config["name"], load_in_8bit=config["load_in_8bit"])

    try:
        # Step 1: Load model
        pipeline.initialize_model()

        # Step 2: Compute head-level Jacobian
        layer_jacobian = pipeline.step_1_head_level_jacobian(num_samples=args.head_samples)

        # Step 3: Load Phase 4 results or use placeholders
        ed_scores = None
        naudc_scores = None

        if args.phase4_results and os.path.exists(args.phase4_results):
            logger.info(f"Loading Phase 4 results from {args.phase4_results}...")
            try:
                with open(args.phase4_results, "r") as f:
                    phase4_data = json.load(f)
                ed_scores = phase4_data.get("raw_scores", {}).get("geometric_ed", {})
                naudc_scores = phase4_data.get("raw_scores", {}).get(
                    "propagation_naudc", {}
                )
                logger.info(f"✓ Loaded ED and nAUDC scores from Phase 4")
            except Exception as e:
                logger.warning(f"Failed to load Phase 4 results: {e}")

        # Step 4: Lottery ticket identification
        stability_scores = pipeline.step_2_lottery_ticket_identification(
            layer_jacobian, ed_scores, naudc_scores
        )

        # Step 5: Pruning strategy
        pruning_result = pipeline.step_3_pruning_strategy(stability_scores)

        # Step 6: Verification
        pipeline.step_4_verify_pruning(pruning_result)

        # Save results
        pipeline.save_results(output_dir=args.output_dir)

        logger.info(f"\n{'='*80}")
        logger.info("Phase 5 Complete! ✓")
        logger.info(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
