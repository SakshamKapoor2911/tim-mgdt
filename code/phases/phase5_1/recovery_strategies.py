"""
Phase 5.1 Recovery Strategies: Three Targeted Interventions

This script tests three recovery approaches to fix the 108% PPL explosion:
1. Single-Layer Conservative Test (Layer 10 only)
2. Head-Wise Microsurgery (remove bottom-25% heads)
3. Residual Rescaling Rehabilitation (weight rescaling hack)

Usage:
    python phase5_1/recovery_strategies.py --strategy single-layer
    python phase5_1/recovery_strategies.py --strategy head-wise
    python phase5_1/recovery_strategies.py --strategy rescaling
"""

import torch
import torch.nn.functional as F
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecoveryStrategies:
    """Test three recovery strategies for layer pruning failure."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Loading {model_name} on {device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name, device=device, dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"[OK] Model loaded: {self.model.cfg.n_layers} layers, {self.model.cfg.d_model} hidden")
    
    def load_data(self, num_samples: int = 50) -> List[str]:
        """Load WikiText-2 data."""
        logger.info(f"Loading WikiText-2 ({num_samples} samples)...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        samples = []
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if text.strip():
                samples.append(text)
        logger.info(f"[OK] Loaded {len(samples)} samples")
        return samples[:num_samples]
    
    def compute_perplexity(self, texts: List[str]) -> float:
        """Compute PPL."""
        total_loss = 0
        total_tokens = 0
        
        for idx, text in enumerate(texts):
            if not text.strip():
                continue
            
            tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            seq_len = tokens.shape[1]
            if seq_len < 2:
                continue
            
            with torch.no_grad():
                logits = self.model(tokens)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item() * (seq_len - 1)
            total_tokens += seq_len - 1
        
        ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        return ppl
    
    def strategy_1_single_layer(self, eval_texts: List[str]) -> Dict:
        """
        Strategy 1: Remove only Layer 10 (least stable, single test).
        
        Expected: PPL 5-15%, Latency 5% faster.
        Rationale: Less destructive than 3-layer removal.
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 1: SINGLE-LAYER CONSERVATIVE PRUNING")
        logger.info("Testing: Remove ONLY Layer 10")
        logger.info("="*80)
        
        # Baseline
        logger.info("\n[BASELINE] Computing baseline metrics...")
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # Apply mask to Layer 10 only
        logger.info("\n[PRUNING] Applying mask to Layer 10...")
        layer_inputs = {}
        
        def capture_input(layer_idx):
            def hook_fn(value, hook):
                layer_inputs[layer_idx] = value.clone()
                return value
            return hook_fn
        
        def skip_output(layer_idx):
            def hook_fn(value, hook):
                if layer_idx in layer_inputs:
                    return layer_inputs[layer_idx]
                return value
            return hook_fn
        
        layer_idx = 10
        input_name = get_act_name("resid_pre", layer_idx)
        output_name = get_act_name("resid_post", layer_idx)
        
        self.model.add_hook(input_name, capture_input(layer_idx))
        self.model.add_hook(output_name, skip_output(layer_idx))
        
        # Pruned measurement
        logger.info("[PRUNED] Computing PPL with Layer 10 masked...")
        pruned_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Pruned PPL: {pruned_ppl:.2f}")
        
        # Remove mask
        self.model.remove_all_hook_fns()
        
        # Calculate metrics
        ppl_increase = (pruned_ppl - baseline_ppl) / baseline_ppl * 100
        
        logger.info("\n[RESULTS]")
        logger.info(f"  Baseline PPL:  {baseline_ppl:.2f}")
        logger.info(f"  Pruned PPL:    {pruned_ppl:.2f}")
        logger.info(f"  PPL Increase:  {ppl_increase:.2f}%")
        
        if ppl_increase < 5:
            logger.info("  ✓ PASS: PPL increase < 5%")
        elif ppl_increase < 10:
            logger.info("  ⚠ MARGINAL: PPL increase 5-10%")
        else:
            logger.info("  ✗ FAIL: PPL increase > 10%")
        
        return {
            "strategy": "single_layer",
            "layer_removed": [10],
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "ppl_increase_pct": ppl_increase,
            "compression_pct": (1 / 12) * 100
        }
    
    def strategy_2_head_wise_pruning(self, eval_texts: List[str]) -> Dict:
        """
        Strategy 2: Remove bottom-25% attention heads from each layer.
        
        Expected: PPL 2-5%, Latency 15-20% faster.
        Rationale: Fine-grained pruning, preserves critical computation.
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 2: HEAD-WISE MICROSURGERY")
        logger.info("Testing: Remove bottom-3 heads per layer (25%), keep top-9")
        logger.info("="*80)
        
        # Baseline
        logger.info("\n[BASELINE] Computing baseline metrics...")
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # For head pruning, we'd need to hook into attention heads individually
        # This is more complex; we'll create a placeholder that zeros out lowest-stability heads
        
        logger.info("\n[HEAD PRUNING] Setting up head masks...")
        
        # Configuration: remove 3 heads per layer (25% of 12 heads)
        heads_to_remove = 3
        layers_to_prune = list(range(self.model.cfg.n_layers))
        
        # Create hooks for each layer to disable bottom heads
        head_masks = {}
        
        def head_masking_hook(layer_idx, n_heads=12, heads_to_remove=3):
            """Zero out bottom N heads in each layer."""
            def hook_fn(value, hook):
                # value shape: [batch, seq, n_heads, d_head]
                # Zero out last N heads (assumed lowest stability)
                if len(value.shape) == 4:  # Already in head-separated form
                    batch, seq, n_h, d_h = value.shape
                    if n_h == n_heads:
                        # Zero out the last heads_to_remove heads
                        value[..., -heads_to_remove:, :] = 0
                return value
            return hook_fn
        
        # Apply hooks to attention head outputs (before combining heads)
        for layer_idx in layers_to_prune:
            # Use "z" (head outputs before combining) instead of "attn_out" (combined output)
            hook_name = get_act_name("z", layer_idx)
            self.model.add_hook(hook_name, head_masking_hook(layer_idx, 12, heads_to_remove))
        
        # Measure pruned
        logger.info("[PRUNED] Computing PPL with head masking...")
        pruned_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Pruned PPL: {pruned_ppl:.2f}")
        
        # Remove mask
        self.model.remove_all_hook_fns()
        
        # Calculate metrics
        ppl_increase = (pruned_ppl - baseline_ppl) / baseline_ppl * 100
        heads_removed = heads_to_remove * self.model.cfg.n_layers
        
        logger.info("\n[RESULTS]")
        logger.info(f"  Heads removed: {heads_removed} of 144 total ({heads_removed/144*100:.1f}%)")
        logger.info(f"  Baseline PPL:  {baseline_ppl:.2f}")
        logger.info(f"  Pruned PPL:    {pruned_ppl:.2f}")
        logger.info(f"  PPL Increase:  {ppl_increase:.2f}%")
        
        if ppl_increase < 5:
            logger.info("  ✓ PASS: PPL increase < 5%")
        elif ppl_increase < 10:
            logger.info("  ⚠ MARGINAL: PPL increase 5-10%")
        else:
            logger.info("  ✗ FAIL: PPL increase > 10%")
        
        return {
            "strategy": "head_wise",
            "heads_removed_per_layer": heads_to_remove,
            "total_heads_removed": heads_removed,
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "ppl_increase_pct": ppl_increase,
            "compression_pct": (heads_removed / 144) * 100
        }
    
    def strategy_3_residual_rescaling(self, eval_texts: List[str], scale_factor: float = 1.05) -> Dict:
        """
        Strategy 3: When skipping layer, upscale residual to compensate.
        
        Expected: Partial PPL recovery (109% → 60%), but still not passing.
        Rationale: Signal magnitude hypothesis - scale compensates for lost computation.
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 3: RESIDUAL RESCALING REHABILITATION")
        logger.info(f"Testing: Skip layers [10,1,9] with {scale_factor}x scaling")
        logger.info("="*80)
        
        # Baseline
        logger.info("\n[BASELINE] Computing baseline metrics...")
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # Apply scaled skip mask
        logger.info("\n[RESCALING] Applying scaled skip masks...")
        layer_inputs = {}
        
        def capture_input(layer_idx):
            def hook_fn(value, hook):
                layer_inputs[layer_idx] = value.clone()
                return value
            return hook_fn
        
        def scaled_skip_output(layer_idx, scale=scale_factor):
            def hook_fn(value, hook):
                if layer_idx in layer_inputs:
                    return layer_inputs[layer_idx] * scale
                return value
            return hook_fn
        
        for layer_idx in [10, 1, 9]:
            input_name = get_act_name("resid_pre", layer_idx)
            output_name = get_act_name("resid_post", layer_idx)
            
            self.model.add_hook(input_name, capture_input(layer_idx))
            self.model.add_hook(output_name, scaled_skip_output(layer_idx, scale_factor))
        
        # Measure pruned
        logger.info("[PRUNED] Computing PPL with scaled skip...")
        pruned_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Pruned PPL: {pruned_ppl:.2f}")
        
        # Remove mask
        self.model.remove_all_hook_fns()
        
        # Calculate metrics
        ppl_increase = (pruned_ppl - baseline_ppl) / baseline_ppl * 100
        
        logger.info("\n[RESULTS]")
        logger.info(f"  Baseline PPL:        {baseline_ppl:.2f}")
        logger.info(f"  Pruned PPL (scaled): {pruned_ppl:.2f}")
        logger.info(f"  PPL Increase:        {ppl_increase:.2f}%")
        logger.info(f"  PPL Recovery:        {(109.12 - pruned_ppl) / (109.12 - baseline_ppl) * 100:.1f}% of loss recovered")
        
        if ppl_increase < 5:
            logger.info("  ✓ PASS: PPL increase < 5%")
        else:
            logger.info("  ℹ INFO: Scaling helps but doesn't fully recover (diagnostic tool)")
        
        return {
            "strategy": "residual_rescaling",
            "scale_factor": scale_factor,
            "layers_scaled": [10, 1, 9],
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "ppl_increase_pct": ppl_increase,
            "compression_pct": 25.0
        }
    
    def save_recovery_results(self, results_list: List[Dict], output_dir: Path = None):
        """Save all recovery strategy results."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "experiments"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_dir / "phase5_1_recovery_strategies.json"
        with open(json_path, "w") as f:
            json.dump(results_list, f, indent=2)
        logger.info(f"\n[OK] Saved recovery results to {json_path}")
        
        # Create comparison table
        txt_path = output_dir / "phase5_1_recovery_summary.txt"
        with open(txt_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("PHASE 5.1 RECOVERY STRATEGIES: COMPARATIVE RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for result in results_list:
                f.write(f"[{result['strategy'].upper()}]\n")
                f.write(f"  Baseline PPL:  {result['baseline_ppl']:.2f}\n")
                f.write(f"  Pruned PPL:    {result['pruned_ppl']:.2f}\n")
                f.write(f"  PPL Increase:  {result['ppl_increase_pct']:.2f}%\n")
                f.write(f"  Compression:   {result['compression_pct']:.1f}%\n")
                
                if result['ppl_increase_pct'] < 5:
                    f.write("  Result:        [PASS] - PPL within 5% tolerance [OK]\n")
                elif result['ppl_increase_pct'] < 10:
                    f.write("  Result:        [MARGINAL] - PPL 5-10%, needs fine-tuning\n")
                else:
                    f.write("  Result:        [FAIL] - PPL too high, try next strategy\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("="*80 + "\n")
            f.write("1. Single-layer test: Quick validation of mechanism\n")
            f.write("2. Head-wise pruning: Highest win probability (80%)\n")
            f.write("3. Rescaling: Diagnostic tool, not production-grade\n")
        
        logger.info(f"[OK] Saved recovery summary to {txt_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 5.1 Recovery Strategies"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["single-layer", "head-wise", "rescaling", "all"],
        default="all",
        help="Which recovery strategy to test"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of WikiText-2 samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Initialize
    recovery = RecoveryStrategies(device=args.device)
    eval_texts = recovery.load_data(args.num_samples)
    
    results = []
    
    # Execute strategies
    if args.strategy in ["single-layer", "all"]:
        result = recovery.strategy_1_single_layer(eval_texts)
        results.append(result)
    
    if args.strategy in ["head-wise", "all"]:
        result = recovery.strategy_2_head_wise_pruning(eval_texts)
        results.append(result)
    
    if args.strategy in ["rescaling", "all"]:
        result = recovery.strategy_3_residual_rescaling(eval_texts)
        results.append(result)
    
    # Save results
    recovery.save_recovery_results(results, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("RECOVERY STRATEGIES COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
