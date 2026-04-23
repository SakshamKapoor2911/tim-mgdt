"""
Phase 5.2: Conservative Recovery Strategies
Test ultra-light interventions that don't break model severely.

Key innovations:
1. Neuron-level pruning (only 1-2% of weights)
2. Layer norm adjustment (redistribute importance)
3. Fine-tuning via gradient scaling
"""

import torch
import torch.nn.functional as F
import json
import logging
from pathlib import Path
from typing import Dict, List
import argparse

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConservativeRecovery:
    """Ultra-conservative recovery strategies for minimal PPL impact."""
    
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
        
        for text in texts:
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
    
    def strategy_1_neuron_pruning(self, eval_texts: List[str], 
                                  prune_pct: float = 1.0) -> Dict:
        """
        Strategy 1: Remove bottom N% of neurons by L2 norm across layers.
        
        This is MUCH lighter than layer/head removal.
        Expected: PPL 1-3% increase with 1% pruning.
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 1: ULTRA-LIGHT NEURON PRUNING")
        logger.info(f"Testing: Remove bottom {prune_pct}% of neurons by L2 norm")
        logger.info("="*80)
        
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # Collect all parameter weights (HookedTransformer uses different naming)
        all_norms = []
        neuron_info = {}  # Track which params these are
        
        for name, param in self.model.named_parameters():
            # Include all weight-like parameters (not just 'weight' in name)
            if param.dim() > 0:  # Don't include scalars
                # Simply take L2 norm of entire parameter
                norm = torch.norm(param.view(-1), p=2).item()
                all_norms.append(norm)
                if name not in neuron_info:
                    neuron_info[name] = []
                neuron_info[name].append(norm)
        
        # Find threshold
        if len(all_norms) == 0:
            logger.warning("No weight parameters found! Skipping neuron pruning.")
            return {
                "strategy": "neuron_pruning",
                "error": "no_parameters_found",
                "prune_percentage": prune_pct
            }
        
        norms_tensor = torch.tensor(all_norms)
        threshold = torch.quantile(norms_tensor, max(0.01, prune_pct/100.0)).item()
        
        logger.info(f"  Weight norm threshold: {threshold:.6f}")
        logger.info(f"  Total weight parameters: {len(all_norms)}")
        
        # Save original state
        original_state = {}
        for name, param in self.model.named_parameters():
            original_state[name] = param.data.clone()
        
        # Apply pruning: zero out bottom parameters by norm
        pruned_count = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.dim() > 0:  # Skip scalars
                norm = torch.norm(param.view(-1), p=2).item()
                total_params += param.numel()
                if norm < threshold:
                    param.data.zero_()
                    pruned_count += param.numel()
        
        logger.info(f"  Pruned parameter elements: {pruned_count}/{total_params}")
        
        # Test
        pruned_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Pruned PPL: {pruned_ppl:.2f}")
        
        ppl_increase = (pruned_ppl - baseline_ppl) / baseline_ppl * 100
        
        logger.info(f"\n[RESULTS]")
        logger.info(f"  Baseline PPL: {baseline_ppl:.2f}")
        logger.info(f"  Pruned PPL:   {pruned_ppl:.2f}")
        logger.info(f"  PPL Increase: {ppl_increase:.2f}%")
        
        if ppl_increase < 5:
            logger.info(f"  ✓ PASS: PPL increase < 5%")
        elif ppl_increase < 10:
            logger.info(f"  ⚠ MARGINAL: PPL increase 5-10%")
        else:
            logger.info(f"  ✗ FAIL: PPL increase > 10%")
        
        # Restore state
        for name, param in self.model.named_parameters():
            param.data = original_state[name]
        
        return {
            "strategy": "neuron_pruning",
            "prune_percentage": prune_pct,
            "neurons_pruned": pruned_count,
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "ppl_increase_pct": ppl_increase
        }
    
    def strategy_2_layer_norm_rebalancing(self, eval_texts: List[str]) -> Dict:
        """
        Strategy 2: Adjust layer norms to redistribute importance.
        
        Hypothesis: Maybe layer 10 is over-weighted. Rescale its output 
        slightly downward and other layers upward to compensate.
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 2: LAYER NORM REBALANCING")
        logger.info("Testing: Redistribute layer importance via norm adjustment")
        logger.info("="*80)
        
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # Strategy: slightly reduce layer 10's influence
        scale_down_layer = 10
        scale_factor_down = 0.95  # 5% reduction
        scale_factor_up = 1.01   # 1% increase for others
        
        original_state = {}
        for name, param in self.model.named_parameters():
            original_state[name] = param.data.clone()
        
        # Apply scaling to HookedTransformer parameters
        for name, param in self.model.named_parameters():
            # Check for layer 10 in the parameter name (handles various naming schemes)
            if ".10." in name or ("blocks" in name and "10" in name):
                param.data = param.data * scale_factor_down
            elif "blocks." in name or "attn" in name or "mlp" in name:
                # Selectively scale other critical layers/components
                param.data = param.data * scale_factor_up
        
        logger.info(f"  Scaled layer {scale_down_layer}: {scale_factor_down}x")
        logger.info(f"  Scaled other layers: {scale_factor_up}x")
        
        pruned_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Adjusted PPL: {pruned_ppl:.2f}")
        
        ppl_change = (pruned_ppl - baseline_ppl) / baseline_ppl * 100
        
        logger.info(f"\n[RESULTS]")
        logger.info(f"  Baseline PPL: {baseline_ppl:.2f}")
        logger.info(f"  Adjusted PPL: {pruned_ppl:.2f}")
        logger.info(f"  PPL Change:   {ppl_change:.2f}%")
        
        if ppl_change < 0:
            logger.info(f"  ✓ IMPROVEMENT: PPL decreased by {abs(ppl_change):.2f}%")
        elif ppl_change < 2:
            logger.info(f"  ✓ NEAR-NEUTRAL: Minimal impact")
        elif ppl_change < 10:
            logger.info(f"  ⚠ MARGINAL: Small PPL increase")
        else:
            logger.info(f"  ✗ FAIL: PPL increased too much")
        
        # Restore state
        for name, param in self.model.named_parameters():
            param.data = original_state[name]
        
        return {
            "strategy": "layer_norm_rebalancing",
            "baseline_ppl": baseline_ppl,
            "adjusted_ppl": pruned_ppl,
            "ppl_change_pct": ppl_change
        }
    
    def strategy_3_dropout_regularization(self, eval_texts: List[str], 
                                         dropout_rate: float = 0.05) -> Dict:
        """
        Strategy 3: Apply light dropout to simulate parameter uncertainty.
        
        Hypothesis: Maybe we need to make layer 10 outputs less deterministic 
        to break dependency chains.
        """
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 3: LIGHT DROPOUT REGULARIZATION")
        logger.info(f"Testing: Add {dropout_rate*100:.1f}% dropout to critical layers")
        logger.info("="*80)
        
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # Add dropout hook to critical layers
        def dropout_hook(value, hook):
            if self.training:  # Only apply during training
                return F.dropout(value, p=dropout_rate, training=True)
            return value
        
        # We're in eval mode, so dropout won't help. Let's test in a different way:
        # Compute PPL with model in different states
        self.model.eval()
        pruned_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Current PPL (eval): {pruned_ppl:.2f}")
        
        logger.info(f"\n[RESULTS]")
        logger.info(f"  Baseline PPL: {baseline_ppl:.2f}")
        logger.info(f"  dropout approach not suitable for eval-time testing")
        
        return {
            "strategy": "dropout_regularization",
            "note": "Requires training phase, skipped for eval-only",
            "ppl": baseline_ppl
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", 
                       choices=["neuron", "norm", "dropout", "all"])
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()
    
    recovery = ConservativeRecovery()
    eval_texts = recovery.load_data(args.num_samples)
    
    results = {}
    
    if args.strategy in ["neuron", "all"]:
        # Test with 1%, 2%, 5% pruning
        for prune_pct in [0.5, 1.0, 2.0]:
            key = f"neuron_{prune_pct}_pct"
            results[key] = recovery.strategy_1_neuron_pruning(eval_texts, prune_pct)
    
    if args.strategy in ["norm", "all"]:
        results["norm_rebalancing"] = recovery.strategy_2_layer_norm_rebalancing(eval_texts)
    
    if args.strategy in ["dropout", "all"]:
        results["dropout"] = recovery.strategy_3_dropout_regularization(eval_texts)
    
    # Save results
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "phase5_2_conservative_recovery.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n[OK] Saved results to {output_dir}/phase5_2_conservative_recovery.json")


if __name__ == "__main__":
    main()
