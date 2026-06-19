"""
Phase 5.3: Hybrid Recovery Strategy
Combine layer norm rebalancing + targeted fine-tuning for optimal recovery.

Key idea: Don't remove layer 10; instead rebalance and selectively adjust weights.
"""

import torch
import torch.nn.functional as F
import json
import logging
from pathlib import Path
from typing import Dict, List
import argparse

from transformer_lens import HookedTransformer
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridStrategy:
    """Combine rebalancing + fine-tuning for optimal recovery."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
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
        
        if total_tokens == 0:
            return 0
        
        ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        return ppl
    
    def strategy_hybrid_rebalance_prune(self, eval_texts: List[str]) -> Dict:
        """
        Hybrid Strategy: Rebalance critical layers + selective pruning.
        
        Approach:
        1. Reduce Layer 10 influence (0.95x) 
        2. Boost other layers (1.02x)
        3. Selectively reduce low-norm weights in layer 10 only
        """
        logger.info("\n" + "="*80)
        logger.info("HYBRID STRATEGY: REBALANCE + SELECTIVE PRUNING")
        logger.info("Testing: Rebalance all layers + prune bottom 2% weights in Layer 10")
        logger.info("="*80)
        
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        # Save original state
        original_state = {}
        for name, param in self.model.named_parameters():
            original_state[name] = param.data.clone()
        
        # Strategy 1: Rebalance all layers
        logger.info("\n[STEP 1] Rebalancing layer importance...")
        for name, param in self.model.named_parameters():
            if ".10." in name or ("blocks" in name and "10" in name):
                param.data = param.data * 0.95
            elif "blocks." in name or "attn" in name or "mlp" in name:
                param.data = param.data * 1.02
        
        # Strategy 2: Selective pruning in layer 10 only
        logger.info("[STEP 2] Selective pruning in Layer 10...")
        pruned_elements = 0
        total_elements_in_layer_10 = 0
        
        for name, param in self.model.named_parameters():
            if ".10." in name or ("blocks" in name and "10" in name):
                # Find bottom 2% smallest magnitude weights
                flat_weights = param.data.view(-1)
                total_elements_in_layer_10 += flat_weights.numel()
                
                # Get L2 norms for each element
                if param.dim() >= 2:
                    # For matrices, compute norm per neuron
                    if param.shape[0] > 1 and param.shape[1] > 1:
                        col_norms = torch.norm(param.data, p=2, dim=0)
                        threshold = torch.quantile(col_norms, 0.02)
                        # Zero out low-norm columns
                        mask = (col_norms > threshold).float()
                        param.data = param.data * mask.unsqueeze(0)
                        pruned_elements += (mask == 0).sum().item()
                else:
                    # For 1D parameters, use element-wise magnitude
                    threshold = torch.quantile(torch.abs(flat_weights), 0.02)
                    mask = (torch.abs(flat_weights) > threshold).float()
                    param.data = mask * flat_weights
                    pruned_elements += (mask == 0).sum().item()
        
        logger.info(f"  Pruned elements: {pruned_elements}/{total_elements_in_layer_10}")
        
        # Test
        hybrid_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Hybrid PPL: {hybrid_ppl:.2f}")
        
        ppl_increase = (hybrid_ppl - baseline_ppl) / baseline_ppl * 100
        
        logger.info(f"\n[RESULTS]")
        logger.info(f"  Baseline PPL:    {baseline_ppl:.2f}")
        logger.info(f"  Hybrid PPL:      {hybrid_ppl:.2f}")
        logger.info(f"  PPL Increase:    {ppl_increase:.2f}%")
        logger.info(f"  Compression:     {pruned_elements}/{total_elements_in_layer_10} elements")
        
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
            "strategy": "hybrid_rebalance_prune",
            "baseline_ppl": baseline_ppl,
            "hybrid_ppl": hybrid_ppl,
            "ppl_increase_pct": ppl_increase,
            "elements_pruned": pruned_elements,
            "total_layer10_elements": total_elements_in_layer_10,
            "compression_pct": (pruned_elements / total_elements_in_layer_10) * 100 if total_elements_in_layer_10 > 0 else 0
        }
    
    def strategy_progressive_rebalance(self, eval_texts: List[str]) -> Dict:
        """
        Test different rebalancing ratios to find optimal compression.
        
        Test: vary the scale factors for layer 10 reduction
        """
        logger.info("\n" + "="*80)
        logger.info("PROGRESSIVE REBALANCE STRATEGY")
        logger.info("Testing: Find optimal scale factor for layer 10")
        logger.info("="*80)
        
        baseline_ppl = self.compute_perplexity(eval_texts)
        logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
        
        results = {}
        scale_factors = [0.90, 0.92, 0.95, 0.98, 1.0]
        
        for scale_factor in scale_factors:
            # Save original state
            original_state = {}
            for name, param in self.model.named_parameters():
                original_state[name] = param.data.clone()
            
            # Apply scaling
            for name, param in self.model.named_parameters():
                if ".10." in name or ("blocks" in name and "10" in name):
                    param.data = param.data * scale_factor
                elif "blocks." in name or "attn" in name or "mlp" in name:
                    param.data = param.data * (1.0 + (1.0 - scale_factor))
            
            test_ppl = self.compute_perplexity(eval_texts)
            ppl_change = (test_ppl - baseline_ppl) / baseline_ppl * 100
            
            logger.info(f"  Scale {scale_factor:.2f}: PPL={test_ppl:.2f}, Change={ppl_change:+.2f}%")
            
            results[f"scale_{scale_factor:.2f}"] = {
                "scale_factor": scale_factor,
                "ppl": test_ppl,
                "ppl_change_pct": ppl_change
            }
            
            # Restore state
            for name, param in self.model.named_parameters():
                param.data = original_state[name]
        
        logger.info(f"\n[OK] Progressive rebalance testing complete")
        
        return {
            "strategy": "progressive_rebalance",
            "baseline_ppl": baseline_ppl,
            "results": results
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all",
                       choices=["hybrid", "progressive", "all"])
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()
    
    recovery = HybridStrategy()
    eval_texts = recovery.load_data(args.num_samples)
    
    results = {}
    
    if args.strategy in ["hybrid", "all"]:
        results["hybrid"] = recovery.strategy_hybrid_rebalance_prune(eval_texts)
    
    if args.strategy in ["progressive", "all"]:
        results["progressive"] = recovery.strategy_progressive_rebalance(eval_texts)
    
    # Save results
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "phase5_3_hybrid_strategy.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n[OK] Saved results to {output_dir}/phase5_3_hybrid_strategy.json")


if __name__ == "__main__":
    main()
