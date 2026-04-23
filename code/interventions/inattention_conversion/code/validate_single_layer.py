"""
Phase 7.1: InAttention Conversion - Single Layer Validation

Validates InAttention conversion on Layer 10 (proven elastic from Phase 6).

Goal: Compare standard vs InAttention attention, measuring:
- PPL impact (should be <5% delta)
- VRAM reduction (target >30%)
- Inference speedup (target >1.5x)
"""

import torch
import torch.nn.functional as F
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
import sys

from transformer_lens import HookedTransformer
from datasets import load_dataset
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from attention_hooks import AttentionHookManager, MemoryProfiler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InAttentionValidator:
    """Validate InAttention conversion on Layer 10."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading {model_name} on {device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name, device=device, dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(
            f"[OK] Model loaded: {self.model.cfg.n_layers} layers, "
            f"{self.model.cfg.d_model} hidden, {self.model.cfg.n_heads} heads"
        )
        
        self.results = {
            "metadata": {
                "model": model_name,
                "device": device,
                "timestamp": datetime.now().isoformat(),
                "phase": "Phase 7.1: InAttention Conversion"
            }
        }
    
    def load_data(self, num_samples: int = 500) -> List[str]:
        """Load WikiText-2 data for validation."""
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
    
    def compute_perplexity_and_time(self, texts: List[str]) -> Tuple[float, int, float, float]:
        """
        Compute PPL and measure inference time.
        
        Returns:
            (ppl, token_count, total_loss, inference_time_sec)
        """
        total_loss = 0
        total_tokens = 0
        
        start_time = time.time()
        
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
        
        inference_time = time.time() - start_time
        
        if total_tokens == 0:
            return 0, 0, 0, 0
        
        ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        return ppl, total_tokens, total_loss, inference_time
    
    def validate_inattention_layer10(self, eval_texts: List[str]) -> Dict:
        """
        Validate InAttention conversion on Layer 10.
        
        Strategy:
        1. Compute baseline (standard attention)
        2. Apply InAttention hook to Layer 10
        3. Measure PPL, VRAM, speed
        4. Compare results
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 7.1: INATTENTION CONVERSION - LAYER 10 VALIDATION")
        logger.info("="*80)
        logger.info(f"Samples: {len(eval_texts)}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # ===== BASELINE: Standard Attention =====
        logger.info("\n--- BASELINE: Standard Attention ---")
        baseline_ppl, baseline_tokens, baseline_loss, baseline_time = \
            self.compute_perplexity_and_time(eval_texts)
        
        logger.info(f"Baseline PPL: {baseline_ppl:.4f}")
        logger.info(f"  Tokens: {baseline_tokens}")
        logger.info(f"  Time: {baseline_time:.2f}s")
        
        # ===== INATTENTION: Layer 10 Only =====
        logger.info("\n--- INATTENTION: Layer 10 ---")
        hook_manager = AttentionHookManager(self.model, layer_idx=10, device=self.device)
        hook_manager.register_inattention_hook(scale_factor=1.0, attend_pos=0)
        
        inattention_ppl, inattention_tokens, inattention_loss, inattention_time = \
            self.compute_perplexity_and_time(eval_texts)
        
        hook_manager.remove_hooks()
        
        logger.info(f"InAttention PPL: {inattention_ppl:.4f}")
        logger.info(f"  Tokens: {inattention_tokens}")
        logger.info(f"  Time: {inattention_time:.2f}s")
        
        # Calculate deltas
        ppl_delta = inattention_ppl - baseline_ppl
        ppl_delta_pct = (ppl_delta / baseline_ppl) * 100
        speedup = baseline_time / inattention_time if inattention_time > 0 else 1.0
        
        logger.info(f"\nResults:")
        logger.info(f"  PPL Delta: {ppl_delta:+.4f} ({ppl_delta_pct:+.2f}%)")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Status: {'✓ PASS' if abs(ppl_delta_pct) < 5.0 else '✗ CHECK'} (target: <5%)")
        
        return {
            "baseline": {
                "ppl": baseline_ppl,
                "tokens": baseline_tokens,
                "loss": baseline_loss,
                "time_sec": baseline_time
            },
            "inattention_layer10": {
                "ppl": inattention_ppl,
                "tokens": inattention_tokens,
                "loss": inattention_loss,
                "time_sec": inattention_time
            },
            "metrics": {
                "ppl_delta": ppl_delta,
                "ppl_delta_pct": ppl_delta_pct,
                "speedup": speedup,
                "pass_criteria": abs(ppl_delta_pct) < 5.0
            }
        }
    
    def validate_progressive_scaling(self, eval_texts: List[str]) -> Dict:
        """
        Test InAttention with different scale factors on Layer 10.
        
        Hypothesis: InAttention + soft scaling might recover PPL further.
        """
        logger.info("\n" + "="*80)
        logger.info("PROGRESSIVE SCALING WITH INATTENTION")
        logger.info("="*80)
        
        # Save original state
        original_state = {}
        for name, param in self.model.named_parameters():
            original_state[name] = param.data.clone()
        
        results_by_config = {}
        scale_factors = [1.0, 0.98, 0.95, 0.90]
        
        for scale in scale_factors:
            # Restore original
            for name, param in self.model.named_parameters():
                param.data = original_state[name].clone()
            
            # Apply scaling if not 1.0
            if scale < 1.0:
                logger.info(f"\nApplying {scale:.2f}x scaling to Layer 10...")
                for name, param in self.model.named_parameters():
                    if ".10." in name or ("blocks" in name and "10" in name):
                        if param.dim() > 0:
                            param.data = param.data * scale
            
            # Add InAttention hook
            hook_manager = AttentionHookManager(self.model, layer_idx=10, device=self.device)
            hook_manager.register_inattention_hook(scale_factor=1.0)
            
            # Test
            ppl, tokens, loss, elapsed_time = self.compute_perplexity_and_time(eval_texts)
            hook_manager.remove_hooks()
            
            ppl_delta_pct = ((ppl - self.results["validation"]["baseline"]["ppl"]) / 
                           self.results["validation"]["baseline"]["ppl"]) * 100
            
            logger.info(f"Scale {scale:.2f}x + InAttention: PPL={ppl:.4f}, Δ={ppl_delta_pct:+.2f}%")
            
            results_by_config[f"inattention_scale_{scale:.2f}x"] = {
                "scale_factor": scale,
                "ppl": ppl,
                "ppl_delta": ppl - self.results["validation"]["baseline"]["ppl"],
                "ppl_delta_pct": ppl_delta_pct,
                "time_sec": elapsed_time
            }
        
        # Restore original
        for name, param in self.model.named_parameters():
            param.data = original_state[name].clone()
        
        return results_by_config
    
    def compare_with_phase6(self) -> Dict:
        """Compare InAttention results with Phase 6 soft scaling baseline."""
        phase6_baseline = 51.43  # From Phase 6 validation (500 samples)
        phase6_optimal = 51.37   # With 0.98x scaling
        
        inattention_ppl = self.results["validation"]["inattention_layer10"]["ppl"]
        baseline_ppl = self.results["validation"]["baseline"]["ppl"]
        
        comparison = {
            "phase6_baseline": phase6_baseline,
            "phase6_optimal_098x": phase6_optimal,
            "phase7_inattention_baseline": baseline_ppl,
            "phase7_inattention_layer10": inattention_ppl,
            "inattention_vs_phase6_baseline": {
                "delta": inattention_ppl - phase6_baseline,
                "delta_pct": ((inattention_ppl - phase6_baseline) / phase6_baseline) * 100
            },
            "inattention_vs_phase6_optimal": {
                "delta": inattention_ppl - phase6_optimal,
                "delta_pct": ((inattention_ppl - phase6_optimal) / phase6_optimal) * 100
            }
        }
        
        return comparison
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive validation report."""
        comp = self.results.get("comparison", {})
        val = self.results.get("validation", {})
        
        markdown = f"""# Phase 7.1 Report: InAttention Conversion on Layer 10

**Date**: {self.results['metadata']['timestamp']}  
**Model**: {self.results['metadata']['model']}  
**Status**: ✓ VALIDATION PHASE  

---

## Executive Summary

Phase 7.1 tests **InAttention conversion** on Layer 10 (proven elastic from Phase 6).

**Goal**: Achieve 4x inference speedup and 30-50% VRAM reduction while maintaining <5% PPL increase.

---

## Results

### Baseline (Standard Attention)
- **PPL**: {val['baseline']['ppl']:.4f}
- **Tokens**: {val['baseline']['tokens']}
- **Time**: {val['baseline']['time_sec']:.2f}s

### InAttention (Layer 10 Only)
- **PPL**: {val['inattention_layer10']['ppl']:.4f}
- **Tokens**: {val['inattention_layer10']['tokens']}
- **Time**: {val['inattention_layer10']['time_sec']:.2f}s
- **Speedup**: {val['metrics']['speedup']:.2f}x

### Impact Assessment
- **PPL Delta**: {val['metrics']['ppl_delta']:+.4f} ({val['metrics']['ppl_delta_pct']:+.2f}%)
- **Pass Criteria** (<5%): {'✓ PASS' if val['metrics']['pass_criteria'] else '✗ NEEDS ADJUSTMENT'}

---

## Comparison with Phase 6

| Metric | Phase 6 Baseline | Phase 6 Optimal | Phase 7.1 InAttention |
|--------|-----------------|-----------------|----------------------|
| PPL | {comp.get('phase6_baseline', 'N/A')} | {comp.get('phase6_optimal_098x', 'N/A')} | {comp.get('phase7_inattention_layer10', 'N/A')} |
| vs Phase 6 Baseline | — | -0.11% | {comp.get('inattention_vs_phase6_baseline', {}).get('delta_pct', 'N/A'):+.2f}% |

---

## Next Steps

1. **If PPL Delta < 5%**: Proceed to progressive layer testing (Layers 4-7)
2. **If PPL Delta > 5%**: Test hybrid scaling (InAttention + 0.98x soft scaling)
3. **Measure VRAM Savings**: Profile actual memory reduction vs theoretical
4. **Speedup Analysis**: Measure actual vs theoretical 4x speedup

---

## Technical Details

**Hook Point**: `blocks.10.attn.hook_pattern`  
**Mechanism**: Force all positions to attend only to position 0's K/V  
**Parameters**: Attend position = 0 (first token)

---

## Files Generated

- `inattention_layer10_results.json`: Raw metrics
- `inattention_layer10_report.md`: This report

---

**Generated by**: Phase 7.1 Validation Script  
**Framework**: TransformerLens + HuggingFace Transformers
"""
        
        return markdown
    
    def run_full_validation(self, num_samples: int = 500) -> None:
        """Run complete InAttention validation."""
        # Load data
        eval_texts = self.load_data(num_samples)
        
        # Run validation
        logger.info("\n" + "="*80)
        logger.info("RUNNING FULL INATTENTION VALIDATION")
        logger.info("="*80)
        
        self.results["validation"] = self.validate_inattention_layer10(eval_texts)
        
        # Progressive scaling test
        logger.info("\nRunning progressive scaling test...")
        progressive = self.validate_progressive_scaling(eval_texts)
        self.results["validation"]["progressive_scaling"] = progressive
        
        # Comparison with Phase 6
        self.results["comparison"] = self.compare_with_phase6()
        
        # Generate report
        markdown_report = self.generate_markdown_report()
        
        # Save results
        self.save_results(markdown_report)
    
    def save_results(self, markdown_report: str) -> None:
        """Save results to JSON and Markdown files."""
        results_dir = Path(__file__).parent.parent / "experiments" / "single_layer" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = results_dir / "inattention_layer10_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n✓ Saved JSON results to: {json_path}")
        
        # Save Markdown
        md_path = results_dir / "inattention_layer10_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        logger.info(f"✓ Saved markdown report to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 7.1: InAttention Validation")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to use")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    logger.info(f"Phase 7.1: InAttention Conversion Validation")
    logger.info(f"  Samples: {args.num_samples}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    
    validator = InAttentionValidator(model_name=args.model, device=args.device)
    validator.run_full_validation(num_samples=args.num_samples)
    
    logger.info("\n✓ Phase 7.1 Validation Complete")


if __name__ == "__main__":
    main()
