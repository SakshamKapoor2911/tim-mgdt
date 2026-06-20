"""
Phase 5.1: Benchmarking Engine
Evaluates accuracy (Perplexity) and latency impact of layer pruning.

Usage:
    python phase5_1/evaluate_pruned_model.py --model gpt2 --layers-to-prune 10 1 9 --dataset wikitext2
    
Outputs:
    - phase5_1_benchmark_results.json: Full metrics (PPL, latency, speedup)
    - phase5_1_benchmark_summary.txt: Human-readable comparison
"""

import torch
import torch.nn.functional as F
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# TransformerLens
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

# HuggingFace
from datasets import load_dataset
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PruningBenchmark:
    """Benchmarks pruned vs. baseline model on WikiText-2."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        """
        Initialize benchmark engine.
        
        Args:
            model_name: HuggingFace model ID (gpt2, gpt2-medium, etc.)
            device: Compute device ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.baseline_metrics = {}
        self.pruned_metrics = {}
        
        logger.info(f"Loading {model_name} on {device}...")
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Load HookedTransformer model."""
        try:
            self.model = HookedTransformer.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=torch.float32
            )
            logger.info(f"[OK] Loaded {self.model_name}: {self.model.cfg.n_layers} layers")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load HuggingFace tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("[OK] Loaded tokenizer")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def load_wikitext2_subset(self, num_samples: int = 100, split: str = "test") -> List[str]:
        """
        Load WikiText-2 dataset subset for evaluation.
        
        Args:
            num_samples: Number of samples to load
            split: Dataset split ('test', 'validation', 'train')
            
        Returns:
            List of text samples
        """
        logger.info(f"Loading WikiText-2 ({split} split, n={num_samples})...")
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            # Extract text field and filter empty samples
            samples = []
            for i in range(min(num_samples, len(dataset))):
                item = dataset[i]
                # Handle both dict and direct text formats
                if isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = str(item)
                if text.strip():
                    samples.append(text)
            logger.info(f"[OK] Loaded {len(samples)} samples from WikiText-2")
            return samples[:num_samples]
        except Exception as e:
            logger.error(f"Failed to load WikiText-2: {e}")
            raise
    
    def compute_perplexity(self, texts: List[str], pruned: bool = False) -> Dict:
        """
        Compute perplexity on a list of texts (no pruning applied).
        
        Args:
            texts: List of text samples
            pruned: If True, apply pruning mask during inference
            
        Returns:
            Dictionary with PPL, loss, num_tokens
        """
        total_loss = 0.0
        total_tokens = 0
        
        logger.info(f"Computing Perplexity (pruned={pruned})...")
        
        for idx, text in enumerate(texts):
            if not text.strip():
                continue
            
            try:
                # Tokenize
                tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                seq_len = tokens.shape[1]
                
                if seq_len < 2:
                    continue
                
                # Forward pass
                if pruned:
                    with torch.no_grad():
                        logits = self.model(tokens)
                else:
                    with torch.no_grad():
                        logits = self.model(tokens)
                
                # Compute loss (next token prediction)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = tokens[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item() * (seq_len - 1)
                total_tokens += seq_len - 1
                
                if (idx + 1) % max(1, len(texts) // 5) == 0:
                    logger.info(f"  Processed {idx + 1}/{len(texts)} samples")
            
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                continue
        
        ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float('inf')
        
        logger.info(f"[OK] PPL = {ppl:.2f} (avg_loss = {total_loss / total_tokens:.4f})")
        
        return {
            "perplexity": ppl,
            "avg_loss": total_loss / total_tokens if total_tokens > 0 else 0.0,
            "total_tokens": total_tokens
        }
    
    def measure_latency(self, texts: List[str], pruned: bool = False, num_repeats: int = 3) -> Dict:
        """
        Measure inference latency (ms per token).
        
        Args:
            texts: List of text samples
            pruned: If True, apply pruning mask
            num_repeats: Number of times to repeat measurement
            
        Returns:
            Dictionary with latency stats
        """
        latencies = []
        token_counts = []
        
        logger.info(f"Measuring Latency (pruned={pruned}, repeats={num_repeats})...")
        
        for text in texts[:5]:  # Use first 5 texts for timing
            if not text.strip():
                continue
            
            tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            seq_len = tokens.shape[1]
            
            # Warmup
            with torch.no_grad():
                _ = self.model(tokens)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # Measure
            start_time = time.time()
            for _ in range(num_repeats):
                with torch.no_grad():
                    _ = self.model(tokens)
                if self.device == "cuda":
                    torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = (end_time - start_time) * 1000 / num_repeats  # ms
            ms_per_token = elapsed / seq_len
            
            latencies.append(ms_per_token)
            token_counts.append(seq_len)
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        logger.info(f"[OK] Latency = {avg_latency:.4f} ms/token (avg seq_len={sum(token_counts)/len(token_counts):.0f})")
        
        return {
            "ms_per_token": avg_latency,
            "latencies": latencies,
            "token_counts": token_counts
        }
    
    def apply_pruning_mask(self, layers_to_prune: List[int]) -> None:
        """
        Apply reversible pruning mask using TransformerLens hooks.
        Skip layer computation by using residual stream bypass.
        
        Args:
            layers_to_prune: List of layer indices to mask
        """
        logger.info(f"Applying pruning mask to layers: {layers_to_prune}...")
        
        # Store residual inputs to use for skipping
        residual_inputs = {}
        
        def capture_resid_input(layer_idx):
            def hook_fn(value, hook):
                residual_inputs[layer_idx] = value.clone()
                return value
            return hook_fn
        
        def skip_layer_output(layer_idx):
            def hook_fn(value, hook):
                # Return the residual input, effectively skipping layer computation
                if layer_idx in residual_inputs:
                    return residual_inputs[layer_idx]
                return value
            return hook_fn
        
        # For each layer to prune, capture input at resid_pre and skip at resid_post
        for layer_idx in layers_to_prune:
            input_hook_name = get_act_name("resid_pre", layer_idx)
            output_hook_name = get_act_name("resid_post", layer_idx)
            
            # Capture input after it enters the layer
            self.model.add_hook(input_hook_name, capture_resid_input(layer_idx))
            # Skip the layer's output by returning the input
            self.model.add_hook(output_hook_name, skip_layer_output(layer_idx))
        
        logger.info(f"[OK] Pruning mask applied to {len(layers_to_prune)} layers (residual skip)")
    
    def remove_pruning_mask(self) -> None:
        """Remove all hooks (revert to baseline)."""
        self.model.remove_all_hook_fns()
        logger.info("[OK] All hooks removed (baseline restored)")
    
    def benchmark_pruning(
        self,
        layers_to_prune: List[int],
        num_samples: int = 100,
        dataset_split: str = "test"
    ) -> Dict:
        """
        Run full benchmark comparing baseline vs. pruned model.
        
        Args:
            layers_to_prune: List of layer indices to prune
            num_samples: Number of samples for evaluation
            dataset_split: WikiText-2 split to use
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("=" * 80)
        logger.info("PHASE 5.1: PRUNING BENCHMARK")
        logger.info("=" * 80)
        
        # Load evaluation data
        eval_texts = self.load_wikitext2_subset(num_samples=num_samples, split=dataset_split)
        
        # Baseline metrics
        logger.info("\n[BASELINE] Computing metrics for full model...")
        baseline_ppl = self.compute_perplexity(eval_texts, pruned=False)
        baseline_lat = self.measure_latency(eval_texts, pruned=False)
        
        self.baseline_metrics = {
            **baseline_ppl,
            **baseline_lat,
            "num_layers": self.model.cfg.n_layers
        }
        
        # Pruned metrics
        logger.info(f"\n[PRUNED] Applying mask to layers {layers_to_prune}...")
        self.apply_pruning_mask(layers_to_prune)
        
        logger.info("[PRUNED] Computing metrics for masked model...")
        pruned_ppl = self.compute_perplexity(eval_texts, pruned=True)
        pruned_lat = self.measure_latency(eval_texts, pruned=True)
        
        self.pruned_metrics = {
            **pruned_ppl,
            **pruned_lat,
            "num_layers": self.model.cfg.n_layers - len(layers_to_prune),
            "layers_pruned": layers_to_prune
        }
        
        # Remove mask
        self.remove_pruning_mask()
        
        # Calculate comparative metrics
        ppl_increase_pct = (
            (pruned_ppl["perplexity"] - baseline_ppl["perplexity"]) 
            / baseline_ppl["perplexity"] * 100
        )
        speedup = (
            baseline_lat["ms_per_token"] / pruned_lat["ms_per_token"]
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)
        logger.info(f"\n[BASELINE] GPT-2 ({self.model.cfg.n_layers} layers)")
        logger.info(f"  Perplexity: {baseline_ppl['perplexity']:.2f}")
        logger.info(f"  Latency: {baseline_lat['ms_per_token']:.4f} ms/token")
        
        logger.info(f"\n[PRUNED] GPT-2 ({self.model.cfg.n_layers - len(layers_to_prune)} layers, removed {layers_to_prune})")
        logger.info(f"  Perplexity: {pruned_ppl['perplexity']:.2f} ({ppl_increase_pct:+.2f}%)")
        logger.info(f"  Latency: {pruned_lat['ms_per_token']:.4f} ms/token")
        
        logger.info(f"\n[IMPROVEMENT]")
        logger.info(f"  Speedup: {speedup:.2f}x ({(1 - 1/speedup) * 100:.1f}% faster)")
        logger.info(f"  PPL Increase: {ppl_increase_pct:.2f}%")
        logger.info(f"  Model Compression: {len(layers_to_prune) / self.model.cfg.n_layers * 100:.1f}%")
        
        if ppl_increase_pct <= 5.0:
            logger.info("\n[SUCCESS] PPL increase within 5% tolerance ✓")
        else:
            logger.warning(f"\n[WARNING] PPL increase ({ppl_increase_pct:.2f}%) exceeds 5% tolerance")
        
        logger.info("=" * 80)
        
        return {
            "baseline": self.baseline_metrics,
            "pruned": self.pruned_metrics,
            "comparison": {
                "ppl_increase_pct": ppl_increase_pct,
                "speedup_factor": speedup,
                "speedup_pct": (1 - 1/speedup) * 100,
                "compression_pct": len(layers_to_prune) / self.model.cfg.n_layers * 100,
                "layers_removed": layers_to_prune
            }
        }
    
    def save_results(self, results: Dict, output_dir: Path = None):
        """Save results to JSON and text files."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "experiments"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON results
        json_path = output_dir / "phase5_1_benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"[OK] Saved JSON results to {json_path}")
        
        # Text summary
        txt_path = output_dir / "phase5_1_benchmark_summary.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 5.1: PRUNING BENCHMARK SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("[BASELINE]\n")
            f.write(f"  Layers: {results['baseline']['num_layers']}\n")
            f.write(f"  Perplexity: {results['baseline']['perplexity']:.2f}\n")
            f.write(f"  Latency: {results['baseline']['ms_per_token']:.4f} ms/token\n\n")
            
            f.write("[PRUNED]\n")
            f.write(f"  Layers: {results['pruned']['num_layers']}\n")
            f.write(f"  Layers Removed: {results['pruned']['layers_pruned']}\n")
            f.write(f"  Perplexity: {results['pruned']['perplexity']:.2f}\n")
            f.write(f"  Latency: {results['pruned']['ms_per_token']:.4f} ms/token\n\n")
            
            f.write("[COMPARISON]\n")
            f.write(f"  PPL Increase: {results['comparison']['ppl_increase_pct']:.2f}%\n")
            f.write(f"  Speedup: {results['comparison']['speedup_factor']:.2f}x ({results['comparison']['speedup_pct']:.1f}% faster)\n")
            f.write(f"  Model Compression: {results['comparison']['compression_pct']:.1f}%\n")
            
            if results['comparison']['ppl_increase_pct'] <= 5.0:
                f.write("\n[VERDICT] PASS - PPL within 5% tolerance [OK]\n")
            else:
                f.write(f"\n[VERDICT] FAIL - PPL increase exceeds tolerance [X]\n")
        
        logger.info(f"[OK] Saved text summary to {txt_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 5.1: Pruning Benchmark Engine"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name (gpt2, gpt2-medium, etc.)"
    )
    parser.add_argument(
        "--layers-to-prune",
        type=int,
        nargs="+",
        default=[10, 1, 9],
        help="Layer indices to prune (space-separated)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of WikiText-2 samples for evaluation"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="WikiText-2 split (test, validation, train)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = PruningBenchmark(model_name=args.model, device=args.device)
    results = benchmark.benchmark_pruning(
        layers_to_prune=args.layers_to_prune,
        num_samples=args.num_samples,
        dataset_split=args.dataset_split
    )
    
    # Save results
    benchmark.save_results(results, output_dir=args.output_dir)
    
    logger.info("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
