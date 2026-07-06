"""
Phase 14: Final Actionable Interpretability Pipeline
Script 1: Exhaustive Sensitivity Measurement (Oracle Compute Cost)

This script drops the flawed proxy metrics and directly measures the Ground Truth 
causal sensitivity (delta_ce) of each layer by ablating it and evaluating perplexity.
It logs the exact wall-clock time (Oracle Compute Cost) to prove that deriving 
the Bathtub Curve is cheap and highly actionable.
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from datetime import datetime

# Import utilities from Phase 11
from code.phases.phase11.utils import get_device, compute_perplexity_c4

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_loss_c4(model, tokenizer, device, num_samples=30):
    """Calculate the cross-entropy loss on a held-out C4 set."""
    ppl = compute_perplexity_c4(model, tokenizer, device, max_samples=num_samples)
    import math
    return math.log(ppl)

class LayerKnockout:
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Introspect architecture
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers = self.model.model.layers
        else:
            raise ValueError("Expected a Llama/Mistral-style model with model.model.layers")

    def measure_sensitivity(self, num_samples=30):
        start_time = time.time()
        logger.info("Computing Baseline C4 Loss...")
        
        baseline_loss = evaluate_loss_c4(self.model, self.tokenizer, self.device, num_samples)
        logger.info(f"Baseline Loss: {baseline_loss:.4f}")
        
        n_layers = len(self.layers)
        sensitivity_scores = {}
        
        logger.info(f"Starting exhaustive layer knockout (Oracle mapping) for {n_layers} layers...")
        for i in range(n_layers):
            layer = self.layers[i]
            
            # To architecturally safely ablate the layer's contribution to the residual stream,
            # we simply zero out the final output projections (o_proj and down_proj).
            # This causes the attention and MLP branches to output 0, meaning the 
            # residual stream passes through unmodified (Identity).
            has_o_proj = hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj")
            has_down_proj = hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj")
            
            if has_o_proj:
                o_proj_weight_orig = layer.self_attn.o_proj.weight.data.clone()
                layer.self_attn.o_proj.weight.data.zero_()
            if has_down_proj:
                down_proj_weight_orig = layer.mlp.down_proj.weight.data.clone()
                layer.mlp.down_proj.weight.data.zero_()
            
            try:
                loss = evaluate_loss_c4(self.model, self.tokenizer, self.device, num_samples)
                delta_ce = loss - baseline_loss
                sensitivity_scores[f"layer_{i}"] = delta_ce
                logger.info(f"Layer {i}: Delta CE = {delta_ce:.4f}")
            finally:
                # Restore weights
                if has_o_proj:
                    layer.self_attn.o_proj.weight.data.copy_(o_proj_weight_orig)
                if has_down_proj:
                    layer.mlp.down_proj.weight.data.copy_(down_proj_weight_orig)

        oracle_compute_time_s = time.time() - start_time
        logger.info(f"Oracle Compute Cost: {oracle_compute_time_s:.2f} seconds.")
        
        return {
            "model": self.model_name,
            "baseline_loss": baseline_loss,
            "oracle_compute_time_s": oracle_compute_time_s,
            "oracle_compute_time_min": oracle_compute_time_s / 60.0,
            "sensitivities": sensitivity_scores
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    device = get_device()
    out_dir = "experiments/phase14_final"
    os.makedirs(out_dir, exist_ok=True)
    
    # Clean model name for filename
    model_short = args.model.split("/")[-1]
    out_file = os.path.join(out_dir, f"{model_short}_sensitivity_map.json")

    knockout = LayerKnockout(args.model, device)
    results = knockout.measure_sensitivity(num_samples=args.samples)
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved Bathtub Curve sensitivity map to {out_file}")

if __name__ == "__main__":
    main()
