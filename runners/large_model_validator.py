import argparse
import os
import torch
import json
import logging
import time
from code.models.model_wrapper import HookedModelWrapper
from code.metrics.geometric_ed import EffectiveDimension
from code.metrics.causal_jacobian import LogitJacobian
from code.metrics.propagation_naudc import PropagationDrift
from code.interventions.ablation import LayerAblation
from code.utils.manifest import create_manifest, save_manifest, update_manifest_status
from scipy.stats import spearmanr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase8_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Phase8Validator")

def run_large_model_validation(model_id, num_samples, precision, device_map, dataset="wikitext2"):
    """
    Phase 8 specialized runner for Mistral-7B and Llama-3-8B.
    Handles large-model specific loading and multi-GPU optimization.
    """
    phase = "8"
    model_family = "mistral" if "mistral" in model_id.lower() else "llama"
    run_id = f"val_{model_family}_{int(time.time())}"
    output_dir = f"experiments/phase{phase}/{model_family}/{dataset}/{run_id}"
    
    # 1. Initialize Manifest
    logger.info(f"Initializing Phase 8 validation for {model_id}")
    manifest = create_manifest(
        phase=phase,
        model_family=model_family,
        model_id=model_id,
        dataset=dataset,
        sample_count=num_samples,
        seed=42,
        precision=precision,
        custom_metadata={"device_map": device_map}
    )
    save_manifest(manifest, output_dir)
    
    try:
        # Load Model with specialized settings for 3090s
        logger.info(f"Loading model {model_id} with precision {precision} and device_map {device_map}...")

        # Map precision string to HookedModelWrapper params
        load_in_8bit = (precision == "int8")
        load_in_4bit = (precision == "int4")

        wrapper = HookedModelWrapper(
            model_id, 
            load_in_8bit=load_in_8bit, 
            load_in_4bit=load_in_4bit,
            device_map=device_map
        )

        
        n_layers = wrapper.model.cfg.n_layers
        logger.info(f"✓ Model loaded with {n_layers} layers")
        
        # 3. Ground Truth Mapping (PPL Delta)
        # For 7B models, we might want to use fewer samples for PPL if it's too slow
        logger.info(f"Step 1: Mapping Ground Truth Sensitivity ({num_samples} samples)...")
        ablation_tool = LayerAblation(wrapper)
        ground_truth = ablation_tool.map_layer_sensitivity(num_samples=num_samples)
        
        # 4. Compute Proxy Triad
        logger.info("Step 2: Computing Proxy Triad...")
        
        # Geometric ED (Cheap)
        ed_scores = EffectiveDimension.compute_all_layers(wrapper, num_samples=num_samples)
        
        # Causal Jacobian (Medium)
        jac_scores = LogitJacobian.compute_all_layers(wrapper, num_samples=num_samples)
        
        # Propagation nAUDC (Expensive on large models)
        drift_scores = PropagationDrift.compute_all_layers(wrapper, num_samples=num_samples)
        
        # 5. Correlation Analysis
        logger.info("Step 3: Correlation Analysis...")
        gt_list = [ground_truth[f"layer_{i}"] for i in range(n_layers)]
        
        correlations = {}
        for name, scores in [("ED", ed_scores), ("Jacobian", jac_scores), ("nAUDC", drift_scores)]:
            score_list = [scores[f"layer_{i}"] for i in range(n_layers)]
            rho, p_val = spearmanr(score_list, gt_list)
            correlations[name] = {
                "spearman_rho": float(rho),
                "p_value": float(p_val)
            }
            logger.info(f"  {name} Correlation: rho={rho:.4f}, p={p_val:.4f}")
            
        # 6. Save Results
        results = {
            "correlations": correlations,
            "raw_scores": {
                "ground_truth": ground_truth,
                "ed": ed_scores,
                "jacobian": jac_scores,
                "naudc": drift_scores
            }
        }
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        update_manifest_status(output_dir, "completed", end_time=True)
        logger.info(f"✓ Phase 8 Validation complete for {model_id}")
        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        update_manifest_status(output_dir, f"failed: {str(e)}", end_time=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8: Large Model Validation Runner")
    parser.add_argument("--model", type=str, required=True, help="HF model ID (e.g., mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--samples", type=int, default=50, help="Number of calibration samples")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8", "int4"], default="int8")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for multi-GPU")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    
    args = parser.parse_args()
    
    # Simple check for 4-bit support warning
    if args.precision == "int4":
        logger.warning("Warning: int4 precision requested, but current HookedModelWrapper only supports int8 via TransformerLens. Falling back to int8 if available.")
    
    run_large_model_validation(
        model_id=args.model,
        num_samples=args.samples,
        precision=args.precision,
        device_map=args.device_map,
        dataset=args.dataset
    )
