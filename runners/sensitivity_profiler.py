import argparse
import os
import torch
import logging
import time
from code.models.model_wrapper import HookedModelWrapper
from code.metrics.geometric_ed import EffectiveDimension
from code.metrics.causal_jacobian import LogitJacobian
from code.metrics.propagation_naudc import PropagationDrift
from code.metrics.fisher_information import FisherInformation
from code.metrics.ensemble_proxy import EnsembleProxy
from code.interventions.ablation import LayerAblation
from code.utils.manifest import create_manifest, save_manifest, update_manifest_status
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_sensitivity_sweep(model_name, num_samples, phase, dataset):
    run_id = f"sensitivity_{int(time.time())}"
    model_family = "gpt2" if "gpt2" in model_name.lower() else "llama" if "llama" in model_name.lower() else "mistral"
    output_dir = f"experiments/phase{phase}/{model_family}/{dataset}/{run_id}"
    
    manifest = create_manifest(phase, model_family, model_name, dataset, num_samples, 42)
    save_manifest(manifest, output_dir)
    
    try:
        wrapper = HookedModelWrapper(model_name, load_in_8bit=True)
        n_layers = wrapper.model.cfg.n_layers
        
        # Ground Truth
        ablation_tool = LayerAblation(wrapper)
        ground_truth = ablation_tool.map_layer_sensitivity(num_samples=num_samples)
        
        # Proxies
        ed_scores = EffectiveDimension.compute_all_layers(wrapper, num_samples=num_samples)
        jac_scores = LogitJacobian.compute_all_layers(wrapper, num_samples=num_samples)
        drift_scores = PropagationDrift.compute_all_layers(wrapper, num_samples=num_samples)
        
        # Correlation
        gt_list = [ground_truth[f"layer_{i}"] for i in range(n_layers)]
        results = {}
        for name, scores in [("ED", ed_scores), ("Jacobian", jac_scores), ("nAUDC", drift_scores)]:
            score_list = [scores[f"layer_{i}"] for i in range(n_layers)]
            rho, p = spearmanr(score_list, gt_list)
            results[name] = {"rho": float(rho), "p": float(p)}
            logger.info(f"{name} Correlation: {rho:.4f}")
            
        import json
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({"correlations": results, "raw": {"gt": ground_truth, "ed": ed_scores, "jac": jac_scores, "drift": drift_scores}}, f, indent=4)
            
        update_manifest_status(output_dir, "completed", end_time=True)
    except Exception as e:
        logger.error(f"Run failed: {e}")
        update_manifest_status(output_dir, f"failed: {str(e)}", end_time=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--phase", type=str, default="2")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    args = parser.parse_args()
    run_sensitivity_sweep(args.model, args.samples, args.phase, args.dataset)
