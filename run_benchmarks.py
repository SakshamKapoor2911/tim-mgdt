import torch
import json
import os
from models.model_wrapper import HookedModelWrapper
from metrics.geometric_ed import EffectiveDimension
from metrics.causal_jacobian import LogitJacobian
from metrics.propagation_naudc import PropagationDrift
from metrics.fisher_information import FisherInformation
from metrics.ensemble_proxy import EnsembleProxy
from interventions.ablation import LayerAblation
from scipy.stats import spearmanr

def run_pipeline(model_name="mistralai/Mistral-7B-v0.1", num_samples=50):
    print(f"--- Starting Pipeline for {model_name} ({num_samples} samples) ---")
    
    # 1. Initialize Model & Tools
    # Note: Use load_in_8bit=True to fit in 16GB VRAM GPUs (A10G, T4, etc.)
    print("Step 1: Initializing Hooked Model Wrapper...")
    wrapper = HookedModelWrapper(model_name, load_in_8bit=True)
    ablation_tool = LayerAblation(wrapper)
    
    results = {}
    
    # Get the actual number of layers in the model
    n_layers = wrapper.model.cfg.n_layers
    print(f"Model has {n_layers} layers")

    # Sample counts: scale proportionally
    n_ablation = num_samples
    n_ed = min(num_samples * 2, 1000)  # ED is cheap (forward-only), use more
    n_jacobian = num_samples
    n_naudc = num_samples
    n_fisher = num_samples

    # 2. Get Ground Truth (Delta PPL)
    print("Step 2: Mapping Ground Truth Sensitivity via Layer-wise Ablation...")
    ground_truth = ablation_tool.map_layer_sensitivity(num_samples=n_ablation)

    # 3. Compute Proxy Triad
    print("Step 3: Computing Mechanistic Proxies (ED, Jacobian, nAUDC, Fisher)...")
    ed_scores = EffectiveDimension.compute_all_layers(wrapper, num_samples=n_ed)
    jacobian_scores = LogitJacobian.compute_all_layers(wrapper, num_samples=n_jacobian)
    drift_scores = PropagationDrift.compute_all_layers(wrapper, num_samples=n_naudc)
    fisher_scores = FisherInformation.compute_all_layers(wrapper, num_samples=n_fisher)
    
    # 3b. Compute Ensemble Proxies (sign-corrected: all metrics aligned to higher=more important)
    print("\nStep 3b: Computing Sign-Corrected Ensemble Proxies...")
    ensemble_mean = EnsembleProxy.compute_sign_corrected_mean(ed_scores, jacobian_scores, drift_scores)
    ensemble_product = EnsembleProxy.compute_sign_corrected_product(ed_scores, jacobian_scores)
    ensemble_weighted = EnsembleProxy.compute_weighted_sum(ed_scores, jacobian_scores, drift_scores)
    print("  \u2713 Ensemble proxies computed (3-way mean, ED\u00d7Jac product, weighted sum)")

    # 4. Statistical Correlation Analysis
    print("\nStep 4: Performing Spearman Correlation Analysis...")
    gt_list = [ground_truth[f"layer_{i}"] for i in range(n_layers)]
    
    metrics_to_test = [
        ("Geometric (ED)", ed_scores),
        ("Causal (Jacobian)", jacobian_scores), 
        ("Propagation (nAUDC)", drift_scores),
        ("Fisher Information", fisher_scores),
        ("Ensemble (3-way Mean)", ensemble_mean),
        ("Ensemble (ED\u00d7invJac)", ensemble_product),
        ("Ensemble (Weighted Sum)", ensemble_weighted),
    ]
    
    for name, scores in metrics_to_test:
        score_list = [scores[f"layer_{i}"] for i in range(n_layers)]
        rho, p_val = spearmanr(score_list, gt_list)
        results[name] = {
            "spearman_rho": float(rho), 
            "p_value": float(p_val)
        }
        print(f"Result for {name}: Spearman Rho = {rho:.4f} (p = {p_val:.4f})")

    # 5. Save Results to JSON for review
    output_path = f"experiments/sensitivity_results_{num_samples}samples.json"
    
    # Include raw per-layer scores for analysis and reproducibility
    raw_scores = {
        "ground_truth_delta_ce": ground_truth,
        "geometric_ed": ed_scores,
        "causal_jacobian": jacobian_scores,
        "propagation_naudc": drift_scores,
        "fisher_information": fisher_scores,
        "ensemble_3way_mean": ensemble_mean,
        "ensemble_ed_x_inv_jacobian": ensemble_product,
        "ensemble_weighted_sum": ensemble_weighted,
    }
    
    all_results = {
        "correlations": results,
        "raw_scores": raw_scores,
        "config": {
            "model": model_name,
            "n_layers": n_layers,
            "num_samples": num_samples,
            "sample_counts": {
                "ablation": n_ablation,
                "ed": n_ed, 
                "jacobian": n_jacobian,
                "naudc": n_naudc,
                "fisher": n_fisher,
            }
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n--- Pipeline Complete. Results saved to {output_path} ---")
    print(f"Total metrics tested: {len(results)}")
    print(f"Samples used: ablation={n_ablation}, ED={n_ed}, Jacobian={n_jacobian}, nAUDC={n_naudc}, Fisher={n_fisher}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    MODEL_FOR_TESTING = "gpt2"
    NUM_SAMPLES = 500  # Number of calibration samples (50 for quick test, 500 for full run)
    
    # UNCOMMENT FOR LARGER MODELS (requires more VRAM):
    # MODEL_FOR_TESTING = "gpt2-medium"  # ~355M params
    # MODEL_FOR_TESTING = "mistralai/Mistral-7B-v0.1"  # needs 16GB+

    # Clear cache before starting to maximize VRAM headroom
    torch.cuda.empty_cache()

    run_pipeline(model_name=MODEL_FOR_TESTING, num_samples=NUM_SAMPLES)
