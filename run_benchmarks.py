import torch
import json
import os
import logging
from models.model_wrapper import HookedModelWrapper
from metrics.geometric_ed import EffectiveDimension
from metrics.causal_jacobian import LogitJacobian
from metrics.propagation_naudc import PropagationDrift
from metrics.fisher_information import FisherInformation
from metrics.ensemble_proxy import EnsembleProxy
from interventions.ablation import LayerAblation
from scipy.stats import spearmanr

# Setup logging for reproducibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(model_name="mistralai/Mistral-7B-v0.1", num_samples=50):
    """
    Run full attribution metrics pipeline: ablation → proxies → correlation analysis.
    
    Args:
        model_name: HuggingFace model identifier (str)
        num_samples: Number of calibration samples (int, >= 10)
    
    Raises:
        ValueError: If inputs are invalid
        Exception: If model loading fails
    """
    # Input validation
    if not isinstance(num_samples, int) or num_samples < 10:
        raise ValueError(f"num_samples must be int >= 10, got {num_samples}")
    if not isinstance(model_name, str):
        raise ValueError(f"model_name must be str, got {type(model_name)}")
    
    # Ensure output directory exists
    os.makedirs("experiments", exist_ok=True)
    
    logger.info(f"Starting Pipeline for {model_name} ({num_samples} samples)")
    
    # 1. Initialize Model & Tools
    logger.info("Step 1: Initializing Hooked Model Wrapper...")
    try:
        wrapper = HookedModelWrapper(model_name, load_in_8bit=True)
        ablation_tool = LayerAblation(wrapper)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise
    
    results = {}
    
    # Get the actual number of layers in the model
    n_layers = wrapper.model.cfg.n_layers
    logger.info(f"Model has {n_layers} layers")

    # Sample counts: scale proportionally
    n_ablation = num_samples
    n_ed = min(num_samples * 2, 1000)  # ED is cheap (forward-only), use more
    n_jacobian = num_samples
    n_naudc = num_samples
    n_fisher = num_samples

    # 2. Get Ground Truth (Delta PPL)
    logger.info("Step 2: Mapping Ground Truth Sensitivity via Layer-wise Ablation...")
    ground_truth = ablation_tool.map_layer_sensitivity(num_samples=n_ablation)

    # 3. Compute Proxy Triad
    logger.info("Step 3: Computing Mechanistic Proxies (ED, Jacobian, nAUDC, Fisher)...")
    ed_scores = EffectiveDimension.compute_all_layers(wrapper, num_samples=n_ed)
    jacobian_scores = LogitJacobian.compute_all_layers(wrapper, num_samples=n_jacobian)
    drift_scores = PropagationDrift.compute_all_layers(wrapper, num_samples=n_naudc)
    fisher_scores = FisherInformation.compute_all_layers(wrapper, num_samples=n_fisher)
    
    # 3b. Compute Ensemble Proxies (sign-corrected: all metrics aligned to higher=more important)
    logger.info("Step 3b: Computing Sign-Corrected Ensemble Proxies...")
    ensemble_mean = EnsembleProxy.compute_sign_corrected_mean(ed_scores, jacobian_scores, drift_scores)
    ensemble_product = EnsembleProxy.compute_sign_corrected_product(ed_scores, jacobian_scores)
    ensemble_weighted = EnsembleProxy.compute_weighted_sum(ed_scores, jacobian_scores, drift_scores)
    logger.info("Ensemble proxies computed (3-way mean, ED×Jac product, weighted sum)")

    # 4. Statistical Correlation Analysis (with sign correction for interpretation)
    logger.info("Step 4: Performing Spearman Correlation Analysis...")
    gt_list = [ground_truth[f"layer_{i}"] for i in range(n_layers)]
    
    # Raw metrics with inverse polarity noted
    metrics_to_test = [
        ("Geometric (ED)", ed_scores, False),
        ("Causal (Jacobian)", jacobian_scores, True),  # Inverted polarity
        ("Propagation (nAUDC)", drift_scores, True),   # Inverted polarity
        ("Fisher Information", fisher_scores, False),
        ("Ensemble (ED+invJac+invnAUDC)", ensemble_mean, False),
        ("Ensemble (ED×invJac)", ensemble_product, False),
        ("Ensemble (Weighted Sum)", ensemble_weighted, False),
    ]
    
    # Keep track of both raw and sign-corrected correlations
    raw_correlations = {}
    sign_corrected_correlations = {}
    
    for name, scores, is_inverted in metrics_to_test:
        score_list = [scores[f"layer_{i}"] for i in range(n_layers)]
        rho_raw, p_val_raw = spearmanr(score_list, gt_list)
        
        # For research: report absolute rho with correction note
        rho_report = abs(rho_raw) if is_inverted else rho_raw
        p_val_report = p_val_raw
        
        if is_inverted:
            label = f"{name} [sign-corrected]"
        else:
            label = name
            
        results[name] = {
            "spearman_rho": float(rho_report), 
            "p_value": float(p_val_report),
            "is_sign_corrected": is_inverted
        }
        raw_correlations[name] = float(rho_raw)
        sign_corrected_correlations[name] = float(rho_report)
        
        logger.info(f"Result for {label}: Spearman Rho = {rho_report:.4f} (p = {p_val_report:.4f})")

    # 5. Save Results to JSON for review
    output_path = f"experiments/sensitivity_results_{num_samples}samples.json"
    
    # Include raw per-layer scores for analysis and reproducibility
    raw_scores = {
        "ground_truth_delta_ce": ground_truth,
        "geometric_ed": ed_scores,
        "causal_jacobian": jacobian_scores,
        "propagation_naudc": drift_scores,
        "fisher_information": fisher_scores,
        "ensemble_mean": ensemble_mean,
        "ensemble_product": ensemble_product,
        "ensemble_weighted": ensemble_weighted,
    }
    
    all_results = {
        "correlations": results,
        "raw_scores": raw_scores,
        "correlation_metadata": {
            "raw_correlations": raw_correlations,
            "sign_corrected_correlations": sign_corrected_correlations,
        },
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
            },
            "notes": "Correlations for Jacobian and nAUDC are sign-corrected (inverted polarity). Report as abs(rho) for clarity."
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"Pipeline Complete. Results saved to {output_path}")
    logger.info(f"Total metrics tested: {len(results)}")
    logger.info(f"Samples used: ablation={n_ablation}, ED={n_ed}, Jacobian={n_jacobian}, nAUDC={n_naudc}, Fisher={n_fisher}")

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
