import os
import json
import argparse
import scipy.stats as stats
import logging
from datetime import datetime
from code.models.registry import get_model
from code.metrics.geometric_ed import EffectiveDimension
from code.metrics.causal_jacobian import LogitJacobian
from code.metrics.propagation_naudc import PropagationDrift
from code.interventions.ablation import LayerAblation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Phase 12A: Proxy Triad Validation on 7B Models")
    parser.add_argument("--model", type=str, default="mistral-7b", help="Model key (e.g. mistral-7b, llama-3-8b)")
    parser.add_argument("--samples", type=int, default=30, help="Number of calibration samples")
    parser.add_argument("--out", type=str, default="experiments/phase12", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_file = os.path.join(args.out, f"{args.model}_proxy_validation.json")

    logger.info(f"Loading {args.model}...")
    wrapper = get_model(args.model)
    
    n_layers = wrapper.model.cfg.n_layers
    results = {
        "model": args.model,
        "n_layers": n_layers,
        "samples": args.samples,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "correlations": {}
    }

    # 1. Ground Truth (Layer Ablation Delta CE)
    logger.info("--- 1. Ground Truth (Layer Ablation) ---")
    ablation = LayerAblation(wrapper)
    ground_truth = ablation.map_layer_sensitivity(num_samples=args.samples)
    
    # 2. Geometric Proxy (Effective Dimension)
    logger.info("--- 2. Geometric Proxy (ED) ---")
    ed_scores = EffectiveDimension.compute_all_layers(wrapper, num_samples=args.samples)
    
    # 3. Causal Proxy (Jacobian Norm)
    logger.info("--- 3. Causal Proxy (Jacobian) ---")
    jac_scores = LogitJacobian.compute_all_layers(wrapper, num_samples=args.samples)
    
    # 4. Propagation Proxy (nAUDC)
    logger.info("--- 4. Propagation Proxy (nAUDC) ---")
    naudc_scores = PropagationDrift.compute_all_layers(wrapper, num_samples=args.samples)

    # Compile vectors
    layers = [f"layer_{i}" for i in range(n_layers)]
    gt_vec = [ground_truth[l] for l in layers]
    ed_vec = [ed_scores[l] for l in layers]
    jac_vec = [jac_scores[l] for l in layers]
    naudc_vec = [naudc_scores[l] for l in layers]
    
    # We combine the three proxies into a single "Ensemble" score by normalizing and taking geometric mean
    # Lower ED is worse (highly compressible = less sensitive) -> so we invert ED (1/ED) for sensitivity
    # Higher Jacobian = more sensitive
    # Higher nAUDC = more drift = more sensitive
    
    # Safe normalization
    def normalize(vec):
        v = [float(x) for x in vec]
        v_min, v_max = min(v), max(v)
        if v_max == v_min: return [0.0] * len(v)
        return [(x - v_min) / (v_max - v_min) for x in v]

    ed_norm = normalize(ed_vec)
    ed_inv_norm = [1.0 - x for x in ed_norm] # Inverted: higher means more sensitive
    jac_norm = normalize(jac_vec)
    naudc_norm = normalize(naudc_vec)
    
    ensemble_vec = [(ed_inv_norm[i] * jac_norm[i] * naudc_norm[i])**(1/3) for i in range(n_layers)]

    for l in layers:
        results["metrics"][l] = {
            "ground_truth_delta_ce": ground_truth[l],
            "ed": ed_scores[l],
            "jacobian": jac_scores[l],
            "naudc": naudc_scores[l],
            "ensemble_score": ensemble_vec[int(l.split("_")[1])]
        }

    # Spearman correlations
    def spearman(a, b):
        rho, pval = stats.spearmanr(a, b)
        return {"rho": float(rho), "p_value": float(pval)}

    results["correlations"]["ed_vs_gt"] = spearman(ed_inv_norm, gt_vec)
    results["correlations"]["jacobian_vs_gt"] = spearman(jac_norm, gt_vec)
    results["correlations"]["naudc_vs_gt"] = spearman(naudc_norm, gt_vec)
    results["correlations"]["ensemble_vs_gt"] = spearman(ensemble_vec, gt_vec)

    logger.info("--- Final Spearman Correlations vs Ground Truth (Delta CE) ---")
    logger.info(f"ED:       {results['correlations']['ed_vs_gt']['rho']:.4f}")
    logger.info(f"Jacobian: {results['correlations']['jacobian_vs_gt']['rho']:.4f}")
    logger.info(f"nAUDC:    {results['correlations']['naudc_vs_gt']['rho']:.4f}")
    logger.info(f"Ensemble: {results['correlations']['ensemble_vs_gt']['rho']:.4f}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved proxy validation results to {out_file}")

if __name__ == "__main__":
    main()
