import argparse
import os
import json
import logging
import time
from code.models.model_wrapper import HookedModelWrapper
from code.advanced_proxies.head_jacobian import HeadJacobian, HeadJacobianConfig
from code.advanced_proxies.circuit_analyzer import CircuitAnalyzer, TicketConfig
from code.advanced_proxies.pruning_engine import PruningEngine, PruningConfig
from code.utils.manifest import create_manifest, save_manifest, update_manifest_status

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CompressionPipeline")

def run_compression(model_name, num_samples, phase, dataset, k_prune):
    run_id = f"compression_{int(time.time())}"
    model_family = "gpt2" if "gpt2" in model_name.lower() else "llama" if "llama" in model_name.lower() else "mistral"
    output_dir = f"experiments/phase{phase}/{model_family}/{dataset}/{run_id}"
    
    manifest = create_manifest(phase, model_family, model_name, dataset, num_samples, 42, 
                             custom_metadata={"k_prune": k_prune})
    save_manifest(manifest, output_dir)
    
    try:
        wrapper = HookedModelWrapper(model_name, load_in_8bit=True)
        
        # 1. Head Jacobian
        hj_config = HeadJacobianConfig(num_samples=num_samples)
        hj_tool = HeadJacobian(hj_config)
        head_scores = hj_tool.compute_head_scores(wrapper)
        layer_jac = hj_tool.aggregate_to_layer_level(head_scores, wrapper.model.cfg.n_layers)
        
        # 2. Circuit Analysis (Using placeholders for ED/nAUDC if not provided)
        # In a real run, you'd load these from a previous sensitivity_profiler manifest
        ed_placeholder = {f"layer_{i}": 1.0 for i in range(wrapper.model.cfg.n_layers)}
        drift_placeholder = {f"layer_{i}": 0.0 for i in range(wrapper.model.cfg.n_layers)}
        
        analyzer = CircuitAnalyzer(TicketConfig())
        stability_scores = analyzer.find_winning_tickets(ed_placeholder, layer_jac, drift_placeholder, wrapper.model.cfg.n_layers)
        
        # 3. Pruning
        engine = PruningEngine(wrapper, PruningConfig())
        pruned_indices, summary = engine.prune_bottom_k_layers(stability_scores, k=k_prune)
        
        # Save Results
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({
                "stability_scores": stability_scores,
                "pruned_layers": pruned_indices,
                "summary": summary
            }, f, indent=4)
            
        update_manifest_status(output_dir, "completed", end_time=True)
        logger.info(f"Compression complete. Pruned: {pruned_indices}")
        
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        update_manifest_status(output_dir, f"failed: {str(e)}", end_time=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--phase", type=str, default="5")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--k", type=int, default=2, help="Number of layers to prune")
    args = parser.parse_args()
    run_compression(args.model, args.samples, args.phase, args.dataset, args.k)
