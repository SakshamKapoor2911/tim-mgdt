"""
Demo/Test version of the benchmark pipeline using mock data
to avoid model loading compatibility issues.
"""
import torch
import json
import os
from metrics.geometric_ed import EffectiveDimension
from metrics.causal_jacobian import LogitJacobian
from metrics.propagation_naudc import PropagationDrift
from interventions.ablation import LayerAblation
from scipy.stats import spearmanr


class MockModel:
    """Mock model configuration for testing."""
    def __init__(self, n_layers=12):
        self.cfg = type('Config', (), {'n_layers': n_layers})()
    
    def eval(self):
        """No-op for compatibility."""
        return self
    
    def zero_grad(self):
        """No-op for compatibility."""
        pass
    
    def __call__(self, input_ids):
        """Simple mock forward pass."""
        batch_size, seq_len = input_ids.shape
        vocab_size = 50257
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return logits
    
    def parameters(self):
        """Mock parameters for iteration."""
        return []
    
    @property
    def blocks(self):
        """Create mock blocks."""
        class MockBlock:
            def __init__(self):
                pass
            def register_forward_hook(self, fn):
                class MockHook:
                    def remove(self):
                        pass
                return MockHook()
            def parameters(self):
                return []
        return [MockBlock() for _ in range(self.cfg.n_layers)]


class MockCache:
    """Mock cache for testing the geometric_ed implementation."""
    def __init__(self, n_layers=12, seq_len=16, d_model=768):
        self.cache = {}
        for i in range(n_layers):
            # Create random activations: [batch=1, seq_len, d_model]
            self.cache[f"blocks.{i}.hook_resid_post"] = torch.randn(1, seq_len, d_model)
    
    def __getitem__(self, key):
        return self.cache[key]


class MockModelWrapper:
    """Mock wrapper for testing without loading actual models."""
    def __init__(self, n_layers=12):
        self.model = MockModel(n_layers=n_layers)
        self.device = torch.device("cpu")
    
    def run_with_cache(self, input_text):
        """Returns dummy logits and mock cache."""
        logits = torch.randn(1, 16, 50257)  # Dummy logits
        cache = MockCache(n_layers=self.model.cfg.n_layers)
        return logits, cache


def run_demo_pipeline():
    """Run the benchmark pipeline with mock data."""
    print("\n" + "="*70)
    print("TIM-MGDT DEMO: Running Sensitivity Ranking Pipeline")
    print("="*70 + "\n")
    
    # Use a smaller model for demo (6 layers instead of 32)
    n_layers = 6
    print(f"Setting up demo with {n_layers}-layer model...\n")
    
    # 1. Initialize Mock Model & Tools
    print("Step 1: Initializing Mock Model Wrapper...")
    wrapper = MockModelWrapper(n_layers=n_layers)
    ablation_tool = LayerAblation(wrapper)
    print(f"✓ Mock model loaded with {n_layers} layers\n")
    
    results = {}

    # 2. Get Ground Truth (Delta PPL) - Uses mock implementation
    print("Step 2: Mapping Ground Truth Sensitivity via Layer-wise Ablation...")
    ground_truth = ablation_tool.map_layer_sensitivity()
    print(f"✓ Ground truth mapping completed")
    for i, (layer, score) in enumerate(list(ground_truth.items())[:3]):
        print(f"  {layer}: {score:.4f}")
    print(f"  ... ({n_layers} total layers)\n")

    # 3. Compute Proxy Triad
    print("Step 3: Computing Mechanistic Proxies (ED, Jacobian, nAUDC)...")
    
    # Geometric ED - Uses REAL torch.svd implementation
    print("  - Computing Effective Dimension (ED)...")
    ed_scores = EffectiveDimension.compute_all_layers(wrapper)
    print(f"    ✓ ED computed for {len(ed_scores)} layers")
    
    # Causal Jacobian - Uses mock implementation
    print("  - Computing Logit Jacobian...")
    jacobian_scores = LogitJacobian.compute_all_layers(wrapper)
    print(f"    ✓ Jacobian computed for {len(jacobian_scores)} layers")
    
    # Propagation Drift - Uses mock implementation
    print("  - Computing Propagation Drift (nAUDC)...")
    drift_scores = PropagationDrift.compute_all_layers(wrapper)
    print(f"    ✓ Drift computed for {len(drift_scores)} layers\n")

    # 4. Statistical Correlation Analysis
    print("Step 4: Performing Spearman Correlation Analysis...\n")
    
    # Build lists dynamically based on actual layer count
    gt_list = [ground_truth[f"layer_{i}"] for i in range(n_layers)]
    
    # Correlation results
    for name, scores in [("Geometric (ED)", ed_scores), 
                        ("Causal (Jacobian)", jacobian_scores), 
                        ("Propagation (nAUDC)", drift_scores)]:
        score_list = [scores[f"layer_{i}"] for i in range(n_layers)]
        rho, p_val = spearmanr(score_list, gt_list)
        results[name] = {
            "spearman_rho": float(rho), 
            "p_value": float(p_val),
            "interpretation": "Strong correlation" if abs(rho) > 0.7 else "Moderate" if abs(rho) > 0.4 else "Weak"
        }
        print(f"{name}:")
        print(f"  Spearman ρ = {rho:7.4f}  |  p-value = {p_val:.6f}")
        print(f"  Interpretation: {results[name]['interpretation']}\n")

    # 5. Save Results to JSON
    output_path = "experiments/sensitivity_results.json"
    os.makedirs("experiments", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print("="*70)
    print(f"✓ Pipeline Complete!")
    print(f"✓ Results saved to: {output_path}\n")
    
    # Display summary
    print("SUMMARY:")  
    print("-" * 70)
    with open(output_path, "r") as f:
        summary = json.load(f)
        print(json.dumps(summary, indent=2))
    print("-" * 70 + "\n")


if __name__ == "__main__":
    run_demo_pipeline()
