import torch
from utils.dataset_manager import load_calibration_data

class EffectiveDimension:
    """Computes Effective Dimension (ED) for each layer's residual stream activations."""
    @staticmethod
    def compute_all_layers(wrapper, variance_threshold=0.95, num_samples=100):
        """
        Uses SVD to determine the number of dimensions required 
        to explain variance_threshold (default 95%) of the activation variance.
        """
        ed_scores = {}
        n_layers = wrapper.model.cfg.n_layers
        
        # Load real calibration texts from WikiText-2 (cached locally)
        texts = load_calibration_data(num_samples=num_samples)
        
        print("[Geometric ED] Computing effective dimension for each layer...")
        all_acts = {i: [] for i in range(n_layers)}
        
        # Collect activations from multiple texts
        for text in texts:
            try:
                _, cache = wrapper.run_with_cache(text)
                
                for layer_idx in range(n_layers):
                    try:
                        act = cache[f"blocks.{layer_idx}.hook_resid_post"]
                        # Flatten: [batch*seq, d_model]
                        act_flat = act.reshape(-1, act.shape[-1]).detach().cpu().float()
                        all_acts[layer_idx].append(act_flat)
                    except KeyError:
                        pass
                
                torch.cuda.empty_cache()
            except Exception as e:
                pass
        
        # Calculate ED for each layer
        print("\nCalculating effective dimensions...")
        for layer_idx in range(n_layers):
            if not all_acts[layer_idx]:
                ed_scores[f"layer_{layer_idx}"] = 1.0
                continue
            
            # Concatenate activations: [total_seq, d_model]
            X = torch.cat(all_acts[layer_idx], dim=0)
            
            if X.shape[0] < 2:
                ed_scores[f"layer_{layer_idx}"] = 1.0
                continue
            
            # Center the data
            X_centered = X - X.mean(dim=0, keepdim=True)
            
            try:
                # Use SVD for ED calculation
                U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
                
                # Calculate cumulative explained variance
                S_sq = S ** 2
                total_var = S_sq.sum()
                cum_var = torch.cumsum(S_sq, dim=0) / total_var
                
                # Find ED: minimum k where cumulative variance >= threshold
                ed_idx = (cum_var >= variance_threshold).nonzero(as_tuple=True)[0]
                if len(ed_idx) > 0:
                    ed_value = ed_idx[0].item() + 1
                else:
                    ed_value = len(S)
                
                ed_scores[f"layer_{layer_idx}"] = float(ed_value)
                if layer_idx % max(1, n_layers // 4) == 0:
                    print(f"  Layer {layer_idx}: ED = {ed_value}")
            except Exception as e:
                ed_scores[f"layer_{layer_idx}"] = 1.0
            
            torch.cuda.empty_cache()
        
        return ed_scores
