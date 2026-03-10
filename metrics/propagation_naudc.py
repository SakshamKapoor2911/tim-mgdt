import torch
import torch.nn.functional as F
from utils.dataset_manager import load_calibration_data


class PropagationDrift:
    """Normalized Area Under the Drift Curve (nAUDC).

    Measures cumulative cosine drift across ALL downstream layers after
    noise injection at layer i:
        nAUDC(L_i) = 1/(D-i) * Σ_{j=i+1}^{D} (1 - CosSim(r_j^orig, r_j^mod))
    """

    @staticmethod
    def compute_all_layers(wrapper, noise_level=0.15, num_samples=30):
        """Compute nAUDC with cumulative downstream drift for each layer."""
        model = wrapper.model
        device = wrapper.device
        n_layers = model.cfg.n_layers

        calibration_texts = load_calibration_data(num_samples=num_samples)

        print("[Propagation nAUDC] Computing cumulative downstream drift...")

        model.eval()
        layer_naudcs = {i: [] for i in range(n_layers)}

        for text_idx, text in enumerate(calibration_texts):
            tokens = model.to_tokens(text)
            if tokens.shape[1] < 2:
                continue

            # Step 1: Get clean activations at all layers
            with torch.no_grad():
                _, clean_cache = model.run_with_cache(tokens)

            clean_acts = {}
            for j in range(n_layers):
                clean_acts[j] = clean_cache[f"blocks.{j}.hook_resid_post"].detach().cpu()

            del clean_cache

            # Step 2: For each injection layer, compute cumulative drift
            for layer_idx in range(n_layers):
                if layer_idx >= n_layers - 1:
                    # Last layer has no downstream layers
                    layer_naudcs[layer_idx].append(0.0)
                    continue

                noisy_acts = {}

                def make_noise_hook():
                    def hook_fn(value, hook):
                        return value + noise_level * torch.randn_like(value)
                    return hook_fn

                def make_capture_hook(j):
                    def hook_fn(value, hook):
                        noisy_acts[j] = value.detach().cpu()
                        return value  # pass through
                    return hook_fn

                # Inject noise at layer_idx, capture activations at all downstream layers
                fwd_hooks = [(f"blocks.{layer_idx}.hook_resid_post", make_noise_hook())]
                for j in range(layer_idx + 1, n_layers):
                    fwd_hooks.append(
                        (f"blocks.{j}.hook_resid_post", make_capture_hook(j))
                    )

                with torch.no_grad():
                    model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

                # nAUDC(L_i) = 1/(D-i) * Σ_{j=i+1}^{D} (1 - CosSim(r_j^orig, r_j^mod))
                drifts = []
                for j in range(layer_idx + 1, n_layers):
                    if j in noisy_acts:
                        clean_flat = clean_acts[j].flatten().unsqueeze(0).float()
                        noisy_flat = noisy_acts[j].flatten().unsqueeze(0).float()
                        cos_sim = F.cosine_similarity(clean_flat, noisy_flat).item()
                        drifts.append(max(0.0, 1.0 - cos_sim))

                naudc = sum(drifts) / len(drifts) if drifts else 0.0
                layer_naudcs[layer_idx].append(naudc)

            torch.cuda.empty_cache()

        # Average across texts
        naudc_scores = {}
        print("\n  Averaging nAUDC scores across texts...")
        for layer_idx in range(n_layers):
            if layer_naudcs[layer_idx]:
                naudc_scores[f"layer_{layer_idx}"] = float(
                    sum(layer_naudcs[layer_idx]) / len(layer_naudcs[layer_idx])
                )
            else:
                naudc_scores[f"layer_{layer_idx}"] = 0.0

            if layer_idx % max(1, n_layers // 4) == 0:
                print(f"  Layer {layer_idx}: nAUDC = {naudc_scores[f'layer_{layer_idx}']:.6f}")

        return naudc_scores
