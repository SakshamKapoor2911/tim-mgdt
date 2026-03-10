import torch
import torch.nn.functional as F
from utils.dataset_manager import load_calibration_data


class LogitJacobian:
    """Computes ||∂Logits/∂h_{L_i}||_2 — gradient of logits w.r.t. layer activations.

    This measures how sensitive the final output logits are to perturbations
    in the residual stream at each layer's output.
    """

    @staticmethod
    def compute_all_layers(wrapper, num_samples=50):
        """Compute activation-gradient Jacobian norm for each layer."""
        model = wrapper.model
        device = wrapper.device
        n_layers = model.cfg.n_layers

        calibration_texts = load_calibration_data(num_samples=num_samples)

        print("[Causal Jacobian] Computing ||∂Logits/∂h_{L_i}||_2 for each layer...")

        model.eval()
        layer_grads = {i: [] for i in range(n_layers)}

        for text_idx, text in enumerate(calibration_texts):
            tokens = model.to_tokens(text)
            if tokens.shape[1] < 2:
                continue

            captured = {}

            def make_hook(idx):
                def hook_fn(value, hook):
                    if value.requires_grad:
                        value.retain_grad()
                    captured[idx] = value
                    return value  # pass through unchanged
                return hook_fn

            hooks = [
                (f"blocks.{i}.hook_resid_post", make_hook(i))
                for i in range(n_layers)
            ]

            try:
                with torch.enable_grad():
                    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                    # Use sum of logits as scalar for backward
                    scalar = logits.sum()
                    scalar.backward()

                for layer_idx in range(n_layers):
                    if layer_idx in captured and captured[layer_idx].grad is not None:
                        grad = captured[layer_idx].grad
                        # RMS-normalized gradient norm
                        grad_norm = grad.norm().item() / (grad.numel() ** 0.5)
                        layer_grads[layer_idx].append(grad_norm)
            except Exception as e:
                if text_idx == 0:
                    print(f"  Warning: Jacobian failed on text {text_idx}: {e}")
            finally:
                model.zero_grad()
                torch.cuda.empty_cache()

        # Average across texts
        jacobian_scores = {}
        print("\n  Averaging Jacobian norms across texts...")
        for layer_idx in range(n_layers):
            if layer_grads[layer_idx]:
                jacobian_scores[f"layer_{layer_idx}"] = float(
                    sum(layer_grads[layer_idx]) / len(layer_grads[layer_idx])
                )
            else:
                jacobian_scores[f"layer_{layer_idx}"] = 0.0

            if layer_idx % max(1, n_layers // 4) == 0:
                print(f"  Layer {layer_idx}: ||∂Logits/∂h||_2 = "
                      f"{jacobian_scores[f'layer_{layer_idx}']:.6f}")

        return jacobian_scores
