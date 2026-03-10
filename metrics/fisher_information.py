"""
Empirical Fisher Information per layer.

Computes F_i = E_{x, y~p(y|x,θ)} [||∂ log p(y|x,θ_i) / ∂θ_i||²]
where y is sampled from the model's own output distribution.

This differs from the Jacobian metric (which measures gradient w.r.t.
activations). Fisher measures parameter sensitivity under the model's
own predictive distribution.
"""

import torch
import torch.nn.functional as F
from utils.dataset_manager import load_calibration_data


class FisherInformation:
    """Empirical Fisher Information scores per layer using sampled labels."""

    @staticmethod
    def compute_all_layers(wrapper, num_samples=50):
        """Compute empirical Fisher: E[||∂ log p(y_sampled|x) / ∂θ||²] per layer."""
        model = wrapper.model
        device = wrapper.device
        n_layers = model.cfg.n_layers

        calibration_texts = load_calibration_data(num_samples=num_samples)

        print("[Fisher Information] Computing empirical Fisher per layer...")

        model.eval()
        layer_fisher = {i: [] for i in range(n_layers)}

        for text_idx, text in enumerate(calibration_texts):
            tokens = model.to_tokens(text)
            if tokens.shape[1] < 2:
                continue

            try:
                with torch.enable_grad():
                    logits = model(tokens)  # [1, seq_len, vocab]

                    # Sample labels from the model's own output distribution
                    with torch.no_grad():
                        probs = F.softmax(logits[:, :-1, :].detach(), dim=-1)
                        sampled = torch.multinomial(
                            probs.reshape(-1, probs.shape[-1]), 1
                        ).reshape(1, -1)

                    # Compute log p(sampled_y | x, θ)
                    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
                    selected = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                    loss = selected.sum()
                    loss.backward()

                for layer_idx in range(n_layers):
                    fisher_score = 0.0
                    param_count = 0
                    for param in model.blocks[layer_idx].parameters():
                        if param.grad is not None:
                            # Fisher = E[||∇ log p||²] — sum of squared gradients
                            fisher_score += (param.grad ** 2).sum().item()
                            param_count += 1

                    if param_count > 0:
                        layer_fisher[layer_idx].append(fisher_score / param_count)
            except Exception as e:
                if text_idx == 0:
                    print(f"  Warning: Fisher failed on text {text_idx}: {e}")
            finally:
                model.zero_grad()
                torch.cuda.empty_cache()

        fisher_scores = {}
        print("\n  Averaging Fisher scores across texts...")
        for layer_idx in range(n_layers):
            if layer_fisher[layer_idx]:
                fisher_scores[f"layer_{layer_idx}"] = float(
                    sum(layer_fisher[layer_idx]) / len(layer_fisher[layer_idx])
                )
            else:
                fisher_scores[f"layer_{layer_idx}"] = 0.0

            if layer_idx % max(1, n_layers // 4) == 0:
                print(f"  Layer {layer_idx}: Fisher = {fisher_scores[f'layer_{layer_idx}']:.6f}")

        return fisher_scores
