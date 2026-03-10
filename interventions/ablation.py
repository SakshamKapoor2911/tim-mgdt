import torch
import torch.nn.functional as F
import numpy as np
from utils.dataset_manager import load_calibration_data


class LayerAblation:
    """Ground truth layer importance via layer-skip ablation measuring ΔCE (cross-entropy)."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.device = wrapper.device

    def map_layer_sensitivity(self, num_samples=50):
        """Compute ground truth sensitivity: skip each layer and measure ΔCE."""
        print("\n[Ground Truth] Starting layer-skip ablation (ΔCE / ΔPPL)...")

        model = self.wrapper.model
        texts = load_calibration_data(num_samples=num_samples)
        n_layers = model.cfg.n_layers

        # Tokenize all texts once
        all_tokens = []
        for text in texts:
            tokens = model.to_tokens(text)
            if tokens.shape[1] >= 2:
                all_tokens.append(tokens)

        print(f"  Using {len(all_tokens)} valid texts ({sum(t.shape[1] for t in all_tokens)} total tokens)")

        # Baseline cross-entropy (no ablation)
        print("  Computing baseline cross-entropy...")
        baseline_ce = self._compute_cross_entropy(model, all_tokens)
        print(f"  Baseline CE: {baseline_ce:.4f} (PPL={np.exp(baseline_ce):.2f})\n")

        sensitivities = {}
        for layer_idx in range(n_layers):
            ablated_ce = self._compute_ablated_cross_entropy(model, all_tokens, layer_idx)
            delta = ablated_ce - baseline_ce  # positive = layer was important
            sensitivities[f"layer_{layer_idx}"] = float(delta)

            if layer_idx % max(1, n_layers // 4) == 0:
                print(f"  Layer {layer_idx}: ΔCE = {delta:+.4f} (ablated PPL={np.exp(ablated_ce):.2f})")

            torch.cuda.empty_cache()

        return sensitivities

    def _compute_cross_entropy(self, model, all_tokens):
        """Average next-token cross-entropy loss across all texts."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for tokens in all_tokens:
                logits = model(tokens)
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = tokens[:, 1:].reshape(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
                total_loss += loss.item()
                total_tokens += shift_labels.numel()

        return total_loss / max(total_tokens, 1)

    def _compute_ablated_cross_entropy(self, model, all_tokens, layer_idx):
        """Cross-entropy with layer skipped: resid_post replaced by resid_pre."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for tokens in all_tokens:
            captured_pre = {}

            def capture_pre(value, hook):
                captured_pre['act'] = value.clone()
                return value

            def replace_post(value, hook):
                return captured_pre['act']

            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        (f"blocks.{layer_idx}.hook_resid_pre", capture_pre),
                        (f"blocks.{layer_idx}.hook_resid_post", replace_post),
                    ]
                )

            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = tokens[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

        return total_loss / max(total_tokens, 1)
