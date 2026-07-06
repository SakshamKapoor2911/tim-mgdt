"""CPU-only tests for Phase 20 mask-guided structured pruning."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from tests._repo_paths import setup_repo_paths

setup_repo_paths()

from code.phases.phase20_mask_guided.mask_guided_width_pruning import (
    aligned_keep_size,
    apply_mask_guided_width_pruning,
    compute_mask_guided_scores,
    masked_axis_score,
    tag_from_sparsity,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=8):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TinyLayer(nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=8):
        super().__init__()
        self.mlp = TinyMLP(hidden_size, intermediate_size)


class TinyBackbone(nn.Module):
    def __init__(self, num_layers=2, hidden_size=4, intermediate_size=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
        )


class TinyModel(nn.Module):
    def __init__(self, hidden_size=4, intermediate_size=8, num_layers=2):
        super().__init__()
        self.config = SimpleNamespace(intermediate_size=intermediate_size)
        self.model = TinyBackbone(num_layers, hidden_size, intermediate_size)


class TestPhase20MaskGuidedPruning(unittest.TestCase):
    def test_tag_from_sparsity(self):
        self.assertEqual(tag_from_sparsity(0.005), "s005")
        self.assertEqual(tag_from_sparsity(0.01), "s01")
        self.assertEqual(tag_from_sparsity(0.2), "s20")

    def test_aligned_keep_size_handles_tiny_and_mistral_sizes(self):
        self.assertEqual(aligned_keep_size(8, 0.25, "none"), 6)
        self.assertEqual(aligned_keep_size(8, 0.25, "ceil"), 8)
        self.assertEqual(aligned_keep_size(14336, 0.005, "ceil"), 14272)
        self.assertEqual(aligned_keep_size(14336, 0.01, "ceil"), 14208)

    def test_masked_axis_score_preserves_expected_shape(self):
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        row_scores = masked_axis_score(weight, axis=1, mask_sparsity=0.0)
        col_scores = masked_axis_score(weight, axis=0, mask_sparsity=0.0)

        self.assertEqual(row_scores.shape, (2,))
        self.assertEqual(col_scores.shape, (3,))
        self.assertTrue(torch.equal(row_scores, torch.tensor([6.0, 15.0])))
        self.assertTrue(torch.equal(col_scores, torch.tensor([5.0, 7.0, 9.0])))

    def test_apply_mask_guided_width_pruning_slices_all_mlp_projections(self):
        torch.manual_seed(0)
        model = TinyModel(hidden_size=4, intermediate_size=8, num_layers=2)
        scores = compute_mask_guided_scores(model, mask_sparsity=0.2)

        stats = apply_mask_guided_width_pruning(
            model,
            scores,
            sparsity=0.25,
            strategy="none",
        )

        self.assertEqual(stats["modules_pruned"], 2)
        self.assertEqual(stats["intermediate_size_before"], 8)
        self.assertEqual(stats["intermediate_size_after_min"], 6)
        self.assertEqual(stats["intermediate_size_after_max"], 6)
        self.assertEqual(model.config.intermediate_size, 6)
        self.assertAlmostEqual(stats["mlp_param_sparsity"], 0.25)

        for layer in model.model.layers:
            self.assertEqual(tuple(layer.mlp.gate_proj.weight.shape), (6, 4))
            self.assertEqual(tuple(layer.mlp.up_proj.weight.shape), (6, 4))
            self.assertEqual(tuple(layer.mlp.down_proj.weight.shape), (4, 6))

        hidden = torch.randn(2, 3, 4)
        output = model.model.layers[0].mlp(hidden)
        self.assertEqual(tuple(output.shape), (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
