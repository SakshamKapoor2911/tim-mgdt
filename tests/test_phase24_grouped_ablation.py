"""CPU-only checks for Phase 24 grouped mask ablation helpers."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from tests._repo_paths import setup_repo_paths

setup_repo_paths()

from code.phases.phase24_grouped_ablation.build_grouped_mask import (
    apply_grouped_layerwise_mask,
    layer_block,
    module_metadata,
    should_prune,
)
from code.phases.phase24_grouped_ablation.bootstrap_gsm8k_ci import decision


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 8, bias=False)


class TinyBackbone(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer() for _ in range(n_layers)])


class TinyModel(nn.Module):
    def __init__(self, n_layers=6):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=n_layers)
        self.model = TinyBackbone(n_layers)
        self.lm_head = nn.Linear(4, 16, bias=False)


class TestPhase24GroupedAblation(unittest.TestCase):
    def test_layer_block_partitions_layers_into_thirds(self):
        blocks = [layer_block(i, 6) for i in range(6)]
        self.assertEqual(blocks, ["early", "early", "mid", "mid", "late", "late"])

    def test_module_metadata_classifies_attention_mlp_and_lm_head(self):
        attn = module_metadata("model.layers.3.self_attn.q_proj", 6)
        mlp = module_metadata("model.layers.4.mlp.gate_proj", 6)
        other = module_metadata("lm_head", 6)

        self.assertEqual(attn["layer_idx"], 3)
        self.assertEqual(attn["module_family"], "attention")
        self.assertEqual(mlp["layer_block"], "late")
        self.assertEqual(mlp["module_family"], "mlp")
        self.assertEqual(other["layer_block"], "non_layer")

    def test_should_prune_reverts_requested_groups(self):
        early_attn = {"layer_block": "early", "module_family": "attention"}
        late_mlp = {"layer_block": "late", "module_family": "mlp"}

        self.assertFalse(should_prune("revert_early", early_attn))
        self.assertTrue(should_prune("revert_early", late_mlp))
        self.assertFalse(should_prune("revert_mlp", late_mlp))
        self.assertTrue(should_prune("revert_attention", late_mlp))

    def test_apply_grouped_mask_keeps_reverted_modules_dense(self):
        model = TinyModel(n_layers=6)
        with torch.no_grad():
            value = 1.0
            for param in model.parameters():
                param.copy_(torch.arange(value, value + param.numel()).view_as(param))
                value += param.numel()

        stats = apply_grouped_layerwise_mask(model, sparsity=0.5, ablation="revert_early")

        self.assertGreater(stats["modules_pruned"], 0)
        self.assertGreater(stats["modules_reverted_to_dense"], 0)
        self.assertEqual(stats["by_layer_block"]["early"]["zero_linear_params"], 0)
        self.assertGreater(stats["by_layer_block"]["mid"]["zero_linear_params"], 0)
        self.assertGreater(stats["by_layer_block"]["late"]["zero_linear_params"], 0)

    def test_decision_triggers_contingency_when_layer_groups_are_close(self):
        rows = [
            {"tag": "revert_early", "strict": 0.35},
            {"tag": "revert_mid", "strict": 0.34},
            {"tag": "revert_late", "strict": 0.33},
        ]
        self.assertEqual(decision(rows)["recommendation"], "run_module_contingency")

    def test_decision_stops_when_one_layer_group_leads(self):
        rows = [
            {"tag": "revert_early", "strict": 0.42},
            {"tag": "revert_mid", "strict": 0.35},
            {"tag": "revert_late", "strict": 0.34},
        ]
        self.assertEqual(decision(rows)["recommendation"], "stop_after_layer_groups")

    def test_decision_reports_weak_signal_after_flat_module_contingency(self):
        rows = [
            {"tag": "revert_early", "strict": 0.37},
            {"tag": "revert_mid", "strict": 0.38},
            {"tag": "revert_late", "strict": 0.373},
            {"tag": "revert_mlp", "strict": 0.40, "bootstrap_ci_low": 0.347},
            {"tag": "revert_attention", "strict": 0.377, "bootstrap_ci_high": 0.43},
        ]

        result = decision(rows)

        self.assertEqual(result["recommendation"], "distributed_or_weak_module_signal")
        self.assertEqual(result["best_module_group"], "revert_mlp")

    def test_decision_reports_module_signal_when_module_gap_is_clear(self):
        rows = [
            {"tag": "revert_early", "strict": 0.37},
            {"tag": "revert_mid", "strict": 0.38},
            {"tag": "revert_late", "strict": 0.373},
            {"tag": "revert_mlp", "strict": 0.44, "bootstrap_ci_low": 0.39},
            {"tag": "revert_attention", "strict": 0.37, "bootstrap_ci_high": 0.38},
        ]

        result = decision(rows)

        self.assertEqual(result["recommendation"], "module_axis_signal")
        self.assertEqual(result["best_module_group"], "revert_mlp")


if __name__ == "__main__":
    unittest.main()
