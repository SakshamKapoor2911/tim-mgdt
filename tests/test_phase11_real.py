#!/usr/bin/env python3
"""Fast Phase 11 verification on a tiny in-process causal LM.

The production Phase 11 code is intended for Mistral/Llama-class models, but
CI and local smoke tests should not download external models or run Wikitext PPL. This
fixture keeps the module structure used by Llama/Mistral and patches the slow
metric calls with deterministic local measurements.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import code.phases.phase11.stage1_structured_pruning as stage1_mod
import code.phases.phase11.stage2_imp_search as stage2_mod
import code.phases.phase11.stage4_activation_healing as stage4_mod
from code.phases.phase11.stage1_structured_pruning import run_stage1
from code.phases.phase11.stage2_imp_search import run_stage2
from code.phases.phase11.stage3_causal_patching import run_stage3
from code.phases.phase11.stage4_activation_healing import run_stage4
from code.phases.phase11.utils import get_device, get_model_config


class TinyBatch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device):
        return TinyBatch({k: v.to(device) for k, v in self.items()})


class TinyTokenizer:
    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "</s>"

    def __call__(self, text, return_tensors="pt"):
        ids = [(ord(ch) % (self.vocab_size - 1)) + 1 for ch in text]
        ids = ids or [1]
        return TinyBatch({"input_ids": torch.tensor([ids], dtype=torch.long)})


class TinySelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden):
        mixed = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        return self.o_proj(torch.tanh(mixed))


class TinyMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TinyLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.self_attn = TinySelfAttention(hidden_size)
        self.mlp = TinyMLP(hidden_size, intermediate_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden):
        hidden = self.norm(hidden + self.self_attn(hidden) + self.mlp(hidden))
        return (hidden,)


class TinyBackbone(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyLayer(hidden_size, intermediate_size) for _ in range(num_layers)]
        )


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=64,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = TinyBackbone(num_layers, hidden_size, intermediate_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        hidden = self.embed_tokens(input_ids % self.config.vocab_size)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                (labels % self.config.vocab_size).reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)


def fast_perplexity(model, tokenizer, device, max_samples=2, batch_size=1, **kwargs):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    sparsity = 1.0 - (nonzero / total)
    return 10.0 + sparsity


def fast_latency(model, tokenizer, device, **kwargs):
    return 1.0


class TestPhase11FastPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.tokenizer = TinyTokenizer()
        cls.model = TinyCausalLM().to(cls.device)
        cls.model.eval()
        cls.patchers = [
            patch.object(stage1_mod, "compute_perplexity", side_effect=fast_perplexity),
            patch.object(stage2_mod, "compute_perplexity", side_effect=fast_perplexity),
            patch.object(stage4_mod, "compute_perplexity", side_effect=fast_perplexity),
            patch.object(stage4_mod, "time_forward_pass", side_effect=fast_latency),
        ]
        for patcher in cls.patchers:
            patcher.start()

    @classmethod
    def tearDownClass(cls):
        for patcher in reversed(cls.patchers):
            patcher.stop()

    def test_0_model_config(self):
        cfg = get_model_config(self.model)
        self.assertEqual(cfg["arch_family"], "llama")
        self.assertEqual(cfg["num_layers"], 2)
        self.assertEqual(cfg["num_heads"], 2)
        self.assertEqual(cfg["hidden_size"], 16)
        self.assertEqual(cfg["head_dim"], 8)
        self.assertGreater(cfg["num_params"], 0)

    def test_1_stage1_structured_pruning(self):
        res = run_stage1(
            self.model,
            self.tokenizer,
            self.device,
            model_name="tiny-llama-fixture",
            ppl_max_samples=2,
        )
        self.assertEqual(res["stage"], "1_structured_pruning")
        self.assertEqual(res["metric_type"], "MEASURED")
        self.assertEqual(res["status"], "complete")
        self.assertEqual(res["plans_count"], 2)
        for plan_name in ["gqa_prune_20", "gqa_prune_40"]:
            self.assertIn(plan_name, res["plans"])
            self.assertGreater(res["plans"][plan_name]["pruned_ppl"], 0)

    def test_2_stage2_imp_search(self):
        res = run_stage2(
            self.model,
            self.tokenizer,
            self.device,
            ppl_max_samples=2,
            ppl_threshold_pct=10.0,
        )
        self.assertEqual(res["stage"], "2_imp_search")
        self.assertEqual(res["status"], "complete")
        self.assertEqual(res["steps_processed"], 5)
        self.assertIsNotNone(res["best_winner"])
        for step in res["search_history"]:
            self.assertEqual(step["metric_type"], "MEASURED")
            self.assertGreater(step["measured_ppl"], 0)
            self.assertTrue(0.0 <= step["actual_sparsity"] <= 1.0)

    def test_3_stage3_causal_patching(self):
        res = run_stage3(self.model, self.tokenizer, self.device)
        self.assertEqual(res["stage"], "3_causal_patching")
        self.assertEqual(res["status"], "complete")
        self.assertEqual(len(res["causal_scores"]), 2)
        for score_info in res["causal_scores"].values():
            self.assertIn("recovery_score", score_info)
            self.assertIn("kl_patched", score_info)
            self.assertIn("kl_baseline", score_info)

    def test_4_stage4_activation_healing(self):
        res = run_stage4(
            self.model,
            self.tokenizer,
            self.device,
            target_sparsity=0.20,
            ppl_max_samples=2,
        )
        self.assertEqual(res["stage"], "4_activation_healing")
        self.assertEqual(res["metric_type"], "MEASURED")
        self.assertEqual(res["status"], "complete")
        self.assertGreater(res["baseline"]["latency_ms"], 0)
        self.assertGreater(res["pruned_before_healing"]["latency_ms"], 0)
        self.assertGreater(res["healed"]["latency_ms"], 0)
        self.assertTrue(0.0 <= res["deployment"]["param_reduction_pct"] <= 100.0)


if __name__ == "__main__":
    unittest.main()
