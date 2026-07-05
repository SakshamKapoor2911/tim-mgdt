#!/usr/bin/env python3
"""
Phase 11: Real Pruning & Measurement Pipeline - Verification Test Suite
Tests all 4 stages of Phase 11 on GPT-2 with real PyTorch operations and actual measurements.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.phases.phase11.utils import get_device, get_model_config, compute_perplexity
from code.phases.phase11.stage1_structured_pruning import run_stage1
from code.phases.phase11.stage2_imp_search import run_stage2
from code.phases.phase11.stage3_causal_patching import run_stage3
from code.phases.phase11.stage4_activation_healing import run_stage4
from code.phases.phase11.run_full_pipeline import run_full_pipeline


class TestPhase11RealPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load GPT-2 once for all tests to run fast."""
        cls.device = get_device()
        cls.model_id = "gpt2"
        
        print(f"\n[TEST] Loading {cls.model_id} on {cls.device} for testing...")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
            
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32  # CPU safe
        )
        cls.model = cls.model.to(cls.device)
        cls.model.eval()

    def test_0_model_config(self):
        """Test model configuration introspection."""
        cfg = get_model_config(self.model)
        self.assertEqual(cfg["arch_family"], "gpt2")
        self.assertEqual(cfg["num_layers"], 12)
        self.assertEqual(cfg["num_heads"], 12)
        self.assertEqual(cfg["hidden_size"], 768)
        self.assertEqual(cfg["head_dim"], 64)
        self.assertGreater(cfg["num_params"], 0)

    def test_1_stage1_structured_pruning(self):
        """Test Stage 1 real structured pruning on GPT-2."""
        # Use a very small sample count for speed in test
        res = run_stage1(
            self.model,
            self.tokenizer,
            self.device,
            model_name=self.model_id,
            ppl_max_samples=2
        )
        self.assertEqual(res["stage"], "1_structured_pruning")
        self.assertEqual(res["metric_type"], "MEASURED")
        self.assertEqual(res["status"], "complete")
        self.assertIn("baseline_ppl", res)
        self.assertEqual(res["plans_count"], 2)
        
        # Verify plans are filled with real metrics
        for plan_name in ["gqa_prune_20", "gqa_prune_40"]:
            self.assertIn(plan_name, res["plans"])
            p = res["plans"][plan_name]
            self.assertEqual(p["metric_type"], "MEASURED")
            self.assertGreater(p["pruned_ppl"], 0)
            self.assertGreater(p["baseline_ppl"], 0)

    def test_2_stage2_imp_search(self):
        """Test Stage 2 iterative magnitude pruning sweep."""
        res = run_stage2(
            self.model,
            self.tokenizer,
            self.device,
            ppl_max_samples=2,
            ppl_threshold_pct=10.0
        )
        self.assertEqual(res["stage"], "2_imp_search")
        self.assertEqual(res["metric_type"], "MEASURED")
        self.assertEqual(res["status"], "complete")
        self.assertEqual(res["steps_processed"], 5)
        self.assertIsNotNone(res["best_winner"])
        
        # Verify the history contains real measurements
        for step in res["search_history"]:
            self.assertEqual(step["metric_type"], "MEASURED")
            self.assertGreater(step["measured_ppl"], 0)
            self.assertTrue(0.0 <= step["actual_sparsity"] <= 1.0)

    def test_3_stage3_causal_patching(self):
        """Test Stage 3 real causal patching with PyTorch forward hooks."""
        res = run_stage3(self.model, self.tokenizer, self.device)
        self.assertEqual(res["stage"], "3_causal_patching")
        self.assertEqual(res["metric_type"], "MEASURED")
        self.assertEqual(res["status"], "complete")
        self.assertEqual(len(res["causal_scores"]), 12)  # GPT-2 has 12 layers
        
        for layer_key, score_info in res["causal_scores"].items():
            self.assertIn("recovery_score", score_info)
            self.assertIn("kl_patched", score_info)
            self.assertIn("kl_baseline", score_info)

    def test_4_stage4_activation_healing(self):
        """Test Stage 4 activation healing and deployment metrics."""
        res = run_stage4(
            self.model,
            self.tokenizer,
            self.device,
            target_sparsity=0.20,
            ppl_max_samples=2
        )
        self.assertEqual(res["stage"], "4_activation_healing")
        self.assertEqual(res["metric_type"], "MEASURED")
        self.assertEqual(res["status"], "complete")
        
        # Verify metrics
        self.assertEqual(res["healed"]["ppl_increase_pct"], ((res["healed"]["ppl"] - res["baseline"]["ppl"]) / res["baseline"]["ppl"]) * 100)
        self.assertGreater(res["baseline"]["latency_ms"], 0)
        self.assertGreater(res["pruned_before_healing"]["latency_ms"], 0)
        self.assertGreater(res["healed"]["latency_ms"], 0)
        self.assertTrue(0.0 <= res["deployment"]["param_reduction_pct"] <= 100.0)


if __name__ == "__main__":
    unittest.main()
