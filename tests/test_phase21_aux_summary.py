"""CPU-only checks for Phase 21 auxiliary summary helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from tests._repo_paths import setup_repo_paths

setup_repo_paths()

from scripts.summarize_phase21_aux import parse_eval_name, sample_diagnostics, sample_path_for_result, strict_values


class TestPhase21AuxSummary(unittest.TestCase):
    def test_parse_eval_name_strips_lm_eval_timestamp_from_custom_label(self):
        model_tag, run_label = parse_eval_name(
            "experiments/phase21_aux_reasoning/"
            "eval_dense_phase25_aux_limit300_2026-07-07T13-01-49_2026-07-07T09-40-00.123456.json"
        )

        self.assertEqual(model_tag, "dense")
        self.assertEqual(run_label, "phase25_aux_limit300_2026-07-07T13-01-49")

    def test_parse_eval_name_preserves_plain_timestamp_label(self):
        model_tag, run_label = parse_eval_name(
            "experiments/phase21_aux_reasoning/eval_dense_2026-07-06T11-00-00.json"
        )

        self.assertEqual(model_tag, "dense")
        self.assertEqual(run_label, "2026-07-06T11-00-00")

    def test_sample_path_and_values_follow_eval_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "eval_s20_run_2026-07-07T09-40-00.123456.json"
            result.write_text("{}\n")
            sample = root / "samples_gsm8k_platinum_cot_2026-07-07T09-40-00.123456.jsonl"
            rows = [
                {
                    "filter": "strict-match",
                    "exact_match": 1.0,
                    "resps": [["The answer is 4"]],
                    "filtered_resps": ["4"],
                },
                {
                    "filter": "strict-match",
                    "exact_match": 0.0,
                    "resps": [["No marker"]],
                    "filtered_resps": [""],
                },
                {
                    "filter": "flexible-extract",
                    "exact_match": 1.0,
                    "resps": [["Ignored"]],
                    "filtered_resps": ["7"],
                },
            ]
            sample.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            path = sample_path_for_result(result, "gsm8k_platinum_cot")
            values = strict_values(path, "exact_match,strict-match")
            diagnostics = sample_diagnostics(path, "exact_match,strict-match")

        self.assertEqual(path, str(sample))
        self.assertEqual(values, [1, 0])
        self.assertEqual(diagnostics["avg_response_words"], 3.0)
        self.assertEqual(diagnostics["answer_marker_rate"], 0.5)
        self.assertEqual(diagnostics["extraction_failure_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
