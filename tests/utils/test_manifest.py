import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unittest
import os
import json
import tempfile
import time
from unittest.mock import patch

from code.utils.manifest import create_manifest, save_manifest, update_manifest_status

class TestManifest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_manifest(self):
        manifest = create_manifest(
            phase="phase1",
            model_family="llama",
            model_id="llama-2-7b",
            dataset="wikitext",
            sample_count=100,
            seed=42,
            precision="fp16",
            custom_metadata={"extra": "data"}
        )

        self.assertEqual(manifest["phase"], "phase1")
        self.assertEqual(manifest["model_family"], "llama")
        self.assertEqual(manifest["model_id"], "llama-2-7b")
        self.assertEqual(manifest["dataset"], "wikitext")
        self.assertEqual(manifest["sample_count"], 100)
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["precision"], "fp16")
        self.assertEqual(manifest["extra"], "data")
        self.assertEqual(manifest["status"], "started")
        self.assertIn("hardware", manifest)
        self.assertIn("start_time", manifest)
        self.assertIn("start_timestamp", manifest)
        self.assertIn("commit_hash", manifest)

    def test_save_manifest(self):
        manifest = {"test": "data"}
        save_manifest(manifest, self.output_dir)

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        self.assertTrue(os.path.exists(manifest_path))

        with open(manifest_path, "r") as f:
            loaded_manifest = json.load(f)

        self.assertEqual(loaded_manifest, manifest)

    def test_update_manifest_status(self):
        manifest = {
            "status": "started",
            "start_timestamp": time.time() - 10 # Started 10 seconds ago
        }
        save_manifest(manifest, self.output_dir)

        update_manifest_status(self.output_dir, "completed")

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            loaded_manifest = json.load(f)

        self.assertEqual(loaded_manifest["status"], "completed")
        self.assertNotIn("end_time", loaded_manifest)

    def test_update_manifest_status_with_end_time(self):
        manifest = {
            "status": "started",
            "start_timestamp": time.time() - 10 # Started 10 seconds ago
        }
        save_manifest(manifest, self.output_dir)

        update_manifest_status(self.output_dir, "completed", end_time=True)

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            loaded_manifest = json.load(f)

        self.assertEqual(loaded_manifest["status"], "completed")
        self.assertIn("end_time", loaded_manifest)
        self.assertIn("end_timestamp", loaded_manifest)
        self.assertIn("duration_sec", loaded_manifest)
        self.assertGreater(loaded_manifest["duration_sec"], 0)

if __name__ == '__main__':
    unittest.main()
