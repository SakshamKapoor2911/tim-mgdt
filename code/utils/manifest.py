import json
import os
import time
import platform
import subprocess
from typing import Any, Dict

def get_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def create_manifest(
    phase: str,
    model_family: str,
    model_id: str,
    dataset: str,
    sample_count: int,
    seed: int,
    precision: str = "fp32",
    custom_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    manifest = {
        "phase": phase,
        "model_family": model_family,
        "model_id": model_id,
        "dataset": dataset,
        "sample_count": sample_count,
        "seed": seed,
        "hardware": {
            "os": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "precision": precision,
        "commit_hash": get_commit_hash(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_timestamp": time.time(),
        "status": "started"
    }
    
    if custom_metadata:
        manifest.update(custom_metadata)
        
    return manifest

def save_manifest(manifest: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

def update_manifest_status(output_dir: str, status: str, end_time: bool = False):
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["status"] = status
        if end_time:
            manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            manifest["end_timestamp"] = time.time()
            manifest["duration_sec"] = manifest["end_timestamp"] - manifest["start_timestamp"]
            
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
