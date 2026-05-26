#!/usr/bin/env python3
"""
TIM-MGDT Phase 10 Parallel GPU Orchestrator
Launches Mistral-7B and Llama-3-8B evaluations simultaneously on separate GPUs
(GPU 0 and GPU 1 on dual 3090 setup) so they can finish independently.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def main():
    print("=" * 80)
    # Visual WOW: modern CLI header
    print("    🚀 TIM-MGDT: DUAL-GPU PARALLEL RUNNER (PHASE 10)")
    print("    Simultaneous Multi-Model Lottery Ticket Evaluations")
    print("=" * 80)

    # 1. Paths to scripts
    mistral_script = REPO_ROOT / "tests" / "test_phase10_mistral7b_real.py"
    llama_script = REPO_ROOT / "tests" / "test_phase10_llama3_8b_real.py"

    # 2. Command builders
    # We set environment variables specifically for each process to target different GPUs
    env_gpu0 = os.environ.copy()
    env_gpu0["CUDA_VISIBLE_DEVICES"] = "0"
    
    env_gpu1 = os.environ.copy()
    env_gpu1["CUDA_VISIBLE_DEVICES"] = "1"

    print("⚡ Starting Parallel GPU Executions...")
    print("👉 GPU 0: Mistral-7B Evaluation")
    print("👉 GPU 1: Llama-3-8B Evaluation")
    print("-" * 80)

    # 3. Launch subprocesses
    t0 = time.time()
    
    p_mistral = subprocess.Popen(
        [sys.executable, str(mistral_script)],
        env=env_gpu0,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    p_llama = subprocess.Popen(
        [sys.executable, str(llama_script)],
        env=env_gpu1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # 4. Monitor runs in real time (non-blocking status check)
    # We read their outputs incrementally or wait until completion
    active_processes = {
        "Mistral-7B (GPU 0)": p_mistral,
        "Llama-3-8B (GPU 1)": p_llama
    }

    completed = {}

    while active_processes:
        for name, p in list(active_processes.items()):
            poll = p.poll()
            if poll is not None:
                # Process completed!
                elapsed = time.time() - t0
                print(f"\n🎉 [COMPLETE] {name} finished in {elapsed:.1f}s with exit code {poll}!")
                
                # Fetch all remaining stdout
                stdout, _ = p.communicate()
                if stdout:
                    print(f"--- Last 15 lines of output for {name} ---")
                    lines = stdout.splitlines()
                    for line in lines[-15:]:
                        print(f"[{name}] {line}")
                    print("-" * 50)
                
                completed[name] = elapsed
                del active_processes[name]
        time.sleep(1)

    print("\n" + "=" * 80)
    print("📊 DUAL-GPU EXECUTION SUMMARY")
    print("=" * 80)
    for name, elapsed in completed.items():
        print(f" • {name:20}: completed in {elapsed:.1f}s")
    
    print(f"\n✓ Output logs and JSON results saved to: experiments/phase10/")
    print(f"✓ Total wall-clock time: {time.time() - t0:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
