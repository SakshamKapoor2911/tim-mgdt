#!/usr/bin/env python3
"""Test Phase 10 stages with full verbose output on GPU machine"""

import sys
sys.path.insert(0, ".")

print("\n" + "="*70)
print("PHASE 10 STAGE TEST - RUNNING ON GPU MACHINE")
print("="*70 + "\n")

from code.phases.phase10.stage1_identification import run_stage1
from code.phases.phase10.stage2_refinement import run_stage2
from code.phases.phase10.stage3_pattern_analysis import run_stage3
from code.phases.phase10.stage4_deployment_metrics import run_stage4
from code.phases.phase10.utils import get_device

device = get_device()
print(f"Device: {device}\n")

print("\n>>> Executing Stage 1...")
r1 = run_stage1()
print(f"Stage 1 Result: {r1}\n")

print("\n>>> Executing Stage 2...")
r2 = run_stage2()
print(f"Stage 2 Result: {r2}\n")

print("\n>>> Executing Stage 3...")
r3 = run_stage3()
print(f"Stage 3 Result: {r3}\n")

print("\n>>> Executing Stage 4...")
r4 = run_stage4()
print(f"Stage 4 Result: {r4}\n")

print("="*70)
print("✓ ALL STAGES EXECUTED SUCCESSFULLY ON GPU MACHINE")
print("="*70 + "\n")
