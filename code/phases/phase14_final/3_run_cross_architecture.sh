#!/bin/bash
# Phase 14: Cross-Architecture Validation on Llama-3-8B
# This script executes the sensitivity measurement and surgical recovery pipeline 
# on Llama-3-8B to prove the Bathtub Curve and surgical allocation generalize.

set -e

MODEL="meta-llama/Meta-Llama-3-8B"

echo "=========================================================="
echo " Starting Phase 14 Cross-Architecture Validation for Llama 3 "
echo "=========================================================="

echo "1. Measuring Layer Sensitivity (Bathtub Curve) & Logging Oracle Compute Cost..."
PYTHONPATH=. python code/phases/phase14_final/1_measure_sensitivity.py --model "$MODEL" --samples 30

echo "\n2. Running Surgical Recovery vs Baselines..."
PYTHONPATH=. python code/phases/phase14_final/2_run_surgical_recovery.py --model "$MODEL" --eval_samples 30 --train_samples 500

echo "\nCross-Architecture Validation Complete."
