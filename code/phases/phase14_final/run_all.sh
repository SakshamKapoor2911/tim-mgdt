#!/bin/bash
set -e

# Mistral 7B Pipeline
echo "=========================================================="
echo " Starting Phase 14 Pipeline for Mistral 7B "
echo "=========================================================="
PYTHONPATH=. python code/phases/phase14_final/1_measure_sensitivity.py --model "mistralai/Mistral-7B-v0.1" --samples 30
PYTHONPATH=. python code/phases/phase14_final/2_run_surgical_recovery.py --model "mistralai/Mistral-7B-v0.1" --eval_samples 30 --train_samples 500

# Llama 3 8B Pipeline (Cross-Architecture Validation)
echo "=========================================================="
echo " Starting Phase 14 Pipeline for Llama-3-8B "
echo "=========================================================="
bash code/phases/phase14_final/3_run_cross_architecture.sh
