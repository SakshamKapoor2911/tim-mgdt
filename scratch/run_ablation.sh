#!/bin/bash
set -e

# All tasks will run on GPU 1 to leave GPU 0 for the ongoing ASDiv evaluation
export CUDA_VISIBLE_DEVICES=1

echo "--- Phase 1: Boundary Slicing (floor) ---"
python3 scratch/prune_subnetworks_aligned.py --strategy floor --output_dir ./checkpoints/Meta-Llama-3-8B_subnetwork_pruned_aligned_floor

echo "--- Phase 2: Structural Healing Loop ---"
echo "Healing ceil..."
python3 scratch/sparse_lora_heal.py --model_id ./checkpoints/Meta-Llama-3-8B_subnetwork_pruned_aligned_ceil --output_dir ./checkpoints/sparse_lora_healed_ceil
echo "Healing floor..."
python3 scratch/sparse_lora_heal.py --model_id ./checkpoints/Meta-Llama-3-8B_subnetwork_pruned_aligned_floor --output_dir ./checkpoints/sparse_lora_healed_floor

echo "--- Phase 3: Downstream Reasoning Crucible ---"
echo "Evaluating ceil..."
venv_eval/bin/python scratch/evaluate_healed.py --tasks gsm8k --pruned_dir ./checkpoints/Meta-Llama-3-8B_subnetwork_pruned_aligned_ceil --lora_dir ./checkpoints/sparse_lora_healed_ceil --output_file ./results/Meta-Llama-3-8B_sparse_healed_eval_ceil.json
echo "Evaluating floor..."
venv_eval/bin/python scratch/evaluate_healed.py --tasks gsm8k --pruned_dir ./checkpoints/Meta-Llama-3-8B_subnetwork_pruned_aligned_floor --lora_dir ./checkpoints/sparse_lora_healed_floor --output_file ./results/Meta-Llama-3-8B_sparse_healed_eval_floor.json

echo "--- Aggregation ---"
python3 scratch/aggregator.py

echo "--- Ablation Study Complete ---"
