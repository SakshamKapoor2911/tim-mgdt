#!/bin/bash
set -e

# Configuration
TASKS="mmlu,gsm8k,arc_challenge"
FEW_SHOT="5"
RESULTS_DIR="./results"

mkdir -p $RESULTS_DIR

# 1. Merge Mistral LoRA
# echo "Merging Mistral-7B-v0.1..."
# python scratch/merge_peft.py \
#     --dropped_base "checkpoints/Mistral-7B-v0.1_dropped_unhealed" \
#     --lora "checkpoints/Mistral-7B-v0.1_boundary_healed_t2_lora" \
#     --output "checkpoints/Mistral-7B-v0.1_boundary_healed_merged"

# 2. Merge Llama-3 LoRA
# echo "Merging Meta-Llama-3-8B..."
# python scratch/merge_peft.py \
#     --dropped_base "checkpoints/Meta-Llama-3-8B_dropped_unhealed" \
#     --lora "checkpoints/Meta-Llama-3-8B_boundary_healed_t2_lora" \
#     --output "checkpoints/Meta-Llama-3-8B_boundary_healed_merged"

# 3. Define Models to Evaluate
MODELS=(
    "mistralai/Mistral-7B-v0.1"                     # 1. Mistral Unpruned Baseline
    "./checkpoints/Mistral-7B-v0.1_dropped_unhealed"        # 2. Mistral Dropped (Unhealed)
    "./checkpoints/Mistral-7B-v0.1_boundary_healed_merged"  # 3. Mistral Merged Healed Checkpoint
    "meta-llama/Meta-Llama-3-8B"                            # 4. Llama-3 Unpruned Baseline
    "./checkpoints/Meta-Llama-3-8B_dropped_unhealed"        # 5. Llama-3 Dropped (Unhealed)
    "./checkpoints/Meta-Llama-3-8B_boundary_healed_merged"  # 6. Llama-3 Merged Healed Checkpoint
)

# 4. Run Evaluations
for MODEL_PATH in "${MODELS[@]}"; do
    echo "======================================================"
    echo "Initiating evaluation for: $MODEL_PATH"
    echo "======================================================"
    
    MODEL_NAME=$(basename "$MODEL_PATH")

    # Phase A: Loglikelihood Tasks (MMLU, ARC) -> FAST on Hugging Face
    if ! ls $RESULTS_DIR/${MODEL_NAME}_hf_eval*.json >/dev/null 2>&1; then
        echo "Running MMLU and ARC via HF (DDP Multi-GPU)..."
        venv_eval/bin/accelerate launch -m lm_eval --model hf \
          --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
          --tasks mmlu,arc_challenge \
          --num_fewshot $FEW_SHOT \
          --batch_size auto \
          --output_path $RESULTS_DIR/${MODEL_NAME}_hf_eval.json
    else
        echo "Skipping Phase A: $RESULTS_DIR/${MODEL_NAME}_hf_eval.json already exists."
    fi

    # Phase B: Generation Tasks (GSM8K) -> FAST on vLLM
    if ! ls $RESULTS_DIR/${MODEL_NAME}_vllm_eval*.json >/dev/null 2>&1; then
        echo "Running GSM8K via vLLM..."
        venv_eval/bin/lm_eval --model vllm \
          --model_args pretrained=$MODEL_PATH,tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.50,max_num_seqs=256,max_num_batched_tokens=8192,enforce_eager=True,max_model_len=4096 \
          --tasks gsm8k \
          --num_fewshot $FEW_SHOT \
          --batch_size auto \
          --output_path $RESULTS_DIR/${MODEL_NAME}_vllm_eval.json
    else
        echo "Skipping Phase B: $RESULTS_DIR/${MODEL_NAME}_vllm_eval.json already exists."
    fi
      
    echo "Finished evaluating $MODEL_NAME."
done

echo "All evaluations complete. Check the $RESULTS_DIR directory."
