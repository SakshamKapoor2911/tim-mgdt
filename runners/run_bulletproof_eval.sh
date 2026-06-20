#!/usr/bin/env bash
# Bulletproof eval runner — sets up environment explicitly for nohup/background use
set -e

REPO=/home/skapoor/tim-mgdt
VENV=$REPO/venv
LOG=$REPO/logs/bulletproof_eval.log

export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

echo "=== bulletproof_eval starting at $(date) ===" 
echo "PYTHONPATH=$PYTHONPATH"
echo "Python: $VENV/bin/python"

$VENV/bin/python -u -m code.phases.phase14_final.4_bulletproof_eval \
    --models mistralai/Mistral-7B-v0.1 meta-llama/Meta-Llama-3-8B \
    --eval_samples 256 \
    --output_file $REPO/experiments/phase14_final/bulletproof_eval.json

echo "=== bulletproof_eval done at $(date) ==="
