#!/bin/bash
#SBATCH --job-name=finkyc-infer
#SBATCH --output=logs/infer_%j.log
#SBATCH --error=logs/infer_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "=========================================================="
echo " Starting FinKYC GPU Inference on Slurm Cluster (VM03)"
echo " Job ID:       $SLURM_JOB_ID"
echo " Node:         $SLURMD_NODENAME"
echo " Date:         $(date)"
echo " User:         $(whoami)"
echo " Working Dir:  $(pwd)"
echo " GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU allocated via Slurm')"
echo "=========================================================="

mkdir -p logs

if [ -f "/storage/tracka/runtime/venv/bin/activate" ]; then
    echo "Activating shared cluster venv at /storage/tracka/runtime/venv..."
    source "/storage/tracka/runtime/venv/bin/activate"
elif [ -f "${HOME}/.venv/bin/activate" ]; then
    source "${HOME}/.venv/bin/activate"
elif [ -f "./.venv/bin/activate" ]; then
    source "./.venv/bin/activate"
fi

PYTHON_BIN="python3"
if [ -f "/storage/tracka/runtime/venv/bin/python3" ]; then
    PYTHON_BIN="/storage/tracka/runtime/venv/bin/python3"
fi
echo "Using Python: $($PYTHON_BIN --version 2>&1) at $PYTHON_BIN"

ADAPTER_PATH="$1"
if [ -z "$ADAPTER_PATH" ]; then
    ADAPTER_PATH=$(ls -td ${HOME}/checkpoints/job-* 2>/dev/null | head -n 1)
fi

echo "Target Adapter Path: $ADAPTER_PATH"

$PYTHON_BIN infer_slurm.py --adapter_path "$ADAPTER_PATH"

echo "=========================================================="
echo " Inference Completed at $(date)"
echo " Output Log: logs/infer_${SLURM_JOB_ID}.log"
echo "=========================================================="
