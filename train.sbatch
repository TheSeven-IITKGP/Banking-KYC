#!/bin/bash
#SBATCH --job-name=finkyc-qwen-qlora
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# ==============================================================================
# FinKYC: Indian Banking KYC QLoRA Fine-Tuning Job on Slurm Cluster (VM03)
# Author: chandan bauri <chandanbauri441@gmail.com>
# Base Model: Qwen/Qwen2.5-7B-Instruct (shared read-only on cluster)
# Checkpoints saved to: ~/checkpoints/job-$SLURM_JOB_ID/
# ==============================================================================

echo "=========================================================="
echo " Starting FinKYC QLoRA Training on Slurm Cluster (VM03)"
echo " Job ID:       $SLURM_JOB_ID"
echo " Node:         $SLURMD_NODENAME"
echo " Date:         $(date)"
echo " User:         $(whoami)"
echo " Working Dir:  $(pwd)"
echo " GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU allocated via Slurm')"
echo "=========================================================="

# Ensure logs directory exists
mkdir -p logs

# Set output directory following cluster convention: ~/checkpoints/job-JOB_ID
OUTPUT_DIR="${HOME}/checkpoints/job-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Activate Python environment on VM if present
if [ -f "${HOME}/.venv/bin/activate" ]; then
    echo "Activating virtualenv at ${HOME}/.venv..."
    source "${HOME}/.venv/bin/activate"
elif [ -f "./.venv/bin/activate" ]; then
    echo "Activating virtualenv at ./.venv..."
    source "./.venv/bin/activate"
fi

# Locate training script (prefers train_slurm.py, falls back to train.py)
TRAIN_SCRIPT="train_slurm.py"
if [ ! -f "$TRAIN_SCRIPT" ] && [ -f "train.py" ]; then
    TRAIN_SCRIPT="train.py"
fi

# Locate training datasets (checks root project dir and data/processed/)
TRAIN_DATA="kyc_train.jsonl"
if [ ! -f "$TRAIN_DATA" ] && [ -f "data/processed/kyc_train.jsonl" ]; then
    TRAIN_DATA="data/processed/kyc_train.jsonl"
fi

VAL_DATA="kyc_val.jsonl"
if [ ! -f "$VAL_DATA" ] && [ -f "data/processed/kyc_val.jsonl" ]; then
    VAL_DATA="data/processed/kyc_val.jsonl"
fi

TEST_DATA="kyc_test.jsonl"
if [ ! -f "$TEST_DATA" ] && [ -f "data/processed/kyc_test.jsonl" ]; then
    TEST_DATA="data/processed/kyc_test.jsonl"
fi

echo "Training Script: $TRAIN_SCRIPT"
echo "Train Data:      $TRAIN_DATA"
echo "Val Data:        $VAL_DATA"
echo "Test Data:       $TEST_DATA"
echo "Output Dir:      $OUTPUT_DIR"

# Execute QLoRA training on Qwen2.5-7B-Instruct
python3 "$TRAIN_SCRIPT" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --lora_r 16 \
    --lora_alpha 32 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4

EXIT_CODE=$?

echo "=========================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " Training job $SLURM_JOB_ID completed successfully!"
    echo " Adapter weights saved to: $OUTPUT_DIR"
    echo " To verify inference on cluster, run:"
    echo "   python3 infer_slurm.py --adapter_path $OUTPUT_DIR"
else
    echo " Training job $SLURM_JOB_ID failed with exit code $EXIT_CODE."
fi
echo " Finished at: $(date)"
echo "=========================================================="

exit $EXIT_CODE
