#!/bin/bash
#SBATCH --job-name=finkyc-qwen-qlora
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "=========================================================="
echo " Starting FinKYC QLoRA Training on Slurm Cluster (VM03)"
echo " Job ID:       $SLURM_JOB_ID"
echo " Node:         $SLURMD_NODENAME"
echo " Date:         $(date)"
echo " User:         $(whoami)"
echo " Working Dir:  $(pwd)"
echo " GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU allocated via Slurm')"
echo "=========================================================="

mkdir -p logs

OUTPUT_DIR="${HOME}/checkpoints/job-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

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

TRAIN_SCRIPT="train_slurm.py"
if [ ! -f "$TRAIN_SCRIPT" ] && [ -f "train.py" ]; then
    TRAIN_SCRIPT="train.py"
fi

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

MODEL_ARG="Qwen/Qwen2.5-7B-Instruct"
if [ -f "/storage/tracka/models/base_model_path.txt" ]; then
    MODEL_ARG=$(cat /storage/tracka/models/base_model_path.txt | tr -d '\n')
    echo "Cluster Model Path: $MODEL_ARG"
fi

echo "Training Script: $TRAIN_SCRIPT"
echo "Train Data:      $TRAIN_DATA"
echo "Val Data:        $VAL_DATA"
echo "Test Data:       $TEST_DATA"
echo "Output Dir:      $OUTPUT_DIR"

$PYTHON_BIN "$TRAIN_SCRIPT" \
    --model_name "$MODEL_ARG" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_steps 100 \
    --lr 2e-4

EXIT_CODE=$?

echo "=========================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " Training job $SLURM_JOB_ID completed successfully!"
    echo " Adapter weights saved to: $OUTPUT_DIR"
    echo " To verify inference on cluster, run:"
    echo "   sbatch infer.sbatch $OUTPUT_DIR"
else
    echo " Training job $SLURM_JOB_ID failed with exit code $EXIT_CODE."
fi
echo " Finished at: $(date)"
echo "=========================================================="

exit $EXIT_CODE
