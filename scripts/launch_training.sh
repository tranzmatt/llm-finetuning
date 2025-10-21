#!/bin/bash
set -e

NUM_GPUS=4
MASTER_PORT=29500
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_SCRIPT="$SCRIPT_DIR/train.py"

echo "============================================================================"
echo "🦥 UML Fine-tuning - Multi-GPU Launcher"
echo "============================================================================"

# Check script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "❌ Training script not found"
    exit 1
fi

# Check GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "GPUs: $NUM_GPUS"
echo "Port: $MASTER_PORT"
echo ""

# Set environment
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Confirm
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting training..."
echo ""

cd "$(dirname "$SCRIPT_DIR")"

# Pass all arguments directly to train.py
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    "$TRAINING_SCRIPT" \
    "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ TRAINING COMPLETE"
    echo "============================================================================"
else
    echo ""
    echo "❌ Training failed"
    exit 1
fi
