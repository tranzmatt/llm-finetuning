#!/bin/bash
set -e

# OPTION 1: Use 3 contiguous A100 GPUs (RECOMMENDED for stability)
# This avoids non-contiguous GPU communication issues
export CUDA_VISIBLE_DEVICES=0,1,2
NUM_GPUS=3

# OPTION 2: Use all 4 A100 GPUs (requires NCCL tuning)
# Uncomment these lines to use all 4 GPUs:
# export CUDA_VISIBLE_DEVICES=0,1,2,4
# NUM_GPUS=4
# export NCCL_P2P_DISABLE=1  # Disable direct P2P between non-contiguous GPUs
# export NCCL_IB_DISABLE=1   # Disable InfiniBand (use PCIe)

MASTER_PORT=29500
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_SCRIPT="$SCRIPT_DIR/train.py"

echo "============================================================================"
echo "🦥 UML Fine-tuning - Multi-GPU Launcher"
echo "============================================================================"
echo "Configuration: $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Physical GPUs: $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')"
echo "============================================================================"

# Check script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "❌ Training script not found"
    exit 1
fi

echo "GPUs: $NUM_GPUS"
echo "Port: $MASTER_PORT"
echo ""

# Set environment
export NCCL_DEBUG=INFO
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
