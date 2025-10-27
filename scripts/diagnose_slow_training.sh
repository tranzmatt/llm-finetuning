#!/bin/bash

# Diagnose Slow Training
# Figure out why training is 151s/step instead of 4s/step

echo "========================================================================"
echo "TRAINING SPEED DIAGNOSIS"
echo "========================================================================"
echo ""

echo "Expected speed: ~4 seconds per step"
echo "Your speed: 151 seconds per step"
echo "That's 37x SLOWER than expected!"
echo ""

echo "Checking for common issues..."
echo ""

echo "1. GPU Configuration:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

echo "2. CPU Configuration:"
echo "----------------------------------------"
lscpu | grep -E "^CPU\(s\)|Model name|Thread|Core"
echo ""

echo "3. Process Info:"
echo "----------------------------------------"
ps aux | grep -E "train.py|torchrun" | grep -v grep | head -5
echo ""

echo "4. System Load:"
echo "----------------------------------------"
uptime
echo ""

echo "5. Disk I/O (could cause slowness):"
echo "----------------------------------------"
iostat -x 1 2 | grep -A 2 "Device"
echo ""

echo "6. Memory Usage:"
echo "----------------------------------------"
free -h
echo ""

echo "========================================================================"
echo "COMMON CAUSES OF SLOW TRAINING"
echo "========================================================================"
echo ""
echo "❌ Torch compile enabled (causes initial slowdown)"
echo "   Fix: Disable torch._dynamo"
echo ""
echo "❌ Gradient checkpointing issues"
echo "   Fix: Verify use_gradient_checkpointing: false"
echo ""
echo "❌ CPU bottleneck (data loading)"
echo "   Fix: Increase num_workers or reduce batch size"
echo ""
echo "❌ Disk I/O bottleneck"
echo "   Fix: Move data to faster storage"
echo ""
echo "❌ Memory swapping"
echo "   Fix: Reduce batch size or model size"
echo ""
echo "❌ Wrong CUDA/PyTorch compilation"
echo "   Fix: Reinstall with correct CUDA version"
echo ""
echo "========================================================================"
echo ""

# Check config
CONFIG_FILE="$HOME/Code/UML/llm-finetuning/config/training_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "Current Config Settings:"
    echo "----------------------------------------"
    grep -E "batch_size|gradient_accumulation|gradient_checkpointing|num_workers" "$CONFIG_FILE" | head -10
    echo ""
fi

echo "========================================================================"
echo "RECOMMENDATIONS"
echo "========================================================================"
echo ""
echo "Based on your error, the likely causes are:"
echo ""
echo "1. MOST LIKELY: Torch compile + fused loss interaction"
echo "   Solution: Apply fix_fused_loss_final.py"
echo ""
echo "2. Gradient checkpointing causing issues"
echo "   Solution: Verify it's disabled in config"
echo ""
echo "3. Data loading bottleneck"
echo "   Solution: Check num_workers setting"
echo ""
echo "Apply the fix and restart from checkpoint-500"
echo "Speed should improve to ~4-5s/step"
echo ""
echo "========================================================================"
