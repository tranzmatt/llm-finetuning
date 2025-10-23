#!/bin/bash

# Quick Training Status Check
# Run this without attaching to tmux

echo "============================================================================"
echo "TRAINING STATUS CHECK"
echo "============================================================================"
echo ""

# Check if tmux session exists
if tmux has-session -t training 2>/dev/null; then
    echo "✓ tmux session 'training' is running"
else
    echo "❌ tmux session 'training' not found"
    exit 1
fi

echo ""
echo "Last 30 lines of output:"
echo "----------------------------------------------------------------------------"
tmux capture-pane -pt training -S -30 && tmux show-buffer
echo "----------------------------------------------------------------------------"

echo ""
echo "GPU Status:"
echo "----------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv
echo ""

echo "Latest Checkpoints:"
echo "----------------------------------------------------------------------------"
ls -lt outputs/checkpoint-* 2>/dev/null | head -5 || echo "No checkpoints yet"
echo ""

echo "Process Info:"
echo "----------------------------------------------------------------------------"
ps aux | grep -E "train.py|torchrun" | grep -v grep | head -3 || echo "No training process found"
echo ""

echo "============================================================================"
echo ""
echo "To see more detail:"
echo "  tmux attach -t training    # Full attach"
echo "  tail -f ~/training_output.log    # If logging enabled"
echo "  watch -n 2 './status.sh'   # Auto-refresh this status"
echo ""
echo "============================================================================"
