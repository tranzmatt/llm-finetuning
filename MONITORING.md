# Monitoring Training Without Attaching to tmux

## Quick Commands

### 1. Peek at Last 30 Lines (Fastest)
```bash
tmux capture-pane -pt training -S -30 && tmux show-buffer
```

### 2. Auto-Refresh View
```bash
watch -n 2 'tmux capture-pane -pt training -S -30 && tmux show-buffer'
```
Press Ctrl+C to stop watching.

### 3. Check Progress
```bash
# See latest checkpoint
ls -lt outputs/checkpoint-* | head -1

# Count total checkpoints
ls -d outputs/checkpoint-* | wc -l

# GPU usage
nvidia-smi
```

### 4. Use Status Script
```bash
chmod +x status.sh
./status.sh

# Or auto-refresh
watch -n 5 './status.sh'
```

## Setup Logging (Do This Once)

If you haven't started training yet, or when you restart:

```bash
# Attach to tmux
tmux attach -t training

# Enable logging (do this BEFORE starting training)
tmux pipe-pane -o 'cat >> ~/training_output.log'

# Start training
make train

# Detach: Ctrl+B, then D
```

Now you can monitor without attaching:
```bash
tail -f ~/training_output.log
```

## Already Training Without Logging?

Enable it now:

```bash
# From outside tmux, enable logging
tmux pipe-pane -t training -o 'cat >> ~/training_output.log'

# Now monitor
tail -f ~/training_output.log
```

## What to Look For

### Training is healthy if:
```
✓ GPU utilization: 90-100%
✓ Memory used: 50-60GB per GPU (stable)
✓ New checkpoints appearing every ~33 minutes
✓ Loss decreasing over time
✓ No error messages
```

### Training may have issues if:
```
⚠️ GPU utilization: <50%
⚠️ Memory: Fluctuating wildly
⚠️ No new checkpoints for >1 hour
⚠️ Loss = NaN or increasing
⚠️ Errors in output
```

## Monitoring Commands Cheat Sheet

```bash
# Quick peek (last 30 lines)
tmux capture-pane -pt training -S -30 && tmux show-buffer

# Watch live (updates every 2 seconds)
watch -n 2 'tmux capture-pane -pt training -S -30 && tmux show-buffer'

# GPU status
nvidia-smi

# Auto-refresh GPU
watch -n 5 nvidia-smi

# Latest checkpoint
ls -lt outputs/checkpoint-* | head -1

# Training log (if enabled)
tail -f ~/training_output.log

# Check for errors (if logging enabled)
grep -i error ~/training_output.log
grep -i "loss.*nan" ~/training_output.log

# Full status
./status.sh

# Auto-refresh status
watch -n 10 './status.sh'
```

## Remote Monitoring (From Your Laptop)

### Option 1: TensorBoard (Best!)

On server:
```bash
tensorboard --logdir=outputs --host=0.0.0.0 --port=6006 &
```

On your laptop:
```bash
ssh -L 6006:localhost:6006 user@your-server
```

Open browser: http://localhost:6006

You'll see:
- Real-time loss curves
- Learning rate schedule
- GPU metrics
- No need to SSH in!

### Option 2: SSH with Status Command

```bash
ssh user@your-server './status.sh'
```

Quick check without staying connected.

### Option 3: Cron Job + Email

Set up a cron job to email you status:
```bash
# Add to crontab (crontab -e)
*/30 * * * * cd ~/Code/UML/llm-finetuning && ./status.sh | mail -s "Training Status" you@email.com
```

Get status email every 30 minutes!

## Parse Training Progress

Extract useful info from logs:

```bash
# Get all loss values
grep -o "loss': [0-9.]*" ~/training_output.log | cut -d' ' -f2

# Latest loss
grep -o "loss': [0-9.]*" ~/training_output.log | tail -1

# Count completed steps
grep -c "{'loss'" ~/training_output.log

# Estimate completion
# If you see step 1000/3264, you're ~30% done
grep -o "[0-9]*/" ~/training_output.log | tail -1
```

## TensorBoard Setup (Detailed)

If not already configured, add to `config/training_config.yaml`:
```yaml
training:
  logging_steps: 10
  logging_dir: ./outputs/logs
  report_to: tensorboard
```

Start TensorBoard:
```bash
# On server
tensorboard --logdir=outputs --host=0.0.0.0 --port=6006 --reload_interval=30 &

# Or use make command if available
make tensorboard &
```

SSH tunnel from laptop:
```bash
ssh -N -L 6006:localhost:6006 user@your-server
```

Open: http://localhost:6006

## Monitoring Dashboard Script

Create a one-command dashboard:

```bash
#!/bin/bash
# dashboard.sh

while true; do
    clear
    echo "========================================"
    echo "Training Dashboard - $(date)"
    echo "========================================"
    echo ""
    
    echo "Last Output:"
    tmux capture-pane -pt training -S -10 && tmux show-buffer | tail -5
    echo ""
    
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo ""
    
    echo "Latest Checkpoint:"
    ls -lt outputs/checkpoint-* 2>/dev/null | head -1 | awk '{print $9, $6, $7, $8}'
    echo ""
    
    echo "Refreshing in 10s... (Ctrl+C to stop)"
    sleep 10
done
```

```bash
chmod +x dashboard.sh
./dashboard.sh
```

## Compare: Attach vs Monitor

| Method | Risk | Info | Use Case |
|--------|------|------|----------|
| **tmux attach** | ⚠️ High | Full | Debugging, interaction |
| **tmux capture** | ✅ None | Recent | Quick peek |
| **Log file** | ✅ None | Full history | Detailed review |
| **GPU monitor** | ✅ None | Resource usage | Health check |
| **Checkpoints** | ✅ None | Progress | Completion estimate |
| **TensorBoard** | ✅ None | Metrics | Best overall |

## Best Practice Workflow

1. **Start training with logging:**
   ```bash
   tmux new -s training
   tmux pipe-pane -o 'cat >> ~/training_output.log'
   make train
   # Detach: Ctrl+B, then D
   ```

2. **Set up TensorBoard:**
   ```bash
   tensorboard --logdir=outputs --host=0.0.0.0 --port=6006 &
   ```

3. **Monitor without attaching:**
   ```bash
   # Option A: Quick status
   ./status.sh
   
   # Option B: Watch logs
   tail -f ~/training_output.log
   
   # Option C: TensorBoard (from laptop)
   ssh -L 6006:localhost:6006 user@server
   # Open http://localhost:6006
   ```

4. **Only attach if needed:**
   ```bash
   tmux attach -t training
   # Investigate, then detach: Ctrl+B, then D
   ```

## Summary

**Never attach unless necessary!** Use:
- `tmux capture-pane` for quick peeks
- Log files for detailed history  
- TensorBoard for best visualization
- Status script for comprehensive checks

This way, **SSH disconnects never interrupt monitoring**.

---

**Download status.sh and run it:**
```bash
chmod +x status.sh
./status.sh
```
