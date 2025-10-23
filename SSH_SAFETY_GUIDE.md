# SSH Disconnection Safety Guide

## Quick Answer

**Your current setup likely checkpoints every ~500 steps = ~33 minutes**

If you lose SSH:
- ✓ Training **continues** in background
- ⚠️ You lose up to **33 minutes** of work
- ✓ Can resume from last checkpoint

## Better: Use tmux or screen (Highly Recommended!)

### Option 1: tmux (Best)

```bash
# Start a tmux session
tmux new -s training

# Run your training
make train

# Detach (training keeps running)
# Press: Ctrl+B, then D

# Reconnect later (even after SSH disconnect)
tmux attach -t training

# List sessions
tmux ls
```

### Option 2: screen

```bash
# Start screen
screen -S training

# Run training
make train

# Detach: Ctrl+A, then D

# Reconnect
screen -r training
```

### Option 3: nohup (Simple but Less Control)

```bash
# Run in background
nohup make train > training.log 2>&1 &

# Get process ID
echo $!

# Monitor
tail -f training.log

# Kill if needed
kill <PID>
```

## Checkpoint Settings

Default (likely):
```yaml
save_strategy: steps
save_steps: 500          # Checkpoint every 500 steps
save_total_limit: 3      # Keep only 3 most recent
```

### Make Checkpoints More Frequent

Edit `config/training_config.yaml`:

```yaml
training:
  save_steps: 250        # Every ~17 minutes instead of ~33
  # or
  save_steps: 100        # Every ~7 minutes (more disk space)
  
  save_total_limit: 5    # Keep more checkpoints
```

**Trade-off:**
- More frequent = less work lost, but more disk space
- 500 steps = ~33 min, 2-3GB per checkpoint
- 250 steps = ~17 min, twice as many checkpoints
- 100 steps = ~7 min, 5x as many checkpoints

## Checkpoint Locations

Checkpoints saved to:
```
./outputs/checkpoint-500/
./outputs/checkpoint-1000/
./outputs/checkpoint-1500/
...
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Training state
- Config

Size: ~2-3GB per checkpoint

## Resume from Checkpoint

If training stops, resume:

```yaml
# config/training_config.yaml
training:
  resume_from_checkpoint: ./outputs/checkpoint-1500
```

Then:
```bash
make train
```

## Monitor Training Remotely

### 1. Check if still running
```bash
# See if process exists
ps aux | grep train.py

# Check GPU usage
nvidia-smi
```

### 2. Watch progress
```bash
# Watch logs
tail -f logs/training_*.log

# Or if using nohup
tail -f training.log
```

### 3. Check for new checkpoints
```bash
watch -n 60 'ls -lt outputs/ | head -10'
```

### 4. TensorBoard (if set up)
```bash
make tensorboard
# Forward port 6006 to view
```

## Best Practice Setup

1. **Start tmux FIRST**
   ```bash
   tmux new -s training
   ```

2. **Run training**
   ```bash
   make train
   ```

3. **Detach safely**
   ```
   Ctrl+B, then D
   ```

4. **Disconnect SSH** (training continues!)

5. **Reconnect anytime**
   ```bash
   ssh your-server
   tmux attach -t training
   ```

## Checkpoint Frequency Recommendations

| save_steps | Interval | Checkpoints (3 epochs) | Disk Space | Risk |
|------------|----------|------------------------|------------|------|
| 500 | ~33 min | ~6 | ~12-18GB | Medium |
| 250 | ~17 min | ~13 | ~26-39GB | Low |
| 100 | ~7 min | ~32 | ~64-96GB | Very Low |

**Recommendation:** 
- If disk space OK: `save_steps: 250`
- If disk limited: `save_steps: 500` + use tmux

## Emergency: Training Stopped

### Check what happened
```bash
# Find the process
ps aux | grep train.py

# Check recent logs
tail -100 logs/training_*.log

# Check last checkpoint
ls -lt outputs/ | head -5
```

### Resume from last checkpoint
```bash
# Find latest checkpoint
LATEST=$(ls -d outputs/checkpoint-* | sort -V | tail -1)
echo "Latest: $LATEST"

# Edit config to resume
# config/training_config.yaml:
#   resume_from_checkpoint: $LATEST

# Restart
make train
```

## Summary

**Use tmux or screen** - this is the professional way to run long training jobs!

With tmux:
- ✅ SSH disconnects don't matter
- ✅ Can reconnect from anywhere
- ✅ See live output anytime
- ✅ Can kill/restart easily
- ✅ No work lost (unless server crashes)

**Plus checkpoints every 250-500 steps** for hardware failure protection.

---

**Run this to check your exact settings:**
```bash
python check_checkpoints.py
```
