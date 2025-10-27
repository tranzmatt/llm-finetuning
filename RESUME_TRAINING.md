# Resume Training from Checkpoint-500

## What Happened

Training crashed after **500 steps (15% complete)** due to fused loss bug.

**Good news:** You have checkpoint-500 saved!

**Bad news:** Training was VERY slow (151s/step instead of 4s/step)

## Fix and Resume

### Step 1: Apply the Fix

```bash
cd ~/Code/UML/llm-finetuning
python fix_fused_loss_final.py
```

This patches `scripts/train.py` to disable fused loss properly.

### Step 2: Edit Config to Resume

Edit `config/training_config.yaml`:

```yaml
training:
  # Add this line to resume from checkpoint
  resume_from_checkpoint: ./outputs/checkpoint-500
  
  # Everything else stays the same
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  # ...
```

### Step 3: Restart Training

```bash
cd ~/Code/UML/llm-finetuning

# If still in tmux
tmux attach -t training

# Start training
make train
```

Training will continue from step 500 and run to step 3264.

**Expected:**
- Remaining steps: 2764
- Time per step: ~4-5s (not 151s!)
- Total time: ~3-4 hours

## Why Was It So Slow?

The combination of:
1. Torch compile trying to optimize
2. Fused loss causing issues
3. Conflict between them

Created a **37x slowdown** (151s instead of 4s).

The fix disables both problematic features, returning to normal speed.

## After Fix - Expected Speed

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Step time | 151s | 4-5s |
| Remaining time | ~116 hours | ~3-4 hours |
| Stability | Crashes | Stable |

## Verification

After restarting, watch for:

```bash
# Good signs:
  1%|██ | 30/3264 [02:00<3:45:00, 4.2s/it]  ← 4.2s per step ✓
  {'loss': 1.234, 'learning_rate': 1.9e-05, 'epoch': 0.5}  ← Loss updates ✓
  GPU utilization: 95%  ← High GPU usage ✓
```

If you see 4-5 seconds per step, it's working correctly!

## Timeline After Fix

```
Already done:  500 steps (21 hours, but was buggy)
Remaining:    2764 steps (~3-4 hours at normal speed)
Total:        3264 steps
```

**You're 15% done, 85% to go** - should finish in ~3-4 hours after fix!

## Commands Summary

```bash
# 1. Apply fix
cd ~/Code/UML/llm-finetuning
python fix_fused_loss_final.py

# 2. Edit config
vim config/training_config.yaml
# Add: resume_from_checkpoint: ./outputs/checkpoint-500

# 3. Resume in tmux
tmux attach -t training
make train

# 4. Monitor
# Wait for step time to show ~4-5 seconds
# If still 151s, something else is wrong
```

## If Still Slow After Fix

Run diagnostics:

```bash
bash diagnose_slow_training.sh
```

This will check:
- GPU utilization
- Memory usage
- CPU load
- Disk I/O
- Config settings

And provide specific recommendations.

## What the Fix Does

The patch adds to `scripts/train.py`:

```python
# Disable fused cross-entropy loss
if hasattr(model, 'config'):
    model.config.use_fused_cross_entropy = False

# Disable torch compile
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
```

This ensures:
- No fused loss crash
- No torch compile slowdown
- Normal training speed
- Stable execution

## Bottom Line

1. Apply `fix_fused_loss_final.py`
2. Add `resume_from_checkpoint: ./outputs/checkpoint-500` to config
3. Run `make train`
4. Training should complete in ~3-4 hours at normal speed
5. No more crashes!

You've already spent 21 hours and got 15% done. With the fix, the remaining 85% will take just 3-4 hours! 🚀
