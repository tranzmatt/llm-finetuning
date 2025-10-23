#!/usr/bin/env python3
"""
Check Checkpoint Configuration
Shows current settings and recommendations for SSH disconnection safety
"""

import yaml
from pathlib import Path

PROJECT_ROOT = Path.cwd()
CONFIG_FILE = PROJECT_ROOT / "config" / "training_config.yaml"

print("\n" + "="*80)
print("CHECKPOINT CONFIGURATION")
print("="*80)

if not CONFIG_FILE.exists():
    print(f"\n❌ Config file not found: {CONFIG_FILE}")
    exit(1)

with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)

training = config.get('training', {})

# Key checkpoint settings
save_strategy = training.get('save_strategy', 'steps')
save_steps = training.get('save_steps', 500)
save_total_limit = training.get('save_total_limit', 3)
max_steps = training.get('max_steps', -1)
num_train_epochs = training.get('num_train_epochs', 3)
per_device_batch = training.get('per_device_train_batch_size', 1)
gradient_accum = training.get('gradient_accumulation_steps', 16)

print(f"\nCurrent Settings:")
print(f"  save_strategy: {save_strategy}")
print(f"  save_steps: {save_steps}")
print(f"  save_total_limit: {save_total_limit}")
print(f"  num_train_epochs: {num_train_epochs}")

# Calculate checkpoint frequency
print(f"\n" + "="*80)
print("CHECKPOINT FREQUENCY")
print("="*80)

# Your data: 58,025 examples, 3 GPUs
total_examples = 58025
num_gpus = 3
examples_per_gpu = total_examples // num_gpus
steps_per_epoch = examples_per_gpu // (per_device_batch * gradient_accum)
total_steps = steps_per_epoch * num_train_epochs

print(f"\nTraining Details:")
print(f"  Total examples: {total_examples:,}")
print(f"  GPUs: {num_gpus}")
print(f"  Batch size per GPU: {per_device_batch}")
print(f"  Gradient accumulation: {gradient_accum}")
print(f"  Effective batch size: {per_device_batch * gradient_accum * num_gpus}")
print(f"\n  Steps per epoch: ~{steps_per_epoch}")
print(f"  Total steps (3 epochs): ~{total_steps}")

# Time estimates (assuming ~4s per step)
step_time = 4.0  # seconds
checkpoint_interval_time = save_steps * step_time / 60  # minutes
checkpoints_per_epoch = steps_per_epoch // save_steps
total_checkpoints = total_steps // save_steps

print(f"\nCheckpoint Timing:")
print(f"  Checkpoint every: {save_steps} steps")
print(f"  Time between checkpoints: ~{checkpoint_interval_time:.1f} minutes")
print(f"  Checkpoints per epoch: ~{checkpoints_per_epoch}")
print(f"  Total checkpoints: ~{total_checkpoints}")
print(f"  Kept on disk: {save_total_limit} (oldest deleted)")

print(f"\n" + "="*80)
print("SSH DISCONNECTION SAFETY")
print("="*80)

max_loss_time = checkpoint_interval_time

print(f"\nIf you lose SSH connection:")
print(f"  ✓ Training continues in background")
print(f"  ✓ Maximum work lost: ~{max_loss_time:.0f} minutes")
print(f"  ✓ Can resume from last checkpoint")

if checkpoint_interval_time > 30:
    print(f"\n⚠️  WARNING: Checkpoints are {checkpoint_interval_time:.0f} minutes apart!")
    print(f"   Consider reducing save_steps for safety")
elif checkpoint_interval_time > 60:
    print(f"\n⚠️  CAUTION: Checkpoints are {checkpoint_interval_time:.0f} minutes apart")
    print(f"   Recommend save_steps=250 (~17 min) or save_steps=100 (~7 min)")
else:
    print(f"\n✓ Good: Checkpoints every {checkpoint_interval_time:.0f} minutes")

print(f"\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"\nFor better SSH safety, you can:")
print(f"\n1. Reduce checkpoint frequency:")
print(f"   save_steps: 500 → 250  (every ~17 min instead of ~33 min)")
print(f"   save_steps: 500 → 100  (every ~7 min instead of ~33 min)")

print(f"\n2. Use tmux/screen (recommended):")
print(f"   tmux new -s training")
print(f"   make train")
print(f"   # Press Ctrl+B, then D to detach")
print(f"   # Reconnect later: tmux attach -t training")

print(f"\n3. Use nohup:")
print(f"   nohup make train > training.log 2>&1 &")
print(f"   # Training continues even if you disconnect")
print(f"   # Monitor: tail -f training.log")

print(f"\n4. Keep more checkpoints:")
print(f"   save_total_limit: 3 → 5 (keeps more checkpoints)")

print(f"\n" + "="*80)
print("CHECKPOINT LOCATIONS")
print("="*80)

output_dir = training.get('output_dir', './outputs')
print(f"\nCheckpoints saved to: {output_dir}")
print(f"\nCheckpoint directories will look like:")
print(f"  {output_dir}/checkpoint-500/")
print(f"  {output_dir}/checkpoint-1000/")
print(f"  {output_dir}/checkpoint-1500/")
print(f"  ...")

print(f"\nTo resume from checkpoint:")
print(f"  # Edit config/training_config.yaml:")
print(f"  resume_from_checkpoint: {output_dir}/checkpoint-1500")
print(f"  # Then: make train")

print(f"\n" + "="*80)
print("MONITORING")
print("="*80)

print(f"\nTo monitor training remotely:")
print(f"  1. TensorBoard:")
print(f"     make tensorboard")
print(f"     # Open: http://localhost:6006")
print(f"\n  2. Watch logs:")
print(f"     tail -f logs/training_*.log")
print(f"\n  3. Check for new checkpoints:")
print(f"     watch -n 60 'ls -lt {output_dir}/ | head -10'")

print(f"\n" + "="*80)
