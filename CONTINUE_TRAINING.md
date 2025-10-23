# How to Continue Training After Initial Run

## Common Scenarios

### Scenario 1: Loss Still Decreasing (Model Learning)
**What to do:** Add more epochs

```yaml
# config/training_config.yaml
training:
  num_train_epochs: 5  # Add 2 more epochs (was 3)
  resume_from_checkpoint: ./outputs/checkpoint-3264  # Final checkpoint
```

**Expected:** Loss continues dropping, quality improves

---

### Scenario 2: Loss Plateaued (Model Converged)
**What to do:** Lower learning rate and continue

```yaml
training:
  num_train_epochs: 2  # 2 more epochs with lower LR
  learning_rate: 0.00005  # 5x lower (was 0.0001)
  resume_from_checkpoint: ./outputs/checkpoint-3264
```

**Expected:** Small improvements, fine-tuning

---

### Scenario 3: Overfitting (Training Good, Validation Bad)
**What to do:** More regularization

```yaml
lora:
  lora_dropout: 0.05  # Add some dropout back
  
training:
  weight_decay: 0.01  # Add regularization
  # Don't add more epochs - model already overfit
```

Or get more training data!

---

### Scenario 4: Not Learning At All (Loss Flat)
**What to do:** Start over with different hyperparameters

```yaml
training:
  learning_rate: 0.0002  # Try higher (was 0.0001)
  # Remove resume_from_checkpoint - start fresh
  
lora:
  r: 128  # More capacity (was 64)
  lora_alpha: 256
```

---

### Scenario 5: Some Diagrams Good, Others Bad
**What to do:** Continue training on filtered data

1. Filter dataset to problem areas
2. Continue with lower LR
3. Focus training on weak points

```yaml
training:
  num_train_epochs: 2
  learning_rate: 0.00003  # Lower LR for fine-tuning
  resume_from_checkpoint: ./outputs/checkpoint-3264
```

---

## Recommended Training Progression

### Phase 1: Initial Training (3 epochs)
```yaml
num_train_epochs: 3
learning_rate: 0.0001
lora_dropout: 0
```

**Check results → Not satisfied?**

### Phase 2: Extended Training (2 more epochs)
```yaml
num_train_epochs: 5  # Total of 5
learning_rate: 0.0001  # Same LR
resume_from_checkpoint: ./outputs/checkpoint-3264
```

**Check results → Better but could be refined?**

### Phase 3: Fine-tuning (2 epochs, lower LR)
```yaml
num_train_epochs: 2  # 2 more epochs
learning_rate: 0.00003  # 3x lower
resume_from_checkpoint: ./outputs/checkpoint-5440  # From phase 2
```

**Check results → Should be very good now!**

---

## How to Check Results

### During Training
```bash
# Watch loss - should be decreasing
tail -f ~/training_output.log | grep "loss"

# Check in TensorBoard
tensorboard --logdir=outputs
```

### After Training
```bash
# Test the model
python scripts/test_model.py

# Generate sample UML diagrams
python scripts/generate_samples.py
```

### What Good Results Look Like
- ✓ Loss: < 0.5 (started ~2.3)
- ✓ UML syntax: Valid PlantUML
- ✓ Diagram structure: Logical and complete
- ✓ Entity names: Relevant to prompt
- ✓ Relationships: Accurate

### What Bad Results Look Like
- ❌ Loss: Still > 1.0
- ❌ Invalid syntax (won't render)
- ❌ Incomplete diagrams
- ❌ Hallucinated entities
- ❌ Wrong diagram type

---

## Quick Decision Tree

```
Is loss still decreasing after 3 epochs?
├─ YES → Add 2 more epochs (same LR)
└─ NO → Loss plateaued?
    ├─ YES → Lower LR, add 2 epochs
    └─ NO → Loss flat from start?
        └─ YES → Restart with higher LR or more capacity
```

---

## Configuration Examples

### Continue Training (2 More Epochs)
```yaml
training:
  output_dir: ./outputs_extended
  num_train_epochs: 2  # Just 2 more
  learning_rate: 0.0001
  resume_from_checkpoint: ./outputs/checkpoint-3264
  
  # Everything else same as original
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  # ...
```

### Fine-tune with Lower LR
```yaml
training:
  output_dir: ./outputs_finetuned
  num_train_epochs: 2
  learning_rate: 0.00003  # 3x lower
  resume_from_checkpoint: ./outputs/checkpoint-3264
  
  # Rest same
```

### Train Longer from Scratch (If Restarting)
```yaml
training:
  output_dir: ./outputs_v2
  num_train_epochs: 5  # More epochs this time
  learning_rate: 0.00015  # Slightly higher
  # NO resume_from_checkpoint
  
lora:
  r: 128  # Double the rank
  lora_alpha: 256
  # Rest same
```

---

## Commands

### Continue Training
```bash
# 1. Edit config to add epochs and set resume_from_checkpoint
# 2. Run training
make train

# Or specify config directly
python scripts/train.py --config config/continue_training.yaml
```

### Compare Results
```bash
# Test original model
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264

# Test extended model  
python scripts/evaluate.py --checkpoint ./outputs_extended/checkpoint-5440

# Compare outputs side-by-side
python scripts/compare_models.py \
  --model1 ./outputs/checkpoint-3264 \
  --model2 ./outputs_extended/checkpoint-5440
```

---

## Best Practices

### ✅ DO:
- Save checkpoints frequently (every 250-500 steps)
- Test model quality before continuing
- Lower LR when fine-tuning
- Keep original checkpoints (don't overwrite)
- Use different output_dir for each training run
- Monitor validation metrics

### ❌ DON'T:
- Train forever without checking results
- Use same LR for fine-tuning
- Overwrite original checkpoints
- Train more if already overfitting
- Ignore validation performance
- Skip evaluation between training runs

---

## Learning Rate Guidelines

| Scenario | Original LR | New LR | Reasoning |
|----------|-------------|--------|-----------|
| Initial training | - | 0.0001 | Standard |
| Continue (loss dropping) | 0.0001 | 0.0001 | Keep same |
| Fine-tune (converged) | 0.0001 | 0.00003 | 3x lower for polish |
| Restart (not learning) | 0.0001 | 0.0002 | 2x higher to learn |
| Overfitting | 0.0001 | 0.00005 | Lower + regularization |

---

## Memory: Model Capacity vs Epochs

**Not learning well?**
- Try: More epochs + same capacity
- Or: More capacity (higher r/alpha) + same epochs

**Learning but slow?**
- Try: More epochs
- Don't: Increase capacity (may overfit)

**Overfit quickly?**
- Try: Dropout, weight decay
- Don't: More epochs or capacity

---

## Expected Timelines

| Action | Time | Output |
|--------|------|--------|
| Add 2 epochs | ~6-7 hours | Better convergence |
| Fine-tune (2 epochs, low LR) | ~6-7 hours | Polish + refinement |
| Restart from scratch | ~10-12 hours | Fresh start |
| Quick test (100 steps) | ~7 minutes | Sanity check |

---

## Quick Checklist

Before continuing training:

- [ ] Evaluated current model quality
- [ ] Checked loss curve (still decreasing?)
- [ ] Backed up current checkpoints
- [ ] Decided on learning rate
- [ ] Set resume_from_checkpoint correctly
- [ ] Changed output_dir (don't overwrite!)
- [ ] Know how many more epochs needed

---

## Example: Complete Extended Training

```bash
# 1. Check current results
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264

# 2. Decide: Need 2 more epochs at same LR

# 3. Edit config
vim config/training_config.yaml
# Set:
#   num_train_epochs: 2
#   resume_from_checkpoint: ./outputs/checkpoint-3264
#   output_dir: ./outputs_epoch5

# 4. Continue training
make train

# 5. After ~6-7 hours, check again
python scripts/evaluate.py --checkpoint ./outputs_epoch5/checkpoint-2176

# 6. If better: Use new model
# 7. If not better enough: Fine-tune with lower LR
```

---

## TL;DR

**Quick continue training:**
1. Edit config: `num_train_epochs: 5` (was 3)
2. Add: `resume_from_checkpoint: ./outputs/checkpoint-3264`
3. Run: `make train`
4. Wait ~6 hours
5. Check if better!

**For fine-tuning:**
- Same as above but also set `learning_rate: 0.00003` (lower)

**For fresh start:**
- Remove `resume_from_checkpoint`
- Increase `r: 128` and `lora_alpha: 256`
- Run `make train`
