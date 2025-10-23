# Model Evaluation and Testing Guide

Now you have evaluation scripts! Here's how to use them:

## Scripts Available

1. **test_model.py** - Quick test (3 prompts)
2. **evaluate.py** - Full evaluation with scoring
3. **compare_models.py** - Compare two checkpoints

## Installation

First, copy the scripts to your project:

```bash
cd ~/Code/UML/llm-finetuning
cp test_model.py scripts/
cp evaluate.py scripts/
cp compare_models.py scripts/
chmod +x scripts/*.py
```

## 1. Quick Test (Fastest)

**When to use:** Just want to see if model works

```bash
# Test your final checkpoint
python scripts/test_model.py --checkpoint ./outputs/checkpoint-3264

# Or the final model
python scripts/test_model.py --checkpoint ./outputs/final_model
```

**Output:**
- Generates 3 sample UML diagrams
- Quick validation
- Takes ~2 minutes

**What you'll see:**
```
TEST 1/3
==================
Prompt: Create a class diagram for a simple banking system

Generated UML:
@startuml
class Account {
  - accountNumber: String
  - balance: Double
  ...
}
@enduml

✓ Has valid PlantUML markers
✓ Contains UML elements
```

---

## 2. Full Evaluation (Recommended)

**When to use:** Systematic quality assessment

```bash
# Evaluate final checkpoint
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264

# Evaluate with custom prompts
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264 \
  --prompts "Create a class diagram for a hotel booking system" \
           "Design sequence diagram for checkout process"
```

**Output:**
- Tests on 5 default prompts (or your custom ones)
- Validates syntax
- Provides quality score
- Gives recommendations
- Saves results to `evaluation_results.txt`

**Takes:** ~5-10 minutes

**What you'll see:**
```
EVALUATION SUMMARY
==================
Results: 4/5 generated valid UML
Success rate: 80.0%

RECOMMENDATIONS
==================
✓ Good! Most outputs are valid.
  Minor improvements possible with more training.
```

---

## 3. Compare Models (When Continuing Training)

**When to use:** Compare original vs extended training

```bash
# Compare epoch 3 vs epoch 5
python scripts/compare_models.py \
  --model1 ./outputs/checkpoint-3264 \
  --model2 ./outputs_extended/checkpoint-5440
```

**Output:**
- Side-by-side comparison
- Scores each model
- Tells you which is better
- Shows both outputs

**Takes:** ~10-15 minutes

**What you'll see:**
```
COMPARISON SUMMARY
==================
Model 1: Average score: 3.8/5
Model 2: Average score: 4.2/5

✓ Model 2 is better (by 0.4 points)
Recommendation: Use Model 2
```

---

## Typical Workflow

### After Initial Training (3 epochs)

```bash
# 1. Quick test
python scripts/test_model.py --checkpoint ./outputs/checkpoint-3264

# 2. If looks good, full evaluation
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264

# 3. Check results in evaluation_results.txt
cat evaluation_results.txt
```

**Decision:**
- Success rate > 80%? ✅ Good enough!
- Success rate 50-80%? ⚠️ Consider 2 more epochs
- Success rate < 50%? ❌ Definitely need more training

### After Extended Training (5 epochs)

```bash
# Compare original (3 epochs) vs extended (5 epochs)
python scripts/compare_models.py \
  --model1 ./outputs/checkpoint-3264 \
  --model2 ./outputs_extended/checkpoint-5440

# Use the better one!
```

---

## Understanding the Scores

### evaluate.py Scoring

**Valid UML = all of:**
- ✓ Has `@startuml`
- ✓ Has `@enduml`
- ✓ Contains UML elements (classes, relationships, etc.)
- ✓ Not too short (> 3 lines)

**Success Rate:**
- 100% = Perfect! 🎉
- 80-99% = Very good ✓
- 50-79% = Acceptable, could improve ⚠️
- < 50% = Needs more training ❌

### compare_models.py Scoring

Each output scored on:
1. Has `@startuml` (1 point)
2. Has `@enduml` (1 point)
3. Has classes/interfaces (1 point)
4. Has relationships (1 point)
5. Sufficient length (1 point)

**Max score: 5/5**

---

## Custom Test Prompts

Create your own test set:

```bash
# Create test_prompts.txt
cat > test_prompts.txt << 'EOF'
Create a class diagram for a social media platform
Design a sequence diagram for posting a photo
Generate an activity diagram for user registration
Create a use case diagram for a mobile banking app
EOF

# Use with evaluate.py
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264 \
  --prompts $(cat test_prompts.txt)
```

---

## Interpreting Results

### Good Signs ✓
```
✓ Has valid PlantUML markers
✓ Contains UML elements
✓ Logical structure
✓ Relevant entity names
✓ Proper relationships
✓ Complete diagram
```

### Warning Signs ⚠️
```
⚠️ Missing markers
⚠️ Incomplete (too short)
⚠️ Syntax errors
⚠️ Generic/template-like
⚠️ Inconsistent style
```

### Bad Signs ❌
```
❌ No UML syntax at all
❌ Hallucinated nonsense
❌ Just repeating prompt
❌ Empty output
❌ Crashes/errors
```

---

## What to Do Based on Results

### Scenario 1: Success Rate 90-100%
**Action:** Model is great! Use it.

```bash
# Save the final model
cp -r ./outputs/checkpoint-3264 ./final_model
```

### Scenario 2: Success Rate 70-89%
**Action:** Good but could be better. Try 2 more epochs.

```yaml
# config/training_config.yaml
training:
  num_train_epochs: 5  # Was 3
  resume_from_checkpoint: ./outputs/checkpoint-3264
  output_dir: ./outputs_extended
```

Then:
```bash
make train
# Wait 6-7 hours
python scripts/compare_models.py \
  --model1 ./outputs/checkpoint-3264 \
  --model2 ./outputs_extended/checkpoint-5440
```

### Scenario 3: Success Rate 50-69%
**Action:** Needs improvement. Fine-tune with lower LR.

```yaml
training:
  num_train_epochs: 2
  learning_rate: 0.00003  # Lower LR
  resume_from_checkpoint: ./outputs/checkpoint-3264
```

### Scenario 4: Success Rate < 50%
**Action:** Model didn't learn well. Need more training or restart.

**Option A:** Add many more epochs
```yaml
num_train_epochs: 7  # 4 more epochs
resume_from_checkpoint: ./outputs/checkpoint-3264
```

**Option B:** Restart with more capacity
```yaml
num_train_epochs: 3
# Remove resume_from_checkpoint
lora:
  r: 128  # Was 64
  lora_alpha: 256
```

---

## Saving Results

All scripts can save output:

```bash
# Save test output
python scripts/test_model.py --checkpoint ./outputs/checkpoint-3264 \
  > test_output.txt

# Save evaluation
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264
# Automatically saves to evaluation_results.txt

# Save comparison
python scripts/compare_models.py --model1 ... --model2 ... \
  > comparison_results.txt
```

---

## Troubleshooting

### Script fails with import error
```bash
# Make sure you're in the venv
source venv/bin/activate

# Install if needed
pip install unsloth transformers
```

### Model not found
```bash
# Check checkpoint exists
ls -la ./outputs/checkpoint-3264/

# Should see:
# - config.json
# - adapter_config.json
# - adapter_model.bin
```

### GPU out of memory
```bash
# Only load one model at a time
# Don't run test + evaluate + compare simultaneously

# Or set smaller max_seq_length temporarily
```

---

## Quick Reference

```bash
# Quick test (2 min)
python scripts/test_model.py --checkpoint <path>

# Full eval (10 min)
python scripts/evaluate.py --checkpoint <path>

# Compare (15 min)
python scripts/compare_models.py --model1 <path1> --model2 <path2>

# View results
cat evaluation_results.txt
```

---

## Example Complete Workflow

```bash
# 1. Training finishes
# outputs/checkpoint-3264/ created

# 2. Quick test
python scripts/test_model.py --checkpoint ./outputs/checkpoint-3264
# Looks promising!

# 3. Full evaluation
python scripts/evaluate.py --checkpoint ./outputs/checkpoint-3264
# Success rate: 75% - could be better

# 4. Train 2 more epochs
vim config/training_config.yaml
# Set num_train_epochs: 5, resume_from_checkpoint, output_dir
make train
# Wait ~6 hours...

# 5. Compare
python scripts/compare_models.py \
  --model1 ./outputs/checkpoint-3264 \
  --model2 ./outputs_extended/checkpoint-5440
# Model 2 is better by 0.6 points!

# 6. Use the better model
cp -r ./outputs_extended/checkpoint-5440 ./production_model
```

Done! 🎉

---

## Summary

**Three scripts, three use cases:**

1. **test_model.py** → Quick check if it works
2. **evaluate.py** → Systematic quality assessment  
3. **compare_models.py** → A/B testing after more training

**Use them to:**
- Decide if training is complete
- Compare before/after extended training
- Make informed decisions about continuing

**They tell you:**
- ✅ Model is good, use it
- ⚠️ Model is OK, could improve with more training
- ❌ Model needs significant more training
