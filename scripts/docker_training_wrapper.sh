#!/bin/bash
# train.sh - Run UML fine-tuning with conversational data

cd /workspace/work

echo "========================================================================"
echo "UML Fine-tuning with Unsloth (Conversational Format)"
echo "========================================================================"
echo "Start time: $(date)"
echo "Script: scripts/unsloth-cli-conversational.py"
echo "Data format: Conversational JSONL"
echo ""

# Check GPUs
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
echo ""

# Check data files
echo "Data files:"
ls -lh data/processed/*_conversational.jsonl
echo ""

echo "Starting training..."
echo "========================================================================"

torchrun \
  --nproc_per_node=3 \
  --master_port=29500 \
  scripts/unsloth-cli-conversational.py \
    --model_name "unsloth/gpt-oss-120b-unsloth-bnb-4bit" \
    --max_seq_length 4096 \
    --load_in_4bit \
    --dataset "data/processed" \
    --r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --bias "none" \
    --use_gradient_checkpointing "unsloth" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --warmup_steps 50 \
    --max_steps 3264 \
    --learning_rate 2e-5 \
    --optim "adamw_8bit" \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --seed 3407 \
    --output_dir "outputs" \
    --report_to "none" \
    --logging_steps 10 \
    --save_model \
    --save_path "outputs/final_model" \
    --save_method "merged_16bit"

echo ""
echo "========================================================================"
echo "Training complete: $(date)"
echo "Model saved to: outputs/final_model"
echo "========================================================================"
