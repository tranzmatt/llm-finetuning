#!/bin/bash
set -e

# Create output directory for logs
mkdir -p logs

docker run -d \
  --gpus '"device=0,1,2"' \
  -v $(pwd):/workspace/work \
  -v ${HOME}/.cache/huggingface:/workspace/.cache/huggingface \
  --name unsloth_training \
  unsloth/unsloth \
  bash -c "cd /workspace/work && torchrun --nproc_per_node=3 --master_port=29500 scripts/unsloth-cli-conversational.py \
    --model_name 'unsloth/gpt-oss-120b-unsloth-bnb-4bit' \
    --max_seq_length 4096 \
    --load_in_4bit \
    --dataset 'data/processed' \
    --r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --bias 'none' \
    --use_gradient_checkpointing 'unsloth' \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --warmup_steps 50 \
    --max_steps 3264 \
    --learning_rate 2e-5 \
    --optim 'adamw_8bit' \
    --weight_decay 0.01 \
    --lr_scheduler_type 'cosine' \
    --seed 3407 \
    --output_dir 'outputs' \
    --report_to 'none' \
    --logging_steps 10 \
    --save_model \
    --save_path 'outputs/final_model' \
    --save_method 'merged_16bit' 2>&1 | tee /workspace/work/logs/training.log"

echo "Training started in background. Monitor with:"
echo "  docker logs -f unsloth_training"
echo "Or check: logs/training.log"
