#!/bin/bash
set -e

# Clean up old container
docker rm -f unsloth_latest 2>/dev/null || true

# Start container with SSH
docker run -d \
  --gpus '"device=0,1,2"' \
  -e JUPYTER_PORT=8000 \
  -e JUPYTER_PASSWORD="mypassword" \
  -e "SSH_KEY=$(cat ~/.ssh/unsloth_container_key.pub)" \
  -e USER_PASSWORD="unsloth2024" \
  -p 9000:8000 -p 2222:22 \
  -v $(pwd):/workspace/work \
  -v ${HOME}/.cache/huggingface:/workspace/.cache/huggingface \
  --name unsloth_latest \
  unsloth/unsloth

echo "Waiting for container to start..."
sleep 5

# Create logs directory
mkdir -p logs

# Start training in background
docker exec -d unsloth_latest bash -c "cd /workspace/work && torchrun --nproc_per_node=3 --master_port=29500 scripts/unsloth-cli-conversational.py \
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
  --save_method 'merged_16bit' > /workspace/work/logs/training.log 2>&1"

echo ""
echo "========================================================================"
echo "Training Started!"
echo "========================================================================"
echo "Container: unsloth_latest"
echo "GPUs: 3 (devices 0, 1, 2)"
echo "Script: scripts/unsloth-cli-conversational.py"
echo "Data: data/processed/*_conversational.jsonl"
echo ""
echo "Monitor:"
echo "  docker logs -f unsloth_latest"
echo "  tail -f logs/training.log"
echo ""
echo "SSH into container:"
echo "  ssh -i ~/.ssh/unsloth_container_key -p 2222 unsloth@localhost"
echo ""
echo "Check GPU usage:"
echo "  watch -n 2 nvidia-smi"
echo "========================================================================"
