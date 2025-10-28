#!/bin/bash
#
set -e

docker stop unsloth_training
docker rm unsloth_training

# docker pull unsloth/unsloth
#
docker run -d -e JUPYTER_PORT=8000 \
  -e JUPYTER_PASSWORD="mypassword" \
  -e "SSH_KEY=$(cat ~/.ssh/unsloth_container_key.pub)" \
  -e USER_PASSWORD="unsloth2024" \
  --gpus '"device=0,1,2"' \
  -p 9000:8000 -p 2222:22 \
  -v $(pwd):/workspace/work \
  -v ${HOME}/.cache/huggingface:/workspace/.cache/huggingface \
  --name unsloth_latest \
  unsloth/unsloth


# ssh -i ~/.ssh/unsloth_container_key -p 2222 unsloth@localhost
