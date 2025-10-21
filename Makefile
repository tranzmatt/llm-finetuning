# Makefile for UML Fine-tuning Project

.PHONY: help setup install validate train test monitor clean

help:
	@echo "🦥 UML Fine-tuning Project - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install dependencies"
	@echo "  make env-check      - Check environment"
	@echo ""
	@echo "Data:"
	@echo "  make validate       - Validate training data"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Launch training"
	@echo "  make monitor        - Monitor GPU usage"
	@echo "  make tensorboard    - Start TensorBoard"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Test latest model"
	@echo "  make interactive    - Interactive chat"
	@echo ""
	@echo "Export:"
	@echo "  make merge          - Merge to 16-bit"
	@echo "  make gguf           - Convert to GGUF"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove temporary files"

install:
	pip install -r requirements.txt

env-check:
	@python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
	@python -c "import unsloth; print('Unsloth: OK')"

validate:
	python scripts/validate_data.py

train:
	cd scripts && ./launch_training.sh

monitor:
	watch -n 1 nvidia-smi

tensorboard:
	tensorboard --logdir logs/

test:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "No model found"; exit 1; fi; \
	python scripts/test_model.py --model_path "$$MODEL"

interactive:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "No model found"; exit 1; fi; \
	python scripts/test_model.py --model_path "$$MODEL" --mode interactive

merge:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "No model found"; exit 1; fi; \
	python scripts/merge_model.py --model_path "$$MODEL" --merge_16bit

gguf:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "No model found"; exit 1; fi; \
	python scripts/merge_model.py --model_path "$$MODEL" --to_gguf q8_0 q4_k_m

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
