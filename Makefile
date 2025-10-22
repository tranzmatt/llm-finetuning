# Makefile for UML Fine-tuning Project

.PHONY: help env install activate validate train monitor test clean clean-env

HOST_PYTHON = /usr/bin/python3
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

help:
	@echo "🦥 UML Fine-tuning Project - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make env            - Create venv and install dependencies"
	@echo "  make install        - Install dependencies (no venv)"
	@echo "  make activate       - Show activation command"
	@echo "  make env-check      - Check environment"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make extract-pdf    - Extract content from PDFs"
	@echo "  make format-data    - Format extracted data to JSONL"
	@echo "  make pipeline       - Run full pipeline (extract → format → validate)"
	@echo "  make validate       - Validate training data"
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
	@echo "  make clean-env      - Remove virtual environment"
	@echo ""

env:
	@echo "🔧 Creating virtual environment..."
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "⚠️  Virtual environment already exists at $(VENV_DIR)"; \
		echo "   Run 'make clean-env' first to recreate it"; \
		exit 1; \
	fi
	$(HOST_PYTHON) -m venv $(VENV_DIR)
	@echo "✓ Virtual environment created"
	@echo ""
	@echo "📦 Installing dependencies..."
	$(VENV_DIR)/bin/pip3 install --upgrade pip
	$(VENV_DIR)/bin/pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
	$(VENV_DIR)/bin/pip3 install unsloth torchao==.0.13.0
	$(VENV_DIR)/bin/pip3 install -r requirements.txt
	@echo ""
	@echo "✅ Environment ready!"
	@echo ""
	@echo "To activate the environment, run:"
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo ""
	@echo "Or use: make activate"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

activate:
	@echo "To activate the virtual environment, run:"
	@echo ""
	@echo "  source $(VENV_DIR)/bin/activate"
	@echo ""
	@echo "Or add to your shell:"
	@echo "  alias activate-uml='source $(PWD)/$(VENV_DIR)/bin/activate'"

env-check:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "🔍 Checking environment (venv)..."; \
		echo ""; \
		$(PYTHON) --version; \
		$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"; \
		$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"; \
		$(PYTHON) -c "import unsloth; print('Unsloth: OK')" 2>/dev/null || echo "Unsloth: Not installed"; \
	else \
		echo "🔍 Checking environment (system)..."; \
		echo ""; \
		python3 --version; \
		python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not installed"; \
		python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "CUDA: Not available"; \
		python3 -c "import unsloth; print('Unsloth: OK')" 2>/dev/null || echo "Unsloth: Not installed"; \
	fi

validate:
	@if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) scripts/validate_data.py; \
	else \
		python3 scripts/validate_data.py; \
	fi

train:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "⚠️  No virtual environment found"; \
		echo "   Run 'make env' first, or use system Python with 'cd scripts && ./launch_training.sh'"; \
		exit 1; \
	fi
	cd scripts && bash launch_training.sh

monitor:
	watch -n 1 nvidia-smi

tensorboard:
	@if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m tensorboard.main --logdir logs/; \
	else \
		tensorboard --logdir logs/; \
	fi

test:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "❌ No model found"; exit 1; fi; \
	echo "Testing: $$MODEL"; \
	if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) scripts/test_model.py --model_path "$$MODEL"; \
	else \
		python3 scripts/test_model.py --model_path "$$MODEL"; \
	fi

interactive:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "❌ No model found"; exit 1; fi; \
	echo "Using: $$MODEL"; \
	if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) scripts/test_model.py --model_path "$$MODEL" --mode interactive; \
	else \
		python3 scripts/test_model.py --model_path "$$MODEL" --mode interactive; \
	fi

merge:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "❌ No model found"; exit 1; fi; \
	echo "Merging: $$MODEL"; \
	if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) scripts/merge_model.py --model_path "$$MODEL" --merge_16bit; \
	else \
		python3 scripts/merge_model.py --model_path "$$MODEL" --merge_16bit; \
	fi

gguf:
	@MODEL=$$(ls -td models/*/final_model 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ]; then echo "❌ No model found"; exit 1; fi; \
	echo "Converting: $$MODEL"; \
	if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) scripts/merge_model.py --model_path "$$MODEL" --to_gguf q8_0 q4_k_m; \
	else \
		python3 scripts/merge_model.py --model_path "$$MODEL" --to_gguf q8_0 q4_k_m; \
	fi

# Add after the validate target

extract-pdf:
	@if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) scripts/extract_pdf.py; \
	else \
		python3 scripts/extract_pdf.py; \
	fi

format-data:
	@if [ -d "$(VENV_DIR)" ]; then \
		$(PYTHON) scripts/format_data.py; \
	else \
		python3 scripts/format_data.py; \
	fi

pipeline: extract-pdf format-data validate
	@echo ""
	@echo "✅ Data pipeline complete!"
	@echo "   1. PDFs extracted to data/extracted/"
	@echo "   2. Data formatted to data/processed/"
	@echo "   3. Data validated"

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	@echo
