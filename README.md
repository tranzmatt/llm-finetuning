# 🦥 UML Diagram Fine-tuning on DGX1

Fine-tuning large language models for UML and PlantUML diagram generation.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data in data/raw/
cp your_data.jsonl data/raw/

# 3. Validate data
python scripts/validate_data.py

# 4. Configure (edit config/training_config.yaml)
# Choose model, set hyperparameters

# 5. Start training
cd scripts && ./launch_training.sh

# 6. Monitor
make tensorboard  # or: tensorboard --logdir logs/

# 7. Test model
python scripts/test_model.py --model_path models/[your-model]/final_model
```

## Available Models

- **gpt-oss-120b-4bit**: 120B parameters, 4K context
- **llama-3.3-70b**: 70B parameters, 8K context  
- **mistral-large**: 123B parameters, 8K context

## Commands

```bash
make help           # Show all commands
make install        # Install dependencies
make validate       # Validate data
make train          # Start training
make monitor        # Monitor GPUs
make test           # Test model
make merge          # Export to 16-bit
```

## Documentation

- `GETTING_STARTED.md` - Quick start guide
- `COMPLETE_GUIDE.md` - Comprehensive documentation
- `PROJECT_SUMMARY.md` - Project overview

## Support

- Issues: Check logs/ directory
- Config: Edit config/*.yaml files
- Scripts: Use --help flag

Built with [Unsloth](https://unsloth.ai) 🦥
