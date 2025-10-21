# 🚀 Getting Started

## 5-Minute Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add data
cp your_data.jsonl data/raw/

# 3. Validate
python scripts/validate_data.py

# 4. Train
cd scripts && ./launch_training.sh
```

## Data Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a UML expert."},
    {"role": "user", "content": "Create a class diagram..."},
    {"role": "assistant", "content": "@startuml..."}
  ]
}
```

## Configuration

Edit `config/training_config.yaml`:
- `selected_model`: Choose model
- `num_train_epochs`: Training duration
- `learning_rate`: Training speed

## Monitoring

```bash
make monitor        # GPU usage
make tensorboard    # Training metrics
```

## Testing

```bash
make test           # Run test cases
make interactive    # Chat with model
```

See README.md for more details.
