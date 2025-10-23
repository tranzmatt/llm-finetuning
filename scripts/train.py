#!/usr/bin/env python3
"""
Main Training Script
Reads configuration from config files and trains the model
"""

import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import multiprocessing

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_yaml_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_all_configs(config_file='training_config.yaml'):
    """Load all configuration files."""
    config_dir = PROJECT_ROOT / "config"
    return {
        'base': load_yaml_config(config_dir / "base_config.yaml"),
        'models': load_yaml_config(config_dir / "model_configs.yaml"),
        'training': load_yaml_config(config_dir / config_file),
    }

def get_model_config(configs, model_name=None):
    """Get configuration for selected model."""
    if model_name is None:
        model_name = configs['training']['selected_model']
    
    if model_name not in configs['models']:
        available = ', '.join(configs['models'].keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    return configs['models'][model_name]

def load_jsonl_data(file_paths):
    """Load and combine multiple JSONL files."""
    all_data = []
    
    for file_path in file_paths:
        file_path = PROJECT_ROOT / file_path
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Loading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"  Error parsing line {line_num}: {e}")
                        continue
    
    print(f"✓ Loaded {len(all_data)} examples total")
    return all_data

def create_dataset(data, tokenizer):
    """Create HuggingFace dataset from JSONL data."""
    
    def format_chat(example):
        if "messages" not in example:
            return {"text": ""}
        
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": text}
        except Exception as e:
            print(f"Warning: Error formatting example: {e}")
            return {"text": ""}
    
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_chat, remove_columns=dataset.column_names, desc="Formatting")
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    
    print(f"✓ Dataset size: {len(dataset)}")
    return dataset

def split_dataset(dataset, val_split, seed):
    """Split dataset into train and validation."""
    if val_split > 0:
        split = dataset.train_test_split(test_size=val_split, seed=seed)
        return split['train'], split['test']
    return dataset, None

def setup_model(model_config, lora_config, local_rank=-1):
    """Load and configure model with LoRA."""
    
    print("\nLoading model...")
    print(f"  Model: {model_config['model_name']}")
    print(f"  Max sequence length: {model_config['max_seq_length']}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['model_name'],
        max_seq_length=model_config['max_seq_length'],
        dtype=getattr(torch, model_config.get('dtype', 'bfloat16')),
        load_in_4bit=model_config['load_in_4bit'],
        load_in_8bit=model_config.get('load_in_8bit', False),
        device_map={"": local_rank} if local_rank != -1 else "auto",
    )
    
    print("\nApplying LoRA...")
    print(f"  Rank: {lora_config['r']}")
    print(f"  Alpha: {lora_config['lora_alpha']}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        target_modules=lora_config['target_modules'],
        use_gradient_checkpointing=lora_config['use_gradient_checkpointing'],
        random_state=lora_config['random_state'],
        use_rslora=lora_config.get('use_rslora', False),
        max_seq_length=model_config['max_seq_length'],
    )
    
    return model, tokenizer

def create_training_args(config, paths, model_config):
    """Create training arguments."""
    
    train_cfg = config['training']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['selected_model']
    output_dir = PROJECT_ROOT / paths['output_dir'] / f"{model_name}_{timestamp}"

    # Convert numeric values that YAML might parse as strings
    learning_rate = float(train_cfg['learning_rate'])
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg['num_train_epochs'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=train_cfg.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        learning_rate=float(train_cfg['learning_rate']),
        warmup_steps=train_cfg['warmup_steps'],
        logging_steps=train_cfg['logging_steps'],
        save_steps=train_cfg['save_steps'],
        save_total_limit=train_cfg['save_total_limit'],
        eval_steps=train_cfg.get('eval_steps', 500),
        eval_strategy="steps" if train_cfg.get('eval_steps') else "no",
        fp16=train_cfg.get('fp16', False),
        bf16=train_cfg.get('bf16', True),
        optim=train_cfg['optim'],
        weight_decay=train_cfg['weight_decay'],
        lr_scheduler_type=train_cfg['lr_scheduler_type'],
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        seed=train_cfg['seed'],
        ddp_find_unused_parameters=False,
        group_by_length=train_cfg.get('group_by_length', True),
        dataset_num_proc=train_cfg.get('dataset_num_proc', multiprocessing.cpu_count() // 2),
        logging_dir=str(PROJECT_ROOT / paths['log_dir'] / f"{model_name}_{timestamp}"),
        report_to=train_cfg.get('report_to', 'tensorboard'),
        max_seq_length=model_config['max_seq_length'],
        packing=train_cfg.get('packing', False),
    )

def print_training_summary(config, train_dataset, eval_dataset, world_size, config_file):
    """Print training configuration summary."""
    
    train_cfg = config['training']
    lora_cfg = config['lora']
    model_name = config['selected_model']
    
    effective_batch = (
        train_cfg['per_device_train_batch_size'] *
        train_cfg['gradient_accumulation_steps'] *
        world_size
    )
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Config file: {config_file}")
    print(f"Model: {model_name}")
    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")
    print(f"Epochs: {train_cfg['num_train_epochs']}")
    print(f"Batch size per device: {train_cfg['per_device_train_batch_size']}")
    print(f"Gradient accumulation: {train_cfg['gradient_accumulation_steps']}")
    print(f"GPUs: {world_size}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Learning rate: {train_cfg['learning_rate']}")
    print(f"LoRA rank: {lora_cfg['r']}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Train model with config files")
    parser.add_argument('--config', type=str, default='training_config.yaml',
                        help='Config file name (in config/ directory)')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank <= 0:
        print("\n" + "="*80)
        print("UML DIAGRAM FINE-TUNING")
        print("="*80)
        print(f"Config: {args.config}")
        print(f"GPUs: {world_size}")
        print("="*80 + "\n")
    
    # Load configs with custom file
    configs = load_all_configs(args.config)
    
    # Apply command-line overrides
    if args.epochs:
        configs['training']['training']['num_train_epochs'] = args.epochs
    if args.batch_size:
        configs['training']['training']['per_device_train_batch_size'] = args.batch_size
    if args.lr:
        configs['training']['training']['learning_rate'] = args.lr
    
    model_config = get_model_config(configs)
    
    # Setup model
    model, tokenizer = setup_model(
        model_config,
        configs['training']['lora'],
        local_rank
    )
    
    # Load data
    if local_rank <= 0:
        print("\nLoading data...")
    
    data_files = configs['base']['data']['train_files']
    raw_data = load_jsonl_data(data_files)
    
    if len(raw_data) == 0:
        print("Error: No training data found!")
        sys.exit(1)
    
    dataset = create_dataset(raw_data, tokenizer)
    
    # Split
    val_split = configs['base']['data'].get('validation_split', 0.0)
    seed = configs['base']['data'].get('seed', 42)
    train_dataset, eval_dataset = split_dataset(dataset, val_split, seed)
    
    if local_rank <= 0:
        print("\n" + "="*80)
        print("SAMPLE")
        print("="*80)
        print(train_dataset[0]["text"][:500] + "...")
        print("="*80 + "\n")
    
    # Training args
    training_args = create_training_args(
        configs['training'],
        configs['base']['paths'],
        model_config
    )
    
    if local_rank <= 0:
        print_training_summary(
            configs['training'], 
            train_dataset, 
            eval_dataset, 
            world_size,
            args.config
        )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    
    if local_rank <= 0:
        print("Starting training...\n")
    
    # Train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save
    if local_rank <= 0:
        output_path = training_args.output_dir + "/final_model"
        print(f"\nSaving to {output_path}...")
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Save config
        config_save_path = Path(output_path) / "training_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(configs, f)
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        print(f"Model: {output_path}")
        print(f"Config: {args.config}")
        print("\nNext steps:")
        print(f"  python scripts/test_model.py --model_path {output_path}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
