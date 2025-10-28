#!/usr/bin/env python3

"""
🦥 Starter Script for Fine-Tuning FastLanguageModel with Unsloth
Modified for conversational JSONL format

This script is adapted from the official unsloth-cli.py to work with
conversational format data (single "text" field) instead of Alpaca format.

Usage for UML training:
    python unsloth-cli-modified.py \
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
        --save_model \
        --save_path "outputs/final_model" \
        --save_method "merged_16bit"

To see a full list of configurable options, use:
    python unsloth-cli-modified.py --help

Happy fine-tuning!
"""

import argparse
import os


def run(args):
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers.utils import strtobool
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    import logging
    from glob import glob
    logging.getLogger('hf-to-gguf').setLevel(logging.WARNING)

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    # Load dataset - MODIFIED FOR CONVERSATIONAL FORMAT
    use_modelscope = strtobool(os.environ.get('UNSLOTH_USE_MODELSCOPE', 'False'))
    if use_modelscope:
        from modelscope import MsDataset
        dataset = MsDataset.load(args.dataset, split="train")
    else:
        # Check if dataset is a local path
        if os.path.exists(args.dataset):
            # Load local JSONL files
            print(f"Loading local dataset from: {args.dataset}")
            data_files = glob(os.path.join(args.dataset, "*_conversational.jsonl"))
            if not data_files:
                # Try loading all JSONL files
                data_files = glob(os.path.join(args.dataset, "*.jsonl"))
            if not data_files:
                raise ValueError(f"No JSONL files found in {args.dataset}")
            print(f"Found {len(data_files)} data files:")
            for f in data_files:
                print(f"  - {f}")
            dataset = load_dataset('json', data_files=data_files, split='train')
        else:
            # Load from HuggingFace hub
            dataset = load_dataset(args.dataset, split="train")
    
    # Data is already in conversational format with "text" field
    # No formatting needed!
    print(f"✓ Loaded {len(dataset)} examples")
    print("Data is ready for training!")

    # Configure training arguments
    training_args = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        max_length=args.max_seq_length,
        dataset_text_field="text",  # Important: tell SFTTrainer which field to use
        dataset_num_proc=2,
        packing=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Train model
    trainer_stats = trainer.train()

    # Save model
    if args.save_model:
        # if args.quantization_method is a list, we will save the model for each quantization method
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(f"Saving model with quantization method: {quantization_method}")
                    model.save_pretrained_gguf(
                        args.save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=args.hub_path,
                            hub_token=args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                print(f"Saving model with quantization method: {args.quantization}")
                model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=quantization_method,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        print("Warning: The model is not saved!")


if __name__ == "__main__":

    # Define argument parser
    parser = argparse.ArgumentParser(description="🦥 Fine-tune your llm faster using unsloth!")

    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument('--model_name', type=str, default="unsloth/llama-3-8b", help="Model name to load")
    model_group.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action='store_true', help="Use 4bit quantization to reduce memory usage")
    model_group.add_argument('--dataset', type=str, default="data/processed", help="Local path to JSONL files")

    lora_group = parser.add_argument_group("🧠 LoRA Options", "These options are used to configure the LoRA model.")
    lora_group.add_argument('--r', type=int, default=16, help="Rank for Lora model, default is 16.  (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_dropout', type=float, default=0.0, help="LoRA dropout rate, default is 0.0 which is optimized.")
    lora_group.add_argument('--bias', type=str, default="none", help="Bias setting for LoRA")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Use gradient checkpointing")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Random state for reproducibility, default is 3407.")
    lora_group.add_argument('--use_rslora', action='store_true', help="Use rank stabilized LoRA")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="Configuration for LoftQ")

   
    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training, default is 2.")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps, default is 4.")
    training_group.add_argument('--warmup_steps', type=int, default=5, help="Number of warmup steps, default is 5.")
    training_group.add_argument('--max_steps', type=int, default=400, help="Maximum number of training steps.")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate, default is 2e-4.")
    training_group.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizer type.")
    training_group.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay, default is 0.01.")
    training_group.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler type, default is 'linear'.")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility, default is 3407.")
    

    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="tensorboard",
        choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"],
        help="The list of integrations to report the results and logs to. Supported platforms are: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations installed, 'none' for no integrations.")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps, default is 1")

    # Saving and pushing arguments
    save_group = parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    save_group.add_argument('--save_model', action='store_true', help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model, default is 'merged_16bit'")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="model", help="Path to save the model")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+",
        help="Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf ")

    push_group = parser.add_argument_group('🚀 Push Model Options')
    push_group.add_argument('--push_model', action='store_true', help="Push the model to Hugging Face hub after training")
    push_group.add_argument('--push_gguf', action='store_true', help="Push the model as GGUF to Hugging Face hub after training")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Path on Hugging Face hub to push the model")
    push_group.add_argument('--hub_token', type=str, help="Token for pushing the model to Hugging Face hub")

    args = parser.parse_args()
    run(args)
