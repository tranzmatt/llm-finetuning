#!/usr/bin/env python3
"""
Model Merging and Export Script
"""

import argparse
import sys
from pathlib import Path
import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).parent.parent

def merge_to_16bit(model_path, output_path=None):
    """Merge to 16-bit."""
    
    if output_path is None:
        output_path = Path(model_path).parent / f"{Path(model_path).name}_merged_16bit"
    
    print(f"\n{'='*80}")
    print("MERGING TO 16-BIT")
    print(f"{'='*80}")
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
    
    print("Merging...")
    model.save_pretrained_merged(
        str(output_path),
        tokenizer,
        save_method="merged_16bit"
    )
    
    print(f"\n✓ Saved: {output_path}\n")
    return output_path

def convert_to_gguf(model_path, quant_methods, output_dir=None):
    """Convert to GGUF."""
    
    if output_dir is None:
        output_dir = Path(model_path).parent / f"{Path(model_path).name}_gguf"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CONVERTING TO GGUF")
    print(f"{'='*80}")
    print(f"Input: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Methods: {', '.join(quant_methods)}")
    print(f"{'='*80}\n")
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
    
    for method in quant_methods:
        print(f"\nConverting to {method}...")
        output_file = output_dir / method
        
        model.save_pretrained_gguf(
            str(output_file),
            tokenizer,
            quantization_method=method
        )
        
        print(f"✓ Saved: {output_file}")
    
    print(f"\n✓ All GGUF models saved to: {output_dir}\n")
    return output_dir

def push_to_hub(model_path, hub_path, token, save_method="merged_16bit"):
    """Push to HF Hub."""
    
    print(f"\n{'='*80}")
    print("PUSHING TO HF HUB")
    print(f"{'='*80}")
    print(f"Local: {model_path}")
    print(f"Hub: {hub_path}")
    print(f"{'='*80}\n")
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
    
    print("Pushing...")
    
    if save_method == "gguf":
        model.push_to_hub_gguf(hub_path=hub_path, tokenizer=tokenizer, token=token)
    else:
        model.push_to_hub_merged(hub_path=hub_path, tokenizer=tokenizer, save_method=save_method, token=token)
    
    print(f"\n✓ Pushed to: https://huggingface.co/{hub_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Merge and export models")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--merge_16bit', action='store_true')
    parser.add_argument('--merge_output', type=str)
    parser.add_argument('--to_gguf', type=str, nargs='+')
    parser.add_argument('--gguf_output', type=str)
    parser.add_argument('--push_to_hub', type=str)
    parser.add_argument('--hub_token', type=str)
    parser.add_argument('--save_method', type=str, default='merged_16bit')
    
    args = parser.parse_args()
    
    model_path = PROJECT_ROOT / args.model_path
    if not model_path.exists():
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model not found: {args.model_path}")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print("MODEL EXPORT")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"{'='*80}")
    
    merged_path = None
    if args.merge_16bit:
        merged_path = merge_to_16bit(str(model_path), args.merge_output)
    
    if args.to_gguf:
        source = merged_path if merged_path else model_path
        convert_to_gguf(str(source), args.to_gguf, args.gguf_output)
    
    if args.push_to_hub:
        if not args.hub_token:
            print("❌ --hub_token required")
            sys.exit(1)
        source = merged_path if merged_path else model_path
        push_to_hub(str(source), args.push_to_hub, args.hub_token, args.save_method)
    
    print(f"\n{'='*80}")
    print("✅ EXPORT COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
