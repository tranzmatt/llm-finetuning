#!/usr/bin/env python3
"""
Quick Test: Generate Sample UML Diagrams
Simple script to test your trained model
"""

import sys
import yaml
from pathlib import Path
import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def test_model(checkpoint_path):
    """Quick test of trained model."""
    
    print("\n" + "="*80)
    print("QUICK MODEL TEST")
    print("="*80)
    print(f"\nCheckpoint: {checkpoint_path}")
    
    # Load configs
    with open(PROJECT_ROOT / "config" / "training_config.yaml") as f:
        training_config = yaml.safe_load(f)
    
    with open(PROJECT_ROOT / "config" / "model_configs.yaml") as f:
        model_configs = yaml.safe_load(f)
    
    selected_model = training_config['selected_model']
    model_config = model_configs[selected_model]
    
    print(f"\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=model_config['max_seq_length'],
        dtype=model_config.get('dtype'),
        load_in_4bit=model_config.get('load_in_4bit', False),
    )
    
    FastLanguageModel.for_inference(model)
    print("✓ Model loaded\n")
    
    # Test prompts
    prompts = [
        "Create a class diagram for a simple banking system",
        "Design a sequence diagram showing user authentication",
        "Generate a class diagram for a pet store application",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print("="*80)
        print(f"TEST {i}/{len(prompts)}")
        print("="*80)
        print(f"\nPrompt: {prompt}")
        print("\n" + "-"*80)
        
        # Format
        formatted = f"""<|start|>system<|message|>You are an expert at creating UML diagrams.
<|start|>user<|message|>{prompt}
<|start|>assistant<|message|>"""
        
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        # Generate
        print("Generating...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract response
        if "<|start|>assistant<|message|>" in generated:
            response = generated.split("<|start|>assistant<|message|>")[-1]
            response = response.split("<|end|>")[0].strip()
        else:
            response = generated
        
        print("\nGenerated UML:\n")
        print(response)
        print("\n")
        
        # Quick validation
        if "@startuml" in response and "@enduml" in response:
            print("✓ Has valid PlantUML markers")
        else:
            print("⚠️  Missing PlantUML markers")
        
        if any(x in response for x in ['class ', 'participant ', '->', 'interface ']):
            print("✓ Contains UML elements")
        else:
            print("⚠️  No recognizable UML elements")
        
        print()
    
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nTo save outputs to a file, use:")
    print("  python test_model.py --checkpoint <path> > test_output.txt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test of trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory')
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    test_model(checkpoint_path)
