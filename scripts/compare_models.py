#!/usr/bin/env python3
"""
Compare Two Model Checkpoints
Compare outputs from different training runs
"""

import sys
import yaml
from pathlib import Path
import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def load_model(checkpoint_path, label):
    """Load model from checkpoint."""
    
    with open(PROJECT_ROOT / "config" / "training_config.yaml") as f:
        training_config = yaml.safe_load(f)
    
    with open(PROJECT_ROOT / "config" / "model_configs.yaml") as f:
        model_configs = yaml.safe_load(f)
    
    selected_model = training_config['selected_model']
    model_config = model_configs[selected_model]
    
    print(f"\nLoading {label}: {checkpoint_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=model_config['max_seq_length'],
        dtype=model_config.get('dtype'),
        load_in_4bit=model_config.get('load_in_4bit', False),
    )
    
    FastLanguageModel.for_inference(model)
    print(f"✓ {label} loaded")
    
    return model, tokenizer

def generate_uml(model, tokenizer, prompt):
    """Generate UML from prompt."""
    
    formatted = f"""<|start|>system<|message|>You are an expert at creating UML diagrams.
<|start|>user<|message|>{prompt}
<|start|>assistant<|message|>"""
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|start|>assistant<|message|>" in generated:
        response = generated.split("<|start|>assistant<|message|>")[-1]
        response = response.split("<|end|>")[0].strip()
        return response
    
    return generated

def score_output(output):
    """Simple scoring of UML output."""
    score = 0
    
    if "@startuml" in output:
        score += 1
    if "@enduml" in output:
        score += 1
    
    if "class " in output or "interface " in output:
        score += 1
    
    if "->" in output or "-->" in output or "--" in output:
        score += 1
    
    lines = [l.strip() for l in output.split('\n') if l.strip()]
    if len(lines) >= 5:
        score += 1
    
    return score

def compare_models(checkpoint1, checkpoint2, test_prompts=None):
    """Compare two model checkpoints."""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Load both models
    model1, tokenizer1 = load_model(checkpoint1, "Model 1")
    model2, tokenizer2 = load_model(checkpoint2, "Model 2")
    
    # Test prompts
    if test_prompts is None:
        test_prompts = [
            "Create a class diagram for an online shopping cart",
            "Design a sequence diagram for password reset",
            "Generate a class diagram for a restaurant reservation system",
        ]
    
    print(f"\n\nComparing on {len(test_prompts)} prompts...")
    print("="*80)
    
    scores1 = []
    scores2 = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST {i}/{len(test_prompts)}")
        print("="*80)
        print(f"\nPrompt: {prompt}")
        
        # Generate from both
        print("\nGenerating from Model 1...")
        output1 = generate_uml(model1, tokenizer1, prompt)
        score1 = score_output(output1)
        scores1.append(score1)
        
        print("Generating from Model 2...")
        output2 = generate_uml(model2, tokenizer2, prompt)
        score2 = score_output(output2)
        scores2.append(score2)
        
        # Show outputs
        print("\n" + "-"*80)
        print("MODEL 1 OUTPUT:")
        print("-"*80)
        print(output1)
        print(f"\nScore: {score1}/5")
        
        print("\n" + "-"*80)
        print("MODEL 2 OUTPUT:")
        print("-"*80)
        print(output2)
        print(f"\nScore: {score2}/5")
        
        # Winner
        print("\n" + "-"*80)
        if score1 > score2:
            print("Winner: Model 1 ✓")
        elif score2 > score1:
            print("Winner: Model 2 ✓")
        else:
            print("Winner: Tie")
    
    # Summary
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    avg1 = sum(scores1) / len(scores1)
    avg2 = sum(scores2) / len(scores2)
    
    print(f"\nModel 1 ({checkpoint1}):")
    print(f"  Average score: {avg1:.2f}/5")
    print(f"  Individual: {scores1}")
    
    print(f"\nModel 2 ({checkpoint2}):")
    print(f"  Average score: {avg2:.2f}/5")
    print(f"  Individual: {scores2}")
    
    print("\n" + "-"*80)
    if avg1 > avg2:
        print(f"✓ Model 1 is better (by {avg1-avg2:.2f} points)")
        print("\nRecommendation: Use Model 1")
    elif avg2 > avg1:
        print(f"✓ Model 2 is better (by {avg2-avg1:.2f} points)")
        print("\nRecommendation: Use Model 2")
    else:
        print("Models perform equally")
        print("\nRecommendation: Either model is fine")
    
    print("\n" + "="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare two model checkpoints')
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first checkpoint')
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second checkpoint')
    parser.add_argument('--prompts', type=str, nargs='+',
                       help='Custom test prompts')
    
    args = parser.parse_args()
    
    # Validate
    model1_path = Path(args.model1)
    model2_path = Path(args.model2)
    
    if not model1_path.exists():
        print(f"❌ Model 1 not found: {model1_path}")
        sys.exit(1)
    
    if not model2_path.exists():
        print(f"❌ Model 2 not found: {model2_path}")
        sys.exit(1)
    
    # Compare
    compare_models(model1_path, model2_path, args.prompts)

if __name__ == "__main__":
    main()
