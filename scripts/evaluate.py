#!/usr/bin/env python3
"""
Evaluate Trained UML Model
Tests the model on sample prompts and checks quality
"""

import sys
import yaml
from pathlib import Path
import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    
    # Load configs
    with open(PROJECT_ROOT / "config" / "training_config.yaml") as f:
        training_config = yaml.safe_load(f)
    
    with open(PROJECT_ROOT / "config" / "model_configs.yaml") as f:
        model_configs = yaml.safe_load(f)
    
    selected_model = training_config['selected_model']
    model_config = model_configs[selected_model]
    
    print(f"\nLoading model from: {checkpoint_path}")
    print(f"Base model: {model_config['model_name']}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=model_config['max_seq_length'],
        dtype=model_config.get('dtype'),
        load_in_4bit=model_config.get('load_in_4bit', False),
    )
    
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded")
    return model, tokenizer

def generate_uml(model, tokenizer, prompt, max_length=512):
    """Generate UML diagram from prompt."""
    
    # Format as conversational
    formatted = f"""<|start|>system<|message|>You are an expert at creating UML diagrams.
<|start|>user<|message|>{prompt}
<|start|>assistant<|message|>"""
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    if "<|start|>assistant<|message|>" in generated:
        response = generated.split("<|start|>assistant<|message|>")[-1]
        response = response.split("<|end|>")[0].strip()
        return response
    
    return generated

def check_plantuml_syntax(uml_code):
    """Basic syntax validation for PlantUML."""
    
    issues = []
    
    # Check for start/end
    if "@startuml" not in uml_code:
        issues.append("Missing @startuml")
    if "@enduml" not in uml_code:
        issues.append("Missing @enduml")
    
    # Check for basic structure
    lines = uml_code.split('\n')
    non_empty = [l.strip() for l in lines if l.strip() and not l.strip().startswith("'")]
    
    if len(non_empty) < 3:
        issues.append("Too short - likely incomplete")
    
    # Check for common syntax patterns
    has_content = any([
        'class ' in uml_code,
        'interface ' in uml_code,
        'participant ' in uml_code,
        '->' in uml_code,
        '-->' in uml_code,
    ])
    
    if not has_content:
        issues.append("No recognizable UML elements")
    
    return issues

def evaluate_model(checkpoint_path, test_prompts=None):
    """Evaluate model on test prompts."""
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Load model
    model, tokenizer = load_model(checkpoint_path)
    
    # Default test prompts
    if test_prompts is None:
        test_prompts = [
            "Create a class diagram for an e-commerce system with User, Product, and Order classes",
            "Design a sequence diagram for a user login process",
            "Generate a class diagram for a simple blog system with Posts, Comments, and Users",
            "Create a use case diagram for an ATM machine",
            "Design a class diagram for a library management system",
        ]
    
    print(f"\nTesting on {len(test_prompts)} prompts...")
    print("="*80)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt[:70]}...")
        print("-"*80)
        
        # Generate
        try:
            output = generate_uml(model, tokenizer, prompt)
            
            # Check syntax
            issues = check_plantuml_syntax(output)
            
            result = {
                'prompt': prompt,
                'output': output,
                'issues': issues,
                'valid': len(issues) == 0
            }
            
            results.append(result)
            
            # Print output
            print("\nGenerated UML:")
            print(output[:500])
            if len(output) > 500:
                print(f"... ({len(output)} chars total)")
            
            # Print validation
            if issues:
                print(f"\n⚠️  Issues found: {', '.join(issues)}")
            else:
                print("\n✓ Syntax looks good!")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'prompt': prompt,
                'output': None,
                'issues': [str(e)],
                'valid': False
            })
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    valid_count = sum(1 for r in results if r['valid'])
    total = len(results)
    
    print(f"\nResults: {valid_count}/{total} generated valid UML")
    print(f"Success rate: {valid_count/total*100:.1f}%")
    
    if valid_count < total:
        print(f"\nCommon issues:")
        all_issues = []
        for r in results:
            all_issues.extend(r['issues'])
        
        from collections import Counter
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common(5):
            print(f"  - {issue}: {count} times")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if valid_count == total:
        print("\n✅ Excellent! Model generates valid UML consistently.")
        print("   Consider testing on more complex prompts.")
    elif valid_count >= total * 0.8:
        print("\n✓ Good! Most outputs are valid.")
        print("   Minor improvements possible with more training.")
    elif valid_count >= total * 0.5:
        print("\n⚠️  Acceptable but could be better.")
        print("   Consider:")
        print("   - Training 2 more epochs")
        print("   - Lower learning rate for fine-tuning")
    else:
        print("\n❌ Model needs more training.")
        print("   Recommendations:")
        print("   - Train 2-3 more epochs")
        print("   - Check if loss is still decreasing")
        print("   - Consider higher learning rate or more capacity")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained UML model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--prompts', type=str, nargs='+',
                       help='Custom test prompts')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Run evaluation
    results = evaluate_model(checkpoint_path, args.prompts)
    
    # Save results
    output_file = PROJECT_ROOT / "evaluation_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"Test {i}\n")
            f.write(f"{'='*80}\n")
            f.write(f"\nPrompt: {r['prompt']}\n\n")
            f.write(f"Output:\n{r['output']}\n\n")
            if r['issues']:
                f.write(f"Issues: {', '.join(r['issues'])}\n")
            else:
                f.write("Status: ✓ Valid\n")
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
