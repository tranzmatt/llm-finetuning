#!/usr/bin/env python3
"""
Model Testing and Inference Script
"""

import sys
import yaml
import torch
import argparse
from pathlib import Path
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_TESTS = [
    {
        "system": "You are a helpful assistant specialized in UML diagrams.",
        "user": "Create a simple class diagram for a Library system with Book and Member classes.",
    },
    {
        "system": "You are a helpful assistant specialized in PlantUML.",
        "user": "Generate a sequence diagram showing a user logging into a system.",
    },
    {
        "system": "You are a helpful assistant specialized in UML diagrams.",
        "user": "Create a use case diagram for an online shopping system.",
    },
]

def load_generation_config():
    """Load generation config."""
    try:
        config_path = PROJECT_ROOT / "config" / "training_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('generation', {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
        })
    except:
        return {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
        }

def load_model(model_path, load_in_4bit=False):
    """Load model for inference."""
    
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    print(f"Path: {model_path}")
    print(f"4-bit: {load_in_4bit}")
    print(f"{'='*80}\n")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )
    
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded!\n")
    return model, tokenizer

def generate_response(model, tokenizer, messages, **gen_kwargs):
    """Generate response."""
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response

def run_tests(model, tokenizer, test_cases, gen_config):
    """Run test cases."""
    
    print(f"\n{'='*80}")
    print(f"RUNNING {len(test_cases)} TEST CASES")
    print(f"{'='*80}\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'─'*80}")
        
        messages = [
            {"role": "system", "content": test["system"]},
            {"role": "user", "content": test["user"]},
        ]
        
        print(f"\n💬 User: {test['user']}")
        print(f"\n⏳ Generating...")
        
        response = generate_response(model, tokenizer, messages, **gen_config)
        
        print(f"\n🤖 Assistant:")
        print(f"{'─'*80}")
        print(response)
        print(f"{'─'*80}")
    
    print(f"\n{'='*80}")
    print("TESTS COMPLETE")
    print(f"{'='*80}\n")

def interactive_mode(model, tokenizer, gen_config):
    """Interactive chat."""
    
    print(f"\n{'='*80}")
    print("INTERACTIVE MODE")
    print(f"{'='*80}")
    print("Commands: 'reset', 'quit'/'exit'")
    print(f"{'='*80}\n")
    
    conversation = []
    system = "You are a helpful assistant specialized in UML and PlantUML diagrams."
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                conversation = []
                print("🔄 Reset")
                continue
            
            if not user_input:
                continue
            
            messages = [{"role": "system", "content": system}]
            messages.extend(conversation)
            messages.append({"role": "user", "content": user_input})
            
            print("\n⏳ Generating...")
            response = generate_response(model, tokenizer, messages, **gen_config)
            
            print(f"\n🤖 Assistant: {response}")
            
            conversation.append({"role": "user", "content": user_input})
            conversation.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

def compare_models(base_model_name, ft_model_path, test_cases, gen_config):
    """Compare base and fine-tuned."""
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}\n")
    
    print("Loading base model...")
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)
    
    print("Loading fine-tuned model...")
    ft_model, ft_tok = load_model(ft_model_path, load_in_4bit=True)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test['user'][:60]}...")
        print(f"{'='*80}")
        
        messages = [
            {"role": "system", "content": test["system"]},
            {"role": "user", "content": test["user"]},
        ]
        
        print(f"\n📦 BASE:")
        print(f"{'─'*80}")
        base_resp = generate_response(base_model, base_tok, messages, max_new_tokens=300, temperature=0.7)
        print(base_resp)
        
        print(f"\n✨ FINE-TUNED:")
        print(f"{'─'*80}")
        ft_resp = generate_response(ft_model, ft_tok, messages, max_new_tokens=300, temperature=0.7)
        print(ft_resp)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'interactive', 'compare'])
    parser.add_argument('--base_model', type=str, default='unsloth/gpt-oss-120b-unsloth-bnb-4bit')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--max_new_tokens', type=int)
    
    args = parser.parse_args()
    
    gen_config = load_generation_config()
    
    if args.temperature:
        gen_config['temperature'] = args.temperature
    if args.max_new_tokens:
        gen_config['max_new_tokens'] = args.max_new_tokens
    
    model_path = PROJECT_ROOT / args.model_path
    if not model_path.exists():
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model not found: {args.model_path}")
            sys.exit(1)
    
    if args.mode == 'test':
        model, tokenizer = load_model(str(model_path), args.load_in_4bit)
        run_tests(model, tokenizer, DEFAULT_TESTS, gen_config)
        
    elif args.mode == 'interactive':
        model, tokenizer = load_model(str(model_path), args.load_in_4bit)
        interactive_mode(model, tokenizer, gen_config)
        
    elif args.mode == 'compare':
        compare_models(args.base_model, str(model_path), DEFAULT_TESTS, gen_config)

if __name__ == "__main__":
    main()
