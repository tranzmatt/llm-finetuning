#!/usr/bin/env python3
"""
Data Formatting Script
Converts extracted content to JSONL training format
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent

def load_extracted_file(filepath: Path) -> Dict:
    """Load extracted JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_to_conversational(extracted_data: Dict) -> List[Dict]:
    """
    Convert extracted data to conversational format.
    
    This is a template - customize based on your data structure!
    """
    
    examples = []
    
    # Example: Convert each page to a training example
    for page_data in extracted_data['pages']:
        page_num = page_data['page']
        text = page_data['text']
        
        # Skip empty pages
        if not text or len(text) < 50:
            continue
        
        # Create a conversational example
        # TODO: Customize this based on your actual data!
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in UML diagrams."
                },
                {
                    "role": "user",
                    "content": f"Based on this requirement: {text[:200]}..."
                },
                {
                    "role": "assistant",
                    "content": "@startuml\n// TODO: Your UML diagram here\n@enduml"
                }
            ]
        }
        
        examples.append(example)
    
    return examples

def format_directory(extracted_dir: Path, output_dir: Path):
    """Format all extracted files."""
    
    extracted_files = list(extracted_dir.glob("*_extracted.json"))
    
    if not extracted_files:
        print("❌ No extracted files found!")
        return
    
    print(f"Found {len(extracted_files)} extracted file(s)")
    
    all_examples = []
    
    for filepath in extracted_files:
        print(f"\nProcessing: {filepath.name}")
        
        try:
            data = load_extracted_file(filepath)
            examples = format_to_conversational(data)
            
            print(f"  ✓ Created {len(examples)} training examples")
            all_examples.extend(examples)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Save to JSONL
    if all_examples:
        output_file = output_dir / "training_data_conversational.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Saved {len(all_examples)} examples to: {output_file}")
    else:
        print("\n⚠️  No examples created!")

def main():
    extracted_dir = PROJECT_ROOT / "data" / "extracted"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("DATA FORMATTING")
    print("="*80)
    print("\n⚠️  NOTE: This is a template script!")
    print("Edit format_to_conversational() to match your data structure.\n")
    
    format_directory(extracted_dir, processed_dir)
    
    print("\n" + "="*80)
    print("FORMATTING COMPLETE")
    print("="*80)
    print("\nNext step: python scripts/validate_data.py")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
