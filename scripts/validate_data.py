#!/usr/bin/env python3
"""
Data Validation Script
Validates JSONL data files and provides statistics
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent

def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    errors = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    errors.append((line_num, str(e)))
    
    return data, errors

def validate_messages(data):
    """Validate message format."""
    results = {
        'valid': 0,
        'invalid': 0,
        'errors': [],
        'role_counts': Counter(),
    }
    
    for idx, record in enumerate(data):
        if 'messages' not in record:
            results['invalid'] += 1
            results['errors'].append((idx, "Missing 'messages' field"))
            continue
        
        messages = record['messages']
        if not isinstance(messages, list):
            results['invalid'] += 1
            results['errors'].append((idx, "'messages' must be a list"))
            continue
        
        valid = True
        for msg in messages:
            if 'role' not in msg or 'content' not in msg:
                results['errors'].append((idx, f"Message missing role or content"))
                valid = False
                break
            results['role_counts'][msg['role']] += 1
        
        if valid:
            results['valid'] += 1
        else:
            results['invalid'] += 1
    
    return results

def print_report(filepath, data, validation, errors):
    """Print validation report."""
    print(f"\n{'='*80}")
    print(f"VALIDATION: {filepath.name}")
    print(f"{'='*80}")
    
    print(f"\n📄 File Parsing:")
    print(f"   Records loaded: {len(data)}")
    if errors:
        print(f"   ⚠️  Parse errors: {len(errors)}")
    else:
        print(f"   ✓ No parse errors")
    
    print(f"\n📋 Format Validation:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Invalid: {validation['invalid']}")
    
    if validation['errors']:
        print(f"   ⚠️  Errors: {len(validation['errors'])}")
        for idx, error in validation['errors'][:5]:
            print(f"      Record {idx}: {error}")
    else:
        print(f"   ✓ All records valid")
    
    print(f"\n👥 Roles:")
    for role, count in validation['role_counts'].most_common():
        print(f"   {role}: {count}")
    
    success_rate = (validation['valid'] / len(data) * 100) if data else 0
    print(f"\n✨ Success Rate: {success_rate:.1f}%")
    
    return success_rate >= 95

def main():
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    # Check directories
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    files_to_check = [
        raw_dir / "plantuml_training_data_conversational.jsonl",
        raw_dir / "uml_training_data_conversational.jsonl",
        processed_dir / "plantuml_training_data_conversational.jsonl",
        processed_dir / "uml_training_data_conversational.jsonl",
    ]
    
    existing_files = [f for f in files_to_check if f.exists()]
    
    if not existing_files:
        print("\n❌ No data files found!")
        print(f"\nPlace files in: {raw_dir}/")
        return False
    
    print(f"\nFound {len(existing_files)} file(s)")
    
    results = {}
    for filepath in existing_files:
        data, errors = load_jsonl(filepath)
        if not data:
            print(f"\n❌ No data in {filepath.name}")
            results[filepath] = False
            continue
        
        validation = validate_messages(data)
        passed = print_report(filepath, data, validation, errors)
        results[filepath] = passed
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for filepath, status in results.items():
        print(f"{'✓' if status else '❌'} {filepath.name}")
    
    print(f"\n{passed}/{total} files passed")
    
    if passed == total:
        print("\n✨ All files valid! Ready to train!")
        return True
    else:
        print("\n⚠️  Some files failed validation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
