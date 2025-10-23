#!/usr/bin/env python3
"""
Analyze Training Data Sequence Lengths
Check if your data fits within max_seq_length
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent

def count_tokens_estimate(text):
    """Rough estimate: ~1.3 tokens per word for English text."""
    # More accurate for code/UML: count by characters
    # Rule of thumb: 1 token ≈ 4 characters for code
    return len(text) // 4

def analyze_file(filepath, max_seq_length=4096):
    """Analyze sequence lengths in a JSONL file."""
    
    print(f"\nAnalyzing: {filepath.name}")
    print("="*80)
    
    lengths = []
    too_long = 0
    total = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Get full conversation length
                if 'messages' in data:
                    full_text = ""
                    for msg in data['messages']:
                        full_text += msg.get('content', '')
                    
                    token_count = count_tokens_estimate(full_text)
                    lengths.append(token_count)
                    total += 1
                    
                    if token_count > max_seq_length:
                        too_long += 1
                        if too_long <= 3:  # Show first 3 examples
                            print(f"\nExample {too_long} (Line {line_num}):")
                            print(f"  Estimated tokens: {token_count}")
                            print(f"  Exceeds limit by: {token_count - max_seq_length} tokens")
                            print(f"  Preview: {full_text[:200]}...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    if not lengths:
        print("No valid data found")
        return
    
    # Statistics
    lengths_sorted = sorted(lengths)
    
    print(f"\nStatistics for {filepath.name}:")
    print("-"*80)
    print(f"Total examples: {total}")
    print(f"Min length: {min(lengths)} tokens")
    print(f"Max length: {max(lengths)} tokens")
    print(f"Mean length: {sum(lengths)//len(lengths)} tokens")
    print(f"Median length: {lengths_sorted[len(lengths_sorted)//2]} tokens")
    print(f"95th percentile: {lengths_sorted[int(len(lengths_sorted)*0.95)]} tokens")
    print(f"99th percentile: {lengths_sorted[int(len(lengths_sorted)*0.99)]} tokens")
    
    print(f"\nWith max_seq_length = {max_seq_length}:")
    print(f"  Examples that fit: {total - too_long} ({(total-too_long)/total*100:.1f}%)")
    print(f"  Examples truncated: {too_long} ({too_long/total*100:.1f}%)")
    
    if too_long > 0:
        pct = too_long / total * 100
        if pct > 10:
            print(f"\n⚠️  WARNING: {pct:.1f}% of examples will be truncated!")
        elif pct > 5:
            print(f"\n⚠️  CAUTION: {pct:.1f}% of examples will be truncated")
        else:
            print(f"\n✓ Only {pct:.1f}% truncated - acceptable")
    else:
        print("\n✅ All examples fit within sequence length!")
    
    return lengths

def main():
    print("\n" + "="*80)
    print("TRAINING DATA SEQUENCE LENGTH ANALYSIS")
    print("="*80)
    
    # Check current model config
    try:
        import yaml
        config_path = PROJECT_ROOT / "config" / "model_configs.yaml"
        with open(config_path, 'r') as f:
            model_configs = yaml.safe_load(f)
        
        with open(PROJECT_ROOT / "config" / "training_config.yaml", 'r') as f:
            training_config = yaml.safe_load(f)
        
        selected_model = training_config['selected_model']
        max_seq = model_configs[selected_model]['max_seq_length']
        
        print(f"\nCurrent configuration:")
        print(f"  Model: {selected_model}")
        print(f"  Max sequence length: {max_seq}")
    except:
        max_seq = 4096
        print(f"\nUsing default max_seq_length: {max_seq}")
    
    # Find data files
    processed_dir = PROJECT_ROOT / "data" / "processed"
    data_files = list(processed_dir.glob("*.jsonl"))
    
    if not data_files:
        print("\n❌ No JSONL files found in data/processed/")
        sys.exit(1)
    
    print(f"\nFound {len(data_files)} file(s)")
    
    all_lengths = []
    for filepath in data_files:
        lengths = analyze_file(filepath, max_seq)
        if lengths:
            all_lengths.extend(lengths)
    
    # Overall statistics
    if all_lengths:
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        
        all_lengths_sorted = sorted(all_lengths)
        too_long = sum(1 for l in all_lengths if l > max_seq)
        
        print(f"\nAll training data combined:")
        print(f"  Total examples: {len(all_lengths)}")
        print(f"  Mean length: {sum(all_lengths)//len(all_lengths)} tokens")
        print(f"  Median length: {all_lengths_sorted[len(all_lengths_sorted)//2]} tokens")
        print(f"  95th percentile: {all_lengths_sorted[int(len(all_lengths_sorted)*0.95)]} tokens")
        print(f"  99th percentile: {all_lengths_sorted[int(len(all_lengths_sorted)*0.99)]} tokens")
        print(f"  Max length: {max(all_lengths)} tokens")
        
        print(f"\n  Examples that fit: {len(all_lengths) - too_long} ({(len(all_lengths)-too_long)/len(all_lengths)*100:.1f}%)")
        print(f"  Examples truncated: {too_long} ({too_long/len(all_lengths)*100:.1f}%)")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if not all_lengths:
        print("\n❌ No data to analyze")
        return
    
    max_length = max(all_lengths)
    pct_truncated = too_long / len(all_lengths) * 100
    
    if pct_truncated == 0:
        print(f"\n✅ PERFECT: All examples fit within {max_seq} tokens")
        print("   No changes needed!")
    
    elif pct_truncated < 5:
        print(f"\n✅ GOOD: Only {pct_truncated:.1f}% of examples truncated")
        print(f"   Current setting ({max_seq}) is fine")
        print("   Minor truncation won't significantly affect quality")
    
    elif pct_truncated < 10:
        print(f"\n⚠️  ACCEPTABLE: {pct_truncated:.1f}% of examples truncated")
        print(f"   Current setting ({max_seq}) is workable")
        print("   Consider increasing if quality is important")
    
    elif pct_truncated < 20:
        print(f"\n⚠️  CONCERNING: {pct_truncated:.1f}% of examples truncated")
        print(f"   Recommend increasing max_seq_length")
    
    else:
        print(f"\n❌ PROBLEM: {pct_truncated:.1f}% of examples truncated!")
        print(f"   You should increase max_seq_length")
    
    # Suggest new length
    if too_long > 0:
        recommended = all_lengths_sorted[int(len(all_lengths_sorted)*0.99)]
        
        # Round up to nearest 512
        recommended = ((recommended + 511) // 512) * 512
        
        print(f"\n📊 Suggested max_seq_length: {recommended}")
        print(f"   This would fit 99% of your examples")
        
        if recommended > max_seq:
            print(f"\n   To change:")
            print(f"   Edit config/model_configs.yaml:")
            print(f"     max_seq_length: {max_seq} → {recommended}")
            
            # Calculate memory impact
            memory_increase = recommended / max_seq
            print(f"\n   ⚠️  Memory impact: ~{memory_increase:.1f}x more VRAM needed")
            print(f"   Current: ~50-60GB per GPU")
            print(f"   With {recommended}: ~{int(50*memory_increase)}-{int(60*memory_increase)}GB per GPU")
            
            if memory_increase > 1.5:
                print(f"\n   ⚠️  This may cause OOM with your current batch size")
                print(f"   You may need to:")
                print(f"     - Reduce batch size")
                print(f"     - Increase gradient accumulation")
                print(f"     - Or keep current length and accept truncation")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
