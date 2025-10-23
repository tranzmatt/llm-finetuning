#!/usr/bin/env python3
"""
GPU Configuration Verification Script
Run this before training to verify CUDA_VISIBLE_DEVICES is set correctly
"""

import os
import torch

print("\n" + "="*80)
print("GPU CONFIGURATION CHECK")
print("="*80)

# Check environment variable
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"\nCUDA_VISIBLE_DEVICES: {cuda_visible}")

if cuda_visible == 'Not set':
    print("⚠️  WARNING: CUDA_VISIBLE_DEVICES not set!")
    print("   You may encounter issues with the display GPU at index 3")
    print("   Run: export CUDA_VISIBLE_DEVICES=0,1,2,4")
elif cuda_visible == '0,1,2,4':
    print("✓ Correctly configured to skip display GPU")
else:
    print(f"⚠️  Custom configuration: {cuda_visible}")

# Check PyTorch
print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("\n❌ CUDA not available! Check your PyTorch installation.")
    exit(1)

# Check devices
device_count = torch.cuda.device_count()
print(f"\nVisible GPU count: {device_count}")

if device_count == 0:
    print("❌ No GPUs visible to PyTorch!")
    exit(1)

print("\nGPU Details:")
print("-" * 80)

a100_count = 0
for i in range(device_count):
    name = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024**3)
    
    is_a100 = "A100" in name
    if is_a100:
        a100_count += 1
    
    status = "✓ A100" if is_a100 else "⚠️  NOT A100"
    print(f"{status} GPU {i}: {name}")
    print(f"        Memory: {memory_gb:.1f} GB")
    print(f"        Compute Capability: {props.major}.{props.minor}")
    
    # Check if this might be the display GPU
    if memory_gb < 10 and "A100" not in name:
        print(f"        ⚠️  WARNING: This looks like a display GPU!")
    print()

print("-" * 80)

# Summary
print("\nSummary:")
print(f"  Total visible GPUs: {device_count}")
print(f"  A100 GPUs: {a100_count}")

if cuda_visible == '0,1,2,4' and a100_count == 4:
    print("\n✅ CONFIGURATION CORRECT")
    print("   All 4 A100 GPUs are visible and ready for training!")
elif a100_count < device_count:
    print("\n⚠️  WARNING: Non-A100 GPU detected!")
    print(f"   Expected 4 A100s, but only found {a100_count}")
    print("   Make sure CUDA_VISIBLE_DEVICES=0,1,2,4")
else:
    print(f"\n⚠️  Unexpected configuration")
    print(f"   Check your CUDA_VISIBLE_DEVICES setting")

# Memory check
print("\nMemory Check:")
for i in range(device_count):
    free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
    total_mem = torch.cuda.mem_get_info(i)[1] / (1024**3)
    used_mem = total_mem - free_mem
    
    print(f"  GPU {i}: {used_mem:.1f} GB used / {total_mem:.1f} GB total ({free_mem:.1f} GB free)")
    
    if used_mem > 5:
        print(f"         ⚠️  WARNING: {used_mem:.1f} GB already in use!")

print("\n" + "="*80)

# Test allocation
try:
    print("\nTest: Allocating 1GB on each GPU...")
    tensors = []
    for i in range(device_count):
        with torch.cuda.device(i):
            # Allocate 1GB
            t = torch.zeros(256 * 1024 * 1024, dtype=torch.float32, device=f'cuda:{i}')
            tensors.append(t)
            print(f"  ✓ GPU {i}: Allocation successful")
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()
    
    print("\n✅ All GPUs can allocate memory successfully!")
    
except Exception as e:
    print(f"\n❌ Memory allocation failed: {e}")
    print("   One or more GPUs may have insufficient memory")

print("\n" + "="*80)
print("READY FOR TRAINING" if a100_count == 4 else "⚠️  FIX CONFIGURATION BEFORE TRAINING")
print("="*80 + "\n")
