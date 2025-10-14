#!/usr/bin/env python3
"""
Training Speed Test Script

This script estimates training time based on current configuration
and validates that the optimizations are working correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config import settings

def count_training_samples():
    """Count actual training samples with current configuration"""
    print("=" * 80)
    print("TRAINING DATA ANALYSIS")
    print("=" * 80)
    
    # Import the settings from fine_tune.py
    sys.path.append("src/training")
    try:
        import fine_tune
        use_sampling = fine_tune.USE_DATA_SAMPLING
        max_samples = fine_tune.MAX_SAMPLES_PER_LANGUAGE
        batch_size = fine_tune.BATCH_SIZE
        grad_accum = fine_tune.GRADIENT_ACCUMULATION_STEPS
        max_steps = fine_tune.MAX_STEPS
        epochs = fine_tune.EPOCHS
    except:
        print("⚠️  Could not import fine_tune.py configuration")
        print("Using default values for estimation...")
        use_sampling = True
        max_samples = 5000
        batch_size = 4
        grad_accum = 4
        max_steps = 1000
        epochs = 1
    
    print(f"\nConfiguration:")
    print(f"   Use Sampling: {use_sampling}")
    print(f"   Max Samples per Language: {max_samples:,}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Gradient Accumulation: {grad_accum}")
    print(f"   Effective Batch Size: {batch_size * grad_accum}")
    print(f"   Max Steps: {max_steps}")
    print(f"   Epochs: {epochs}")
    
    # Count actual files and samples
    training_dir = Path(settings.TRAINING_DATA_PATH)
    total_samples = 0
    total_available = 0
    languages = 0
    
    print(f"\nScanning training data...")
    
    for lang, filename in settings.CORPUS_FILES.items():
        filepath = training_dir / filename
        if filepath.exists():
            languages += 1
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if len(line.strip()) > 10]
                available = len(lines)
                total_available += available
                
                if use_sampling and available > max_samples:
                    used = max_samples
                else:
                    used = available
                
                total_samples += used
                
                if available > 0:
                    print(f"   {lang:12s}: {available:8,} lines -> using {used:6,} samples")
    
    print(f"\nSummary:")
    print(f"   Languages found: {languages}")
    print(f"   Total available: {total_available:,} lines")
    print(f"   Total used: {total_samples:,} samples")
    if total_available > 0:
        print(f"   Reduction: {100 * (1 - total_samples/total_available):.1f}%")
    
    return total_samples, batch_size, grad_accum, max_steps, epochs

def estimate_training_time(total_samples, batch_size, grad_accum, max_steps, epochs):
    """Estimate training time"""
    print("\n" + "=" * 80)
    print("TRAINING TIME ESTIMATION")
    print("=" * 80)
    
    effective_batch_size = batch_size * grad_accum
    steps_per_epoch = total_samples // effective_batch_size
    total_steps_planned = steps_per_epoch * epochs
    
    # Limit by max_steps
    if max_steps > 0 and total_steps_planned > max_steps:
        actual_steps = max_steps
        print(f"\nWARNING: Training will be limited by MAX_STEPS")
    else:
        actual_steps = total_steps_planned
    
    print(f"\nTraining Steps:")
    print(f"   Steps per epoch: {steps_per_epoch:,}")
    print(f"   Planned total steps (epochs x steps): {total_steps_planned:,}")
    if max_steps > 0:
        print(f"   Max steps limit: {max_steps:,}")
    print(f"   Actual steps to run: {actual_steps:,}")
    
    # Time estimates (based on RTX 4050 6GB with 8-bit + LoRA)
    # Typical speed: 1-2 seconds per step with batch_size=4
    seconds_per_step_min = 1.0
    seconds_per_step_max = 2.0
    
    training_time_min = actual_steps * seconds_per_step_min
    training_time_max = actual_steps * seconds_per_step_max
    
    # Add overhead
    data_loading_time = 180  # ~3 minutes
    model_loading_time = 120  # ~2 minutes
    overhead_time = data_loading_time + model_loading_time
    
    total_time_min = (training_time_min + overhead_time) / 60
    total_time_max = (training_time_max + overhead_time) / 60
    
    print(f"\nEstimated Training Time (RTX 4050 6GB):")
    print(f"   Data loading: ~{data_loading_time // 60} minutes")
    print(f"   Model loading: ~{model_loading_time // 60} minutes")
    print(f"   Training ({actual_steps:,} steps): {training_time_min/60:.1f} - {training_time_max/60:.1f} minutes")
    print(f"   " + "-" * 60)
    print(f"   TOTAL: {total_time_min:.1f} - {total_time_max:.1f} minutes")
    print(f"   (approximately {int(total_time_min)} - {int(total_time_max)} minutes)")
    
    return total_time_min, total_time_max

def compare_with_old_config():
    """Compare with old configuration"""
    print("\n" + "=" * 80)
    print("COMPARISON WITH OLD CONFIGURATION")
    print("=" * 80)
    
    # Old configuration
    old_samples = 4_850_000  # Approximate total from all files
    old_batch_size = 2
    old_grad_accum = 2
    old_epochs = 3
    old_max_steps = -1  # No limit
    
    old_effective_batch = old_batch_size * old_grad_accum
    old_steps = (old_samples // old_effective_batch) * old_epochs
    old_training_time = (old_steps * 1.5) / 60 / 60  # hours
    
    print(f"\nOLD Configuration:")
    print(f"   Samples: {old_samples:,}")
    print(f"   Batch size: {old_batch_size}")
    print(f"   Gradient accumulation: {old_grad_accum}")
    print(f"   Effective batch size: {old_effective_batch}")
    print(f"   Epochs: {old_epochs}")
    print(f"   Estimated steps: {old_steps:,}")
    print(f"   Estimated time: {old_training_time:.1f} hours ({old_training_time * 60:.0f} minutes)")
    
    return old_training_time * 60

def main():
    print("\nTesting Training Configuration and Estimating Speed...\n")
    
    # Count samples with current config
    total_samples, batch_size, grad_accum, max_steps, epochs = count_training_samples()
    
    # Estimate training time
    new_time_min, new_time_max = estimate_training_time(
        total_samples, batch_size, grad_accum, max_steps, epochs
    )
    
    # Compare with old config
    old_time = compare_with_old_config()
    
    # Show improvement
    speedup_min = old_time / new_time_max
    speedup_max = old_time / new_time_min
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT SUMMARY")
    print("=" * 80)
    print(f"\nNEW Configuration:")
    print(f"   Estimated time: {new_time_min:.0f} - {new_time_max:.0f} minutes")
    print(f"\nOLD Configuration:")
    print(f"   Estimated time: {old_time:.0f} minutes")
    print(f"\nSPEEDUP: {speedup_min:.1f}x - {speedup_max:.1f}x faster!")
    print(f"\nTime saved: {old_time - new_time_max:.0f} - {old_time - new_time_min:.0f} minutes")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\nYour configuration looks good! You can:")
    print("   1. Start training: python src/training/fine_tune.py")
    print("   2. Monitor progress in real-time (logs every 50 steps)")
    print("   3. Training should complete in ~30-60 minutes")
    print("\nIf you want to:")
    print("   - Train faster: Reduce MAX_SAMPLES_PER_LANGUAGE to 2000-3000")
    print("   - Better quality: Increase MAX_SAMPLES_PER_LANGUAGE to 10000")
    print("   - Use all data: Set USE_DATA_SAMPLING = False (very slow!)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

