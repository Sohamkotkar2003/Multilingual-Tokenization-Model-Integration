#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Dataset Cleanup Script
Fixes:
1. Bad train/val splits (creates proper 90/10 splits)
2. Removes duplicates
3. Removes empty lines
4. Ensures data quality
"""

import os
import sys
import io
from pathlib import Path
from collections import OrderedDict
import random

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Languages that need fixing based on analysis
LANGUAGES_TO_FIX = {
    'bho': {'reason': 'High duplicates (63.7%)', 'fix_duplicates': True, 'fix_split': False, 'fix_empty': False},
    'bo': {'reason': 'High duplicates (16.8%), bad split (99.9%)', 'fix_duplicates': True, 'fix_split': True, 'fix_empty': False},
    'dz': {'reason': 'Duplicates (14.8%), empty lines', 'fix_duplicates': True, 'fix_split': False, 'fix_empty': True},
    'fa_AF': {'reason': 'Bad split (99.6%), empty lines (49.4%)', 'fix_duplicates': False, 'fix_split': True, 'fix_empty': True},
    'lus': {'reason': 'Duplicates (13.8%), empty lines (5.4%)', 'fix_duplicates': True, 'fix_split': False, 'fix_empty': True},
    'my': {'reason': 'Bad split (98%), empty lines (30.4%)', 'fix_duplicates': False, 'fix_split': True, 'fix_empty': True},
    'ps': {'reason': 'Duplicates (32.2%), empty lines (28%)', 'fix_duplicates': True, 'fix_split': False, 'fix_empty': True},
    'si': {'reason': 'Empty lines (36.2%), duplicates (11.9%)', 'fix_duplicates': True, 'fix_split': False, 'fix_empty': True},
    'th': {'reason': 'Bad split (99.2%), empty lines (38.5%)', 'fix_duplicates': False, 'fix_split': True, 'fix_empty': True},
    'vi': {'reason': 'Bad split (99.2%), empty lines (41.5%)', 'fix_duplicates': False, 'fix_split': True, 'fix_empty': True},
}

def read_and_clean_file(filepath):
    """Read file and return cleaned, deduplicated lines"""
    print(f"  Reading: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove empty lines and strip whitespace
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    print(f"    Original lines: {len(lines):,}")
    print(f"    After removing empty: {len(cleaned_lines):,}")
    
    return cleaned_lines

def deduplicate_lines(lines):
    """Remove duplicates while preserving order"""
    seen = set()
    deduplicated = []
    
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduplicated.append(line)
    
    duplicates_removed = len(lines) - len(deduplicated)
    if duplicates_removed > 0:
        print(f"    Removed {duplicates_removed:,} duplicates ({duplicates_removed/len(lines)*100:.1f}%)")
    
    return deduplicated

def create_split(lines, train_ratio=0.90):
    """Create train/val split"""
    # Shuffle for random split
    random.seed(42)  # Fixed seed for reproducibility
    shuffled = lines.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_lines = shuffled[:split_idx]
    val_lines = shuffled[split_idx:]
    
    return train_lines, val_lines

def write_file(filepath, lines):
    """Write lines to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  ✅ Wrote {len(lines):,} lines ({file_size_mb:.2f} MB) to {os.path.basename(filepath)}")

def fix_language(lang_code, config):
    """Fix a single language dataset"""
    print(f"\n{'='*80}")
    print(f"Fixing: {lang_code.upper()}")
    print(f"Reason: {config['reason']}")
    print('='*80)
    
    train_file = Path(f'data/training/{lang_code}_train.txt')
    val_file = Path(f'data/validation/{lang_code}_val.txt')
    
    # Read both files
    train_lines = read_and_clean_file(train_file)
    val_lines = read_and_clean_file(val_file)
    
    if not train_lines and not val_lines:
        print(f"  ❌ No data found for {lang_code}")
        return
    
    # Combine all data
    all_lines = train_lines + val_lines
    print(f"\n  Total combined lines: {len(all_lines):,}")
    
    # Apply fixes
    if config['fix_empty']:
        # Already done in read_and_clean_file
        pass
    
    if config['fix_duplicates']:
        print(f"\n  Deduplicating...")
        all_lines = deduplicate_lines(all_lines)
        print(f"  After deduplication: {len(all_lines):,} lines")
    
    if config['fix_split']:
        print(f"\n  Creating new 90/10 split...")
        train_lines, val_lines = create_split(all_lines, train_ratio=0.90)
    else:
        # Keep existing split ratio if it's reasonable
        if len(all_lines) > 0:
            current_train_ratio = len(train_lines) / len(all_lines)
            if 0.85 <= current_train_ratio <= 0.95:
                print(f"\n  Keeping existing split ratio ({current_train_ratio*100:.1f}% train)")
                # Just use the cleaned data
                pass
            else:
                print(f"\n  Current split ({current_train_ratio*100:.1f}%) is outside 85-95% range, creating 90/10 split...")
                train_lines, val_lines = create_split(all_lines, train_ratio=0.90)
        else:
            train_lines, val_lines = create_split(all_lines, train_ratio=0.90)
    
    # Write back
    print(f"\n  Writing cleaned files...")
    write_file(train_file, train_lines)
    write_file(val_file, val_lines)
    
    # Summary
    final_train_ratio = len(train_lines) / (len(train_lines) + len(val_lines)) * 100
    print(f"\n  📊 Final Stats:")
    print(f"    Train: {len(train_lines):,} lines ({final_train_ratio:.1f}%)")
    print(f"    Val:   {len(val_lines):,} lines ({100-final_train_ratio:.1f}%)")
    print(f"  ✅ {lang_code.upper()} fixed successfully!")

def main():
    print("="*80)
    print("COMPREHENSIVE DATASET CLEANUP")
    print("="*80)
    print(f"\nLanguages to fix: {len(LANGUAGES_TO_FIX)}")
    for lang, config in LANGUAGES_TO_FIX.items():
        print(f"  - {lang}: {config['reason']}")
    
    print("\n" + "="*80)
    print("Starting cleanup process...")
    print("="*80)
    
    for lang_code, config in LANGUAGES_TO_FIX.items():
        try:
            fix_language(lang_code, config)
        except Exception as e:
            print(f"\n  ❌ Error fixing {lang_code}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("CLEANUP COMPLETE!")
    print("="*80)
    print("\nAll datasets have been cleaned and fixed.")
    print("Run analyze_all_datasets.py again to verify improvements.")

if __name__ == "__main__":
    main()

