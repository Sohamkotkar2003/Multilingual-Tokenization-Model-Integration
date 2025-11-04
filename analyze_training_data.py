#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze training data quality"""

import glob
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def analyze_file(filepath, max_lines=1000):
    """Analyze a training file for quality issues"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()[:max_lines] if line.strip()]
    
    # Statistics
    total_lines = len(lines)
    total_chars = sum(len(line) for line in lines)
    avg_length = total_chars / total_lines if total_lines > 0 else 0
    
    # Check for English mixing
    english_heavy = 0
    for line in lines:
        english_chars = sum(1 for c in line if c.isalpha() and ord(c) < 128)
        total_alpha = sum(1 for c in line if c.isalpha())
        if total_alpha > 0 and english_chars / total_alpha > 0.5:
            english_heavy += 1
    
    return {
        'file': filepath.split('\\')[-1],
        'total_lines': total_lines,
        'avg_length': avg_length,
        'english_heavy_pct': (english_heavy / total_lines * 100) if total_lines > 0 else 0
    }

print("\n" + "="*80)
print("  TRAINING DATA QUALITY ANALYSIS")
print("="*80 + "\n")

# Analyze all training files
results = []
for filepath in sorted(glob.glob('data/training/*.txt')):
    result = analyze_file(filepath, max_lines=1000)
    results.append(result)
    
    status = "✅" if result['english_heavy_pct'] < 10 else "⚠️"
    print(f"{status} {result['file']:20s} | Lines: {result['total_lines']:4d} | Avg Len: {result['avg_length']:6.1f} | English-heavy: {result['english_heavy_pct']:5.1f}%")

print("\n" + "="*80)
print("  SUMMARY")
print("="*80 + "\n")

clean_files = [r for r in results if r['english_heavy_pct'] < 10]
mixed_files = [r for r in results if r['english_heavy_pct'] >= 10]

print(f"✅ Clean files: {len(clean_files)}/{len(results)}")
print(f"⚠️ Mixed content files: {len(mixed_files)}/{len(results)}")

if mixed_files:
    print(f"\n⚠️ Files with >10% English content:")
    for r in sorted(mixed_files, key=lambda x: x['english_heavy_pct'], reverse=True):
        print(f"   {r['file']:20s} - {r['english_heavy_pct']:5.1f}% English")

print(f"\n📊 VERDICT:")
if len(mixed_files) == 0:
    print("   ✅ Your training data is CLEAN! Issue is not data quality.")
    print("   ✅ The English mixing is a MODEL LIMITATION, not data issue.")
    print("   ✅ Colab training WILL help, but data is already good!")
else:
    print(f"   ⚠️ {len(mixed_files)} files have English mixing in training data")
    print("   ✅ Cleaning this data could improve results WITHOUT Colab!")

print()

