#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Dataset Quality Analysis
Checks all 33 languages for:
1. Line counts and sizes
2. Data quality (empty lines, duplicates, script correctness)
3. Train/Val split ratios
4. Character distribution and language detection
"""

import os
import sys
import io
from pathlib import Path
from collections import Counter
import unicodedata

# Force UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Language to script mapping
LANGUAGE_SCRIPTS = {
    'as': 'Bengali',  # Assamese uses Bengali script
    'awa': 'Devanagari',
    'bd': 'Devanagari',  # Bodo
    'bho': 'Devanagari',
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'dz': 'Tibetan',
    'en': 'Latin',
    'fa_AF': 'Arabic',  # Dari
    'gu': 'Gujarati',
    'hi': 'Devanagari',
    'hne': 'Devanagari',  # Chhattisgarhi
    'kn': 'Kannada',
    'ks': 'Arabic',  # Kashmiri (Perso-Arabic)
    'lus': 'Latin',  # Mizo
    'mag': 'Devanagari',
    'mai': 'Devanagari',
    'ml': 'Malayalam',
    'mni': 'Meetei_Mayek',  # Meitei
    'mr': 'Devanagari',
    'my': 'Myanmar',
    'ne': 'Devanagari',
    'or': 'Odia',
    'pa': 'Gurmukhi',  # Punjabi
    'ps': 'Arabic',  # Pashto
    'sat': 'Ol_Chiki',  # Santali
    'sa': 'Devanagari',
    'sd': 'Arabic',  # Sindhi
    'si': 'Sinhala',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'ur': 'Arabic',
    'vi': 'Latin',  # Vietnamese
    'gom': 'Devanagari',  # Konkani
    'tcy': 'Kannada',  # Tulu
    'raj': 'Devanagari',  # Rajasthani
    'doi': 'Devanagari',  # Dogri
    'kha': 'Latin',  # Khasi
}

def get_script_name(char):
    """Get Unicode script name for a character"""
    try:
        return unicodedata.name(char).split()[0]
    except:
        return 'Unknown'

def analyze_script_distribution(text_sample):
    """Analyze script distribution in text sample"""
    scripts = Counter()
    for char in text_sample:
        if char.strip():  # Skip whitespace
            try:
                script = unicodedata.name(char).split()[0]
                scripts[script] += 1
            except:
                pass
    return scripts

def analyze_file(filepath, expected_script):
    """Analyze a single dataset file"""
    if not os.path.exists(filepath):
        return None
    
    print(f"\n  Analyzing: {os.path.basename(filepath)}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Basic stats
    total_lines = len(lines)
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    empty_lines = total_lines - len(non_empty_lines)
    
    # Size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    # Duplicates
    unique_lines = set(non_empty_lines)
    duplicates = len(non_empty_lines) - len(unique_lines)
    
    # Average line length
    if non_empty_lines:
        avg_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        min_length = min(len(line) for line in non_empty_lines)
        max_length = max(len(line) for line in non_empty_lines)
    else:
        avg_length = min_length = max_length = 0
    
    # Sample analysis (first 10000 chars)
    sample_text = ''.join(non_empty_lines[:100])[:10000]
    script_dist = analyze_script_distribution(sample_text)
    
    # Check if expected script is dominant
    total_chars = sum(script_dist.values())
    expected_script_chars = sum(count for script, count in script_dist.items() 
                                 if expected_script.lower() in script.lower())
    script_ratio = expected_script_chars / total_chars if total_chars > 0 else 0
    
    # Quality assessment
    quality_issues = []
    if empty_lines > total_lines * 0.01:  # More than 1% empty
        quality_issues.append(f"{empty_lines} empty lines ({empty_lines/total_lines*100:.1f}%)")
    if duplicates > len(non_empty_lines) * 0.05:  # More than 5% duplicates
        quality_issues.append(f"{duplicates} duplicates ({duplicates/len(non_empty_lines)*100:.1f}%)")
    if script_ratio < 0.7 and expected_script != 'Latin':  # Less than 70% expected script
        quality_issues.append(f"Only {script_ratio*100:.1f}% {expected_script} script")
    if avg_length < 20:
        quality_issues.append(f"Very short lines (avg: {avg_length:.1f} chars)")
    if avg_length > 500:
        quality_issues.append(f"Very long lines (avg: {avg_length:.1f} chars)")
    
    return {
        'total_lines': total_lines,
        'non_empty_lines': len(non_empty_lines),
        'empty_lines': empty_lines,
        'duplicates': duplicates,
        'file_size_mb': file_size_mb,
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'script_ratio': script_ratio,
        'quality_issues': quality_issues,
        'top_scripts': script_dist.most_common(3)
    }

def main():
    print("="*80)
    print("COMPREHENSIVE DATASET QUALITY ANALYSIS")
    print("="*80)
    
    train_dir = Path('data/training')
    val_dir = Path('data/validation')
    
    results = {}
    
    for lang_code, expected_script in sorted(LANGUAGE_SCRIPTS.items()):
        print(f"\n{'='*80}")
        print(f"Language: {lang_code.upper()} (Expected script: {expected_script})")
        print('='*80)
        
        # Analyze training file
        train_file = train_dir / f"{lang_code}_train.txt"
        train_stats = analyze_file(train_file, expected_script)
        
        # Analyze validation file
        val_file = val_dir / f"{lang_code}_val.txt"
        val_stats = analyze_file(val_file, expected_script)
        
        if train_stats and val_stats:
            # Calculate split ratio
            total_samples = train_stats['non_empty_lines'] + val_stats['non_empty_lines']
            train_ratio = train_stats['non_empty_lines'] / total_samples * 100
            val_ratio = val_stats['non_empty_lines'] / total_samples * 100
            
            print(f"\n  SUMMARY:")
            print(f"    Training:   {train_stats['non_empty_lines']:>10,} lines ({train_stats['file_size_mb']:>8.2f} MB)")
            print(f"    Validation: {val_stats['non_empty_lines']:>10,} lines ({val_stats['file_size_mb']:>8.2f} MB)")
            print(f"    Split Ratio: {train_ratio:.1f}% train / {val_ratio:.1f}% val")
            print(f"    Avg Length: Train={train_stats['avg_length']:.1f}, Val={val_stats['avg_length']:.1f}")
            print(f"    Script Match: Train={train_stats['script_ratio']*100:.1f}%, Val={val_stats['script_ratio']*100:.1f}%")
            
            # Quality assessment
            all_issues = train_stats['quality_issues'] + val_stats['quality_issues']
            if all_issues:
                print(f"\n  ⚠️  QUALITY ISSUES:")
                for issue in all_issues:
                    print(f"      - {issue}")
            else:
                print(f"\n  ✅ QUALITY: GOOD")
            
            # Split assessment
            if 85 <= train_ratio <= 95:
                print(f"  ✅ SPLIT: GOOD ({train_ratio:.1f}% train)")
            else:
                print(f"  ⚠️  SPLIT: Unusual ratio ({train_ratio:.1f}% train, expected 85-95%)")
            
            results[lang_code] = {
                'train': train_stats,
                'val': val_stats,
                'split_ratio': train_ratio
            }
        elif train_stats:
            print(f"\n  ❌ MISSING: Validation file not found")
        elif val_stats:
            print(f"\n  ❌ MISSING: Training file not found")
        else:
            print(f"\n  ❌ MISSING: Both files not found")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total_train_size = sum(r['train']['file_size_mb'] for r in results.values())
    total_val_size = sum(r['val']['file_size_mb'] for r in results.values())
    total_train_lines = sum(r['train']['non_empty_lines'] for r in results.values())
    total_val_lines = sum(r['val']['non_empty_lines'] for r in results.values())
    
    print(f"\nTotal Languages: {len(results)}")
    print(f"Total Training Data: {total_train_lines:,} lines ({total_train_size:.2f} GB)")
    print(f"Total Validation Data: {total_val_lines:,} lines ({total_val_size:.2f} GB)")
    print(f"Overall Split: {total_train_lines/(total_train_lines+total_val_lines)*100:.1f}% train / {total_val_lines/(total_train_lines+total_val_lines)*100:.1f}% val")
    
    # Languages with issues
    issues_found = []
    for lang_code, data in results.items():
        if data['train']['quality_issues'] or data['val']['quality_issues']:
            issues_found.append(lang_code)
    
    if issues_found:
        print(f"\n⚠️  Languages with quality issues: {', '.join(issues_found)}")
    else:
        print(f"\n✅ All datasets passed quality checks!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

