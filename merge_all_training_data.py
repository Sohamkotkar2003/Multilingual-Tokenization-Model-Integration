#!/usr/bin/env python3
"""
Merge all training data from 21 original + 8 new languages
Creates combined dataset for training unified adapter
"""

import sys
import io
from pathlib import Path
import shutil
import json

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

print("="*80)
print("🔄 MERGING ALL TRAINING DATA")
print("="*80)

# Directories
ORIGINAL_DATA = Path("data/training")
NEW_DATA = Path("data/training_new")
ORIGINAL_VALIDATION = Path("data/validation")
OUTPUT_TRAIN = Path("data/training_merged")
OUTPUT_VAL = Path("data/validation_merged")

# Create output directories
OUTPUT_TRAIN.mkdir(exist_ok=True)
OUTPUT_VAL.mkdir(exist_ok=True)

print("\n📋 Source directories:")
print(f"   - Original (21 langs): {ORIGINAL_DATA}/")
print(f"   - New (8 langs): {NEW_DATA}/")
print(f"   - Original validation: {ORIGINAL_VALIDATION}/")
print(f"\n📁 Output directories:")
print(f"   - Training: {OUTPUT_TRAIN}/")
print(f"   - Validation: {OUTPUT_VAL}/")

# Copy original 21 languages
print("\n" + "="*80)
print("📥 COPYING ORIGINAL 21 LANGUAGES")
print("="*80)

original_files = list(ORIGINAL_DATA.glob("*.txt"))
print(f"\nFound {len(original_files)} files in {ORIGINAL_DATA}/")

copied_count = 0
for file_path in original_files:
    dest = OUTPUT_TRAIN / file_path.name
    shutil.copy2(file_path, dest)
    file_size = file_path.stat().st_size / 1024  # KB
    print(f"   ✅ Copied: {file_path.name} ({file_size:.1f} KB)")
    copied_count += 1

print(f"\n✅ Copied {copied_count} original language files")

# Copy validation data
print("\n" + "="*80)
print("📥 COPYING VALIDATION DATA")
print("="*80)

if ORIGINAL_VALIDATION.exists():
    validation_files = list(ORIGINAL_VALIDATION.glob("*.txt"))
    print(f"\nFound {len(validation_files)} validation files")
    
    val_copied = 0
    for file_path in validation_files:
        dest = OUTPUT_VAL / file_path.name
        shutil.copy2(file_path, dest)
        file_size = file_path.stat().st_size / 1024
        print(f"   ✅ Copied: {file_path.name} ({file_size:.1f} KB)")
        val_copied += 1
    
    print(f"\n✅ Copied {val_copied} validation files")
else:
    print("⚠️  No validation directory found, skipping")

# Process new 8 languages from JSONL
print("\n" + "="*80)
print("📥 CONVERTING NEW 8 LANGUAGES (JSONL → TXT)")
print("="*80)

NEW_LANGUAGES = {
    'sinhala': 'si',
    'tibetan': 'bo',
    'dzongkha': 'dz',
    'pashto': 'ps',
    'dari': 'fa_AF',
    'vietnamese': 'vi',
    'thai': 'th',
    'burmese': 'my'
}

new_converted = 0
total_samples = 0

for lang_name, lang_code in NEW_LANGUAGES.items():
    # Find all files for this language (could be lang_wiki.jsonl, lang_news.jsonl, etc.)
    jsonl_files = list(NEW_DATA.glob(f"{lang_name}*.jsonl"))
    
    if not jsonl_files:
        print(f"   ⚠️  Missing: {lang_name} (no files found)")
        continue
    
    # Read all JSONL files for this language and extract text
    texts = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get('text', '').strip()
                    if text and len(text) > 50:  # Skip very short texts
                        texts.append(text)
                except:
                    continue
    
    if not texts:
        print(f"   ⚠️  No valid texts in {lang_name}")
        continue
    
    # Split 80/20 for train/validation
    split_idx = int(len(texts) * 0.8)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Write training file
    train_file = OUTPUT_TRAIN / f"{lang_code}_train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_texts))
    
    # Write validation file
    val_file = OUTPUT_VAL / f"{lang_code}_val.txt"
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_texts))
    
    train_size = train_file.stat().st_size / 1024
    val_size = val_file.stat().st_size / 1024
    
    print(f"   ✅ {lang_name.capitalize():12} - {len(train_texts):4} train, {len(val_texts):3} val ({train_size:.1f} KB + {val_size:.1f} KB)")
    
    new_converted += 1
    total_samples += len(texts)

print(f"\n✅ Converted {new_converted} new languages ({total_samples} total samples)")

# Final summary
print("\n" + "="*80)
print("📊 MERGE COMPLETE!")
print("="*80)

train_files = list(OUTPUT_TRAIN.glob("*.txt"))
val_files = list(OUTPUT_VAL.glob("*.txt"))

total_train_size = sum(f.stat().st_size for f in train_files) / (1024 * 1024)  # MB
total_val_size = sum(f.stat().st_size for f in val_files) / (1024 * 1024)

print(f"\n📁 Training data: {OUTPUT_TRAIN}/")
print(f"   - Files: {len(train_files)}")
print(f"   - Total size: {total_train_size:.2f} MB")

print(f"\n📁 Validation data: {OUTPUT_VAL}/")
print(f"   - Files: {len(val_files)}")
print(f"   - Total size: {total_val_size:.2f} MB")

print(f"\n🌍 TOTAL LANGUAGES: {len(train_files)}")
print(f"   - Original: 21 languages")
print(f"   - New: 8 languages")
print(f"   - Combined: {len(train_files)} languages")

print("\n" + "="*80)
print("✅ Ready to train unified adapter!")
print("="*80)
print("\n📋 NEXT STEP:")
print("   Use train_single_language_local.py but modify it to use merged data")
print("   OR upload to Colab for faster training")
print(f"\n   Training data: {OUTPUT_TRAIN.absolute()}")
print(f"   Validation data: {OUTPUT_VAL.absolute()}")

