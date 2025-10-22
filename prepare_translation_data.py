#!/usr/bin/env python3
"""
Convert monolingual training data into instruction-tuning format for translation

This creates translation pairs from your monolingual data using back-translation
or simple instruction formatting.
"""

import random
from pathlib import Path
from typing import List, Tuple

# Language mapping
LANGUAGES = {
    'hi': 'Hindi',
    'bn': 'Bengali', 
    'ta': 'Tamil',
    'te': 'Telugu',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'ur': 'Urdu',
    'pa': 'Punjabi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'or': 'Odia',
    'as': 'Assamese',
    'ne': 'Nepali',
    'sa': 'Sanskrit',
    'mai': 'Maithili',
    'bd': 'Bodo',
    'mni': 'Meitei',
    'sat': 'Santali',
    'ks': 'Kashmiri',
    'sd': 'Sindhi',
    'en': 'English'
}

def create_instruction_pairs(lang_code: str, texts: List[str], max_samples: int = 500) -> List[str]:
    """
    Create instruction-tuning pairs from monolingual text
    
    Format: "Instruction: [task]\nInput: [input]\nOutput: [output]"
    """
    lang_name = LANGUAGES.get(lang_code, lang_code)
    pairs = []
    
    # Sample texts
    sampled_texts = random.sample(texts, min(len(texts), max_samples))
    
    for text in sampled_texts:
        text = text.strip()
        if not text or len(text) < 10:  # Skip very short texts
            continue
        
        # Create different instruction formats
        templates = [
            f"Provide text in {lang_name}.\n{text}",
            f"Generate a sentence in {lang_name}.\n{text}",
            f"Write in {lang_name}.\n{text}",
            f"Say this in {lang_name}.\n{text}",
            f"{lang_name} sentence:\n{text}",
        ]
        
        # Randomly choose a template
        formatted = random.choice(templates)
        pairs.append(formatted)
    
    return pairs

def main():
    print("="*80)
    print("CREATING INSTRUCTION-TUNING DATASET")
    print("="*80)
    
    input_dir = Path("data/training")
    output_dir = Path("data/instruction_tuning")
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    SAMPLES_PER_LANGUAGE = 500  # Adjust this
    
    all_pairs = []
    stats = {}
    
    # Process each language file
    for txt_file in sorted(input_dir.glob("*_train.txt")):
        lang_code = txt_file.stem.split('_')[0]  # e.g., 'hi' from 'hi_train.txt'
        
        if lang_code not in LANGUAGES:
            print(f"\nSkipping {txt_file.name} - unknown language code")
            continue
        
        lang_name = LANGUAGES[lang_code]
        print(f"\nProcessing {lang_name} ({lang_code})...")
        
        # Read file
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"  Error reading {txt_file.name}: {e}")
            continue
        
        print(f"  Found {len(lines):,} lines")
        
        # Create instruction pairs
        pairs = create_instruction_pairs(lang_code, lines, SAMPLES_PER_LANGUAGE)
        
        print(f"  Created {len(pairs):,} instruction pairs")
        
        all_pairs.extend(pairs)
        stats[lang_name] = len(pairs)
        
        # Save individual language file
        lang_output_file = output_dir / f"{lang_code}_instruction.txt"
        with open(lang_output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pairs))
        
        print(f"  Saved to {lang_output_file}")
    
    # Shuffle and save combined file
    random.shuffle(all_pairs)
    
    combined_file = output_dir / "all_languages_instruction.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_pairs))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTotal instruction pairs: {len(all_pairs):,}")
    print(f"Saved to: {combined_file}")
    
    print("\nSamples per language:")
    for lang, count in sorted(stats.items()):
        print(f"  {lang:15s}: {count:,}")
    
    print("\n" + "="*80)
    print("Sample outputs:")
    print("="*80)
    
    for i, sample in enumerate(random.sample(all_pairs, min(5, len(all_pairs))), 1):
        print(f"\n{i}. {sample[:200]}...")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Upload the file to Google Drive:")
    print(f"   {combined_file}")
    print("\n2. In Colab Cell 3, update the path:")
    print(f"   data_folder = \"/content/drive/MyDrive/instruction_tuning_data\"")
    print("\n3. Upload the .txt file to that Drive folder")
    print("\n4. Re-run training!")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

