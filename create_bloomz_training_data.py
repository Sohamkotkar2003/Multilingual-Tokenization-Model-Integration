#!/usr/bin/env python3
"""
Create BLOOMZ-compatible instruction tuning dataset

This creates simple language generation tasks that BLOOMZ can learn from.
"""

import random
from pathlib import Path

# Language mapping
LANGUAGES = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
    'gu': 'Gujarati', 'mr': 'Marathi', 'ur': 'Urdu', 'pa': 'Punjabi',
    'kn': 'Kannada', 'ml': 'Malayalam', 'or': 'Odia', 'as': 'Assamese',
    'ne': 'Nepali', 'sa': 'Sanskrit', 'mai': 'Maithili', 'bd': 'Bodo',
    'mni': 'Meitei', 'sat': 'Santali', 'ks': 'Kashmiri', 'sd': 'Sindhi',
    'en': 'English'
}

def main():
    print("="*80)
    print("CREATING BLOOMZ TRAINING DATASET")
    print("="*80)
    
    input_dir = Path("data/training")
    output_file = Path("bloomz_training_data.txt")
    
    SAMPLES_PER_LANGUAGE = 500
    
    all_samples = []
    
    for txt_file in sorted(input_dir.glob("*_train.txt")):
        lang_code = txt_file.stem.split('_')[0]
        
        if lang_code not in LANGUAGES:
            continue
        
        lang_name = LANGUAGES[lang_code]
        print(f"\nProcessing {lang_name}...")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and len(line.strip()) > 20]
        
        # Sample and create simple instruction format
        sampled = random.sample(lines, min(len(lines), SAMPLES_PER_LANGUAGE))
        
        for text in sampled:
            # Simple format: just the native language text
            # BLOOMZ will learn the language patterns
            all_samples.append(text)
        
        print(f"  Added {len(sampled):,} samples")
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(sample + '\n')
    
    print("\n" + "="*80)
    print(f"✅ Created {len(all_samples):,} training samples")
    print(f"📁 Saved to: {output_file}")
    print("="*80)
    
    print("\n📋 NEXT STEPS:")
    print("1. Upload bloomz_training_data.txt to Google Drive")
    print("2. In Colab, update Cell 3:")
    print('   data_folder = "/content/drive/MyDrive/"')
    print('   txt_files = [data_folder + "bloomz_training_data.txt"]')
    print("3. Re-run training with more epochs (5-10) and samples (10k-20k)")
    print("="*80)

if __name__ == "__main__":
    main()

