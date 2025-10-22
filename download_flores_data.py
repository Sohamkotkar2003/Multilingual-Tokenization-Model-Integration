#!/usr/bin/env python3
"""
Download and prepare FLORES-101 parallel translation data

This is the FASTEST way to get high-quality parallel data for adapter training.
FLORES has 3001 sentences in 101 languages - perfect for testing!
"""

import urllib.request
import tarfile
import os
from pathlib import Path

# FLORES language codes (just 3-letter codes)
FLORES_LANGS = {
    'hi': 'hin',  # Hindi
    'bn': 'ben',  # Bengali
    'ta': 'tam',  # Tamil
    'te': 'tel',  # Telugu
    'gu': 'guj',  # Gujarati
    'mr': 'mar',  # Marathi
    'ur': 'urd',  # Urdu
    'pa': 'pan',  # Punjabi
    'kn': 'kan',  # Kannada
    'ml': 'mal',  # Malayalam
    'or': 'ory',  # Odia
    'as': 'asm',  # Assamese
    'ne': 'npi',  # Nepali
}

def download_flores():
    """Download FLORES-101 dataset"""
    print("="*80)
    print("DOWNLOADING FLORES-101 DATASET")
    print("="*80)
    
    url = "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"
    output_file = "flores101_dataset.tar.gz"
    output_dir = "flores101_dataset"
    
    if os.path.exists(output_dir):
        print(f"\n{output_dir} already exists, skipping download")
        return output_dir
    
    print(f"\nDownloading from {url}...")
    print("   (This may take a few minutes - ~50MB)")
    
    urllib.request.urlretrieve(url, output_file)
    print(f"Downloaded {output_file}")
    
    print(f"\nExtracting...")
    with tarfile.open(output_file, 'r:gz') as tar:
        tar.extractall()
    
    print(f"Extracted to {output_dir}/")
    
    # Cleanup
    os.remove(output_file)
    print(f"Removed {output_file}")
    
    return output_dir

def create_translation_pairs(flores_dir, output_file="flores_training_data.txt"):
    """Create translation instruction pairs from FLORES"""
    print("\n" + "="*80)
    print("CREATING TRANSLATION PAIRS")
    print("="*80)
    
    flores_path = Path(flores_dir)
    all_pairs = []
    
    # Use dev and devtest splits (combined = 2009 + 1012 = 3021 sentences)
    splits = ['dev', 'devtest']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # English file  
        en_file = flores_path / split / f"eng.{split}"
        
        if not en_file.exists():
            print(f"  Warning: {en_file} not found, skipping")
            continue
        
        # Read English
        with open(en_file, 'r', encoding='utf-8') as f:
            en_lines = [line.strip() for line in f]
        
        print(f"  Found {len(en_lines)} English sentences")
        
        # Process each target language
        for lang_code, flores_code in FLORES_LANGS.items():
            target_file = flores_path / split / f"{flores_code}.{split}"
            
            if not target_file.exists():
                print(f"    Warning {lang_code}: File not found, skipping")
                continue
            
            # Read target language
            with open(target_file, 'r', encoding='utf-8') as f:
                target_lines = [line.strip() for line in f]
            
            # Create pairs
            lang_name = {
                'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
                'gu': 'Gujarati', 'mr': 'Marathi', 'ur': 'Urdu', 'pa': 'Punjabi',
                'kn': 'Kannada', 'ml': 'Malayalam', 'or': 'Odia', 'as': 'Assamese',
                'ne': 'Nepali'
            }[lang_code]
            
            for en_sent, target_sent in zip(en_lines, target_lines):
                # Format: "Translate to [Language]: [English]\n[Translation]"
                pair = f"Translate to {lang_name}: {en_sent}\n{target_sent}"
                all_pairs.append(pair)
            
            print(f"    {lang_name}: {len(target_lines)} pairs")
    
    # Shuffle
    import random
    random.shuffle(all_pairs)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(pair + '\n\n')  # Double newline to separate pairs
    
    print(f"\n{'='*80}")
    print(f"CREATED {len(all_pairs):,} TRANSLATION PAIRS")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}")
    
    return output_file, len(all_pairs)

def show_samples(output_file, n=5):
    """Show sample pairs"""
    print(f"\nSAMPLE PAIRS:")
    print("="*80)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pairs = content.split('\n\n')
    
    import random
    for i, pair in enumerate(random.sample(pairs[:100], min(n, len(pairs))), 1):
        if pair.strip():
            lines = pair.strip().split('\n')
            if len(lines) >= 2:
                print(f"\n{i}.")
                print(f"  Instruction: {lines[0][:80]}...")
                print(f"  Response:    {lines[1][:80]}...")

def main():
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("\nFLORES-101 PARALLEL DATA DOWNLOADER")
    print("\nThis will download ~50MB of high-quality parallel translation data")
    print("covering 13 Indian languages + English.\n")
    
    # Download
    flores_dir = download_flores()
    
    # Create pairs
    output_file, count = create_translation_pairs(flores_dir)
    
    # Show samples
    show_samples(output_file, n=5)
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print(f"\n1. Upload '{output_file}' to Google Drive")
    print("\n2. In Colab Cell 3, use:")
    print("   data_folder = '/content/drive/MyDrive/'")
    print(f"   txt_files = [data_folder + '{output_file}']")
    print("\n3. Update Cell 2 config:")
    print("   'max_samples': " + str(count) + "  # Use all FLORES data")
    print("   'num_epochs': 5  # More epochs for small dataset")
    print("\n4. Train in Colab!")
    print("\n5. Test with test_colab_adapter.py")
    print("\n" + "="*80)
    print("\nEXPECTED IMPROVEMENT:")
    print("   - Proper translations instead of gibberish")
    print("   - Model follows 'Translate to [Lang]:' instructions")
    print("   - Much better than monolingual training!")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

