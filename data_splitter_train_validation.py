import os
import random
from collections import defaultdict
import re

# You'll need to install these packages:
# pip install langdetect polyglot
# For polyglot, you might also need: pip install pyicu pycld2

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("Warning: langdetect not available. Install with: pip install langdetect")
    LANGDETECT_AVAILABLE = False

def detect_language_simple(text):
    """
    Simple heuristic-based language detection for the specific languages
    """
    text = text.strip()
    if not text:
        return 'unknown'
    
    # Devanagari script detection (Hindi, Marathi, Sanskrit)
    devanagari_chars = re.findall(r'[\u0900-\u097F]', text)
    devanagari_ratio = len(devanagari_chars) / len(text) if len(text) > 0 else 0
    
    # Latin script detection (English)
    latin_chars = re.findall(r'[a-zA-Z]', text)
    latin_ratio = len(latin_chars) / len(text) if len(text) > 0 else 0
    
    if latin_ratio > 0.7:
        return 'en'
    elif devanagari_ratio > 0.3:
        # For Devanagari scripts, we'll need additional heuristics
        # Sanskrit often has more complex conjuncts and specific vocabulary
        # Hindi has more Persian/Urdu loanwords
        # Marathi has specific patterns
        
        # Simple heuristics (you might want to improve these)
        if any(word in text for word in ['और', 'का', 'के', 'की', 'में', 'से', 'को', 'है', 'हैं']):
            return 'hi'  # Common Hindi words
        elif any(word in text for word in ['आणि', 'च्या', 'मध्ये', 'आहे', 'होते']):
            return 'mr'  # Common Marathi words
        else:
            # Default to Sanskrit if Devanagari but no clear Hindi/Marathi markers
            return 'sa'
    else:
        return 'unknown'

def detect_language_advanced(text):
    """
    Use langdetect library if available
    """
    if not LANGDETECT_AVAILABLE:
        return detect_language_simple(text)
    
    try:
        detected = detect(text)
        # Map detected languages to our expected codes
        lang_mapping = {
            'en': 'en',
            'hi': 'hi',
            'mr': 'mr',
            'sa': 'sa',
            'ne': 'sa',  # Sometimes Sanskrit is detected as Nepali
        }
        return lang_mapping.get(detected, detect_language_simple(text))
    except LangDetectException:
        return detect_language_simple(text)

def analyze_corpus_structure(file_path, sample_size=1000):
    """
    Analyze the corpus to understand language distribution and boundaries
    """
    print(f"Analyzing corpus structure from {file_path}...")
    
    languages_detected = []
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            
            line = line.strip()
            if line:
                lang = detect_language_advanced(line)
                languages_detected.append((i, lang))
                line_count += 1
            
            if i % 100 == 0:
                print(f"Analyzed {i} lines...")
    
    # Find language boundaries
    language_segments = []
    current_lang = languages_detected[0][1] if languages_detected else 'unknown'
    segment_start = 0
    
    for line_num, lang in languages_detected:
        if lang != current_lang:
            language_segments.append((segment_start, line_num - 1, current_lang))
            segment_start = line_num
            current_lang = lang
    
    # Add the last segment
    if languages_detected:
        language_segments.append((segment_start, line_count - 1, current_lang))
    
    print(f"\nDetected language segments in first {sample_size} lines:")
    for start, end, lang in language_segments:
        print(f"Lines {start}-{end}: {lang}")
    
    return language_segments

def split_corpus_by_language(input_file, output_dir='output', train_ratio=0.8, sample_analysis=5000, 
                           balance_validation=False, min_val_samples=1000):
    """
    Split the corpus into language-specific training and validation files
    
    Args:
        balance_validation: If True, ensures each language gets at least min_val_samples in validation
        min_val_samples: Minimum number of validation samples per language (if balance_validation=True)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # First, analyze a sample to understand the structure
    print("Step 1: Analyzing corpus structure...")
    analyze_corpus_structure(input_file, sample_analysis)
    
    # Now process the entire file
    print("\nStep 2: Processing entire corpus...")
    
    language_lines = defaultdict(list)
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                lang = detect_language_advanced(line)
                language_lines[lang].append(line)
                total_lines += 1
            
            if i % 10000 == 0:
                print(f"Processed {i} lines...")
    
    print(f"\nTotal lines processed: {total_lines}")
    print("Language distribution:")
    for lang, lines in language_lines.items():
        print(f"  {lang}: {len(lines)} lines ({len(lines)/total_lines*100:.1f}%)")
    
    # Check if we have the expected 4 languages
    expected_langs = {'en', 'hi', 'mr', 'sa'}
    detected_langs = set(language_lines.keys())
    
    if not expected_langs.issubset(detected_langs):
        missing = expected_langs - detected_langs
        print(f"\n⚠️  Warning: Expected languages not detected: {missing}")
        print("This might indicate detection issues. Consider manual verification.")
    
    # Split each language into train/validation
    print(f"\nStep 3: Creating train/validation splits...")
    
    split_summary = {}
    
    for lang, lines in language_lines.items():
        if len(lines) < 10:  # Skip languages with very few samples
            print(f"Skipping {lang} - too few samples ({len(lines)})")
            continue
        
        # Shuffle lines for random split
        random.shuffle(lines)
        
        # Calculate split point
        if balance_validation and len(lines) > min_val_samples * 2:
            # Ensure minimum validation samples
            val_count = max(int(len(lines) * (1 - train_ratio)), min_val_samples)
            train_count = len(lines) - val_count
            actual_train_ratio = train_count / len(lines)
        else:
            # Standard ratio split
            train_count = int(len(lines) * train_ratio)
            val_count = len(lines) - train_count
            actual_train_ratio = train_ratio
        
        train_lines = lines[:train_count]
        val_lines = lines[train_count:train_count + val_count]
        
        # Write training file
        train_file = os.path.join(output_dir, f'{lang}_train.txt')
        with open(train_file, 'w', encoding='utf-8') as f:
            for line in train_lines:
                f.write(line + '\n')
        
        # Write validation file
        val_file = os.path.join(output_dir, f'{lang}_val.txt')
        with open(val_file, 'w', encoding='utf-8') as f:
            for line in val_lines:
                f.write(line + '\n')
        
        split_summary[lang] = {
            'total': len(lines),
            'train': len(train_lines),
            'val': len(val_lines),
            'train_ratio': actual_train_ratio
        }
        
        print(f"Created {lang}: {len(train_lines)} train, {len(val_lines)} val "
              f"(ratio: {actual_train_ratio:.2f})")
    
    # Print detailed summary
    print(f"\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    for lang, stats in split_summary.items():
        print(f"{lang.upper()}:")
        print(f"  Total: {stats['total']:,} lines")
        print(f"  Train: {stats['train']:,} lines ({stats['train_ratio']:.1%})")
        print(f"  Val:   {stats['val']:,} lines ({1-stats['train_ratio']:.1%})")
        print()
    
    print(f"Files created in '{output_dir}' directory:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.txt'):
            file_path = os.path.join(output_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"  {file}: {size_mb:.2f} MB ({line_count:,} lines)")
    
    # Validation balance check
    val_counts = {lang: stats['val'] for lang, stats in split_summary.items()}
    min_val = min(val_counts.values()) if val_counts else 0
    max_val = max(val_counts.values()) if val_counts else 0
    
    print(f"\nValidation set balance:")
    print(f"  Smallest: {min_val:,} lines")
    print(f"  Largest:  {max_val:,} lines")
    if min_val > 0:
        balance_ratio = min_val / max_val
        print(f"  Balance ratio: {balance_ratio:.2f} (1.0 = perfectly balanced)")
        if balance_ratio < 0.1:
            print("  ⚠️  Validation sets are very unbalanced!")
        elif balance_ratio > 0.5:
            print("  ✅ Validation sets are reasonably balanced")

def main():
    # Configuration
    INPUT_FILE = 'data/multilingual_corpus.txt'  # Change this to your file path
    OUTPUT_DIR = 'language_splits'
    TRAIN_RATIO = 0.8  # 80% train, 20% validation
    SAMPLE_SIZE = 5000  # Number of lines to analyze for structure
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    print("Language Detection and Corpus Splitter")
    print("=" * 50)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train/Val ratio: {TRAIN_RATIO:.1f}")
    print(f"Sample analysis size: {SAMPLE_SIZE}")
    print()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        print("Please update the INPUT_FILE variable with the correct path.")
        return
    
    try:
        split_corpus_by_language(
            INPUT_FILE, 
            OUTPUT_DIR, 
            TRAIN_RATIO, 
            SAMPLE_SIZE,
            balance_validation=True,  # Set to True for more balanced validation sets
            min_val_samples=1000      # Minimum validation samples per language
        )
        print("\n✅ Corpus splitting completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your input file and try again.")

if __name__ == "__main__":
    main()