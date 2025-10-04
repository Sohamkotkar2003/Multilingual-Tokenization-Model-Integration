#!/usr/bin/env python3
"""
Data Status Checker

This script checks what data is available and in what format,
helping you understand what you can do with your current setup.
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_files():
    """Check what data files are available"""
    logger.info("=" * 80)
    logger.info("DATA STATUS CHECK")
    logger.info("=" * 80)
    
    # Check training data directory
    training_dir = Path(settings.TRAINING_DATA_PATH)
    if not training_dir.exists():
        logger.error(f"Training data directory not found: {training_dir}")
        return False
    
    logger.info(f"Training data directory: {training_dir}")
    
    # Check for different file types
    file_types = {
        "Text files (.txt)": list(training_dir.glob("*.txt")),
        "XML files (.xml)": list(training_dir.glob("*.xml")),
        "Compressed XML (.xml.bz2)": list(training_dir.glob("*.xml.bz2")),
        "Other files": [f for f in training_dir.iterdir() if f.is_file() and not f.suffix in ['.txt', '.xml', '.bz2']]
    }
    
    total_files = 0
    for file_type, files in file_types.items():
        if files:
            logger.info(f"\n{file_type}: {len(files)} files")
            for file in files[:5]:  # Show first 5 files
                size_mb = file.stat().st_size / (1024 * 1024)
                logger.info(f"  - {file.name} ({size_mb:.2f} MB)")
            if len(files) > 5:
                logger.info(f"  ... and {len(files) - 5} more files")
            total_files += len(files)
        else:
            logger.info(f"\n{file_type}: No files found")
    
    logger.info(f"\nTotal files in training directory: {total_files}")
    
    # Check validation data directory
    validation_dir = Path(settings.VALIDATION_DATA_PATH)
    if validation_dir.exists():
        val_files = list(validation_dir.glob("*.txt"))
        logger.info(f"\nValidation data: {len(val_files)} files")
    else:
        logger.info(f"\nValidation data directory not found: {validation_dir}")
    
    return total_files > 0

def check_sample_data_quality():
    """Check the quality of sample data"""
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE DATA QUALITY CHECK")
    logger.info("=" * 80)
    
    training_dir = Path(settings.TRAINING_DATA_PATH)
    txt_files = list(training_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning("No .txt files found for quality check")
        return
    
    for txt_file in txt_files[:3]:  # Check first 3 files
        logger.info(f"\nChecking {txt_file.name}:")
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            logger.info(f"  Total lines: {len(lines)}")
            
            # Check line lengths
            line_lengths = [len(line.strip()) for line in lines if line.strip()]
            if line_lengths:
                avg_length = sum(line_lengths) / len(line_lengths)
                min_length = min(line_lengths)
                max_length = max(line_lengths)
                logger.info(f"  Average line length: {avg_length:.1f} characters")
                logger.info(f"  Min/Max line length: {min_length}/{max_length} characters")
            
            # Show sample content
            sample_lines = [line.strip() for line in lines[:3] if line.strip()]
            logger.info(f"  Sample content:")
            for i, line in enumerate(sample_lines, 1):
                preview = line[:100] + "..." if len(line) > 100 else line
                logger.info(f"    {i}. {preview}")
                
        except Exception as e:
            logger.error(f"  Error reading {txt_file.name}: {e}")

def check_training_readiness():
    """Check if the data is ready for training"""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING READINESS CHECK")
    logger.info("=" * 80)
    
    # Check if we have the required corpus files
    required_files = list(settings.CORPUS_FILES.values())
    training_dir = Path(settings.TRAINING_DATA_PATH)
    
    available_files = []
    missing_files = []
    
    for filename in required_files:
        filepath = training_dir / filename
        if filepath.exists():
            available_files.append(filename)
        else:
            missing_files.append(filename)
    
    logger.info(f"Required corpus files: {len(required_files)}")
    logger.info(f"Available files: {len(available_files)}")
    logger.info(f"Missing files: {len(missing_files)}")
    
    if available_files:
        logger.info(f"\nAvailable files:")
        for filename in available_files:
            filepath = training_dir / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ {filename} ({size_mb:.2f} MB)")
    
    if missing_files:
        logger.info(f"\nMissing files:")
        for filename in missing_files:
            logger.info(f"  ❌ {filename}")
    
    # Check if we can start training
    if len(available_files) >= len(required_files) * 0.5:  # At least 50% of files
        logger.info(f"\n✅ READY FOR TRAINING!")
        logger.info(f"You have {len(available_files)}/{len(required_files)} required files.")
        logger.info("You can start training with:")
        logger.info("  python src/training/train_multilingual_tokenizer.py")
        logger.info("  python src/training/fine_tune.py")
    else:
        logger.info(f"\n⚠️  PARTIALLY READY FOR TRAINING")
        logger.info(f"You have {len(available_files)}/{len(required_files)} required files.")
        logger.info("Consider running data collection first:")
        logger.info("  python src/data_processing/corpus_collector.py --external")

def check_xml_files():
    """Check if there are XML files that need extraction"""
    logger.info("\n" + "=" * 80)
    logger.info("XML FILES CHECK")
    logger.info("=" * 80)
    
    training_dir = Path(settings.TRAINING_DATA_PATH)
    xml_files = list(training_dir.glob("*.xml.bz2"))
    
    if xml_files:
        logger.info(f"Found {len(xml_files)} compressed XML files:")
        for xml_file in xml_files:
            size_mb = xml_file.stat().st_size / (1024 * 1024)
            logger.info(f"  - {xml_file.name} ({size_mb:.2f} MB)")
        
        logger.info(f"\nThese files need to be extracted before training.")
        logger.info("Run Wikipedia extraction:")
        logger.info("  python src/data_processing/wikipedia_extractor.py")
        logger.info("Or run the complete pipeline:")
        logger.info("  python run_complete_pipeline.py --external")
    else:
        logger.info("No compressed XML files found.")
        logger.info("If you want to use real Wikipedia data, run:")
        logger.info("  python src/data_processing/corpus_collector.py --external")

def main():
    """Main function"""
    logger.info("🔍 Checking your multilingual tokenization data status...")
    
    # Check if data directories exist
    if not os.path.exists(settings.TRAINING_DATA_PATH):
        logger.error(f"Training data directory not found: {settings.TRAINING_DATA_PATH}")
        logger.info("Please run: python src/data_processing/corpus_collector.py")
        return
    
    # Run all checks
    has_data = check_data_files()
    check_sample_data_quality()
    check_training_readiness()
    check_xml_files()
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    if has_data:
        logger.info("✅ You have data available for training!")
        logger.info("\nRecommended next steps:")
        logger.info("1. If you have .xml.bz2 files: python run_complete_pipeline.py --external")
        logger.info("2. If you only have .txt files: python src/training/train_multilingual_tokenizer.py")
        logger.info("3. To fine-tune a model: python src/training/fine_tune.py")
    else:
        logger.info("❌ No data found for training.")
        logger.info("\nPlease run data collection first:")
        logger.info("python src/data_processing/corpus_collector.py --external")

if __name__ == "__main__":
    main()
