#!/usr/bin/env python3
"""
Complete Multilingual Tokenization Pipeline

This script runs the complete pipeline from data collection to model training:
1. Collect data from external sources (Wikipedia, etc.)
2. Extract and clean text from various formats (.xml.bz2, etc.)
3. Process through MCP pipeline
4. Train multilingual tokenizer
5. Fine-tune language model

Usage:
    python run_complete_pipeline.py                    # Run with sample data
    python run_complete_pipeline.py --external         # Download real data
    python run_complete_pipeline.py --extract-only     # Only extract Wikipedia data
    python run_complete_pipeline.py --tokenizer-only   # Only train tokenizer
    python run_complete_pipeline.py --finetune-only    # Only fine-tune model
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config import settings
from src.data_processing.corpus_collector import CorpusCollector
from src.data_processing.mcp_pipeline import MCPPipeline
from src.training.train_multilingual_tokenizer import MultilingualTokenizerTrainer
from src.training.fine_tune import main as fine_tune_main

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_collection(use_external_sources: bool = False):
    """Run data collection step"""
    logger.info("=" * 80)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 80)
    
    collector = CorpusCollector()
    collector.collect_corpora(use_external_sources=use_external_sources)
    
    logger.info("✅ Data collection completed!")
    return True

def run_wikipedia_extraction():
    """Run Wikipedia extraction step"""
    logger.info("=" * 80)
    logger.info("STEP 2: WIKIPEDIA EXTRACTION")
    logger.info("=" * 80)
    
    try:
        from src.data_processing.wikipedia_extractor import WikipediaExtractor
        extractor = WikipediaExtractor()
        all_stats = extractor.extract_all_wikipedia_dumps(max_articles_per_language=5000)
        extractor.print_extraction_summary(all_stats)
        
        logger.info("✅ Wikipedia extraction completed!")
        return True
    except Exception as e:
        logger.error(f"Wikipedia extraction failed: {e}")
        return False

def run_mcp_preprocessing():
    """Run MCP preprocessing step"""
    logger.info("=" * 80)
    logger.info("STEP 3: MCP PREPROCESSING")
    logger.info("=" * 80)
    
    try:
        mcp = MCPPipeline()
        processed_file = mcp.process_all_corpora()
        mcp.print_statistics()
        
        logger.info(f"✅ MCP preprocessing completed! Processed file: {processed_file}")
        return processed_file
    except Exception as e:
        logger.error(f"MCP preprocessing failed: {e}")
        return None

def run_tokenizer_training(processed_file: str = None):
    """Run tokenizer training step"""
    logger.info("=" * 80)
    logger.info("STEP 4: TOKENIZER TRAINING")
    logger.info("=" * 80)
    
    try:
        trainer = MultilingualTokenizerTrainer()
        
        if processed_file and os.path.exists(processed_file):
            logger.info(f"Using processed file: {processed_file}")
            model_path, vocab_path = trainer.train_tokenizer(processed_file)
        else:
            logger.info("No processed file found, collecting and preprocessing data...")
            processed_file = trainer.collect_and_preprocess_data(use_external_sources=False)
            model_path, vocab_path = trainer.train_tokenizer(processed_file)
        
        # Validate tokenizer
        validation_scores = trainer.validate_tokenizer(model_path)
        switching_results = trainer.test_language_switching(model_path)
        
        trainer.print_training_summary()
        
        logger.info("✅ Tokenizer training completed!")
        return model_path, vocab_path
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}")
        return None, None

def run_model_finetuning():
    """Run model fine-tuning step"""
    logger.info("=" * 80)
    logger.info("STEP 5: MODEL FINE-TUNING")
    logger.info("=" * 80)
    
    try:
        # Run the fine-tuning script
        fine_tune_main()
        
        logger.info("✅ Model fine-tuning completed!")
        return True
    except Exception as e:
        logger.error(f"Model fine-tuning failed: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    # Check if data directories exist
    if not os.path.exists(settings.TRAINING_DATA_PATH):
        logger.error(f"Training data directory not found: {settings.TRAINING_DATA_PATH}")
        return False
    
    if not os.path.exists(settings.VALIDATION_DATA_PATH):
        logger.error(f"Validation data directory not found: {settings.VALIDATION_DATA_PATH}")
        return False
    
    # Check if required packages are installed
    required_packages = [
        'sentencepiece', 'transformers', 'torch', 'datasets', 
        'peft', 'bitsandbytes', 'requests', 'bz2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All prerequisites met!")
    return True

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description="Run complete multilingual tokenization pipeline")
    parser.add_argument("--external", action="store_true",
                       help="Download data from external sources (Wikipedia, etc.)")
    parser.add_argument("--extract-only", action="store_true",
                       help="Only run Wikipedia extraction step")
    parser.add_argument("--tokenizer-only", action="store_true",
                       help="Only run tokenizer training step")
    parser.add_argument("--finetune-only", action="store_true",
                       help="Only run model fine-tuning step")
    parser.add_argument("--skip-collection", action="store_true",
                       help="Skip data collection step")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip Wikipedia extraction step")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip MCP preprocessing step")
    parser.add_argument("--skip-tokenizer", action="store_true",
                       help="Skip tokenizer training step")
    parser.add_argument("--skip-finetune", action="store_true",
                       help="Skip model fine-tuning step")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        sys.exit(1)
    
    logger.info("🚀 Starting Complete Multilingual Tokenization Pipeline")
    logger.info(f"External sources: {'Yes' if args.external else 'No'}")
    logger.info(f"Steps to run: {[step for step in ['collection', 'extraction', 'preprocessing', 'tokenizer', 'finetune'] if not getattr(args, f'skip_{step}', False)]}")
    
    success = True
    processed_file = None
    
    try:
        # Step 1: Data Collection
        if not args.skip_collection and not args.extract_only and not args.tokenizer_only and not args.finetune_only:
            success = run_data_collection(use_external_sources=args.external)
            if not success:
                logger.error("Data collection failed. Exiting.")
                sys.exit(1)
        
        # Step 2: Wikipedia Extraction
        if not args.skip_extraction and not args.tokenizer_only and not args.finetune_only:
            if args.extract_only:
                success = run_wikipedia_extraction()
            else:
                success = run_wikipedia_extraction()
            if not success:
                logger.warning("Wikipedia extraction failed, but continuing with sample data...")
        
        # Step 3: MCP Preprocessing
        if not args.skip_preprocessing and not args.tokenizer_only and not args.finetune_only:
            processed_file = run_mcp_preprocessing()
            if not processed_file:
                logger.error("MCP preprocessing failed. Exiting.")
                sys.exit(1)
        
        # Step 4: Tokenizer Training
        if not args.skip_tokenizer and not args.finetune_only:
            model_path, vocab_path = run_tokenizer_training(processed_file)
            if not model_path:
                logger.error("Tokenizer training failed. Exiting.")
                sys.exit(1)
        
        # Step 5: Model Fine-tuning
        if not args.skip_finetune:
            success = run_model_finetuning()
            if not success:
                logger.error("Model fine-tuning failed. Exiting.")
                sys.exit(1)
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Test the API: python main.py")
        logger.info("2. Run evaluation: python src/evaluation/metrics.py")
        logger.info("3. Check logs: tail -f logs/api.log")
        logger.info("\nYour multilingual tokenization model is ready! 🚀")
        
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
