"""
Multilingual Tokenizer Training Script for 21 Indian Languages

This script implements the complete training pipeline as specified in the requirements:
1. Use MCP pipeline for data preprocessing
2. Train multilingual tokenizer on combined 21 language dataset
3. Validate tokenization quality across all languages
4. Prepare for large-scale Gurukul integration

Based on the requirements:
- Support for 21 Indian languages
- MCP preprocessing for robust data handling
- SentencePiece training with proper Unicode normalization
- Integration with Indigenous NLP + Vaani TTS
"""

import os
import sys
import logging
import sentencepiece as spm
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import MCP pipeline and settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings
from src.data_processing.mcp_pipeline import MCPPipeline
from src.data_processing.corpus_collector import CorpusCollector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualTokenizerTrainer:
    """
    Advanced multilingual tokenizer trainer for 21 Indian languages
    
    This class implements the complete training pipeline:
    1. Corpus collection and MCP preprocessing
    2. Multilingual tokenizer training
    3. Quality validation across all languages
    4. Integration preparation
    """
    
    def __init__(self):
        self.stats = {
            "total_sentences": 0,
            "language_counts": {},
            "script_counts": {},
            "vocab_size": 0,
            "training_time": 0,
            "validation_scores": {}
        }
        
        # Ensure directories exist
        settings.create_directories()
        
        # Initialize MCP pipeline
        self.mcp_pipeline = MCPPipeline()
        
        # Initialize corpus collector
        self.corpus_collector = CorpusCollector()
    
    def collect_and_preprocess_data(self, use_external_sources: bool = False) -> str:
        """
        Collect and preprocess data using MCP pipeline
        
        Args:
            use_external_sources: Whether to download from external sources
            
        Returns:
            Path to processed data file
        """
        logger.info("Starting data collection and preprocessing...")
        
        # Step 1: Collect corpora
        logger.info("Step 1: Collecting corpora...")
        self.corpus_collector.collect_corpora(use_external_sources=use_external_sources)
        
        # Step 2: Run MCP preprocessing
        logger.info("Step 2: Running MCP preprocessing...")
        processed_file = self.mcp_pipeline.process_all_corpora()
        
        # Update stats
        self.stats.update(self.mcp_pipeline.stats)
        
        logger.info(f"Data preprocessing completed. Processed file: {processed_file}")
        return processed_file
    
    def train_tokenizer(self, processed_file: str) -> Tuple[str, str]:
        """
        Train multilingual tokenizer using SentencePiece
        
        Args:
            processed_file: Path to processed training data
            
        Returns:
            Tuple of (model_path, vocab_path)
        """
        logger.info("Starting multilingual tokenizer training...")
        
        # Ensure model directory exists
        os.makedirs(settings.FINE_TUNED_MODEL_PATH, exist_ok=True)
        
        # Output paths
        model_prefix = os.path.join(settings.FINE_TUNED_MODEL_PATH, "multilingual_tokenizer")
        model_path = f"{model_prefix}.model"
        vocab_path = f"{model_prefix}.vocab"
        
        # Enhanced SentencePiece training arguments for 21 languages
        spm_args = [
            f'--input={processed_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={settings.SP_VOCAB_SIZE}',
            f'--model_type={settings.SP_MODEL_TYPE}',
            f'--character_coverage={settings.SP_CHARACTER_COVERAGE}',
            f'--input_sentence_size={min(settings.SP_INPUT_SENTENCE_SIZE, self.stats["total_sentences"])}',
            '--shuffle_input_sentence=true' if settings.SP_SHUFFLE_INPUT_SENTENCE else '--shuffle_input_sentence=false',
            
            # Multilingual-specific settings
            '--split_by_unicode_script=false',  # Don't split scripts for better multilingual support
            '--split_by_whitespace=true',
            '--split_by_number=true',
            '--treat_whitespace_as_suffix=false',
            '--allow_whitespace_only_pieces=true',
            '--split_digits=false',  # Keep numbers intact
            '--byte_fallback=true',  # Handle unknown characters gracefully
            
            # Unicode normalization for Indian languages
            '--normalization_rule_name=nfkc',  # Use NFKC normalization
            '--remove_extra_whitespaces=true',
            '--add_dummy_prefix=false',  # Don't add dummy prefix for better multilingual support
            
            # Enhanced settings for 21 languages
            '--hard_vocab_limit=false',  # Allow flexible vocabulary size
        ]
        
        # Train the tokenizer
        try:
            import time
            start_time = time.time()
            
            logger.info("Training SentencePiece tokenizer...")
            logger.info(f"Training arguments: {' '.join(spm_args)}")
            
            spm.SentencePieceTrainer.train(' '.join(spm_args))
            
            training_time = time.time() - start_time
            self.stats["training_time"] = training_time
            
            logger.info(f"Tokenizer training completed successfully!")
            logger.info(f"Training time: {training_time:.2f} seconds")
            logger.info(f"Model saved as: {model_path}")
            logger.info(f"Vocabulary saved as: {vocab_path}")
            
            # Get vocabulary size
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            self.stats["vocab_size"] = sp.get_piece_size()
            
            return model_path, vocab_path
            
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")
            raise
    
    def validate_tokenizer(self, model_path: str) -> Dict[str, Dict[str, float]]:
        """
        Validate tokenizer quality across all languages
        
        Args:
            model_path: Path to trained tokenizer model
            
        Returns:
            Validation scores for each language
        """
        logger.info("Validating tokenizer across all languages...")
        
        # Load the tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        validation_scores = {}
        
        # Test sentences for all 21 languages
        test_sentences = {
            "assamese": "নমস্কাৰ, আপুনি কেনেকৈ আছে? মই অসমীয়া শিকি আছোঁ।",
            "bengali": "নমস্কার, আপনি কেমন আছেন? আমি বাংলা শিখছি।",
            "bodo": "जायो, नों माब्ला आं? आं बड़ो खालाम लागोन आं।",
            "english": "Hello, how are you? I am learning English.",
            "gujurati": "નમસ્કાર, તમે કેમ છો? હું ગુજરાતી શીખી રહ્યો છું।",
            "hindi": "नमस्ते, आप कैसे हैं? मैं हिंदी सीख रहा हूं।",
            "kannada": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ? ನಾನು ಕನ್ನಡ ಕಲಿಯುತ್ತಿದ್ದೇನೆ।",
            "kashmiri": "السلام علیکم، تہ کیوی چھیو? می کٲشُر کران چھوس۔",
            "maithili": "प्रणाम, अहाँ कहाँ छी? हम मैथिली सीख रहल छी।",
            "malyalam": "നമസ്കാരം, നിങ്ങൾ എങ്ങനെയുണ്ട്? ഞാൻ മലയാളം പഠിക്കുന്നു।",
            "marathi": "नमस्कार, तुम्ही कसे आहात? मी मराठी शिकत आहे।",
            "meitei": "খুরুমজরি, নুং কিদা লৈরিবনো? ঈনা মৈতৈলোন নরবা লৈরিবদি।",
            "nepali": "नमस्कार, तपाईं कसरी हुनुहुन्छ? म नेपाली सिक्दै छु।",
            "odia": "ନମସ୍କାର, ଆପଣ କିପରି ଅଛନ୍ତି? ମୁଁ ଓଡ଼ିଆ ଶିଖୁଛି।",
            "punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ? ਮੈਂ ਪੰਜਾਬੀ ਸਿੱਖ ਰਿਹਾ ਹਾਂ।",
            "sanskrit": "नमस्कारः, भवान् कथं वर्तते? अहं संस्कृतं पठामि।",
            "santali": "ᱡᱚᱦᱟᱨ, ᱟᱢ ᱠᱮᱢᱚᱱ ᱢᱮᱱᱟᱢ? ᱟᱢ ᱥᱟᱱᱛᱟᱲᱤ ᱪᱮᱫᱟᱜ ᱠᱟᱱᱟᱢ।",
            "sindhi": "السلام علیڪم، تون ڪيئن آهين؟ مان سنڌي سکي رهيو آهيان۔",
            "tamil": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? நான் தமிழ் கற்கிறேன்।",
            "telugu": "నమస్కారం, మీరు ఎలా ఉన్నారు? నేను తెలుగు నేర్చుకుంటున్నాను।",
            "urdu": "السلام علیکم، آپ کیسے ہیں؟ میں اردو سیکھ رہا ہوں۔"
        }
        
        for language, sentence in test_sentences.items():
            if language not in settings.SUPPORTED_LANGUAGES:
                continue
                
            try:
                # Tokenize
                tokens = sp.encode_as_pieces(sentence)
                token_ids = sp.encode_as_ids(sentence)
                
                # Detokenize
                detokenized = sp.decode_pieces(tokens)
                
                # Calculate metrics
                token_count = len(tokens)
                unique_tokens = len(set(tokens))
                token_diversity = unique_tokens / token_count if token_count > 0 else 0
                
                # Check for lossless reconstruction
                is_lossless = sentence.strip() == detokenized.strip()
                
                # Calculate compression ratio
                original_chars = len(sentence)
                token_chars = sum(len(token) for token in tokens)
                compression_ratio = token_chars / original_chars if original_chars > 0 else 1.0
                
                validation_scores[language] = {
                    "token_count": token_count,
                    "unique_tokens": unique_tokens,
                    "token_diversity": token_diversity,
                    "is_lossless": is_lossless,
                    "compression_ratio": compression_ratio,
                    "original_text": sentence,
                    "detokenized_text": detokenized
                }
                
                logger.info(f"{language}: {token_count} tokens, diversity={token_diversity:.3f}, lossless={is_lossless}")
                
            except Exception as e:
                logger.error(f"Error validating {language}: {e}")
                validation_scores[language] = {
                    "error": str(e),
                    "token_count": 0,
                    "unique_tokens": 0,
                    "token_diversity": 0,
                    "is_lossless": False,
                    "compression_ratio": 1.0
                }
        
        self.stats["validation_scores"] = validation_scores
        return validation_scores
    
    def test_language_switching(self, model_path: str) -> Dict[str, float]:
        """
        Test language switching capability
        
        Args:
            model_path: Path to trained tokenizer model
            
        Returns:
            Language switching test results
        """
        logger.info("Testing language switching capability...")
        
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        # Test mixed-language sentences across multiple scripts
        mixed_sentences = [
            "Hello, नमस्ते, வணக்கம் - how are you?",
            "मैं learning Tamil and தமிழ் at the same time",
            "This is a test of multilingual tokenization across scripts",
            "संस्कृत, English, and தமிழ் in one sentence",
            "Testing 21 languages: हिंदी, தமிழ், తెలుగు, ಕನ್ನಡ, বাংলা",
            "ਪੰਜਾਬੀ, ગુજરાતી, മലയാളം, ଓଡ଼ିଆ mixing together",
            "اردو and English in one sentence with نेपाली",
            "Bodo बड़ो, Santali ᱥᱟᱱᱛᱟᱲᱤ, Meitei মৈতৈলোন together"
        ]
        
        switching_results = {}
        
        for i, sentence in enumerate(mixed_sentences):
            try:
                tokens = sp.encode_as_pieces(sentence)
                detokenized = sp.decode_pieces(tokens)
                
                # Check if all scripts are preserved
                scripts_preserved = True
                for char in sentence:
                    if char.isalpha():
                        # Check if character is preserved in detokenized text
                        if char not in detokenized:
                            scripts_preserved = False
                            break
                
                switching_results[f"mixed_sentence_{i+1}"] = {
                    "original": sentence,
                    "tokens": tokens,
                    "detokenized": detokenized,
                    "scripts_preserved": scripts_preserved,
                    "token_count": len(tokens)
                }
                
                logger.info(f"Mixed sentence {i+1}: {len(tokens)} tokens, scripts_preserved={scripts_preserved}")
                
            except Exception as e:
                logger.error(f"Error testing mixed sentence {i+1}: {e}")
                switching_results[f"mixed_sentence_{i+1}"] = {
                    "error": str(e),
                    "scripts_preserved": False
                }
        
        return switching_results
    
    def print_training_summary(self):
        """Print comprehensive training summary"""
        logger.info("\n" + "=" * 80)
        logger.info("MULTILINGUAL TOKENIZER TRAINING SUMMARY - 21 LANGUAGES")
        logger.info("=" * 80)
        
        logger.info(f"Total sentences processed: {self.stats['total_sentences']:,}")
        logger.info(f"Vocabulary size: {self.stats['vocab_size']:,}")
        logger.info(f"Training time: {self.stats['training_time']:.2f} seconds")
        
        logger.info(f"\nLanguage distribution:")
        for lang, count in sorted(self.stats['language_counts'].items()):
            percentage = (count / self.stats['total_sentences']) * 100 if self.stats['total_sentences'] > 0 else 0
            logger.info(f"  {lang}: {count:,} sentences ({percentage:.1f}%)")
        
        logger.info(f"\nScript distribution:")
        for script, count in sorted(self.stats['script_counts'].items()):
            percentage = (count / self.stats['total_sentences']) * 100 if self.stats['total_sentences'] > 0 else 0
            logger.info(f"  {script}: {count:,} sentences ({percentage:.1f}%)")
        
        if self.stats['validation_scores']:
            logger.info(f"\nValidation Results (21 Languages):")
            logger.info("-" * 50)
            
            for lang, scores in self.stats['validation_scores'].items():
                if 'error' not in scores:
                    logger.info(f"{lang.upper()}:")
                    logger.info(f"  Tokens: {scores['token_count']}")
                    logger.info(f"  Diversity: {scores['token_diversity']:.3f}")
                    logger.info(f"  Lossless: {'✓' if scores['is_lossless'] else '✗'}")
                    logger.info(f"  Compression: {scores['compression_ratio']:.3f}")
                else:
                    logger.info(f"{lang.upper()}: ERROR - {scores['error']}")
        
        logger.info("=" * 80)
        logger.info("\nSupported Languages (21):")
        logger.info(", ".join(settings.SUPPORTED_LANGUAGES))
        logger.info("=" * 80)
    
    def cleanup(self):
        """Clean up temporary files"""
        self.mcp_pipeline.cleanup()


def main():
    """Main function to run multilingual tokenizer training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train multilingual tokenizer for 21 Indian languages")
    parser.add_argument("--external-data", action="store_true",
                       help="Attempt to download external data sources")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip data preprocessing (use existing processed data)")
    parser.add_argument("--processed-file", type=str,
                       help="Path to existing processed data file")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultilingualTokenizerTrainer()
    
    try:
        # Step 1: Collect and preprocess data
        if args.skip_preprocessing and args.processed_file:
            processed_file = args.processed_file
            logger.info(f"Using existing processed file: {processed_file}")
        else:
            processed_file = trainer.collect_and_preprocess_data(use_external_sources=args.external_data)
        
        # Step 2: Train tokenizer
        model_path, vocab_path = trainer.train_tokenizer(processed_file)
        
        # Step 3: Validate tokenizer
        validation_scores = trainer.validate_tokenizer(model_path)
        
        # Step 4: Test language switching
        switching_results = trainer.test_language_switching(model_path)
        
        # Step 5: Print summary
        trainer.print_training_summary()
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model path: {model_path}")
        logger.info(f"Vocabulary path: {vocab_path}")
        logger.info("\nNext steps:")
        logger.info("1. Update settings.py with new tokenizer paths")
        logger.info("2. Run fine-tuning: python src/training/fine_tune.py")
        logger.info("3. Start API: python main.py")
        logger.info("4. Test with: python src/evaluation/metrics.py")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()