"""
SentencePiece Tokenizer Training Script for Multilingual Support

This script trains a custom SentencePiece tokenizer on Hindi, Sanskrit, Marathi, and English
text data with proper handling of Devanagari scripts and ligatures.

Usage:
    python train_tokenizer.py

The script will:
1. Collect training data from all language files
2. Apply proper Unicode normalization for Devanagari
3. Train a BPE tokenizer with appropriate vocabulary size
4. Save the tokenizer model for use with the API

Requirements:
    - Training data files in data/training/ directory
    - sentencepiece library installed
"""

import os
import logging
import unicodedata
import sentencepiece as spm
from typing import List, Dict
import tempfile
from core import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualTokenizerTrainer:
    def __init__(self):
        self.temp_combined_file = None
        self.stats = {
            "total_sentences": 0,
            "language_counts": {},
            "devanagari_sentences": 0,
            "latin_sentences": 0
        }

    def normalize_devanagari_text(self, text: str) -> str:
        """
        Apply proper Unicode normalization for Devanagari text
        This ensures consistent handling of ligatures and combining characters
        """
        # Apply NFC normalization to combine base characters with diacritics
        normalized = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters that can cause tokenization issues
        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            normalized = normalized.replace(char, '')
        
        return normalized.strip()

    def detect_script(self, text: str) -> str:
        """Detect the primary script of the text"""
        devanagari_count = sum(1 for c in text if 
                             settings.DEVANAGARI_UNICODE_RANGE[0] <= ord(c) <= settings.DEVANAGARI_UNICODE_RANGE[1])
        latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return "mixed"
        
        if devanagari_count / total_alpha > 0.5:
            return "devanagari"
        elif latin_count / total_alpha > 0.5:
            return "latin"
        else:
            return "mixed"

    def prepare_training_data(self) -> str:
        """
        Collect and prepare training data from all language files
        Returns: Path to the combined training file
        """
        logger.info("Preparing multilingual training data...")
        
        # Create temporary file for combined training data
        temp_fd, self.temp_combined_file = tempfile.mkstemp(suffix='.txt', prefix='multilingual_training_')
        
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as outfile:
            for lang, filename in settings.CORPUS_FILES.items():
                filepath = os.path.join(settings.TRAINING_DATA_PATH, filename)
                
                if not os.path.exists(filepath):
                    logger.warning(f"Training file not found: {filepath}")
                    
                    # Create sample data if file doesn't exist
                    sample_data = self.create_sample_data(lang)
                    os.makedirs(settings.TRAINING_DATA_PATH, exist_ok=True)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(sample_data)
                    logger.info(f"Created sample data for {lang}")
                
                # Process the file
                lang_sentence_count = 0
                with open(filepath, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if len(line) < 10:  # Skip very short lines
                            continue
                        
                        # Apply normalization (especially important for Devanagari)
                        normalized_line = self.normalize_devanagari_text(line)
                        
                        if len(normalized_line) < 10:
                            continue
                        
                        # Write to combined file
                        outfile.write(normalized_line + '\n')
                        lang_sentence_count += 1
                        self.stats["total_sentences"] += 1
                        
                        # Update script statistics
                        script = self.detect_script(normalized_line)
                        if script == "devanagari":
                            self.stats["devanagari_sentences"] += 1
                        elif script == "latin":
                            self.stats["latin_sentences"] += 1
                
                self.stats["language_counts"][lang] = lang_sentence_count
                logger.info(f"Loaded {lang_sentence_count} sentences from {lang}")
        
        logger.info(f"Total training sentences: {self.stats['total_sentences']}")
        logger.info(f"Devanagari sentences: {self.stats['devanagari_sentences']}")
        logger.info(f"Latin sentences: {self.stats['latin_sentences']}")
        
        return self.temp_combined_file

    def create_sample_data(self, language: str) -> str:
        """Create sample training data if files don't exist"""
        sample_data = {
            "hindi": """
नमस्ते, आप कैसे हैं?
मैं हिंदी भाषा सीख रहा हूं।
भारत एक बहुत सुंदर देश है।
यह पुस्तक बहुत अच्छी है।
हमें अपनी भाषा पर गर्व होना चाहिए।
शिक्षा सबसे महत्वपूर्ण है।
स्वतंत्रता दिवस एक राष्ट्रीय त्योहार है।
गणित एक कठिन विषय हो सकता है।
प्रकृति की सुंदरता अद्भुत है।
हमें पर्यावरण की रक्षा करनी चाहिए।
            """.strip(),
            
            "sanskrit": """
नमस्कार, कथं वर्तसे?
संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति।
वेदाः संस्कृते लिखिताः सन्ति।
धर्मो रक्षति रक्षितः।
सत्यं शिवं सुन्दरम्।
विद्या ददाति विनयं।
यत्र नार्यस्तु पूज्यन्ते रमन्ते तत्र देवताः।
सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।
वसुधैव कुटुम्बकम्।
अहिंसा परमो धर्मः।
            """.strip(),
            
            "marathi": """
नमस्कार, तुम्ही कसे आहात?
मराठी ही महाराष्ट्राची भाषा आहे।
मला मराठी भाषा आवडते।
शिक्षण हा सर्वात महत्वाचा मुद्दा आहे।
पुस्तक वाचणे हा चांगला सवय आहे।
निसर्गाचे संरक्षण करणे आवश्यक आहे।
सणउत्सव आपल्या संस्कृतीचा भाग आहेत।
कष्ट केल्याशिवाय काहीही मिळत नाही।
एकता आणि अखंडता महत्वपूर्ण आहे।
सत्य आणि न्याय हेच खरे मूल्य आहेत।
            """.strip(),
            
            "english": """
Hello, how are you today?
Learning multiple languages is beneficial.
Education is the foundation of progress.
Technology has transformed our lives.
Reading books expands our knowledge.
Environmental protection is crucial.
Cultural diversity makes us stronger.
Hard work leads to success.
Respect for others is important.
Science and innovation drive development.
            """.strip()
        }
        
        return sample_data.get(language, "Sample training data for multilingual tokenizer.")

    def train_tokenizer(self, training_file: str):
        """Train the SentencePiece tokenizer"""
        logger.info("Training SentencePiece tokenizer...")
        
        # Ensure model directory exists
        os.makedirs(settings.FINE_TUNED_MODEL_PATH, exist_ok=True)
        
        # Output paths
        model_prefix = os.path.join(settings.FINE_TUNED_MODEL_PATH, "multi_tokenizer")
        
        # SentencePiece training arguments
        spm_args = [
            f'--input={training_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={settings.SP_VOCAB_SIZE}',
            f'--model_type={settings.SP_MODEL_TYPE}',
            f'--character_coverage={settings.SP_CHARACTER_COVERAGE}',
            f'--input_sentence_size={min(settings.SP_INPUT_SENTENCE_SIZE, self.stats["total_sentences"])}',
            '--shuffle_input_sentence=true' if settings.SP_SHUFFLE_INPUT_SENTENCE else '--shuffle_input_sentence=false',
            '--split_by_unicode_script=false',  # Don't split scripts to handle multilingual better
            '--split_by_whitespace=true',
            '--split_by_number=true',
            '--treat_whitespace_as_suffix=false',
            '--allow_whitespace_only_pieces=true',
            '--split_digits=false',  # Keep numbers intact
            '--byte_fallback=true',  # Handle unknown characters gracefully
            # Devanagari-specific settings
            '--normalization_rule_name=nfc',  # Use NFC normalization
            '--remove_extra_whitespaces=true',
            '--add_dummy_prefix=false',  # Don't add dummy prefix for better multilingual support
        ]
        
        # Train the tokenizer
        try:
            spm.SentencePieceTrainer.train(' '.join(spm_args))
            logger.info(f"Tokenizer training completed successfully!")
            logger.info(f"Model saved as: {model_prefix}.model")
            logger.info(f"Vocabulary saved as: {model_prefix}.vocab")
            
            # Update settings paths
            self.update_settings_paths(f"{model_prefix}.model", f"{model_prefix}.vocab")
            
        except Exception as e:
            logger.error(f"Tokenizer training failed: {e}")
            raise

    def update_settings_paths(self, model_path: str, vocab_path: str):
        """Update settings.py with correct tokenizer paths"""
        try:
            settings_file = "core/settings.py"
            if os.path.exists(settings_file):
                with open(settings_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update the tokenizer model path
                old_model_line = 'TOKENIZER_MODEL_PATH = "model/multi_tokenizer.model"'
                new_model_line = f'TOKENIZER_MODEL_PATH = "{model_path}"'
                content = content.replace(old_model_line, new_model_line)
                
                # Update the vocab path
                old_vocab_line = 'TOKENIZER_VOCAB_PATH = "model/multi_tokenizer.vocab"'
                new_vocab_line = f'TOKENIZER_VOCAB_PATH = "{vocab_path}"'
                content = content.replace(old_vocab_line, new_vocab_line)
                
                with open(settings_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("Updated settings.py with new tokenizer paths")
        except Exception as e:
            logger.warning(f"Could not update settings.py: {e}")

    def test_tokenizer(self, model_path: str):
        """Test the trained tokenizer with sample texts"""
        logger.info("Testing trained tokenizer...")
        
        # Load the tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        # Test sentences in different languages
        test_sentences = {
            "hindi": "नमस्ते, आप कैसे हैं?",
            "sanskrit": "संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति।",
            "marathi": "मराठी ही महाराष्ट्राची भाषा आहे।",
            "english": "Hello, how are you today?"
        }
        
        logger.info("\nTokenization Test Results:")
        logger.info("=" * 50)
        
        for lang, sentence in test_sentences.items():
            tokens = sp.encode_as_pieces(sentence)
            ids = sp.encode_as_ids(sentence)
            reconstructed = sp.decode_pieces(tokens)
            
            logger.info(f"\n{lang.upper()}:")
            logger.info(f"Original: {sentence}")
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Token count: {len(tokens)}")
            logger.info(f"Reconstructed: {reconstructed}")
            logger.info(f"Lossless: {'✓' if sentence == reconstructed else '✗'}")
        
        # Test vocabulary stats
        vocab_size = sp.get_piece_size()
        logger.info(f"\nVocabulary size: {vocab_size}")
        logger.info(f"Target vocabulary size: {settings.SP_VOCAB_SIZE}")
        
        # Test some special tokens
        logger.info("\nSpecial tokens:")
        logger.info(f"BOS: {sp.bos_id()} -> {sp.id_to_piece(sp.bos_id())}")
        logger.info(f"EOS: {sp.eos_id()} -> {sp.id_to_piece(sp.eos_id())}")
        logger.info(f"UNK: {sp.unk_id()} -> {sp.id_to_piece(sp.unk_id())}")
        logger.info(f"PAD: {sp.pad_id()} -> {sp.id_to_piece(sp.pad_id())}")

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_combined_file and os.path.exists(self.temp_combined_file):
            os.unlink(self.temp_combined_file)
            logger.info("Cleaned up temporary files")

    def print_statistics(self):
        """Print training data statistics"""
        logger.info("\nTraining Data Statistics:")
        logger.info("=" * 50)
        logger.info(f"Total sentences: {self.stats['total_sentences']}")
        logger.info(f"Devanagari sentences: {self.stats['devanagari_sentences']}")
        logger.info(f"Latin sentences: {self.stats['latin_sentences']}")
        logger.info(f"Mixed/Other sentences: {self.stats['total_sentences'] - self.stats['devanagari_sentences'] - self.stats['latin_sentences']}")
        
        logger.info("\nLanguage distribution:")
        for lang, count in self.stats['language_counts'].items():
            percentage = (count / self.stats['total_sentences']) * 100 if self.stats['total_sentences'] > 0 else 0
            logger.info(f"  {lang}: {count} sentences ({percentage:.1f}%)")


def main():
    """Main function to train the multilingual tokenizer"""
    logger.info("=" * 60)
    logger.info("Multilingual SentencePiece Tokenizer Training")
    logger.info("=" * 60)
    logger.info(f"Supported languages: {', '.join(settings.SUPPORTED_LANGUAGES)}")
    logger.info(f"Vocabulary size: {settings.SP_VOCAB_SIZE}")
    logger.info(f"Model type: {settings.SP_MODEL_TYPE}")
    logger.info(f"Character coverage: {settings.SP_CHARACTER_COVERAGE}")
    
    trainer = MultilingualTokenizerTrainer()
    
    try:
        # Step 1: Prepare training data
        training_file = trainer.prepare_training_data()
        trainer.print_statistics()
        
        # Step 2: Train tokenizer
        trainer.train_tokenizer(training_file)
        
        # Step 3: Test tokenizer
        model_path = os.path.join(settings.FINE_TUNED_MODEL_PATH, "multi_tokenizer.model")
        trainer.test_tokenizer(model_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("Tokenizer training completed successfully!")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Restart your API server (app.py)")
        logger.info("2. Test the /tokenize endpoint with multilingual text")
        logger.info("3. Use the trained tokenizer for fine-tuning")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()