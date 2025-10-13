"""
Multi-Corpus Preprocessing (MCP) Pipeline for 21 Indian Languages

This module implements the MCP pipeline with support for:
1. Assamese, 2. Bengali, 3. Bodo, 4. English, 5. Gujarati, 6. Hindi,
7. Kannada, 8. Kashmiri, 9. Maithili, 10. Malayalam, 11. Marathi,
12. Meitei (Manipuri), 13. Nepali, 14. Odia, 15. Punjabi, 16. Sanskrit,
17. Santali, 18. Sindhi, 19. Tamil, 20. Telugu, 21. Urdu

Handles robust preprocessing including Unicode normalization, cleaning,
sentence segmentation, and deduplication for diverse Indian scripts.
"""

import os
import re
import unicodedata
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import tempfile
import json
from pathlib import Path

# Import settings
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPPipeline:
    """
    Multi-Corpus Preprocessing Pipeline for 21 Indian Languages
    
    Supported languages and their scripts:
    - Devanagari: Hindi, Sanskrit, Marathi, Nepali, Bodo, Maithili
    - Bengali-Assamese: Bengali, Assamese
    - Tamil: Tamil
    - Telugu: Telugu
    - Kannada: Kannada
    - Malayalam: Malayalam
    - Gujarati: Gujarati
    - Gurmukhi: Punjabi
    - Odia: Odia
    - Perso-Arabic: Urdu, Kashmiri, Sindhi
    - Meitei Mayek: Meitei (Manipuri)
    - Ol Chiki: Santali
    - Latin: English
    """
    
    # Unicode ranges for all supported scripts
    SCRIPT_RANGES = {
        'devanagari': (0x0900, 0x097F),
        'bengali': (0x0980, 0x09FF),
        'gurmukhi': (0x0A00, 0x0A7F),
        'gujarati': (0x0A80, 0x0AFF),
        'odia': (0x0B00, 0x0B7F),
        'tamil': (0x0B80, 0x0BFF),
        'telugu': (0x0C00, 0x0C7F),
        'kannada': (0x0C80, 0x0CFF),
        'malayalam': (0x0D00, 0x0D7F),
        'arabic': (0x0600, 0x06FF),  # For Urdu, Kashmiri, Sindhi
        'arabic_supplement': (0x0750, 0x077F),
        'arabic_extended': (0x08A0, 0x08FF),
        'meitei': (0xABC0, 0xABFF),  # Meetei Mayek
        'ol_chiki': (0x1C50, 0x1C7F),  # Santali
        'latin': (0x0020, 0x007F),
    }
    
    # Language to script mapping
    LANGUAGE_SCRIPTS = {
        'assamese': ['bengali'],
        'bengali': ['bengali'],
        'bodo': ['devanagari'],
        'english': ['latin'],
        'gujarati': ['gujarati'],
        'hindi': ['devanagari'],
        'kannada': ['kannada'],
        'kashmiri': ['arabic', 'arabic_supplement', 'arabic_extended', 'devanagari'],
        'maithili': ['devanagari'],
        'malayalam': ['malayalam'],
        'marathi': ['devanagari'],
        'meitei': ['meitei', 'bengali'],  # Can be written in both scripts
        'nepali': ['devanagari'],
        'odia': ['odia'],
        'punjabi': ['gurmukhi'],
        'sanskrit': ['devanagari'],
        'santali': ['ol_chiki', 'devanagari', 'bengali'],  # Multiple scripts
        'sindhi': ['arabic', 'arabic_supplement', 'devanagari'],
        'tamil': ['tamil'],
        'telugu': ['telugu'],
        'urdu': ['arabic', 'arabic_supplement', 'arabic_extended'],
    }
    
    def __init__(self):
        self.stats = {
            "total_sentences": 0,
            "language_counts": defaultdict(int),
            "script_counts": defaultdict(int),
            "duplicates_removed": 0,
            "noise_removed": 0,
            "normalization_applied": 0
        }
        self.temp_files = []
        self.sentence_hashes = set()
        
    def normalize_text(self, text: str, language: str) -> str:
        """
        Apply language-specific Unicode normalization
        
        Args:
            text: Input text to normalize
            language: Language code for script-specific handling
            
        Returns:
            Normalized text
        """
        # Remove zero-width characters that can cause tokenization issues
        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u200e', '\u200f']
        for char in zero_width_chars:
            text = text.replace(char, '')
        
        # Apply NFC normalization to combine base characters with diacritics
        normalized = unicodedata.normalize('NFC', text)
        
        # Language-specific normalization based on script
        if language in ["hindi", "sanskrit", "marathi", "nepali", "bodo", "maithili"]:
            normalized = self._normalize_devanagari(normalized)
        elif language in ["bengali", "assamese"]:
            normalized = self._normalize_bengali(normalized)
        elif language == "tamil":
            normalized = self._normalize_tamil(normalized)
        elif language == "telugu":
            normalized = self._normalize_telugu(normalized)
        elif language == "kannada":
            normalized = self._normalize_kannada(normalized)
        elif language == "gujarati":
            normalized = self._normalize_gujarati(normalized)
        elif language == "punjabi":
            normalized = self._normalize_punjabi(normalized)
        elif language == "odia":
            normalized = self._normalize_odia(normalized)
        elif language == "malayalam":
            normalized = self._normalize_malayalam(normalized)
        elif language in ["urdu", "kashmiri", "sindhi"]:
            normalized = self._normalize_perso_arabic(normalized)
        elif language == "meitei":
            normalized = self._normalize_meitei(normalized)
        elif language == "santali":
            normalized = self._normalize_santali(normalized)
        elif language == "english":
            normalized = self._normalize_english(normalized)
        
        self.stats["normalization_applied"] += 1
        return normalized.strip()
    
    def _normalize_devanagari(self, text: str) -> str:
        """Normalize Devanagari script text (Hindi, Sanskrit, Marathi, Nepali, Bodo, Maithili)"""
        # Normalize nukta combinations
        text = text.replace('\u0929', '\u0928\u093C')  # ऩ -> ऩ
        text = text.replace('\u0931', '\u0930\u093C')  # ऱ -> ऱ
        text = text.replace('\u0934', '\u0933\u093C')  # ऴ -> ऴ
        text = text.replace('\u0958', '\u0915\u093C')  # क़
        text = text.replace('\u0959', '\u0916\u093C')  # ख़
        text = text.replace('\u095A', '\u0917\u093C')  # ग़
        text = text.replace('\u095B', '\u091C\u093C')  # ज़
        text = text.replace('\u095C', '\u0921\u093C')  # ड़
        text = text.replace('\u095D', '\u0922\u093C')  # ढ़
        text = text.replace('\u095E', '\u092B\u093C')  # फ़
        text = text.replace('\u095F', '\u092F\u093C')  # य़
        return text
    
    def _normalize_bengali(self, text: str) -> str:
        """Normalize Bengali-Assamese script text"""
        # Normalize common variations
        text = text.replace('\u09CE', '\u09A4')  # ৎ -> ত (khanda ta)
        return text
    
    def _normalize_tamil(self, text: str) -> str:
        """Normalize Tamil script text"""
        # Tamil has fewer variations but normalize common patterns
        return text
    
    def _normalize_telugu(self, text: str) -> str:
        """Normalize Telugu script text"""
        return text
    
    def _normalize_kannada(self, text: str) -> str:
        """Normalize Kannada script text"""
        return text
    
    def _normalize_gujarati(self, text: str) -> str:
        """Normalize Gujarati script text"""
        return text
    
    def _normalize_punjabi(self, text: str) -> str:
        """Normalize Punjabi (Gurmukhi) script text"""
        # Normalize Gurmukhi specific characters
        return text
    
    def _normalize_odia(self, text: str) -> str:
        """Normalize Odia script text"""
        return text
    
    def _normalize_malayalam(self, text: str) -> str:
        """Normalize Malayalam script text"""
        # Normalize chillu characters and other Malayalam-specific features
        return text
    
    def _normalize_perso_arabic(self, text: str) -> str:
        """Normalize Perso-Arabic script text (Urdu, Kashmiri, Sindhi)"""
        # Normalize Arabic presentation forms
        text = unicodedata.normalize('NFKC', text)
        # Remove Arabic diacritics if needed (optional)
        # text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        return text
    
    def _normalize_meitei(self, text: str) -> str:
        """Normalize Meitei Mayek script text"""
        # Meetei Mayek normalization
        return text
    
    def _normalize_santali(self, text: str) -> str:
        """Normalize Santali (Ol Chiki) script text"""
        # Ol Chiki normalization
        return text
    
    def _normalize_english(self, text: str) -> str:
        """Normalize English text"""
        # Basic normalization for English
        text = text.lower() if text.isupper() else text  # Preserve mixed case
        return text
    
    def clean_text(self, text: str, language: str) -> str:
        """
        Clean text by removing noise, HTML tags, and irrelevant content
        
        Args:
            text: Input text to clean
            language: Language code for language-specific cleaning
            
        Returns:
            Cleaned text
        """
        original_length = len(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Language-specific cleaning based on allowed scripts
        allowed_scripts = self.LANGUAGE_SCRIPTS.get(language, ['latin'])
        
        # Build pattern for allowed characters
        allowed_ranges = []
        for script in allowed_scripts:
            if script in self.SCRIPT_RANGES:
                start, end = self.SCRIPT_RANGES[script]
                allowed_ranges.append(f'\\u{start:04X}-\\u{end:04X}')
        
        # Always allow basic Latin (spaces, punctuation, numbers)
        allowed_ranges.append('\\u0020-\\u007F')
        
        # Create regex pattern
        if allowed_ranges:
            pattern = f'[^{"".join(allowed_ranges)}]'
            text = re.sub(pattern, '', text)
        
        # Remove empty lines and very short lines
        if len(text.strip()) < 10:
            return ""
        
        cleaned_length = len(text)
        if original_length != cleaned_length:
            self.stats["noise_removed"] += 1
        
        return text.strip()
    
    def segment_sentences(self, text: str, language: str) -> List[str]:
        """
        Segment text into sentences using language-specific rules
        
        Args:
            text: Input text to segment
            language: Language code for language-specific segmentation
            
        Returns:
            List of sentences
        """
        # Sentence endings by script type
        if language in ["hindi", "sanskrit", "marathi", "nepali", "bodo", "maithili",
                       "bengali", "assamese", "gujarati", "punjabi", "odia"]:
            # Indic scripts use danda (।) and double danda (॥)
            pattern = r'[।॥.!?]+'
        elif language in ["urdu", "kashmiri", "sindhi"]:
            # Perso-Arabic uses different punctuation
            pattern = r'[۔؟!.?]+'
        elif language == "tamil":
            pattern = r'[.!?]+'
        elif language in ["telugu", "kannada", "malayalam"]:
            pattern = r'[.!?।]+'
        elif language in ["meitei", "santali"]:
            pattern = r'[.!?।]+'
        else:  # English and others
            pattern = r'[.!?]+'
        
        # Split on sentence endings
        sentences = re.split(pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Minimum sentence length: 10 characters or 3 words
            if len(sentence) >= 10 and len(sentence.split()) >= 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def detect_script(self, text: str) -> str:
        """
        Detect the primary script of the text
        
        Args:
            text: Input text
            
        Returns:
            Script name
        """
        script_counts = defaultdict(int)
        total_chars = 0
        
        for char in text:
            if char.isalpha() or ord(char) > 127:  # Non-ASCII characters
                total_chars += 1
                for script, (start, end) in self.SCRIPT_RANGES.items():
                    if start <= ord(char) <= end:
                        script_counts[script] += 1
                        break
        
        if total_chars == 0:
            return "unknown"
        
        # Return the script with the highest character count
        return max(script_counts.items(), key=lambda x: x[1])[0] if script_counts else "latin"
    
    def deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        """
        Remove duplicate sentences using hashing
        
        Args:
            sentences: List of sentences to deduplicate
            
        Returns:
            List of unique sentences
        """
        unique_sentences = []
        seen_hashes = set()
        
        for sentence in sentences:
            # Create hash of normalized sentence
            sentence_hash = hashlib.md5(sentence.encode('utf-8')).hexdigest()
            
            if sentence_hash not in seen_hashes:
                seen_hashes.add(sentence_hash)
                unique_sentences.append(sentence)
            else:
                self.stats["duplicates_removed"] += 1
        
        return unique_sentences
    
    def prepare_for_tokenization(self, sentences: List[str], language: str) -> List[str]:
        """
        Prepare sentences for SentencePiece tokenization
        
        Args:
            sentences: List of sentences
            language: Language code
            
        Returns:
            List of prepared sentences
        """
        prepared_sentences = []
        
        for sentence in sentences:
            # Add language tag for multilingual corpora
            if language != "english":
                # Format: <2language> sentence
                tagged_sentence = f"<2{language}> {sentence}"
            else:
                tagged_sentence = sentence
            
            prepared_sentences.append(tagged_sentence)
        
        return prepared_sentences
    
    def process_corpus_file(self, filepath: str, language: str) -> List[str]:
        """
        Process a single corpus file through the complete MCP pipeline
        
        Args:
            filepath: Path to the corpus file
            language: Language code
            
        Returns:
            List of processed sentences
        """
        logger.info(f"Processing {language} corpus: {filepath}")
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return []
        
        processed_sentences = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} lines from {filepath}")
                    
                    # Clean the line
                    cleaned_line = self.clean_text(line, language)
                    if not cleaned_line:
                        continue
                    
                    # Normalize the text
                    normalized_line = self.normalize_text(cleaned_line, language)
                    if not normalized_line:
                        continue
                    
                    # Segment into sentences
                    sentences = self.segment_sentences(normalized_line, language)
                    
                    # Add to processed sentences
                    processed_sentences.extend(sentences)
                    
                    # Update statistics
                    self.stats["language_counts"][language] += len(sentences)
                    self.stats["total_sentences"] += len(sentences)
                    
                    # Detect script for statistics
                    script = self.detect_script(normalized_line)
                    self.stats["script_counts"][script] += 1
        
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return []
        
        logger.info(f"Processed {len(processed_sentences)} sentences from {language}")
        return processed_sentences
    
    def process_all_corpora(self) -> str:
        """
        Process all corpus files through the MCP pipeline
        
        Returns:
            Path to the combined processed file
        """
        logger.info("Starting MCP pipeline for all corpora")
        logger.info(f"Processing 21 Indian languages: {', '.join(sorted(self.LANGUAGE_SCRIPTS.keys()))}")
        
        # Create temporary file for combined processed data
        temp_fd, temp_file = tempfile.mkstemp(suffix='.txt', prefix='mcp_processed_')
        self.temp_files.append(temp_file)
        
        all_processed_sentences = []
        
        # Process each language corpus
        for language, filename in settings.CORPUS_FILES.items():
            if language not in self.LANGUAGE_SCRIPTS:
                logger.warning(f"Language '{language}' not in supported list, skipping")
                continue
                
            filepath = os.path.join(settings.TRAINING_DATA_PATH, filename)
            
            # Process the corpus file
            processed_sentences = self.process_corpus_file(filepath, language)
            
            if processed_sentences:
                # Deduplicate sentences for this language
                unique_sentences = self.deduplicate_sentences(processed_sentences)
                
                # Prepare for tokenization
                prepared_sentences = self.prepare_for_tokenization(unique_sentences, language)
                
                # Add to combined list
                all_processed_sentences.extend(prepared_sentences)
                
                logger.info(f"Added {len(prepared_sentences)} unique sentences from {language}")
        
        # Final deduplication across all languages
        logger.info("Performing final deduplication across all languages")
        all_processed_sentences = self.deduplicate_sentences(all_processed_sentences)
        
        # Write to temporary file
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            for sentence in all_processed_sentences:
                f.write(sentence + '\n')
        
        logger.info(f"Combined processed data written to: {temp_file}")
        logger.info(f"Total processed sentences: {len(all_processed_sentences)}")
        
        return temp_file
    
    def print_statistics(self):
        """Print comprehensive MCP pipeline statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("MCP Pipeline Statistics - 21 Indian Languages")
        logger.info("=" * 80)
        logger.info(f"Total sentences processed: {self.stats['total_sentences']:,}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']:,}")
        logger.info(f"Noise removed: {self.stats['noise_removed']:,}")
        logger.info(f"Normalization applied: {self.stats['normalization_applied']:,}")
        
        logger.info("\nLanguage distribution:")
        for lang, count in sorted(self.stats['language_counts'].items()):
            percentage = (count / self.stats['total_sentences']) * 100 if self.stats['total_sentences'] > 0 else 0
            logger.info(f"  {lang:12s}: {count:10,} sentences ({percentage:5.1f}%)")
        
        logger.info("\nScript distribution:")
        for script, count in sorted(self.stats['script_counts'].items()):
            logger.info(f"  {script:20s}: {count:10,} occurrences")
        
        logger.info("=" * 80)
    
    def cleanup(self):
        # """Clean up temporary files"""
        # for temp_file in self.temp_files:
        #     if os.path.exists(temp_file):
        #         os.unlink(temp_file)
        #         logger.info(f"Cleaned up temporary file: {temp_file}")
        pass


def main():
    """Main function to run the MCP pipeline"""
    logger.info("=" * 80)
    logger.info("Multi-Corpus Preprocessing (MCP) Pipeline")
    logger.info("Supporting 21 Indian Languages")
    logger.info("=" * 80)
    
    supported_langs = [
        "Assamese", "Bengali", "Bodo", "English", "Gujarati", "Hindi",
        "Kannada", "Kashmiri", "Maithili", "Malayalam", "Marathi",
        "Meitei", "Nepali", "Odia", "Punjabi", "Sanskrit",
        "Santali", "Sindhi", "Tamil", "Telugu", "Urdu"
    ]
    logger.info(f"Languages: {', '.join(supported_langs)}")
    logger.info("=" * 80 + "\n")
    
    # Create MCP pipeline
    mcp = MCPPipeline()
    
    try:
        # Process all corpora
        processed_file = mcp.process_all_corpora()
        
        # Print statistics
        mcp.print_statistics()
        
        logger.info(f"\nMCP pipeline completed successfully!")
        logger.info(f"Processed data saved to: {processed_file}")
        logger.info("\nNext steps:")
        logger.info("1. Use this processed data to train the multilingual tokenizer")
        logger.info("2. Run: python src/training/train_tokenizer.py")
        logger.info("3. Fine-tune the language model with the new tokenizer")
        
    except Exception as e:
        logger.error(f"MCP pipeline failed: {e}")
        raise
    finally:
        mcp.cleanup()


if __name__ == "__main__":
    main()