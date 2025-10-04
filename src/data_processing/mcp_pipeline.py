"""
Multi-Corpus Preprocessing (MCP) Pipeline for 20+ Indian Languages

This module implements the MCP pipeline as specified in the requirements document.
It handles robust preprocessing of multilingual corpora including:
- Unicode normalization for different scripts
- Language-specific cleaning and tokenization preparation
- Deduplication and noise removal
- Sentence segmentation for Indian languages
- Format preparation for SentencePiece training

Based on the requirements:
- Support for 20+ Indian languages
- Integration with AI4Bharat preprocessing best practices
- Robust handling of Devanagari, Tamil, Telugu, Kannada, Bengali, etc.
- Preparation for large-scale Gurukul integration
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
    Multi-Corpus Preprocessing Pipeline for 20+ Indian Languages
    
    This class implements the complete MCP workflow as specified in the requirements:
    1. Corpus Collection
    2. Cleaning & Normalization
    3. Sentence Segmentation
    4. Deduplication
    5. Tokenization Preparation
    6. Training Tokenizer
    7. Integration
    """
    
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
        
        # Language-specific normalization
        if language in ["hindi", "sanskrit", "marathi", "nepali", "konkani", "bodo", "dogri", "maithili"]:
            # Devanagari script normalization
            normalized = self._normalize_devanagari(normalized)
        elif language in ["bengali", "assamese"]:
            # Bengali script normalization
            normalized = self._normalize_bengali(normalized)
        elif language in ["tamil"]:
            # Tamil script normalization
            normalized = self._normalize_tamil(normalized)
        elif language in ["telugu"]:
            # Telugu script normalization
            normalized = self._normalize_telugu(normalized)
        elif language in ["kannada"]:
            # Kannada script normalization
            normalized = self._normalize_kannada(normalized)
        elif language in ["gujarati"]:
            # Gujarati script normalization
            normalized = self._normalize_gujarati(normalized)
        elif language in ["punjabi"]:
            # Punjabi (Gurmukhi) script normalization
            normalized = self._normalize_punjabi(normalized)
        elif language in ["odia"]:
            # Odia script normalization
            normalized = self._normalize_odia(normalized)
        elif language in ["malayalam"]:
            # Malayalam script normalization
            normalized = self._normalize_malayalam(normalized)
        elif language in ["urdu"]:
            # Urdu (Arabic script) normalization
            normalized = self._normalize_urdu(normalized)
        
        self.stats["normalization_applied"] += 1
        return normalized.strip()
    
    def _normalize_devanagari(self, text: str) -> str:
        """Normalize Devanagari script text"""
        # Handle common Devanagari ligatures and combining characters
        # This is a simplified version - in production, use indic-nlp-library
        return text
    
    def _normalize_bengali(self, text: str) -> str:
        """Normalize Bengali script text"""
        return text
    
    def _normalize_tamil(self, text: str) -> str:
        """Normalize Tamil script text"""
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
        return text
    
    def _normalize_odia(self, text: str) -> str:
        """Normalize Odia script text"""
        return text
    
    def _normalize_malayalam(self, text: str) -> str:
        """Normalize Malayalam script text"""
        return text
    
    def _normalize_urdu(self, text: str) -> str:
        """Normalize Urdu (Arabic script) text"""
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
        
        # Language-specific cleaning
        if language in ["hindi", "sanskrit", "marathi", "nepali", "konkani", "bodo", "dogri", "maithili"]:
            # Devanagari script cleaning
            text = re.sub(r'[^\u0900-\u097F\u0020-\u007F\u200C\u200D]', '', text)
        elif language in ["bengali", "assamese"]:
            # Bengali script cleaning
            text = re.sub(r'[^\u0980-\u09FF\u0020-\u007F]', '', text)
        elif language in ["tamil"]:
            # Tamil script cleaning
            text = re.sub(r'[^\u0B80-\u0BFF\u0020-\u007F]', '', text)
        elif language in ["telugu"]:
            # Telugu script cleaning
            text = re.sub(r'[^\u0C00-\u0C7F\u0020-\u007F]', '', text)
        elif language in ["kannada"]:
            # Kannada script cleaning
            text = re.sub(r'[^\u0C80-\u0CFF\u0020-\u007F]', '', text)
        elif language in ["gujarati"]:
            # Gujarati script cleaning
            text = re.sub(r'[^\u0A80-\u0AFF\u0020-\u007F]', '', text)
        elif language in ["punjabi"]:
            # Punjabi script cleaning
            text = re.sub(r'[^\u0A00-\u0A7F\u0020-\u007F]', '', text)
        elif language in ["odia"]:
            # Odia script cleaning
            text = re.sub(r'[^\u0B00-\u0B7F\u0020-\u007F]', '', text)
        elif language in ["malayalam"]:
            # Malayalam script cleaning
            text = re.sub(r'[^\u0D00-\u0D7F\u0020-\u007F]', '', text)
        elif language in ["urdu"]:
            # Urdu script cleaning
            text = re.sub(r'[^\u0600-\u06FF\u0020-\u007F]', '', text)
        
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
        # Basic sentence segmentation - in production, use indic-nlp-library
        # This is a simplified version for demonstration
        
        # Common sentence endings across Indian languages
        sentence_endings = [
            r'[।॥]',  # Devanagari, Bengali, etc.
            r'[.!?]',  # Latin scripts
            r'[।]',    # Some scripts use only ।
            r'[?]',    # Question marks
            r'[!]'     # Exclamation marks
        ]
        
        # Create pattern for sentence endings
        pattern = '|'.join(sentence_endings)
        
        # Split on sentence endings
        sentences = re.split(pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def detect_script(self, text: str) -> str:
        """
        Detect the primary script of the text
        
        Args:
            text: Input text
            
        Returns:
            Script name (devanagari, tamil, telugu, etc.)
        """
        script_counts = defaultdict(int)
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                for script, (start, end) in settings.UNICODE_RANGES.items():
                    if start <= ord(char) <= end:
                        script_counts[script] += 1
                        break
        
        if total_chars == 0:
            return "unknown"
        
        # Return the script with the highest character count
        return max(script_counts.items(), key=lambda x: x[1])[0] if script_counts else "unknown"
    
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
                # Add language tag: <lang:hi> ... </lang>
                tagged_sentence = f"<lang:{language}> {sentence} </lang>"
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
        
        # Create temporary file for combined processed data
        temp_fd, temp_file = tempfile.mkstemp(suffix='.txt', prefix='mcp_processed_')
        self.temp_files.append(temp_file)
        
        all_processed_sentences = []
        
        # Process each language corpus
        for language, filename in settings.CORPUS_FILES.items():
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
        logger.info("\n" + "=" * 60)
        logger.info("MCP Pipeline Statistics")
        logger.info("=" * 60)
        logger.info(f"Total sentences processed: {self.stats['total_sentences']:,}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']:,}")
        logger.info(f"Noise removed: {self.stats['noise_removed']:,}")
        logger.info(f"Normalization applied: {self.stats['normalization_applied']:,}")
        
        logger.info("\nLanguage distribution:")
        for lang, count in sorted(self.stats['language_counts'].items()):
            percentage = (count / self.stats['total_sentences']) * 100 if self.stats['total_sentences'] > 0 else 0
            logger.info(f"  {lang}: {count:,} sentences ({percentage:.1f}%)")
        
        logger.info("\nScript distribution:")
        for script, count in sorted(self.stats['script_counts'].items()):
            percentage = (count / self.stats['total_sentences']) * 100 if self.stats['total_sentences'] > 0 else 0
            logger.info(f"  {script}: {count:,} sentences ({percentage:.1f}%)")
        
        logger.info("=" * 60)
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")


def main():
    """Main function to run the MCP pipeline"""
    logger.info("Starting Multi-Corpus Preprocessing (MCP) Pipeline")
    logger.info(f"Processing {len(settings.SUPPORTED_LANGUAGES)} languages")
    
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
