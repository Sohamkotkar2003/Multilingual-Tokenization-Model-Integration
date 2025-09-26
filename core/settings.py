"""
Configuration settings for the Multilingual Tokenization & Inference API
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
API_TITLE = "Multilingual Tokenization & Inference API"
API_DESCRIPTION = "API for Hindi, Sanskrit, Marathi, and English tokenization and text generation"
API_VERSION = "1.0.0"
DEBUG_MODE = True

# Model and Tokenizer Paths
TOKENIZER_MODEL_PATH = BASE_DIR / "multi_tokenizer.model"
TOKENIZER_VOCAB_PATH = BASE_DIR / "multi_tokenizer.vocab"
TOKENIZER_MERGE_PATH = BASE_DIR / "multi_tokenizer_merge.txt"

# Model Configuration
MODEL_NAME = "gpt2"  # Replace with your actual model name
MODEL_PATH = BASE_DIR / "models" / "multilingual_model"  # Path to your trained model
MAX_GENERATION_LENGTH = 100
NUM_RETURN_SEQUENCES = 1
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Supported Languages
SUPPORTED_LANGUAGES = ["hindi", "sanskrit", "marathi", "english"]
DEFAULT_LANGUAGE = "english"

# Language Detection Configuration
DEVANAGARI_UNICODE_RANGE = (0x0900, 0x097F)
LANGUAGE_CONFIDENCE_THRESHOLD = 0.5
ENGLISH_RATIO_THRESHOLD = 0.8
DEVANAGARI_RATIO_THRESHOLD = 0.5

# Language-specific keywords for detection
LANGUAGE_KEYWORDS = {
    "sanskrit": ["संस्कृत", "श्लोक", "मन्त्र", "वेद", "उपनिषद्", "गीता"],
    "marathi": ["महाराष्ट्र", "आहे", "आहेत", "होते", "होता", "म्हणजे", "काय", "कसे"],
    "hindi": ["है", "हैं", "था", "थी", "होगा", "होगी", "क्या", "कैसे", "यह", "वह"]
}

# Training Configuration (for future use)
TRAINING_DATA_PATH = BASE_DIR / "data" / "training"
VALIDATION_DATA_PATH = BASE_DIR / "data" / "validation"
CORPUS_FILES = {
    "hindi": "hindi_corpus.txt",
    "sanskrit": "sanskrit_corpus.txt", 
    "marathi": "marathi_corpus.txt",
    "english": "english_corpus.txt"
}

# SentencePiece Training Parameters
SP_VOCAB_SIZE = 32000
SP_MODEL_TYPE = "bpe"  # or "unigram"
SP_CHARACTER_COVERAGE = 0.9995
SP_INPUT_SENTENCE_SIZE = 10000000
SP_SHUFFLE_INPUT_SENTENCE = True

# FastText Configuration (if using for language detection)
FASTTEXT_MODEL_PATH = BASE_DIR / "models" / "lid.176.bin"
FASTTEXT_DETECTION_THRESHOLD = 0.7

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "logs" / "api.log"

# Create necessary directories
DIRECTORIES_TO_CREATE = [
    BASE_DIR / "models",
    BASE_DIR / "data" / "training",
    BASE_DIR / "data" / "validation", 
    BASE_DIR / "logs"
]

def create_directories():
    """Create necessary directories if they don't exist"""
    for directory in DIRECTORIES_TO_CREATE:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_config():
    """Get model configuration dictionary"""
    return {
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "max_length": MAX_GENERATION_LENGTH,
        "num_return_sequences": NUM_RETURN_SEQUENCES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": DO_SAMPLE
    }

def get_tokenizer_config():
    """Get tokenizer configuration dictionary"""
    return {
        "model_path": str(TOKENIZER_MODEL_PATH),
        "vocab_path": str(TOKENIZER_VOCAB_PATH),
        "vocab_size": SP_VOCAB_SIZE,
        "model_type": SP_MODEL_TYPE,
        "character_coverage": SP_CHARACTER_COVERAGE
    }

def get_api_config():
    """Get API configuration dictionary"""
    return {
        "host": API_HOST,
        "port": API_PORT,
        "title": API_TITLE,
        "description": API_DESCRIPTION,
        "version": API_VERSION,
        "debug": DEBUG_MODE
    }

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    DEBUG_MODE = False
    API_HOST = "0.0.0.0"
    LOG_LEVEL = "WARNING"

if os.getenv("ENVIRONMENT") == "development":
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"