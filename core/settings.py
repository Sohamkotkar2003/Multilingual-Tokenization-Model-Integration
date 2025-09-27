"""
Configuration settings for the Multilingual Tokenization & Inference API
(Updated to support decoder-only LM, KB/Vaani hooks, and fine-tune script)
"""
import os

DEFAULT_MAX_LENGTH = 1024  # Optimal for most generation tasks
TRAINING_MAX_LENGTH = 1024  # For fine-tuning
INFERENCE_MAX_LENGTH = 1536  # For inference (can be higher since no gradient computation)

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
API_TITLE = "Multilingual Tokenization & Inference API"
API_DESCRIPTION = "API for Hindi, Sanskrit, Marathi, and English tokenization and text generation"
API_VERSION = "1.0.1"
DEBUG_MODE = True

# Model and Tokenizer Paths
TOKENIZER_MODEL_PATH = "model/multi_tokenizer.model"   # SentencePiece .model (optional)
TOKENIZER_VOCAB_PATH = "model/multi_tokenizer.vocab"
TOKENIZER_MERGE_PATH = "model/tokenizer_merge.txt"

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "AhinsaAI/ahinsa0.5-llama3.2-3B")  # HF model name fallback (decoder-only)
# MODEL_NAME = os.getenv("MODEL_NAME", "bigscience/bloom-560m")  # HF model name fallback (decoder-only)
MODEL_PATH = os.getenv("MODEL_PATH", "")  # local checkpoint folder (if used). Empty -> use MODEL_NAME
# Note: Set MODEL_PATH to "mbart_finetuned" in settings.py after training to use fine-tuned model

# Generation params
MAX_GENERATION_LENGTH = 256
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
    "sanskrit": ["संस्कृत", "श्लोक", "मन्त्र", "वेद", "उपनिषद्", "गीता", "धर्मो"],
    "marathi": ["महाराष्ट्र", "आहे", "आहेत", "होते", "होता", "म्हणजे", "काय", "कसे", "एकदा"],
    "hindi": ["है", "हैं", "था", "थी", "होगा", "होगी", "क्या", "कैसे", "यह", "वह", "एक"]
}

# Training / Fine-tuning Configuration
TRAINING_DATA_PATH = "data/training"
VALIDATION_DATA_PATH = "data/validation"
CORPUS_FILES = {
    "hindi": "hi_train.txt",
    "sanskrit": "sa_train.txt",
    "marathi": "mr_train.txt",
    "english": "en_train.txt"
}
FINE_TUNED_MODEL_PATH = "model"

# SentencePiece Training Parameters
SP_VOCAB_SIZE = 32000
SP_MODEL_TYPE = "bpe"  # or "unigram"
SP_CHARACTER_COVERAGE = 0.9995
SP_INPUT_SENTENCE_SIZE = 10000000
SP_SHUFFLE_INPUT_SENTENCE = True

# FastText Configuration
FASTTEXT_MODEL_PATH = "models/lid.176.bin"
FASTTEXT_DETECTION_THRESHOLD = 0.7

# KB and TTS (Vaani) integration endpoints
KB_ENDPOINT = os.getenv("KB_ENDPOINT", "http://127.0.0.1:8000")
KB_TIMEOUT = float(os.getenv("KB_TIMEOUT", 10.0))

VAANI_ENDPOINT = os.getenv("VAANI_ENDPOINT", "")
VAANI_TIMEOUT = float(os.getenv("VAANI_TIMEOUT", 10.0))

# Device and performance options
USE_FP16_IF_GPU = True

# Quantization Configuration for Inference
USE_4BIT_QUANTIZATION = True  # Enable 4-bit quantization for faster inference
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",  # NormalFloat4 quantization
    "bnb_4bit_compute_dtype": "float16",  # Compute dtype for 4-bit base models
    "bnb_4bit_use_double_quant": True,  # Use double quantization for better accuracy
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/api.log"

# Enhanced logging format
DETAILED_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"

# Create necessary directories
DIRECTORIES_TO_CREATE = [
    "data/training",
    "data/validation",
    "logs",
    "model"
]

def create_directories():
    """Create necessary directories if they don't exist"""
    created_dirs = []
    for directory in DIRECTORIES_TO_CREATE:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")
    return created_dirs

def get_model_config():
    """Get model configuration dictionary"""
    return {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "max_new_tokens": MAX_GENERATION_LENGTH,
        "num_return_sequences": NUM_RETURN_SEQUENCES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": DO_SAMPLE,
        "use_4bit_quantization": USE_4BIT_QUANTIZATION,
        "quantization_config": QUANTIZATION_CONFIG
    }

def get_tokenizer_config():
    """Get tokenizer configuration dictionary"""
    return {
        "model_path": TOKENIZER_MODEL_PATH,
        "vocab_path": TOKENIZER_VOCAB_PATH,
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

def get_logging_config():
    """Get logging configuration dictionary"""
    return {
        "level": LOG_LEVEL,
        "format": LOG_FORMAT,
        "detailed_format": DETAILED_LOG_FORMAT,
        "log_file": LOG_FILE,
        "log_file_exists": os.path.exists(LOG_FILE) if LOG_FILE else False
    }

def print_startup_info():
    """Print startup information to console"""
    print("\n" + "=" * 80)
    print(f"🚀 {API_TITLE} v{API_VERSION}")
    print("=" * 80)
    print(f"📍 Host: {API_HOST}:{API_PORT}")
    print(f"🔧 Debug Mode: {DEBUG_MODE}")
    print(f"📊 Log Level: {LOG_LEVEL}")
    print(f"📝 Log File: {LOG_FILE}")
    print(f"🌐 Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"🤖 Model name (fallback): {MODEL_NAME}")
    print(f"📚 Tokenizer (SentencePiece): {TOKENIZER_MODEL_PATH}")
    if MODEL_PATH:
        print(f"📂 Local model path: {MODEL_PATH}")
    print(f"⚡ 4-bit Quantization: {'Enabled' if USE_4BIT_QUANTIZATION else 'Disabled'}")
    print("=" * 80)

# Environment-specific overrides
ENV = os.getenv("ENVIRONMENT", "development").lower()

if ENV == "production":
    DEBUG_MODE = False
    API_HOST = "0.0.0.0"
    LOG_LEVEL = "WARNING"
    print("🔒 Production environment detected")
elif ENV == "development":
    DEBUG_MODE = True
    LOG_LEVEL = LOG_LEVEL or "DEBUG"
    print("🔧 Development environment detected")
elif ENV == "testing":
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    API_PORT = 8001
    print("🧪 Testing environment detected")

# Override settings from environment variables if present
API_HOST = os.getenv("API_HOST", API_HOST)
API_PORT = int(os.getenv("API_PORT", API_PORT))
LOG_LEVEL = os.getenv("LOG_LEVEL", LOG_LEVEL).upper()
DEBUG_MODE = os.getenv("DEBUG_MODE", str(DEBUG_MODE)).lower() == "true"

# Validate LOG_LEVEL
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in VALID_LOG_LEVELS:
    print(f"⚠️  Invalid LOG_LEVEL '{LOG_LEVEL}', defaulting to 'INFO'")
    LOG_LEVEL = "INFO"

# Print configuration on import
if os.getenv("SILENT_START", "false").lower() != "true":
    print_startup_info()
