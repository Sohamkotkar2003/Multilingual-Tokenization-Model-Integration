from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sentencepiece as smp
import uvicorn
import os
from typing import Optional
import logging
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Import settings
from core import settings

# Configure logging BEFORE creating the logger
# Create log directory first
settings.create_directories()

# Set up file and console handlers
log_handlers = [
    logging.StreamHandler(),  # Console output
]

# Add file handler if log file path is specified
if hasattr(settings, 'LOG_FILE') and settings.LOG_FILE:
    log_handlers.append(logging.FileHandler(settings.LOG_FILE, encoding='utf-8'))

# Configure root logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
    handlers=log_handlers,
    force=True  # Override any existing logging configuration
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Test logging immediately
logger.info("=== API Starting - Logging Configuration Test ===")
logger.debug("Debug logging is working")
logger.info("Info logging is working") 
logger.warning("Warning logging is working")

# Initialize FastAPI app with settings
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Global variables for model and tokenizer
sp_tokenizer = None
model = None

class TextRequest(BaseModel):
    text: str
    language: Optional[str] = None  # Optional language hint

class TokenizeResponse(BaseModel):
    language: str
    tokens: list
    token_ids: list
    input_text: str

class GenerateResponse(BaseModel):
    language: str
    generated_text: str
    input_text: str

class LanguageDetectResponse(BaseModel):
    language: str
    confidence: float

def load_models():
    """Load the tokenizer and model on startup"""
    global sp_tokenizer, model
    
    logger.info("Starting model loading process...")
    
    try:
        # Load SentencePiece tokenizer
        logger.info(f"Checking tokenizer path: {settings.TOKENIZER_MODEL_PATH}")
        if settings.TOKENIZER_MODEL_PATH:
            sp_tokenizer = smp.SentencePieceProcessor()
            sp_tokenizer.Load(str(settings.TOKENIZER_MODEL_PATH))
            logger.info(f"✅ Custom multilingual tokenizer loaded from {settings.TOKENIZER_MODEL_PATH}")
        else:
            logger.warning(f"❌ Custom tokenizer not found at {settings.TOKENIZER_MODEL_PATH}")
            logger.info("Will need to create tokenizer before using the API")
            sp_tokenizer = None
        
        # Load model - placeholder implementation
        logger.info(f"Checking model path: {settings.MODEL_PATH}")
        if settings.MODEL_PATH:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(settings.MODEL_PATH), dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            model.eval()
            if torch.cuda.is_available():
                model.to("cuda")
            logger.info(f"✅ Model loaded from checkpoint: {settings.MODEL_PATH}")
        else:
            # model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME)
            logger.warning(f"❌ Model path not found at {settings.MODEL_PATH}")
            logger.info(f"Would load {settings.MODEL_NAME} as fallback if implemented")
        
        if model is not None:
            logger.info("Model loading completed successfully")
        else:
            logger.warning("Model loading completed, but model is not available")

        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}", exc_info=True)
        sp_tokenizer = None
        model = None

def detect_language(text: str) -> tuple:
    """
    Enhanced language detection using settings configuration
    """
    logger.debug(f"Detecting language for text: '{text[:50]}...'")
    
    # Count character types using settings
    devanagari_chars = sum(1 for char in text 
                          if settings.DEVANAGARI_UNICODE_RANGE[0] <= ord(char) <= settings.DEVANAGARI_UNICODE_RANGE[1])
    english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    total_chars = len([c for c in text if c.isalpha()])
    
    logger.debug(f"Character analysis - Devanagari: {devanagari_chars}, English: {english_chars}, Total: {total_chars}")
    
    if total_chars == 0:
        logger.debug(f"No alphabetic characters found, returning default: {settings.DEFAULT_LANGUAGE}")
        return settings.DEFAULT_LANGUAGE, 0.0
    
    devanagari_ratio = devanagari_chars / total_chars
    english_ratio = english_chars / total_chars
    
    logger.debug(f"Language ratios - Devanagari: {devanagari_ratio:.2f}, English: {english_ratio:.2f}")
    
    # Check English first
    if english_ratio > settings.ENGLISH_RATIO_THRESHOLD:
        logger.debug(f"Detected English (ratio: {english_ratio:.2f})")
        return "english", english_ratio
    
    # Check Devanagari languages
    if devanagari_ratio > settings.DEVANAGARI_RATIO_THRESHOLD:
        text_lower = text.lower()
        
        # Check for Sanskrit keywords
        sanskrit_matches = sum(1 for keyword in settings.LANGUAGE_KEYWORDS["sanskrit"] 
                              if keyword in text)
        if sanskrit_matches > 0:
            confidence = min(devanagari_ratio + (sanskrit_matches * 0.1), 1.0)
            logger.debug(f"Detected Sanskrit (matches: {sanskrit_matches}, confidence: {confidence:.2f})")
            return "sanskrit", confidence
        
        # Check for Marathi keywords
        marathi_matches = sum(1 for keyword in settings.LANGUAGE_KEYWORDS["marathi"] 
                             if keyword in text)
        if marathi_matches > 0:
            confidence = min(devanagari_ratio + (marathi_matches * 0.1), 1.0)
            logger.debug(f"Detected Marathi (matches: {marathi_matches}, confidence: {confidence:.2f})")
            return "marathi", confidence
        
        # Check for Hindi keywords
        hindi_matches = sum(1 for keyword in settings.LANGUAGE_KEYWORDS["hindi"] 
                           if keyword in text)
        if hindi_matches > 0:
            confidence = min(devanagari_ratio + (hindi_matches * 0.1), 1.0)
            logger.debug(f"Detected Hindi (matches: {hindi_matches}, confidence: {confidence:.2f})")
            return "hindi", confidence
        
        # Default to Hindi for Devanagari text without specific keywords
        logger.debug(f"Default to Hindi for Devanagari text (ratio: {devanagari_ratio:.2f})")
        return "hindi", devanagari_ratio
    
    # Mixed or unknown
    max_confidence = max(devanagari_ratio, english_ratio)
    if max_confidence > settings.LANGUAGE_CONFIDENCE_THRESHOLD:
        logger.debug(f"Detected mixed language (confidence: {max_confidence:.2f})")
        return "mixed", max_confidence
    else:
        logger.debug(f"Low confidence detection, using default: {settings.DEFAULT_LANGUAGE}")
        return settings.DEFAULT_LANGUAGE, max_confidence

@app.get("/")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint called")
    
    response_data = {
        "status": "API is up and running",
        "tokenizer_loaded": sp_tokenizer is not None,
        "model_loaded": model is not None,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "api_version": settings.API_VERSION,
        "model_config": settings.get_model_config(),
        "tokenizer_config": {
            "vocab_size": sp_tokenizer.GetPieceSize() if sp_tokenizer else 0,
            "model_path": str(settings.TOKENIZER_MODEL_PATH)
        }
    }
    
    logger.info(f"Health check response: tokenizer_loaded={sp_tokenizer is not None}, model_loaded={model is not None}")
    return response_data

@app.post("/language-detect", response_model=LanguageDetectResponse)
async def detect_language_endpoint(request: TextRequest):
    """Detect the language of input text"""
    logger.info(f"Language detection requested for text: '{request.text[:100]}...'")
    
    try:
        language, confidence = detect_language(request.text)
        
        logger.info(f"Language detection result: {language} (confidence: {confidence:.2f})")
        
        return LanguageDetectResponse(
            language=language,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error in language detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TextRequest):
    """Tokenize input text using the multilingual tokenizer"""
    logger.info(f"Tokenization requested for text: '{request.text[:100]}...' (language: {request.language})")
    
    if sp_tokenizer is None:
        logger.error("Tokenization failed: Tokenizer not loaded")
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Detect language if not provided
        if request.language is None:
            detected_lang, confidence = detect_language(request.text)
            logger.info(f"Auto-detected language: {detected_lang} (confidence: {confidence:.2f})")
        else:
            detected_lang = request.language.lower()
            logger.info(f"Using provided language: {detected_lang}")
            
        # Validate language
        if detected_lang not in settings.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language detected: {detected_lang}, falling back to {settings.DEFAULT_LANGUAGE}")
            detected_lang = settings.DEFAULT_LANGUAGE
        
        # Tokenize using SentencePiece
        tokens = sp_tokenizer.EncodeAsPieces(request.text)
        token_ids = sp_tokenizer.EncodeAsIds(request.text)
        
        logger.info(f"✅ Tokenization successful: {len(tokens)} tokens for {detected_lang} text")
        logger.debug(f"Tokens: {tokens[:10]}...")  # Log first 10 tokens
        
        return TokenizeResponse(
            language=detected_lang,
            tokens=tokens,
            token_ids=token_ids,
            input_text=request.text
        )
    except Exception as e:
        logger.error(f"Error in tokenization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: TextRequest):
    """Generate text using the integrated multilingual model"""
    logger.info(f"Text generation requested for: '{request.text[:100]}...' (language: {request.language})")
    
    if model is None:
        logger.error("Text generation failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if sp_tokenizer is None:
        logger.error("Text generation failed: Tokenizer not loaded")
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Detect language if not provided
        if request.language is None:
            detected_lang, confidence = detect_language(request.text)
            logger.info(f"Auto-detected language for generation: {detected_lang} (confidence: {confidence:.2f})")
        else:
            detected_lang = request.language.lower()
            logger.info(f"Using provided language for generation: {detected_lang}")
            
        # Validate language
        if detected_lang not in settings.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language for generation: {detected_lang}, falling back to {settings.DEFAULT_LANGUAGE}")
            detected_lang = settings.DEFAULT_LANGUAGE
        
        # Tokenize input
        token_ids = sp_tokenizer.EncodeAsIds(request.text)
        input_ids = torch.tensor([token_ids])
        
        logger.info(f"Generating text for {detected_lang} with {len(token_ids)} input tokens")
        logger.debug(f"Input token IDs: {token_ids[:10]}...")
        
        # Generate text using settings configuration
        # This is a placeholder - replace with your actual generation logic
        with torch.no_grad():
            # Placeholder: echo back input with some modification
            generated_ids = token_ids + token_ids[:5]  # Simple repetition for demo
            logger.info("Using placeholder generation logic (echo + repeat)")
        
        # Decode generated tokens
        generated_text = sp_tokenizer.DecodeIds(generated_ids)
        
        logger.info(f"✅ Text generation successful: {len(generated_ids)} tokens generated")
        logger.debug(f"Generated text: '{generated_text[:100]}...'")
        
        return GenerateResponse(
            language=detected_lang,
            generated_text=generated_text,
            input_text=request.text
        )
    except Exception as e:
        logger.error(f"Error in text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model and tokenizer"""
    logger.info("Model info endpoint called")
    
    tokenizer_info = settings.get_tokenizer_config()
    model_info = settings.get_model_config()
    
    response_data = {
        "tokenizer_info": {
            **tokenizer_info,
            "loaded": sp_tokenizer is not None,
            "actual_vocab_size": sp_tokenizer.GetPieceSize() if sp_tokenizer else 0,
        },
        "model_info": {
            **model_info,
            "loaded": model is not None,
        },
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "language_keywords": settings.LANGUAGE_KEYWORDS,
        "api_version": settings.API_VERSION,
        "debug_mode": settings.DEBUG_MODE
    }
    
    logger.debug(f"Model info response: {response_data}")
    return response_data

@app.get("/config")
async def get_configuration():
    """Get current API configuration"""
    logger.info("Configuration endpoint called")
    
    config_data = {
        "api": settings.get_api_config(),
        "model": settings.get_model_config(),
        "tokenizer": settings.get_tokenizer_config(),
        "languages": {
            "supported": settings.SUPPORTED_LANGUAGES,
            "default": settings.DEFAULT_LANGUAGE,
            "detection_thresholds": {
                "english_ratio": settings.ENGLISH_RATIO_THRESHOLD,
                "devanagari_ratio": settings.DEVANAGARI_RATIO_THRESHOLD,
                "confidence": settings.LANGUAGE_CONFIDENCE_THRESHOLD
            }
        }
    }
    
    logger.debug("Configuration data retrieved")
    return config_data

@app.on_event("startup")
async def startup_event():
    """Startup event to log configuration"""
    logger.info("=" * 60)
    logger.info(f"🚀 Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info("=" * 60)
    logger.info(f"📍 Debug mode: {settings.DEBUG_MODE}")
    logger.info(f"🌐 Supported languages: {settings.SUPPORTED_LANGUAGES}")
    logger.info(f"📝 Tokenizer path: {settings.TOKENIZER_MODEL_PATH}")
    logger.info(f"🤖 Model path: {settings.MODEL_PATH}")
    logger.info(f"📊 Log level: {settings.LOG_LEVEL}")
    logger.info(f"📁 Log file: {getattr(settings, 'LOG_FILE', 'Console only')}")
    logger.info("=" * 60)
    
    # Load models after startup
    load_models()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=" * 60)
    logger.info("🔌 Shutting down API")
    logger.info("=" * 60)

# Add a logging test endpoint for debugging
@app.get("/test-logging")
async def test_logging():
    """Test endpoint to verify logging is working"""
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    return {
        "message": "Logging test completed - check console and log file",
        "log_level": settings.LOG_LEVEL,
        "log_file": str(getattr(settings, 'LOG_FILE', 'Not configured')),
        "handlers": [str(handler) for handler in logger.handlers]
    }

if __name__ == "__main__":
    # Get API configuration from settings
    api_config = settings.get_api_config()
    
    logger.info(f"Starting server on {api_config['host']}:{api_config['port']}")
    
    # Use uvicorn to run the application
    uvicorn.run(
        "app:app",
        host=api_config["host"],
        port=api_config["port"],
        reload=api_config["debug"],
        log_level=settings.LOG_LEVEL.lower()
    )