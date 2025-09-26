from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sentencepiece as smp
import uvicorn
import os
from typing import Optional
import logging
from pathlib import Path

# Import settings
import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create necessary directories
settings.create_directories()

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
    
    try:
        # Load SentencePiece tokenizer
        if settings.TOKENIZER_MODEL_PATH.exists():
            sp_tokenizer = smp.SentencePieceProcessor()
            sp_tokenizer.Load(str(settings.TOKENIZER_MODEL_PATH))
            logger.info(f"Custom multilingual tokenizer loaded from {settings.TOKENIZER_MODEL_PATH}")
        else:
            logger.warning(f"Custom tokenizer not found at {settings.TOKENIZER_MODEL_PATH}, will need to create one")
            sp_tokenizer = None
        
        # Load model - placeholder implementation
        # Replace this with your actual model loading logic
        if settings.MODEL_PATH.exists():
            # model = AutoModelForCausalLM.from_pretrained(str(settings.MODEL_PATH))
            logger.info(f"Model loading placeholder - integrate your decoder-only model from {settings.MODEL_PATH}")
        else:
            # model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME)
            logger.info(f"Model path not found, would load {settings.MODEL_NAME} as fallback")
        
        model = None  # Placeholder
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        sp_tokenizer = None
        model = None

# Load models when the module is imported
load_models()

def detect_language(text: str) -> tuple:
    """
    Enhanced language detection using settings configuration
    """
    # Count character types using settings
    devanagari_chars = sum(1 for char in text 
                          if settings.DEVANAGARI_UNICODE_RANGE[0] <= ord(char) <= settings.DEVANAGARI_UNICODE_RANGE[1])
    english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return settings.DEFAULT_LANGUAGE, 0.0
    
    devanagari_ratio = devanagari_chars / total_chars
    english_ratio = english_chars / total_chars
    
    # Check English first
    if english_ratio > settings.ENGLISH_RATIO_THRESHOLD:
        return "english", english_ratio
    
    # Check Devanagari languages
    if devanagari_ratio > settings.DEVANAGARI_RATIO_THRESHOLD:
        # Use keyword-based detection for specific languages
        text_lower = text.lower()
        
        # Check for Sanskrit keywords
        sanskrit_matches = sum(1 for keyword in settings.LANGUAGE_KEYWORDS["sanskrit"] 
                              if keyword in text)
        if sanskrit_matches > 0:
            confidence = min(devanagari_ratio + (sanskrit_matches * 0.1), 1.0)
            return "sanskrit", confidence
        
        # Check for Marathi keywords
        marathi_matches = sum(1 for keyword in settings.LANGUAGE_KEYWORDS["marathi"] 
                             if keyword in text)
        if marathi_matches > 0:
            confidence = min(devanagari_ratio + (marathi_matches * 0.1), 1.0)
            return "marathi", confidence
        
        # Check for Hindi keywords
        hindi_matches = sum(1 for keyword in settings.LANGUAGE_KEYWORDS["hindi"] 
                           if keyword in text)
        if hindi_matches > 0:
            confidence = min(devanagari_ratio + (hindi_matches * 0.1), 1.0)
            return "hindi", confidence
        
        # Default to Hindi for Devanagari text without specific keywords
        return "hindi", devanagari_ratio
    
    # Mixed or unknown
    max_confidence = max(devanagari_ratio, english_ratio)
    if max_confidence > settings.LANGUAGE_CONFIDENCE_THRESHOLD:
        return "mixed", max_confidence
    else:
        return settings.DEFAULT_LANGUAGE, max_confidence

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
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

@app.post("/language-detect", response_model=LanguageDetectResponse)
async def detect_language_endpoint(request: TextRequest):
    """Detect the language of input text"""
    try:
        language, confidence = detect_language(request.text)
        return LanguageDetectResponse(
            language=language,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TextRequest):
    """Tokenize input text using the multilingual tokenizer"""
    if sp_tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Detect language if not provided
        if request.language is None:
            detected_lang, _ = detect_language(request.text)
        else:
            detected_lang = request.language.lower()
            
        # Validate language
        if detected_lang not in settings.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language detected: {detected_lang}")
            detected_lang = settings.DEFAULT_LANGUAGE
        
        # Tokenize using SentencePiece
        tokens = sp_tokenizer.EncodeAsPieces(request.text)
        token_ids = sp_tokenizer.EncodeAsIds(request.text)
        
        logger.info(f"Tokenized text in {detected_lang}: {len(tokens)} tokens")
        
        return TokenizeResponse(
            language=detected_lang,
            tokens=tokens,
            token_ids=token_ids,
            input_text=request.text
        )
    except Exception as e:
        logger.error(f"Error in tokenization: {e}")
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: TextRequest):
    """Generate text using the integrated multilingual model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if sp_tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Detect language if not provided
        if request.language is None:
            detected_lang, _ = detect_language(request.text)
        else:
            detected_lang = request.language.lower()
            
        # Validate language
        if detected_lang not in settings.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language detected: {detected_lang}")
            detected_lang = settings.DEFAULT_LANGUAGE
        
        # Tokenize input
        token_ids = sp_tokenizer.EncodeAsIds(request.text)
        input_ids = torch.tensor([token_ids])
        
        logger.info(f"Generating text for {detected_lang} with {len(token_ids)} input tokens")
        
        # Generate text using settings configuration
        # This is a placeholder - replace with your actual generation logic
        with torch.no_grad():
            # Use settings for generation parameters
            # outputs = model.generate(
            #     input_ids,
            #     max_length=settings.MAX_GENERATION_LENGTH,
            #     num_return_sequences=settings.NUM_RETURN_SEQUENCES,
            #     temperature=settings.TEMPERATURE,
            #     top_p=settings.TOP_P,
            #     do_sample=settings.DO_SAMPLE,
            #     pad_token_id=sp_tokenizer.pad_id()
            # )
            
            # Placeholder: echo back input with some modification
            generated_ids = token_ids + token_ids[:5]  # Simple repetition for demo
        
        # Decode generated tokens
        generated_text = sp_tokenizer.DecodeIds(generated_ids)
        
        logger.info(f"Generated {len(generated_ids)} tokens")
        
        return GenerateResponse(
            language=detected_lang,
            generated_text=generated_text,
            input_text=request.text
        )
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model and tokenizer"""
    tokenizer_info = settings.get_tokenizer_config()
    model_info = settings.get_model_config()
    
    return {
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

@app.get("/config")
async def get_configuration():
    """Get current API configuration"""
    return {
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

@app.on_event("startup")
async def startup_event():
    """Startup event to log configuration"""
    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG_MODE}")
    logger.info(f"Supported languages: {settings.SUPPORTED_LANGUAGES}")
    logger.info(f"Tokenizer path: {settings.TOKENIZER_MODEL_PATH}")
    logger.info(f"Model path: {settings.MODEL_PATH}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API")

if __name__ == "__main__":
    # Get API configuration from settings
    api_config = settings.get_api_config()
    
    # Use uvicorn to run the application
    uvicorn.run(
        "app:app",
        host=api_config["host"],
        port=api_config["port"],
        reload=api_config["debug"],
        log_level=settings.LOG_LEVEL.lower()
    )