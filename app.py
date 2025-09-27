from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sentencepiece as spm
import uvicorn
import os
from typing import Optional
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import settings
from core import settings

# =============================================================================
# Logging Setup
# =============================================================================
# Ensure directories exist first
settings.create_directories()

log_handlers = [logging.StreamHandler()]  # Console output
if settings.LOG_FILE:
    log_handlers.append(logging.FileHandler(settings.LOG_FILE, encoding="utf-8"))

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format=settings.LOG_FORMAT,
    handlers=log_handlers,
    force=True
)
logger = logging.getLogger(__name__)

logger.info("=== API Starting - Logging Config Applied ===")

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    debug=settings.DEBUG_MODE
)

# Global variables
sp_tokenizer = None
model = None
hf_tokenizer = None  # Hugging Face tokenizer for decoder-only LM

# =============================================================================
# Request / Response Schemas
# =============================================================================
class TextRequest(BaseModel):
    text: str
    language: Optional[str] = None

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

# =============================================================================
# Model / Tokenizer Loading
# =============================================================================
def load_models():
    """Load SentencePiece tokenizer and HuggingFace model with optional 4-bit quantization"""
    global sp_tokenizer, model, hf_tokenizer

    try:
        # Load SentencePiece tokenizer
        if settings.TOKENIZER_MODEL_PATH and os.path.exists(settings.TOKENIZER_MODEL_PATH):
            sp_tokenizer = spm.SentencePieceProcessor()
            sp_tokenizer.Load(str(settings.TOKENIZER_MODEL_PATH))
            logger.info(f"✅ SentencePiece tokenizer loaded from {settings.TOKENIZER_MODEL_PATH}")
        else:
            logger.warning(f"❌ Tokenizer file not found at {settings.TOKENIZER_MODEL_PATH}")
            sp_tokenizer = None

        # Load HuggingFace model + tokenizer
        model_source = settings.MODEL_PATH if settings.MODEL_PATH else settings.MODEL_NAME
        hf_tokenizer = AutoTokenizer.from_pretrained(model_source)
        
        # Configure quantization if enabled
        quantization_config = None
        if settings.USE_4BIT_QUANTIZATION and torch.cuda.is_available():
            try:
                quantization_config = BitsAndBytesConfig(**settings.QUANTIZATION_CONFIG)
                logger.info("🔧 Using 4-bit quantization for faster inference")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create quantization config: {e}")
                logger.info("🔄 Falling back to standard model loading")
                quantization_config = None
        
        # Load model with or without quantization
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() and settings.USE_FP16_IF_GPU else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"  # Let bitsandbytes handle device placement
        
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        model.eval()
        
        # Move to GPU only if not using quantization (quantization handles device placement)
        if torch.cuda.is_available() and not quantization_config:
            model.to("cuda")
            logger.info(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
        elif quantization_config:
            logger.info("✅ Model loaded with 4-bit quantization")
        else:
            logger.info("✅ Model loaded on CPU")
            
        logger.info(f"✅ Model loaded: {model_source}")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}", exc_info=True)
        sp_tokenizer = None
        model = None
        hf_tokenizer = None

# =============================================================================
# Language Detection
# =============================================================================
def detect_language(text: str) -> tuple:
    """Detect language based on Unicode ranges + keywords"""
    logger.debug(f"Detecting language for text: '{text[:50]}...'")

    devanagari_chars = sum(1 for c in text if settings.DEVANAGARI_UNICODE_RANGE[0] <= ord(c) <= settings.DEVANAGARI_UNICODE_RANGE[1])
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total_chars = len([c for c in text if c.isalpha()])

    if total_chars == 0:
        return settings.DEFAULT_LANGUAGE, 0.0

    devanagari_ratio = devanagari_chars / total_chars
    english_ratio = english_chars / total_chars

    if english_ratio > settings.ENGLISH_RATIO_THRESHOLD:
        return "english", english_ratio

    if devanagari_ratio > settings.DEVANAGARI_RATIO_THRESHOLD:
        for lang, keywords in settings.LANGUAGE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return lang, min(devanagari_ratio + 0.2, 1.0)
        return "hindi", devanagari_ratio

    max_conf = max(devanagari_ratio, english_ratio)
    if max_conf > settings.LANGUAGE_CONFIDENCE_THRESHOLD:
        return "mixed", max_conf
    return settings.DEFAULT_LANGUAGE, max_conf

# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/")
async def health_check():
    return {
        "status": "API is running",
        "tokenizer_loaded": sp_tokenizer is not None,
        "model_loaded": model is not None,
        "api_version": settings.API_VERSION,
        "supported_languages": settings.SUPPORTED_LANGUAGES
    }

@app.post("/language-detect", response_model=LanguageDetectResponse)
async def detect_language_endpoint(request: TextRequest):
    try:
        lang, conf = detect_language(request.text)
        return LanguageDetectResponse(language=lang, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TextRequest):
    if not sp_tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")

    detected_lang, _ = detect_language(request.text) if not request.language else (request.language, 1.0)
    if detected_lang not in settings.SUPPORTED_LANGUAGES:
        detected_lang = settings.DEFAULT_LANGUAGE

    tokens = sp_tokenizer.EncodeAsPieces(request.text)
    ids = sp_tokenizer.EncodeAsIds(request.text)

    return TokenizeResponse(language=detected_lang, tokens=tokens, token_ids=ids, input_text=request.text)

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: TextRequest):
    if not model or not hf_tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    detected_lang, _ = detect_language(request.text) if not request.language else (request.language, 1.0)
    if detected_lang not in settings.SUPPORTED_LANGUAGES:
        detected_lang = settings.DEFAULT_LANGUAGE

    try:
        # Tokenize input text
        inputs = hf_tokenizer(request.text, return_tensors="pt")
        
        # Move inputs to the same device as the model
        # For quantized models, device_map="auto" handles placement automatically
        if hasattr(model, 'device') and model.device.type != 'cpu':
            inputs = inputs.to(model.device)
        elif torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=settings.MAX_GENERATION_LENGTH,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=settings.DO_SAMPLE,
                num_return_sequences=settings.NUM_RETURN_SEQUENCES,
                pad_token_id=hf_tokenizer.eos_token_id  # Ensure proper padding
            )
        generated_text = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return GenerateResponse(language=detected_lang, generated_text=generated_text, input_text=request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {e}")

@app.get("/config")
async def get_configuration():
    return {
        "api": settings.get_api_config(),
        "model": settings.get_model_config(),
        "tokenizer": settings.get_tokenizer_config(),
        "languages": {
            "supported": settings.SUPPORTED_LANGUAGES,
            "default": settings.DEFAULT_LANGUAGE
        },
        "kb_endpoint": settings.KB_ENDPOINT,
        "vaani_endpoint": settings.VAANI_ENDPOINT
    }

# =============================================================================
# Startup / Shutdown Hooks
# =============================================================================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 API Startup")
    load_models()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔌 API Shutdown")

# =============================================================================
# Run with uvicorn
# =============================================================================
if __name__ == "__main__":
    cfg = settings.get_api_config()
    uvicorn.run("app:app", host=cfg["host"], port=cfg["port"], reload=cfg["debug"], log_level=settings.LOG_LEVEL.lower())
