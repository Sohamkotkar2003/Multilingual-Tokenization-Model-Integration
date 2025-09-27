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

"""
Enhanced API endpoints with Knowledge Base integration

Add these endpoints to your existing app.py to enable the complete Q&A pipeline.
These endpoints provide the missing KB integration functionality.

Instructions:
1. Save the kb_integration.py file in your project directory
2. Add these imports and endpoints to your existing app.py file
3. The endpoints will integrate with your existing FastAPI app
"""

import time
import uuid
import torch
from fastapi import HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# Import the KB integration module
from kb_integration import process_qa_query, get_kb_stats, QueryType

# Additional request/response models for KB integration
class QARequest(BaseModel):
    text: str = Field(..., description="User query text")
    language: Optional[str] = Field(None, description="Override detected language")
    user_id: Optional[str] = Field(None, description="User identifier for session tracking")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation context")
    context: Optional[str] = Field(None, description="Additional context for the query")
    generate_response: bool = Field(True, description="Whether to generate a response using the language model")
    max_response_length: Optional[int] = Field(256, description="Maximum response length")

class QAResponse(BaseModel):
    answer: str
    generated_response: Optional[str] = None
    query: str
    language: str
    confidence: float
    sources: list
    query_type: str
    processing_time: float
    metadata: Dict[str, Any]

class MultilingualConversationRequest(BaseModel):
    text: str
    language: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    generate_response: bool = Field(True, description="Whether to generate a response or just process")
    max_response_length: Optional[int] = Field(256, description="Maximum response length")

class MultilingualConversationResponse(BaseModel):
    user_query: str
    kb_answer: str
    generated_response: Optional[str] = None
    language: str
    confidence: float
    query_type: str
    sources: list
    session_id: str
    processing_time: float
    metadata: Dict[str, Any]

# Add these endpoints to your existing FastAPI app

@app.post("/qa", response_model=QAResponse)
async def knowledge_base_qa(request: QARequest):
    """
    Complete Q&A pipeline with Knowledge Base integration
    
    This endpoint processes user queries through the complete pipeline:
    User → Language Detection → KB Query → Response Generation
    """
    try:
        start_time = time.time()
        
        # Detect language if not provided
        detected_lang = request.language
        if not detected_lang:
            detected_lang, _ = detect_language(request.text)
            if detected_lang not in settings.SUPPORTED_LANGUAGES:
                detected_lang = settings.DEFAULT_LANGUAGE
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process through KB integration
        answer, metadata = await process_qa_query(
            query_text=request.text,
            detected_language=detected_lang,
            user_id=request.user_id,
            session_id=session_id
        )
        
        generated_response = None
        
        # Generate enhanced response using language model if requested
        if request.generate_response and model and hf_tokenizer:
            try:
                logger.info(f"Starting response generation for query: {request.text[:50]}...")
                # Create prompt combining KB answer with original query
                if detected_lang == "hindi":
                    prompt = f"प्रश्न: {request.text}\nज्ञान आधार का उत्तर: {answer}\nबेहतर उत्तर:"
                elif detected_lang == "sanskrit":
                    prompt = f"प्रश्नः {request.text}\nज्ञान आधारस्य उत्तरम्: {answer}\nउत्तमं उत्तरम्:"
                elif detected_lang == "marathi":
                    prompt = f"प्रश्न: {request.text}\nज्ञान आधाराचे उत्तर: {answer}\nचांगले उत्तर:"
                else:
                    prompt = f"Question: {request.text}\nKnowledge Base Answer: {answer}\nImproved Answer:"
                
                logger.info(f"Generated prompt: {prompt[:100]}...")
                
                # Generate response
                inputs = hf_tokenizer(prompt, return_tensors="pt")
                
                # Move inputs to the same device as the model
                if hasattr(model, 'device') and model.device.type != 'cpu':
                    inputs = inputs.to(model.device)
                elif torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                logger.info(f"Starting model generation with max_new_tokens: {min(request.max_response_length or 256, settings.MAX_GENERATION_LENGTH)}")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(request.max_response_length or 256, settings.MAX_GENERATION_LENGTH),
                        temperature=settings.TEMPERATURE,
                        top_p=settings.TOP_P,
                        do_sample=settings.DO_SAMPLE,
                        pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                        eos_token_id=hf_tokenizer.eos_token_id
                    )
                
                full_response = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Full model response: {full_response[:200]}...")
                
                # Extract only the generated part after the prompt
                if ":" in full_response:
                    generated_response = full_response.split(":")[-1].strip()
                else:
                    generated_response = full_response[len(prompt):].strip()
                
                logger.info(f"Extracted generated response: {generated_response[:100]}...")
                    
            except Exception as e:
                logger.error(f"Response generation failed: {e}", exc_info=True)
                generated_response = None
        else:
            logger.info(f"Generation skipped - generate_response: {request.generate_response}, model: {model is not None}, tokenizer: {hf_tokenizer is not None}")
        
        return QAResponse(
            answer=answer,
            generated_response=generated_response,
            query=request.text,
            language=detected_lang,
            confidence=metadata.get("kb_confidence", 0.5),
            sources=metadata.get("kb_sources", []),
            query_type=metadata.get("query_type", "conversational"),
            processing_time=time.time() - start_time,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Q&A processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Q&A processing failed: {e}")

@app.post("/multilingual-conversation", response_model=MultilingualConversationResponse)
async def multilingual_conversation(request: MultilingualConversationRequest):
    """
    Complete multilingual conversation pipeline
    
    This endpoint handles:
    1. Language detection
    2. KB query processing
    3. Optional response generation using the LM
    4. Conversation context tracking
    """
    try:
        start_time = time.time()
        
        # Detect language
        detected_lang = request.language
        if not detected_lang:
            detected_lang, _ = detect_language(request.text)
            if detected_lang not in settings.SUPPORTED_LANGUAGES:
                detected_lang = settings.DEFAULT_LANGUAGE
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process KB query
        kb_answer, metadata = await process_qa_query(
            query_text=request.text,
            detected_language=detected_lang,
            user_id=request.user_id,
            session_id=session_id
        )
        
        generated_response = None
        
        # Generate additional response using LM if requested
        if request.generate_response and model and hf_tokenizer:
            try:
                # Create prompt combining KB answer with original query
                if detected_lang == "hindi":
                    prompt = f"प्रश्न: {request.text}\nज्ञान आधार का उत्तर: {kb_answer}\nबेहतर उत्तर:"
                elif detected_lang == "sanskrit":
                    prompt = f"प्रश्नः {request.text}\nज्ञान आधारस्य उत्तरम्: {kb_answer}\nउत्तमं उत्तरम्:"
                elif detected_lang == "marathi":
                    prompt = f"प्रश्न: {request.text}\nज्ञान आधाराचे उत्तर: {kb_answer}\nचांगले उत्तर:"
                else:
                    prompt = f"Question: {request.text}\nKnowledge Base Answer: {kb_answer}\nImproved Answer:"
                
                # Generate response
                inputs = hf_tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(request.max_response_length or 256, settings.MAX_GENERATION_LENGTH),
                        temperature=settings.TEMPERATURE,
                        top_p=settings.TOP_P,
                        do_sample=settings.DO_SAMPLE,
                        pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                        eos_token_id=hf_tokenizer.eos_token_id
                    )
                
                full_response = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the generated part after the prompt
                if ":" in full_response:
                    generated_response = full_response.split(":")[-1].strip()
                else:
                    generated_response = full_response[len(prompt):].strip()
                    
            except Exception as e:
                logger.warning(f"Response generation failed: {e}")
                generated_response = None
        
        return MultilingualConversationResponse(
            user_query=request.text,
            kb_answer=kb_answer,
            generated_response=generated_response,
            language=detected_lang,
            confidence=metadata.get("kb_confidence", 0.5),
            query_type=metadata.get("query_type", "conversational"),
            sources=metadata.get("kb_sources", []),
            session_id=session_id,
            processing_time=time.time() - start_time,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Multilingual conversation processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversation processing failed: {e}")

@app.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        from kb_integration import qa_orchestrator
        
        if session_id in qa_orchestrator.conversation_history:
            history = qa_orchestrator.conversation_history[session_id]
            
            # Format history as Q&A pairs
            formatted_history = []
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    formatted_history.append({
                        "query": history[i],
                        "response": history[i + 1],
                        "timestamp": i // 2 + 1  # Simple sequence number
                    })
            
            return {
                "session_id": session_id,
                "history": formatted_history,
                "total_exchanges": len(formatted_history)
            }
        else:
            return {
                "session_id": session_id,
                "history": [],
                "total_exchanges": 0
            }
            
    except Exception as e:
        logger.error(f"Failed to retrieve conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {e}")

@app.delete("/conversation/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        from kb_integration import qa_orchestrator
        
        if session_id in qa_orchestrator.conversation_history:
            del qa_orchestrator.conversation_history[session_id]
            return {"message": f"Conversation history cleared for session {session_id}"}
        else:
            return {"message": f"No history found for session {session_id}"}
            
    except Exception as e:
        logger.error(f"Failed to clear conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {e}")

@app.get("/stats")
async def get_api_statistics():
    """Get comprehensive API statistics including KB integration stats"""
    try:
        kb_stats = get_kb_stats()
        
        # Combine with existing health check info
        base_stats = {
            "status": "API is running",
            "tokenizer_loaded": sp_tokenizer is not None,
            "model_loaded": model is not None,
            "api_version": settings.API_VERSION,
            "supported_languages": settings.SUPPORTED_LANGUAGES
        }
        
        return {
            **base_stats,
            "kb_integration": kb_stats,
            "endpoints": {
                "tokenization": "/tokenize",
                "generation": "/generate", 
                "language_detection": "/language-detect",
                "knowledge_base_qa": "/qa",
                "multilingual_conversation": "/multilingual-conversation",
                "conversation_history": "/conversation/{session_id}/history"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

@app.post("/test-language-switching")
async def test_language_switching(request: QARequest):
    """
    Test endpoint for mid-conversation language switching capability
    
    This tests the requirement from Day 6: "Test switching between languages mid-conversation"
    """
    try:
        results = []
        
        # Test with multiple language variations of the same query
        test_queries = [
            {"text": request.text, "lang": "original"},
            {"text": "Please explain in English", "lang": "english"},
            {"text": "हिंदी में बताइए", "lang": "hindi"}, 
            {"text": "मराठीत सांगा", "lang": "marathi"}
        ]
        
        session_id = request.session_id or str(uuid.uuid4())
        
        for test_query in test_queries:
            detected_lang, _ = detect_language(test_query["text"])
            if detected_lang not in settings.SUPPORTED_LANGUAGES:
                detected_lang = settings.DEFAULT_LANGUAGE
                
            answer, metadata = await process_qa_query(
                query_text=test_query["text"],
                detected_language=detected_lang,
                user_id=request.user_id,
                session_id=session_id
            )
            
            generated_response = None
            
            # Generate enhanced response using language model if requested
            if request.generate_response and model and hf_tokenizer:
                try:
                    logger.info(f"Starting response generation for language switching query: {test_query['text'][:50]}...")
                    # Create prompt combining KB answer with original query
                    if detected_lang == "hindi":
                        prompt = f"प्रश्न: {test_query['text']}\nज्ञान आधार का उत्तर: {answer}\nबेहतर उत्तर:"
                    elif detected_lang == "sanskrit":
                        prompt = f"प्रश्नः {test_query['text']}\nज्ञान आधारस्य उत्तरम्: {answer}\nउत्तमं उत्तरम्:"
                    elif detected_lang == "marathi":
                        prompt = f"प्रश्न: {test_query['text']}\nज्ञान आधाराचे उत्तर: {answer}\nचांगले उत्तर:"
                    else:
                        prompt = f"Question: {test_query['text']}\nKnowledge Base Answer: {answer}\nImproved Answer:"
                    
                    logger.info(f"Generated prompt for {detected_lang}: {prompt[:100]}...")
                    
                    # Generate response
                    inputs = hf_tokenizer(prompt, return_tensors="pt")
                    
                    # Move inputs to the same device as the model
                    if hasattr(model, 'device') and model.device.type != 'cpu':
                        inputs = inputs.to(model.device)
                    elif torch.cuda.is_available():
                        inputs = inputs.to("cuda")
                    
                    logger.info(f"Starting model generation for {detected_lang} with max_new_tokens: {min(request.max_response_length or 256, settings.MAX_GENERATION_LENGTH)}")
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=min(request.max_response_length or 256, settings.MAX_GENERATION_LENGTH),
                            temperature=settings.TEMPERATURE,
                            top_p=settings.TOP_P,
                            do_sample=settings.DO_SAMPLE,
                            pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                            eos_token_id=hf_tokenizer.eos_token_id
                        )
                    
                    full_response = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"Full model response for {detected_lang}: {full_response[:200]}...")
                    
                    # Extract only the generated part after the prompt
                    if ":" in full_response:
                        generated_response = full_response.split(":")[-1].strip()
                    else:
                        generated_response = full_response[len(prompt):].strip()
                    
                    logger.info(f"Extracted generated response for {detected_lang}: {generated_response[:100]}...")
                        
                except Exception as e:
                    logger.error(f"Response generation failed for {detected_lang}: {e}", exc_info=True)
                    generated_response = None
            else:
                logger.info(f"Generation skipped for {detected_lang} - generate_response: {request.generate_response}, model: {model is not None}, tokenizer: {hf_tokenizer is not None}")
            
            results.append({
                "query": test_query["text"],
                "expected_language": test_query["lang"],
                "detected_language": detected_lang,
                "answer": answer,
                "generated_response": generated_response,
                "confidence": metadata.get("kb_confidence", 0.0)
            })
        
        return {
            "test_type": "language_switching",
            "session_id": session_id,
            "results": results,
            "switching_successful": len(set(r["detected_language"] for r in results)) > 1
        }
        
    except Exception as e:
        logger.error(f"Language switching test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language switching test failed: {e}")

# Health check endpoint with enhanced status
@app.get("/health")
async def health_check_detailed():
    """Enhanced health check with all system components"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "tokenizer": {
                    "loaded": sp_tokenizer is not None,
                    "status": "ok" if sp_tokenizer else "not_loaded"
                },
                "model": {
                    "loaded": model is not None,
                    "status": "ok" if model else "not_loaded",
                    "device": "cuda" if torch.cuda.is_available() and model else "cpu"
                },
                "kb_integration": {
                    "configured": bool(settings.KB_ENDPOINT),
                    "endpoint": settings.KB_ENDPOINT or "not_configured (using mock responses)",
                    "status": "configured" if settings.KB_ENDPOINT else "mock_mode",
                    "note": "KB endpoint not configured - using enhanced mock responses for QA queries"
                },
                "vaani_tts": {
                    "configured": bool(settings.VAANI_ENDPOINT),
                    "endpoint": settings.VAANI_ENDPOINT or "not_configured",
                    "status": "configured" if settings.VAANI_ENDPOINT else "not_configured"
                }
            },
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "api_version": settings.API_VERSION
        }
        
        # Check if any critical components are missing
        critical_issues = []
        if not sp_tokenizer:
            critical_issues.append("tokenizer_not_loaded")
        if not model:
            critical_issues.append("model_not_loaded")
            
        if critical_issues:
            health_status["status"] = "degraded"
            health_status["issues"] = critical_issues
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

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
    import platform
    
    cfg = settings.get_api_config()
    
    # Windows-specific uvicorn configuration to avoid file descriptor issues
    if platform.system() == "Windows":
        uvicorn.run(
            "app:app", 
            host=cfg["host"], 
            port=cfg["port"], 
            reload=False,  # Disable reload on Windows to avoid file descriptor issues
            log_level=settings.LOG_LEVEL.lower(),
            workers=1,  # Single worker to avoid multiprocessing issues
            loop="asyncio"  # Use asyncio loop explicitly
        )
    else:
        uvicorn.run(
            "app:app", 
            host=cfg["host"], 
            port=cfg["port"], 
            reload=cfg["debug"], 
            log_level=settings.LOG_LEVEL.lower()
        )
