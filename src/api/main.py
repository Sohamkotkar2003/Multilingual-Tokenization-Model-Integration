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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings

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
cuda_generation_disabled = False  # Flag to track CUDA issues
model_reloaded_cpu = False  # Flag to track if model was reloaded in CPU mode

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
from src.services.knowledge_base import process_qa_query, get_kb_stats, QueryType

# Additional request/response models for KB integration
class QARequest(BaseModel):
    text: str = Field(..., description="User query text", alias="query")
    language: Optional[str] = Field(None, description="Override detected language")
    user_id: Optional[str] = Field(None, description="User identifier for session tracking")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation context")
    context: Optional[str] = Field(None, description="Additional context for the query")
    generate_response: bool = Field(True, description="Whether to generate a response using the language model")
    max_response_length: Optional[int] = Field(256, description="Maximum response length")
    
    class Config:
        populate_by_name = True

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
        # Skip generation if CUDA issues are detected
        global cuda_generation_disabled
        force_cpu_only = os.getenv("FORCE_CPU_GENERATION", "false").lower() == "true"
        if request.generate_response and model and hf_tokenizer and not force_cpu_only and not cuda_generation_disabled:
            try:
                logger.info(f"Starting response generation for query: {request.text[:50]}...")
                # Create prompt combining KB answer with original query
                if detected_lang == "hindi":
                    prompt = f"प्रश्न: {request.text}\n\nसंक्षिप्त और सटीक उत्तर दें:"
                elif detected_lang == "sanskrit":
                    prompt = f"प्रश्नः {request.text}\n\nसंक्षिप्तं सटीकं च उत्तरं ददातु:"
                elif detected_lang == "marathi":
                    prompt = f"प्रश्न: {request.text}\n\nसंक्षिप्त आणि अचूक उत्तर द्या:"
                elif detected_lang == "tamil":
                    prompt = f"கேள்வி: {request.text}\n\nசுருக்கமான மற்றும் துல்லியமான பதில் கொடுங்கள்:"
                elif detected_lang == "telugu":
                    prompt = f"ప్రశ్న: {request.text}\n\nసంక్షిప్తమైన మరియు ఖచ్చితమైన సమాధానం ఇవ్వండి:"
                elif detected_lang == "bengali":
                    prompt = f"প্রশ্ন: {request.text}\n\nসংক্ষিপ্ত এবং সঠিক উত্তর দিন:"
                else:
                    prompt = f"Question: {request.text}\n\nProvide a concise and accurate answer:"
                
                logger.info(f"Generated prompt: {prompt[:100]}...")
                
                # Generate response
                inputs = hf_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # Validate token IDs to prevent CUDA assertion errors
                input_ids = inputs['input_ids']
                vocab_size = hf_tokenizer.vocab_size
                
                # Check for invalid token IDs
                if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                    logger.warning(f"Invalid token IDs detected. Vocab size: {vocab_size}, Min: {input_ids.min()}, Max: {input_ids.max()}")
                    # Filter out invalid tokens
                    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                    inputs['input_ids'] = input_ids
                
                # Move inputs to the same device as the model
                if hasattr(model, 'device') and model.device.type != 'cpu':
                    inputs = inputs.to(model.device)
                elif torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                logger.info(f"Starting model generation with max_new_tokens: {min(request.max_response_length or 150, settings.MAX_GENERATION_LENGTH)}")
                
                with torch.no_grad():
                    try:
                        # Try GPU generation first
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=min(request.max_response_length or 150, settings.MAX_GENERATION_LENGTH),
                            temperature=0.7,  # Reduced temperature for stability
                            top_p=0.9,
                            top_k=40,  # Reduced top_k for stability
                            do_sample=True,
                            repetition_penalty=1.1,  # Reduced repetition penalty
                            no_repeat_ngram_size=2,  # Reduced n-gram size
                            pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                            eos_token_id=hf_tokenizer.eos_token_id,
                            use_cache=True  # Enable KV cache
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "device-side assert" in str(e):
                            logger.error(f"CUDA error during generation: {e}")
                            logger.info("Attempting CPU fallback with fresh tokenization...")
                            
                            # Clear CUDA cache and restart with fresh tokenization
                            try:
                                torch.cuda.empty_cache()
                                # Re-tokenize on CPU to avoid CUDA context issues
                                inputs_cpu = hf_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                                
                                # Validate token IDs for CPU inputs
                                input_ids_cpu = inputs_cpu['input_ids']
                                vocab_size = hf_tokenizer.vocab_size
                                
                                if torch.any(input_ids_cpu >= vocab_size) or torch.any(input_ids_cpu < 0):
                                    logger.warning(f"Invalid token IDs in CPU fallback. Clamping to valid range.")
                                    input_ids_cpu = torch.clamp(input_ids_cpu, 0, vocab_size - 1)
                                    inputs_cpu['input_ids'] = input_ids_cpu
                                
                                # Move model to CPU
                                model_cpu = model.to("cpu")
                                logger.info("Model moved to CPU for fallback generation")
                                
                                outputs = model_cpu.generate(
                                    **inputs_cpu,
                                    max_new_tokens=50,  # Very conservative for CPU
                                    temperature=0.6,
                                    do_sample=True,
                                    pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                                    eos_token_id=hf_tokenizer.eos_token_id
                                )
                                logger.info("CPU fallback generation successful")
                                
                            except Exception as cpu_error:
                                logger.error(f"CPU fallback also failed: {cpu_error}")
                                
                                # Try to reload model in CPU-only mode
                                if not model_reloaded_cpu:
                                    try:
                                        reload_model_cpu_only()
                                        # Try generation one more time with reloaded model
                                        if model is not None:
                                            logger.info("Attempting generation with reloaded CPU model...")
                                            outputs = model.generate(
                                                **inputs_cpu,
                                                max_new_tokens=30,  # Very conservative
                                                temperature=0.5,
                                                do_sample=True,
                                                pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                                                eos_token_id=hf_tokenizer.eos_token_id
                                            )
                                            logger.info("Generation successful with reloaded CPU model")
                                        else:
                                            outputs = None
                                    except Exception as reload_error:
                                        logger.error(f"Model reload and generation failed: {reload_error}")
                                        # Disable generation for future requests
                                        cuda_generation_disabled = True
                                        logger.warning("CUDA generation disabled for this session due to persistent errors")
                                        outputs = None
                                else:
                                    # Already tried reloading, disable generation
                                    cuda_generation_disabled = True
                                    logger.warning("CUDA generation disabled for this session due to persistent errors")
                                    outputs = None
                        else:
                            raise e
                
                if outputs is not None:
                    full_response = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"Full model response: {full_response[:200]}...")
                    
                    # Extract only the generated part after the prompt
                    if ":" in full_response:
                        generated_response = full_response.split(":")[-1].strip()
                    else:
                        generated_response = full_response[len(prompt):].strip()
                    
                    logger.info(f"Extracted generated response: {generated_response[:100]}...")
                else:
                    logger.warning("No outputs generated - using KB answer only")
                    generated_response = None
                    
            except Exception as e:
                logger.error(f"Response generation failed: {e}", exc_info=True)
                generated_response = None
        else:
            if cuda_generation_disabled:
                logger.info("Generation skipped - CUDA generation disabled due to previous errors")
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
                    prompt = f"प्रश्न: {request.text}\n\nसंक्षिप्त और सटीक उत्तर दें:"
                elif detected_lang == "sanskrit":
                    prompt = f"प्रश्नः {request.text}\n\nसंक्षिप्तं सटीकं च उत्तरं ददातु:"
                elif detected_lang == "marathi":
                    prompt = f"प्रश्न: {request.text}\n\nसंक्षिप्त आणि अचूक उत्तर द्या:"
                elif detected_lang == "tamil":
                    prompt = f"கேள்வி: {request.text}\n\nசுருக்கமான மற்றும் துல்லியமான பதில் கொடுங்கள்:"
                elif detected_lang == "telugu":
                    prompt = f"ప్రశ్న: {request.text}\n\nసంక్షిప్తమైన మరియు ఖచ్చితమైన సమాధానం ఇవ్వండి:"
                elif detected_lang == "bengali":
                    prompt = f"প্রশ্ন: {request.text}\n\nসংক্ষিপ্ত এবং সঠিক উত্তর দিন:"
                else:
                    prompt = f"Question: {request.text}\n\nProvide a concise and accurate answer:"
                
                # Generate response
                inputs = hf_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # Validate token IDs to prevent CUDA assertion errors
                input_ids = inputs['input_ids']
                vocab_size = hf_tokenizer.vocab_size
                
                # Check for invalid token IDs
                if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                    logger.warning(f"Invalid token IDs detected. Vocab size: {vocab_size}, Min: {input_ids.min()}, Max: {input_ids.max()}")
                    # Filter out invalid tokens
                    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                    inputs['input_ids'] = input_ids
                
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
                
                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=min(request.max_response_length or 150, settings.MAX_GENERATION_LENGTH),
                            temperature=0.7,  # Reduced temperature for stability
                            top_p=0.9,
                            top_k=40,  # Reduced top_k for stability
                            do_sample=True,
                            repetition_penalty=1.1,  # Reduced repetition penalty
                            no_repeat_ngram_size=2,  # Reduced n-gram size
                            pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                            eos_token_id=hf_tokenizer.eos_token_id,
                            early_stopping=True,
                            use_cache=True
                        )
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "device-side assert" in str(e):
                            logger.error(f"CUDA error during generation: {e}")
                            # Fallback to CPU generation
                            logger.info("Falling back to CPU generation with conservative settings")
                            inputs = inputs.to("cpu")
                            model_cpu = model.to("cpu")
                            outputs = model_cpu.generate(
                                **inputs,
                                max_new_tokens=50,
                                temperature=0.6,
                                do_sample=True,
                                pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                                eos_token_id=hf_tokenizer.eos_token_id,
                                early_stopping=True
                            )
                        else:
                            raise e
                
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
        from src.services.knowledge_base import qa_orchestrator
        
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
        from src.services.knowledge_base import qa_orchestrator
        
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
                        prompt = f"प्रश्न: {test_query['text']}\n\nसंक्षिप्त और सटीक उत्तर दें:"
                    elif detected_lang == "sanskrit":
                        prompt = f"प्रश्नः {test_query['text']}\n\nसंक्षिप्तं सटीकं च उत्तरं ददातु:"
                    elif detected_lang == "marathi":
                        prompt = f"प्रश्न: {test_query['text']}\n\nसंक्षिप्त आणि अचूक उत्तर द्या:"
                    elif detected_lang == "tamil":
                        prompt = f"கேள்வி: {test_query['text']}\n\nசுருக்கமான மற்றும் துல்லியமான பதில் கொடுங்கள்:"
                    elif detected_lang == "telugu":
                        prompt = f"ప్రశ్న: {test_query['text']}\n\nసంక్షిప్తమైన మరియు ఖచ్చితమైన సమాధానం ఇవ్వండి:"
                    elif detected_lang == "bengali":
                        prompt = f"প্রশ্ন: {test_query['text']}\n\nসংক্ষিপ্ত এবং সঠিক উত্তর দিন:"
                    else:
                        prompt = f"Question: {test_query['text']}\n\nProvide a concise and accurate answer:"
                    
                    logger.info(f"Generated prompt for {detected_lang}: {prompt[:100]}...")
                    
                    # Generate response
                    inputs = hf_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    
                    # Validate token IDs to prevent CUDA assertion errors
                    input_ids = inputs['input_ids']
                    vocab_size = hf_tokenizer.vocab_size
                    
                    # Check for invalid token IDs
                    if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                        logger.warning(f"Invalid token IDs detected for {detected_lang}. Vocab size: {vocab_size}, Min: {input_ids.min()}, Max: {input_ids.max()}")
                        # Filter out invalid tokens
                        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                        inputs['input_ids'] = input_ids
                    
                    # Move inputs to the same device as the model
                    if hasattr(model, 'device') and model.device.type != 'cpu':
                        inputs = inputs.to(model.device)
                    elif torch.cuda.is_available():
                        inputs = inputs.to("cuda")
                    
                    logger.info(f"Starting model generation for {detected_lang} with max_new_tokens: {min(request.max_response_length or 150, settings.MAX_GENERATION_LENGTH)}")
                    
                    with torch.no_grad():
                        try:
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=min(request.max_response_length or 150, settings.MAX_GENERATION_LENGTH),
                                temperature=0.7,  # Reduced temperature for stability
                                top_p=0.9,
                                top_k=40,  # Reduced top_k for stability
                                do_sample=True,
                                repetition_penalty=1.1,  # Reduced repetition penalty
                                no_repeat_ngram_size=2,  # Reduced n-gram size
                                pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                                eos_token_id=hf_tokenizer.eos_token_id,
                                early_stopping=True,
                                use_cache=True
                            )
                        except RuntimeError as e:
                            if "CUDA" in str(e) or "device-side assert" in str(e):
                                logger.error(f"CUDA error during generation for {detected_lang}: {e}")
                                # Fallback to CPU generation
                                logger.info(f"Falling back to CPU generation for {detected_lang} with conservative settings")
                                inputs = inputs.to("cpu")
                                model_cpu = model.to("cpu")
                                outputs = model_cpu.generate(
                                    **inputs,
                                    max_new_tokens=50,
                                    temperature=0.6,
                                    do_sample=True,
                                    pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                                    eos_token_id=hf_tokenizer.eos_token_id,
                                    early_stopping=True
                                )
                            else:
                                raise e
                    
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
def reload_model_cpu_only():
    """Reload model in CPU-only mode when CUDA fails"""
    global model, model_reloaded_cpu
    
    try:
        logger.info("🔄 Reloading model in CPU-only mode due to CUDA issues...")
        
        # Clear existing model
        if model is not None:
            del model
            torch.cuda.empty_cache()
        
        # Reload model with CPU-only configuration
        model_source = settings.MODEL_PATH if settings.MODEL_PATH else settings.MODEL_NAME
        
        # Check if we have a LoRA adapter
        adapter_path = None
        base_model_name = None
        
        if os.path.exists(model_source) and os.path.exists(os.path.join(model_source, "adapter_config.json")):
            adapter_path = model_source
            import json
            with open(os.path.join(adapter_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "bigscience/bloom-560m")
        else:
            base_model_name = model_source
        
        # Load base model on CPU only
        model_kwargs = {
            "torch_dtype": torch.float32,  # Use float32 for CPU
            "device_map": "cpu",  # Force CPU
        }
        
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        
        # Load LoRA adapter if available
        if adapter_path:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info(f"✅ LoRA adapter loaded on CPU from {adapter_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load LoRA adapter on CPU: {e}")
        
        model.eval()
        model_reloaded_cpu = True
        logger.info("✅ Model successfully reloaded in CPU-only mode")
        
    except Exception as e:
        logger.error(f"❌ Failed to reload model in CPU mode: {e}")
        model = None

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
        
        # Check if we have a LoRA adapter
        adapter_path = None
        base_model_name = None
        
        if os.path.exists(model_source) and os.path.exists(os.path.join(model_source, "adapter_config.json")):
            # This is a LoRA adapter directory
            adapter_path = model_source
            # Read the base model from adapter config
            import json
            with open(os.path.join(adapter_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "bigscience/bloom-560m")
            logger.info(f"🔧 Detected LoRA adapter at {adapter_path}")
            logger.info(f"🔧 Base model: {base_model_name}")
        else:
            # This is a regular model
            base_model_name = model_source
        
        # Load tokenizer from adapter if available, otherwise from base model
        if adapter_path:
            hf_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        else:
            hf_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Check if CPU-only mode is forced
        force_cpu_only = os.getenv("FORCE_CPU_GENERATION", "false").lower() == "true"
        
        # Configure quantization if enabled (but not in CPU-only mode)
        quantization_config = None
        if settings.USE_4BIT_QUANTIZATION and torch.cuda.is_available() and not force_cpu_only:
            try:
                quantization_config = BitsAndBytesConfig(**settings.QUANTIZATION_CONFIG)
                logger.info("🔧 Using 4-bit quantization for faster inference")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create quantization config: {e}")
                logger.info("🔄 Falling back to standard model loading")
                quantization_config = None
        
        # Load base model with or without quantization
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() and settings.USE_FP16_IF_GPU and not force_cpu_only else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"  # Let bitsandbytes handle device placement
        elif force_cpu_only:
            model_kwargs["device_map"] = "cpu"  # Force CPU loading
            logger.info("🔧 Forcing CPU-only model loading")
        
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        
        # Load LoRA adapter if available
        if adapter_path:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
                logger.info(f"✅ LoRA adapter loaded from {adapter_path}")
            except ImportError:
                logger.error("❌ PEFT library not found. Cannot load LoRA adapter.")
                logger.info("Install with: pip install peft")
                raise Exception("PEFT library required for LoRA adapter loading")
            except Exception as e:
                logger.error(f"❌ Failed to load LoRA adapter: {e}")
                raise e
        
        model.eval()
        
        # Move to GPU only if not using quantization and not forcing CPU-only mode
        if torch.cuda.is_available() and not quantization_config and not force_cpu_only:
            model.to("cuda")
            logger.info(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
        elif quantization_config:
            logger.info("✅ Model loaded with 4-bit quantization")
        elif force_cpu_only:
            logger.info("✅ Model loaded on CPU (forced CPU-only mode)")
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
    """Enhanced language detection for 20+ Indian languages based on Unicode ranges + keywords"""
    logger.debug(f"Detecting language for text: '{text[:50]}...'")

    # Count characters in each script
    script_counts = {}
    total_chars = 0
    
    for char in text:
        if char.isalpha():
            total_chars += 1
            for script, (start, end) in settings.UNICODE_RANGES.items():
                if start <= ord(char) <= end:
                    script_counts[script] = script_counts.get(script, 0) + 1
                    break

    if total_chars == 0:
        return settings.DEFAULT_LANGUAGE, 0.0

    # Calculate ratios for each script
    script_ratios = {script: count / total_chars for script, count in script_counts.items()}
    
    # Check for English first (highest priority)
    english_ratio = script_ratios.get("latin", 0)
    if english_ratio > settings.ENGLISH_RATIO_THRESHOLD:
        return "english", english_ratio

    # Check for Devanagari-based languages
    devanagari_ratio = script_ratios.get("devanagari", 0)
    if devanagari_ratio > settings.DEVANAGARI_RATIO_THRESHOLD:
        # Use keyword matching to distinguish between Devanagari languages
        # Check Sanskrit first (highest priority for classical text)
        if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("sanskrit", [])):
            return "sanskrit", min(devanagari_ratio + 0.3, 1.0)
        
        # Check other Devanagari languages
        for lang, keywords in settings.LANGUAGE_KEYWORDS.items():
            if lang in ["hindi", "marathi", "nepali", "konkani", "bodo", "dogri", "maithili"]:
                if any(kw in text for kw in keywords):
                    return lang, min(devanagari_ratio + 0.2, 1.0)
        return "hindi", devanagari_ratio  # Default to Hindi for Devanagari

    # Check for other scripts
    for script, ratio in script_ratios.items():
        if ratio > 0.3:  # Threshold for script detection
            if script == "tamil":
                # Check Tamil keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("tamil", [])):
                    return "tamil", ratio
            elif script == "telugu":
                # Check Telugu keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("telugu", [])):
                    return "telugu", ratio
            elif script == "kannada":
                # Check Kannada keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("kannada", [])):
                    return "kannada", ratio
            elif script == "bengali":
                # Check Bengali/Assamese keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("bengali", [])):
                    return "bengali", ratio
                elif any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("assamese", [])):
                    return "assamese", ratio
            elif script == "gujarati":
                # Check Gujarati keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("gujarati", [])):
                    return "gujarati", ratio
            elif script == "punjabi":
                # Check Punjabi keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("punjabi", [])):
                    return "punjabi", ratio
            elif script == "odia":
                # Check Odia keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("odia", [])):
                    return "odia", ratio
            elif script == "malayalam":
                # Check Malayalam keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("malayalam", [])):
                    return "malayalam", ratio
            elif script == "urdu":
                # Check Urdu keywords
                if any(kw in text for kw in settings.LANGUAGE_KEYWORDS.get("urdu", [])):
                    return "urdu", ratio

    # Fallback: check all language keywords
    for lang, keywords in settings.LANGUAGE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return lang, 0.6  # Medium confidence for keyword match

    # If no script or keyword match, return the most likely script
    if script_ratios:
        most_likely_script = max(script_ratios.items(), key=lambda x: x[1])
        if most_likely_script[1] > settings.LANGUAGE_CONFIDENCE_THRESHOLD:
            # Map script to default language
            script_to_lang = {
                "devanagari": "hindi",
                "tamil": "tamil",
                "telugu": "telugu",
                "kannada": "kannada",
                "bengali": "bengali",
                "gujarati": "gujarati",
                "punjabi": "punjabi",
                "odia": "odia",
                "malayalam": "malayalam",
                "urdu": "urdu",
                "latin": "english"
            }
            return script_to_lang.get(most_likely_script[0], settings.DEFAULT_LANGUAGE), most_likely_script[1]

    return settings.DEFAULT_LANGUAGE, 0.0

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
        inputs = hf_tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Validate token IDs to prevent CUDA assertion errors
        input_ids = inputs['input_ids']
        vocab_size = hf_tokenizer.vocab_size
        
        # Check for invalid token IDs
        if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
            logger.warning(f"Invalid token IDs detected. Vocab size: {vocab_size}, Min: {input_ids.min()}, Max: {input_ids.max()}")
            # Filter out invalid tokens
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            inputs['input_ids'] = input_ids
        
        # Move inputs to the same device as the model
        # For quantized models, device_map="auto" handles placement automatically
        if hasattr(model, 'device') and model.device.type != 'cpu':
            inputs = inputs.to(model.device)
        elif torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(settings.MAX_GENERATION_LENGTH, 100),  # Reduce max tokens to avoid memory issues
                    temperature=settings.TEMPERATURE,
                    top_p=settings.TOP_P,
                    do_sample=settings.DO_SAMPLE,
                    num_return_sequences=1,  # Reduce to 1 to avoid memory issues
                    pad_token_id=hf_tokenizer.eos_token_id,  # Ensure proper padding
                    repetition_penalty=1.1,  # Add repetition penalty
                    no_repeat_ngram_size=2,  # Prevent repetition
                    early_stopping=True  # Stop early if possible
                )
            except RuntimeError as e:
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    logger.error(f"CUDA error during generation: {e}")
                    # Fallback to CPU generation
                    logger.info("Falling back to CPU generation")
                    inputs = inputs.to("cpu")
                    model_cpu = model.to("cpu")
                    outputs = model_cpu.generate(
                        **inputs,
                        max_new_tokens=50,  # Very conservative for CPU
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=hf_tokenizer.eos_token_id
                    )
                else:
                    raise e
        generated_text = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return GenerateResponse(language=detected_lang, generated_text=generated_text, input_text=request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {e}")

@app.get("/config")
async def get_configuration():
    global cuda_generation_disabled
    return {
        "api": settings.get_api_config(),
        "model": settings.get_model_config(),
        "tokenizer": settings.get_tokenizer_config(),
        "languages": {
            "supported": settings.SUPPORTED_LANGUAGES,
            "default": settings.DEFAULT_LANGUAGE
        },
        "kb_endpoint": settings.KB_ENDPOINT,
        "vaani_endpoint": settings.VAANI_ENDPOINT,
        "generation": {
            "cuda_generation_disabled": cuda_generation_disabled,
            "force_cpu_only": os.getenv("FORCE_CPU_GENERATION", "false").lower() == "true"
        }
    }

@app.post("/reset-generation")
async def reset_generation_status():
    """Reset CUDA generation status after fixing issues"""
    global cuda_generation_disabled, model_reloaded_cpu
    cuda_generation_disabled = False
    model_reloaded_cpu = False
    torch.cuda.empty_cache()  # Clear CUDA cache
    
    # Reload models normally (with GPU if available)
    load_models()
    
    return {
        "message": "CUDA generation status reset and models reloaded",
        "cuda_generation_disabled": False,
        "model_reloaded_cpu": False,
        "cuda_cache_cleared": True,
        "models_reloaded": True
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
