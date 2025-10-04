# Indigenous NLP + Vaani TTS Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the Multilingual Tokenization Model with Indigenous NLP (Nisarg) and Vaani TTS (Karthikeya) systems. The integration enables seamless multilingual text processing, generation, and voice synthesis across 20+ Indian languages.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [API Integration](#api-integration)
5. [Indigenous NLP Integration](#indigenous-nlp-integration)
6. [Vaani TTS Integration](#vaani-tts-integration)
7. [Complete Pipeline Integration](#complete-pipeline-integration)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Language        │───▶│  Multilingual   │
│   (Text/Audio)  │    │  Detection       │    │  Tokenizer      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Vaani TTS     │◀───│  Response        │◀───│  Language       │
│   (Karthikeya)  │    │  Generation      │    │  Model          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐
│   Audio Output  │◀───│  Indigenous NLP  │
│   (Voice)       │    │  (Nisarg)        │
└─────────────────┘    └──────────────────┘
```

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 20GB+ disk space

### Dependencies
```bash
# Core dependencies
pip install fastapi uvicorn torch transformers
pip install sentencepiece peft bitsandbytes
pip install aiohttp redis nginx

# TTS dependencies
pip install TTS gTTS pyttsx3

# NLP dependencies
pip install indic-nlp-library sacremoses nltk
```

### External Services
- **Indigenous NLP (Nisarg)**: Text preprocessing and analysis
- **Vaani TTS (Karthikeya)**: Text-to-speech synthesis
- **Redis**: Caching layer (optional)
- **Nginx**: Load balancing (optional)

## Installation & Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Multilingual-Tokenization-Model-Integration
pip install -r requirements.txt
```

### 2. Configure Settings
Edit `config/settings.py`:
```python
# TTS Integration Settings
VAANI_ENDPOINT = "http://localhost:8001"  # Vaani TTS service
VAANI_TIMEOUT = 120.0

# Indigenous NLP Settings
INDIGENOUS_NLP_ENDPOINT = "http://localhost:8002"  # Nisarg service
INDIGENOUS_NLP_TIMEOUT = 60.0

# KB Integration
KB_ENDPOINT = "http://localhost:8003"  # Knowledge Base service
KB_TIMEOUT = 120.0
```

### 3. Start Services
```bash
# Start the multilingual API
python main.py

# Start Vaani TTS (in separate terminal)
python src/services/vaani_tts_service.py

# Start Indigenous NLP (in separate terminal)
python src/services/indigenous_nlp_service.py
```

## API Integration

### Core Endpoints

#### 1. Language Detection
```python
import requests

def detect_language(text):
    response = requests.post(
        "http://localhost:8000/language-detect",
        json={"text": text}
    )
    return response.json()

# Example
result = detect_language("नमस्ते, आप कैसे हैं?")
print(f"Language: {result['language']}, Confidence: {result['confidence']}")
```

#### 2. Text Tokenization
```python
def tokenize_text(text, language=None):
    response = requests.post(
        "http://localhost:8000/tokenize",
        json={"text": text, "language": language}
    )
    return response.json()

# Example
result = tokenize_text("Hello, how are you?", "english")
print(f"Tokens: {result['tokens']}")
```

#### 3. Text Generation
```python
def generate_text(prompt, language=None, max_length=256):
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "text": prompt,
            "language": language,
            "max_length": max_length
        }
    )
    return response.json()

# Example
result = generate_text("Tell me about India", "hindi")
print(f"Generated: {result['generated_text']}")
```

#### 4. Multilingual Conversation
```python
def multilingual_conversation(text, session_id=None, user_id=None):
    response = requests.post(
        "http://localhost:8000/multilingual-conversation",
        json={
            "text": text,
            "session_id": session_id,
            "user_id": user_id,
            "generate_response": True
        }
    )
    return response.json()

# Example
result = multilingual_conversation(
    "भारत के बारे में बताइए",
    session_id="user123_session456"
)
print(f"Response: {result['generated_response']}")
```

## Indigenous NLP Integration

### 1. Text Preprocessing
```python
class IndigenousNLPIntegration:
    def __init__(self, endpoint="http://localhost:8002"):
        self.endpoint = endpoint
    
    def preprocess_text(self, text, language):
        """Preprocess text using Indigenous NLP (Nisarg)"""
        response = requests.post(
            f"{self.endpoint}/preprocess",
            json={
                "text": text,
                "language": language,
                "normalize": True,
                "clean": True
            }
        )
        return response.json()
    
    def analyze_sentiment(self, text, language):
        """Analyze sentiment using Indigenous NLP"""
        response = requests.post(
            f"{self.endpoint}/sentiment",
            json={"text": text, "language": language}
        )
        return response.json()
    
    def extract_entities(self, text, language):
        """Extract named entities"""
        response = requests.post(
            f"{self.endpoint}/entities",
            json={"text": text, "language": language}
        )
        return response.json()

# Usage
nlp = IndigenousNLPIntegration()
preprocessed = nlp.preprocess_text("नमस्ते, आप कैसे हैं?", "hindi")
```

### 2. Language-Specific Processing
```python
def process_multilingual_text(text, target_language="hindi"):
    """Complete text processing pipeline"""
    
    # Step 1: Detect language
    lang_result = detect_language(text)
    detected_lang = lang_result['language']
    
    # Step 2: Preprocess with Indigenous NLP
    nlp = IndigenousNLPIntegration()
    preprocessed = nlp.preprocess_text(text, detected_lang)
    
    # Step 3: Generate response
    response = generate_text(
        preprocessed['processed_text'],
        language=target_language
    )
    
    return {
        "original_text": text,
        "detected_language": detected_lang,
        "preprocessed_text": preprocessed['processed_text'],
        "generated_response": response['generated_text'],
        "target_language": target_language
    }
```

## Vaani TTS Integration

### 1. Text-to-Speech Service
```python
class VaaniTTSIntegration:
    def __init__(self, endpoint="http://localhost:8001"):
        self.endpoint = endpoint
    
    def synthesize_speech(self, text, language, voice=None):
        """Convert text to speech using Vaani TTS"""
        response = requests.post(
            f"{self.endpoint}/synthesize",
            json={
                "text": text,
                "language": language,
                "voice": voice,
                "format": "wav",
                "sample_rate": 22050
            }
        )
        return response.json()
    
    def get_available_voices(self, language):
        """Get available voices for a language"""
        response = requests.get(
            f"{self.endpoint}/voices/{language}"
        )
        return response.json()
    
    def stream_speech(self, text, language, chunk_size=1024):
        """Stream speech synthesis for long texts"""
        response = requests.post(
            f"{self.endpoint}/stream",
            json={"text": text, "language": language},
            stream=True
        )
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            yield chunk

# Usage
tts = VaaniTTSIntegration()
audio_result = tts.synthesize_speech("नमस्ते, आप कैसे हैं?", "hindi")
```

### 2. Multilingual TTS Pipeline
```python
def multilingual_tts_pipeline(text, target_language="hindi"):
    """Complete multilingual TTS pipeline"""
    
    # Step 1: Process text
    processed = process_multilingual_text(text, target_language)
    
    # Step 2: Synthesize speech
    tts = VaaniTTSIntegration()
    audio_result = tts.synthesize_speech(
        processed['generated_response'],
        target_language
    )
    
    return {
        "text_response": processed['generated_response'],
        "audio_url": audio_result['audio_url'],
        "language": target_language,
        "duration": audio_result['duration'],
        "file_size": audio_result['file_size']
    }
```

## Complete Pipeline Integration

### 1. End-to-End Integration Class
```python
class CompleteMultilingualPipeline:
    def __init__(self):
        self.api_endpoint = "http://localhost:8000"
        self.tts_endpoint = "http://localhost:8001"
        self.nlp_endpoint = "http://localhost:8002"
        self.kb_endpoint = "http://localhost:8003"
    
    def process_user_input(self, text, user_id=None, session_id=None):
        """Complete pipeline: Input → Processing → Response → Audio"""
        
        # Step 1: Language detection and preprocessing
        lang_result = detect_language(text)
        detected_lang = lang_result['language']
        
        # Step 2: Indigenous NLP preprocessing
        nlp = IndigenousNLPIntegration(self.nlp_endpoint)
        preprocessed = nlp.preprocess_text(text, detected_lang)
        
        # Step 3: Knowledge Base query
        kb_response = requests.post(
            f"{self.api_endpoint}/qa",
            json={
                "text": preprocessed['processed_text'],
                "language": detected_lang,
                "user_id": user_id,
                "session_id": session_id
            }
        ).json()
        
        # Step 4: Generate enhanced response
        if kb_response.get('generated_response'):
            final_text = kb_response['generated_response']
        else:
            final_text = kb_response['answer']
        
        # Step 5: Synthesize speech
        tts = VaaniTTSIntegration(self.tts_endpoint)
        audio_result = tts.synthesize_speech(final_text, detected_lang)
        
        return {
            "user_input": text,
            "detected_language": detected_lang,
            "text_response": final_text,
            "audio_url": audio_result['audio_url'],
            "session_id": session_id,
            "processing_time": kb_response.get('processing_time', 0),
            "confidence": lang_result['confidence']
        }
    
    def handle_conversation(self, messages, user_id, session_id):
        """Handle multi-turn conversations"""
        responses = []
        
        for message in messages:
            result = self.process_user_input(
                message['text'],
                user_id=user_id,
                session_id=session_id
            )
            responses.append(result)
        
        return responses

# Usage
pipeline = CompleteMultilingualPipeline()

# Single interaction
result = pipeline.process_user_input(
    "भारत के बारे में बताइए",
    user_id="user123",
    session_id="session456"
)

print(f"Response: {result['text_response']}")
print(f"Audio: {result['audio_url']}")

# Multi-turn conversation
conversation = [
    {"text": "नमस्ते, आप कैसे हैं?"},
    {"text": "भारत के बारे में बताइए"},
    {"text": "धन्यवाद"}
]

responses = pipeline.handle_conversation(
    conversation,
    user_id="user123",
    session_id="session456"
)
```

### 2. WebSocket Integration for Real-time
```python
import asyncio
import websockets
import json

class RealTimeMultilingualPipeline:
    def __init__(self):
        self.pipeline = CompleteMultilingualPipeline()
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections for real-time processing"""
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data['type'] == 'text_input':
                    result = self.pipeline.process_user_input(
                        data['text'],
                        user_id=data.get('user_id'),
                        session_id=data.get('session_id')
                    )
                    
                    await websocket.send(json.dumps({
                        'type': 'response',
                        'data': result
                    }))
                
                elif data['type'] == 'language_switch':
                    # Handle mid-conversation language switching
                    result = self.pipeline.process_user_input(
                        data['text'],
                        user_id=data.get('user_id'),
                        session_id=data.get('session_id')
                    )
                    
                    await websocket.send(json.dumps({
                        'type': 'language_switched',
                        'data': result
                    }))
                
            except Exception as e:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': str(e)
                }))

# Start WebSocket server
start_server = websockets.serve(
    RealTimeMultilingualPipeline().handle_websocket,
    "localhost", 8765
)

asyncio.get_event_loop().run_until_complete(start_server)
```

## Testing & Validation

### 1. Unit Tests
```python
import unittest
import requests

class TestMultilingualIntegration(unittest.TestCase):
    def setUp(self):
        self.api_url = "http://localhost:8000"
        self.pipeline = CompleteMultilingualPipeline()
    
    def test_language_detection(self):
        """Test language detection accuracy"""
        test_cases = [
            ("नमस्ते, आप कैसे हैं?", "hindi"),
            ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "tamil"),
            ("Hello, how are you?", "english")
        ]
        
        for text, expected_lang in test_cases:
            result = detect_language(text)
            self.assertEqual(result['language'], expected_lang)
            self.assertGreater(result['confidence'], 0.7)
    
    def test_multilingual_conversation(self):
        """Test multilingual conversation handling"""
        result = self.pipeline.process_user_input(
            "भारत के बारे में बताइए",
            user_id="test_user",
            session_id="test_session"
        )
        
        self.assertIsNotNone(result['text_response'])
        self.assertIsNotNone(result['audio_url'])
        self.assertEqual(result['detected_language'], 'hindi')
    
    def test_language_switching(self):
        """Test mid-conversation language switching"""
        # Start with Hindi
        result1 = self.pipeline.process_user_input(
            "नमस्ते", user_id="test", session_id="test"
        )
        
        # Switch to Tamil
        result2 = self.pipeline.process_user_input(
            "வணக்கம்", user_id="test", session_id="test"
        )
        
        self.assertEqual(result1['detected_language'], 'hindi')
        self.assertEqual(result2['detected_language'], 'tamil')

if __name__ == '__main__':
    unittest.main()
```

### 2. Integration Tests
```bash
# Test API endpoints
python -m pytest tests/test_api_integration.py

# Test TTS integration
python -m pytest tests/test_tts_integration.py

# Test complete pipeline
python -m pytest tests/test_complete_pipeline.py
```

### 3. Load Testing
```python
import asyncio
import aiohttp
import time

async def load_test_api(concurrent_users=100, requests_per_user=10):
    """Load test the multilingual API"""
    
    async def make_request(session, user_id):
        """Make a single request"""
        async with session.post(
            "http://localhost:8000/multilingual-conversation",
            json={
                "text": "Test message",
                "user_id": f"user_{user_id}",
                "session_id": f"session_{user_id}"
            }
        ) as response:
            return await response.json()
    
    async def user_simulation(session, user_id):
        """Simulate a user making multiple requests"""
        results = []
        for i in range(requests_per_user):
            start_time = time.time()
            result = await make_request(session, user_id)
            end_time = time.time()
            
            results.append({
                "user_id": user_id,
                "request_id": i,
                "response_time": end_time - start_time,
                "success": "error" not in result
            })
        
        return results
    
    # Run load test
    async with aiohttp.ClientSession() as session:
        tasks = [
            user_simulation(session, i) 
            for i in range(concurrent_users)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        all_results = [r for user_results in results for r in user_results]
        success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
        avg_response_time = sum(r['response_time'] for r in all_results) / len(all_results)
        
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        print(f"Total Requests: {len(all_results)}")

# Run load test
asyncio.run(load_test_api(concurrent_users=50, requests_per_user=5))
```

## Troubleshooting

### Common Issues

#### 1. Language Detection Issues
```python
# Debug language detection
def debug_language_detection(text):
    from src.api.main import detect_language
    
    lang, conf = detect_language(text)
    print(f"Text: {text}")
    print(f"Detected: {lang} (confidence: {conf})")
    
    # Check Unicode ranges
    for char in text:
        if char.isalpha():
            print(f"Character '{char}': U+{ord(char):04X}")

# Example
debug_language_detection("नमस्ते, आप कैसे हैं?")
```

#### 2. TTS Integration Issues
```python
# Test TTS connectivity
def test_tts_connection():
    try:
        response = requests.get("http://localhost:8001/health")
        print(f"TTS Service Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("TTS Service not available")

# Test voice availability
def test_voice_availability():
    tts = VaaniTTSIntegration()
    voices = tts.get_available_voices("hindi")
    print(f"Available Hindi voices: {voices}")
```

#### 3. Memory Issues
```python
# Monitor memory usage
import psutil
import torch

def monitor_memory():
    """Monitor system and GPU memory usage"""
    # System memory
    memory = psutil.virtual_memory()
    print(f"System Memory: {memory.percent}% used")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_reserved:.2f}GB reserved")

# Call periodically
monitor_memory()
```

### Performance Optimization

#### 1. Caching
```python
import redis
import json

class CachedMultilingualPipeline(CompleteMultilingualPipeline):
    def __init__(self, redis_url="redis://localhost:6379"):
        super().__init__()
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = 3600  # 1 hour
    
    def process_user_input(self, text, user_id=None, session_id=None):
        # Check cache first
        cache_key = f"pipeline:{hash(text)}:{user_id}:{session_id}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # Process normally
        result = super().process_user_input(text, user_id, session_id)
        
        # Cache result
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(result)
        )
        
        return result
```

#### 2. Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncMultilingualPipeline:
    def __init__(self):
        self.pipeline = CompleteMultilingualPipeline()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_user_input_async(self, text, user_id=None, session_id=None):
        """Async version of process_user_input"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.pipeline.process_user_input,
            text, user_id, session_id
        )
        return result
    
    async def batch_process(self, texts, user_id=None, session_id=None):
        """Process multiple texts concurrently"""
        tasks = [
            self.process_user_input_async(text, user_id, session_id)
            for text in texts
        ]
        return await asyncio.gather(*tasks)

# Usage
async def main():
    pipeline = AsyncMultilingualPipeline()
    
    texts = [
        "नमस्ते, आप कैसे हैं?",
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "Hello, how are you?"
    ]
    
    results = await pipeline.batch_process(texts)
    for result in results:
        print(f"Response: {result['text_response']}")

asyncio.run(main())
```

## Deployment

### Docker Compose Setup
```yaml
# docker-compose.integration.yml
version: '3.8'
services:
  multilingual-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - VAANI_ENDPOINT=http://vaani-tts:8001
      - INDIGENOUS_NLP_ENDPOINT=http://indigenous-nlp:8002
      - KB_ENDPOINT=http://knowledge-base:8003
    depends_on:
      - redis
      - vaani-tts
      - indigenous-nlp
      - knowledge-base
  
  vaani-tts:
    image: vaani-tts:latest
    ports:
      - "8001:8001"
    volumes:
      - ./audio_cache:/app/cache
  
  indigenous-nlp:
    image: indigenous-nlp:latest
    ports:
      - "8002:8002"
  
  knowledge-base:
    image: knowledge-base:latest
    ports:
      - "8003:8003"
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - multilingual-api
```

### Production Deployment
```bash
# Deploy with Docker Compose
docker-compose -f docker-compose.integration.yml up -d

# Scale services
docker-compose up --scale multilingual-api=3

# Monitor services
docker-compose logs -f multilingual-api
```

## Conclusion

This integration guide provides comprehensive instructions for connecting the Multilingual Tokenization Model with Indigenous NLP and Vaani TTS systems. The integration enables:

- **Seamless multilingual processing** across 20+ Indian languages
- **Real-time voice synthesis** with language-specific TTS
- **Advanced text preprocessing** using Indigenous NLP
- **Scalable deployment** with Docker and load balancing
- **Production-ready performance** with caching and optimization

For additional support or questions, please refer to the API documentation at `http://localhost:8000/docs` or open an issue in the repository.

---

**Last Updated**: December 2024  
**Version**: 2.0  
**Compatibility**: Python 3.8+, CUDA 11.0+
